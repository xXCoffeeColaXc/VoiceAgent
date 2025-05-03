import os, re
import asyncio
from typing import Optional, Dict, List
from langchain.schema import Document
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_community.vectorstores.pgvector import PGVector
from langchain.chains.retrieval_qa.base import RetrievalQA
from livekit.agents import function_tool, RunContext
from db import get_db_connection_str
from tools.arxiv_tools import download_arxiv_pdf

COLLECTION_NAME = "documents"


def preprocess_text(text: str) -> str:
    text = text.replace("\x00", "")
    text = re.sub(r"[\x01-\x08\x0B-\x0C\x0E-\x1F\x7F-\x9F]+", "", text)
    text = text.replace("\ufffd", "")
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def clear_document(doc: Document) -> Document:
    doc.page_content = preprocess_text(doc.page_content)
    return doc


def ingest_document_to_pgvector(
    file_path: str,
    doc_id: str,
    chunk_size: int = 1000,
    chunk_overlap: int = 200,
    embedding_model: str = "text-embedding-3-small",
) -> PGVector:
    # 1) Load & chunk
    loader = PyPDFLoader(file_path)
    docs = loader.load()
    splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    chunks = splitter.split_documents(docs)

    # 2) Clean & inject doc_id
    for i, chunk in enumerate(chunks):
        chunk = clear_document(chunk)
        chunk.metadata["doc_id"] = doc_id
        chunk.metadata["chunk_id"] = f"{doc_id}_{i:04d}"

    # 3) Embed & store (dropping old collection if re-ingest)
    embeddings = OpenAIEmbeddings(model=embedding_model)
    store = PGVector.from_documents(
        documents=chunks,
        embedding=embeddings,
        connection_string=get_db_connection_str(),
        collection_name=COLLECTION_NAME,
        pre_delete_collection=True,  # ← wipe prior chunks for this collection
        use_jsonb=True,
    )
    return store


@function_tool(
    name="ingest_document",
    description=("Fetch a PDF from arXiv (by `arxiv_id` or `pdf_url`), then chunk, embed, "
                 "and store it in the vector database under a `doc_id`."),
)
async def ingest_document(
    context: RunContext,
    arxiv_id: Optional[str] = None,
    pdf_url: Optional[str] = None,
) -> Dict[str, str]:
    # download
    result = await download_arxiv_pdf(context, arxiv_id=arxiv_id, pdf_url=pdf_url)
    file_path = result["file_path"]
    doc_id = arxiv_id or os.path.splitext(os.path.basename(file_path))[0]

    # ingest in background
    await asyncio.to_thread(ingest_document_to_pgvector, file_path, doc_id)

    return {"doc_id": doc_id, "status": "ingested"}


@function_tool(
    name="query_document",
    description=("Given a user question, retrieve relevant text chunks from the PGVector store "
                 "and answer using the chat LLM. Returns a short, direct answer."),
)
async def query_document(
    context: RunContext,
    query: str,
    k: Optional[int] = None,
) -> Dict[str, str]:
    k = k or 4
    embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
    store = PGVector.from_existing_index(
        embedding=embeddings,
        collection_name=COLLECTION_NAME,
        connection_string=get_db_connection_str(),
    )
    retriever = store.as_retriever(search_kwargs={"k": k})
    llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)
    qa_chain = RetrievalQA.from_chain_type(llm=llm,
                                           chain_type="stuff",
                                           retriever=retriever,
                                           return_source_documents=False)
    answer = await asyncio.to_thread(qa_chain.run, query)
    return {"answer": answer}
