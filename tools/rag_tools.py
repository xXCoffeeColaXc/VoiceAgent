import os, re
import asyncio
from typing import Optional, Dict, List, Any
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

    meta0 = docs[0].metadata
    title = meta0.get("title") or os.path.basename(file_path)
    authors = meta0.get("author") or meta0.get("creator")
    published = meta0.get("moddate") or meta0.get("creationdate")
    total_pages = meta0.get("total_pages")

    splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    chunks = splitter.split_documents(docs)

    # 2) Clean & inject doc_id
    for i, chunk in enumerate(chunks):
        chunk = clear_document(chunk)
        chunk.metadata.update({
            "doc_id": doc_id,
            "chunk_id": f"{doc_id}_{i:04d}",
            "title": title,
            "authors": authors,
            "published": published,
            "total_pages": total_pages,
            "source": file_path,
        })

    # 3) Embed & store (dropping old collection if re-ingest)
    embeddings = OpenAIEmbeddings(model=embedding_model)
    store = PGVector.from_documents(
        documents=chunks,
        embedding=embeddings,
        connection_string=get_db_connection_str(),
        collection_name=COLLECTION_NAME,
        pre_delete_collection=False,  # ← wipe prior chunks for this collection
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
    description=("Given a user question (and optional doc_id), retrieve relevant text chunks from the PGVector store "
                 "and answer the question, and tell me which document(s) you used."),
)
async def query_document(
    context: RunContext,
    query: str,
    doc_id: Optional[str] = None,
    k: Optional[int] = None,
) -> Dict[str, Any]:
    k = k or 4

    # 1) Connect to pgvector
    embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
    store = PGVector.from_existing_index(
        embedding=embeddings,
        collection_name=COLLECTION_NAME,
        connection_string=get_db_connection_str(),
    )

    # 2) Build retriever (filter by doc_id if given)
    filter_dict = {"doc_id": doc_id} if doc_id else None
    retriever = store.as_retriever(search_kwargs={"k": k}, filter=filter_dict)

    # 3) Retrieval-QA chain
    llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)
    qa_chain = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=retriever,
        return_source_documents=True,
    )

    # 4) Run chain in thread
    res = await asyncio.to_thread(lambda: qa_chain.invoke({"query": query}))

    used_doc_ids = sorted({d.metadata.get("doc_id") for d in res["source_documents"] if d.metadata.get("doc_id")})

    return {
        "answer": res["result"],
        "source_doc_ids": used_doc_ids,
    }


@function_tool(
    name="list_documents",
    description="List all ingested papers with their doc_id and title.",
)
async def list_documents(context: RunContext) -> Dict[str, List[Dict]]:
    from sqlalchemy import create_engine, text

    eng = create_engine(get_db_connection_str())
    with eng.begin() as conn:
        rows = conn.execute(
            text("""
              SELECT DISTINCT
                     (cmetadata->>'doc_id')  AS doc_id,
                     (cmetadata->>'title')   AS title
              FROM   langchain_pg_embedding
              WHERE  (cmetadata->>'doc_id') IS NOT NULL
                AND  (cmetadata->>'title')  IS NOT NULL
              """)).fetchall()

    return {"documents": [{"doc_id": r[0], "title": r[1]} for r in rows]}


@function_tool(
    name="describe_document",
    description="Return metadata (title, authors, published date, pages) for a given doc_id.",
)
async def describe_document(context: RunContext, doc_id: str) -> Dict[str, str]:
    from sqlalchemy import create_engine, text

    eng = create_engine(get_db_connection_str())
    with eng.begin() as conn:
        row = conn.execute(
            text("""
              SELECT cmetadata
              FROM   langchain_pg_embedding
              WHERE  (cmetadata->>'doc_id') = :doc
              LIMIT 1
              """),
            {
                "doc": doc_id
            },
        ).scalar()

    if row is None:
        return {"error": f"doc_id {doc_id} not found"}

    meta = row  # JSONB dict
    return {
        "doc_id": doc_id,
        "title": meta.get("title"),
        "authors": meta.get("authors"),
        "published": meta.get("published"),
        "total_pages": meta.get("total_pages"),
        "source": meta.get("source"),
    }
