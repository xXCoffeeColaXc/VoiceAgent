import aiohttp, feedparser, urllib.parse
from livekit.agents import function_tool, RunContext
from typing import Optional, List, Dict
import re
import arxiv
import asyncio
import os
import tempfile

ARXIV_API = "http://export.arxiv.org/api/query"


def preprocess_text(text: str):
    # Remove inline asterisks and their content
    text = re.sub(r"\*[^\*]*\*", "", text)
    # Remove leftover lone asterisks
    text = re.sub(r"\*+", "", text)
    # remove non-alphanumeric symbols at the beginning of lines
    text = re.sub(r"^\W+", "", text, flags=re.MULTILINE)

    return text


@function_tool(
    name="search_arxiv",
    description=("Search arXiv and return up to N results with title, authors, "
                 "summary, published date, PDF link, and arXiv ID."
                 ""
                 "Use when the user asks for research papers."),
)
async def search_arxiv(
    context: RunContext,
    query: str,
    max_results: Optional[int] = None,  # <- optional, *no default in schema*
    sort_by: Optional[str] = None,
    sort_order: Optional[str] = None,
) -> List[dict]:
    # ---- fallback values implemented here, *after* the schema check ----
    max_results = max_results or 3
    sort_by = sort_by or "submittedDate"
    sort_order = sort_order or "descending"

    url = (f"{ARXIV_API}?search_query={urllib.parse.quote_plus(query)}"
           f"&max_results={max_results}&sortBy={sort_by}&sortOrder={sort_order}")

    async with aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=20)) as sess:
        async with sess.get(url) as resp:
            xml = await resp.text()

    feed = feedparser.parse(xml)
    out = []
    for i, entry in enumerate(feed.entries, start=1):
        pdf = next((l.href for l in entry.links if l.type == "application/pdf"), None)
        out.append({
            "id": entry.id.split("/abs/")[-1],
            "title": f"{i}. {preprocess_text(entry.title)}",
            "authors": [a.name for a in getattr(entry, "authors", [])],
            "summary": preprocess_text(entry.summary),
            "published": entry.published,
            "pdf_url": pdf,
        })
    return out


@function_tool(
    name="download_arxiv_pdf",
    description=("Download an arXiv paper PDF. "
                 "Provide either `arxiv_id` (e.g. '2404.12345v3') or `pdf_url`. "
                 "Returns the local file path."),
)
async def download_arxiv_pdf(
    context: RunContext,
    arxiv_id: Optional[str] = None,
    pdf_url: Optional[str] = None,
) -> Dict[str, str]:
    # 1. Resolve PDF URL if only arxiv_id was provided
    if arxiv_id and not pdf_url:
        client = arxiv.Client()  # instantiate the arXiv client
        search = arxiv.Search(id_list=[arxiv_id])
        paper = next(client.results(search))
        # offload the blocking download to a thread to keep the event loop free
        pdf_path = await asyncio.to_thread(paper.download_pdf)  # auto-saves to temp file
    else:
        pdf_path = await asyncio.to_thread(
            lambda: arxiv.Client()._download(pdf_url)  # placeholder if downloading via URL
        )

    # 2. Move to our own temp directory for consistency
    tmp_dir = tempfile.gettempdir()  # use host temp dir
    file_name = os.path.basename(pdf_path)
    local_path = os.path.join(tmp_dir, file_name)

    # 3. Copy into final location (binary-safe)
    with open(pdf_path, "rb") as src, open(local_path, "wb") as dst:
        dst.write(src.read())  # write in binary mode

    return {"file_path": local_path}
