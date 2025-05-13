import aiohttp, feedparser, urllib.parse
from livekit.agents import function_tool, RunContext
from typing import Optional, List, Dict
import re
import arxiv
import asyncio
import os
import tempfile

ARXIV_API = "http://export.arxiv.org/api/query"


def _clean_markdown(text: str) -> str:
    # strip *…* and any leading symbols
    text = re.sub(r"\*[^\*]*\*", "", text)
    text = re.sub(r"\*+", "", text)
    text = re.sub(r"http[s]?://\S+", "", text)
    text = re.sub(r"^\W+", "", text, flags=re.MULTILINE)
    return text.strip()


@function_tool(
    name="search_arxiv",
    description=("Search arXiv by keyword and return up to N results. "
                 "By default returns only `id` and `title` for each paper. "
                 "Set `titles_only=false` to include authors, summary, published date, and pdf_url."),
)
async def search_arxiv(
    context: RunContext,
    query: str,
    max_results: Optional[int] = None,
    sort_by: Optional[str] = None,
    sort_order: Optional[str] = None,
    titles_only: Optional[bool] = None,
) -> List[Dict]:
    # apply defaults
    max_results = max_results or 3
    sort_by = sort_by or "submittedDate"
    sort_order = sort_order or "descending"
    titles_only = True if titles_only is None else titles_only

    url = (f"{ARXIV_API}"
           f"?search_query={urllib.parse.quote_plus(query)}"
           f"&max_results={max_results}"
           f"&sortBy={sort_by}&sortOrder={sort_order}")

    print(f"Fetching arXiv search results from: {url}")
    async with aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=20)) as sess:
        async with sess.get(url) as resp:
            xml = await resp.text()

    feed = feedparser.parse(xml)
    results = []
    for i, entry in enumerate(feed.entries, start=1):
        paper_id = entry.id.split("/abs/")[-1]
        print(f"Found paper: {paper_id}")
        title = f"{i}. {_clean_markdown(entry.title)}"
        if titles_only:
            results.append({"id": paper_id, "title": title})
        else:
            pdf_url = next((l.href for l in entry.links if l.type == "application/pdf"), None)
            results.append({
                "id": paper_id,
                "title": title,
                "authors": [a.name for a in getattr(entry, "authors", [])],
                "summary": _clean_markdown(entry.summary),
                "published": entry.published,
                "pdf_url": pdf_url,
            })
    return results


@function_tool(
    name="download_arxiv_pdf",
    description=("Download the PDF of an arXiv paper given its `arxiv_id` or direct `pdf_url`. "
                 "Returns the local file path. Does not ingest into vector store."),
)
async def download_arxiv_pdf(
    context: RunContext,
    arxiv_id: Optional[str] = None,
    pdf_url: Optional[str] = None,
) -> Dict[str, str]:
    # 1) If we're given an arXiv ID, use the arxiv Client
    if arxiv_id and not pdf_url:
        client = arxiv.Client()
        search = arxiv.Search(id_list=[arxiv_id])
        paper = next(client.results(search))
        # download in background thread
        pdf_path = await asyncio.to_thread(paper.download_pdf)

    # 2) Otherwise, if they've supplied a direct PDF URL, fetch with aiohttp
    elif pdf_url:
        # derive a safe filename
        parsed = urllib.parse.urlparse(pdf_url)
        file_name = os.path.basename(parsed.path) or f"{arxiv_id or 'paper'}.pdf"
        tmp_dir = tempfile.gettempdir()
        local_path = os.path.join(tmp_dir, file_name)

        async with aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=30)) as sess:
            async with sess.get(pdf_url) as resp:
                resp.raise_for_status()
                # write in binary chunks
                with open(local_path, "wb") as f:
                    async for chunk in resp.content.iter_chunked(65536):
                        f.write(chunk)
        pdf_path = local_path

    else:
        raise ValueError("Must provide either `arxiv_id` or `pdf_url`")

    return {"file_path": pdf_path}
