# tools/arxiv_tools.py
import aiohttp, feedparser, urllib.parse
from livekit.agents import function_tool, RunContext
from typing import Optional, List
import re

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
