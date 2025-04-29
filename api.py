import requests, feedparser

BASE_URL = "http://export.arxiv.org/api/query"
""" API parameters:
Parameter	        | Purpose
search_query        | Boolean / keyword search across title, abstract, author, category, etc.
id_list             | Fetch specific papers by ID (2404.12345,2404.99999v2).
max_results         | Limit count (default = 10, max = 30000).
sortBy & sortOrder  | relevance, lastUpdatedDate, etc.
"""

query = "cat:cs.CL AND deep learning"  # Boolean / keyword search across title, abstract, author, category, etc.
max_results = 5
url = f"{BASE_URL}?search_query={query}&max_results={max_results}"
feed = feedparser.parse(requests.get(url, timeout=30).text)  # entry.title, entry.summary, entry.authors

for i, entry in enumerate(feed.entries):
    title = entry.title
    pdf_url = next(l.href for l in entry.links if l.type == "application/pdf")
    print(f"{i}. {title}\nPDF: {pdf_url}\n")

choice = int(input("Enter the number of the document you want to download: ")) - 1
if 0 <= choice < len(feed.entries):
    selected_entry = feed.entries[choice]
    pdf_url = next(l.href for l in selected_entry.links if l.type == "application/pdf")
    pdf_bytes = requests.get(pdf_url).content
    open("paper.pdf", "wb").write(pdf_bytes)
    print(f"Downloaded: {selected_entry.title}")
else:
    print("Invalid choice.")
