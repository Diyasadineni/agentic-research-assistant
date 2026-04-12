from tavily import TavilyClient
from langchain.tools import tool
from dotenv import load_dotenv
import os

load_dotenv() 
tavily = TavilyClient(api_key=os.getenv("TAVILY_API_KEY"))

@tool
def search_web(query: str) -> str:
    """Search the web for a given query and return results."""
    results = tavily.search(query=query, max_results=5)
    output = []
    for r in results["results"]:
        output.append(f"Title: {r['title']}\nURL: {r['url']}\nSummary: {r['content']}\n")
    return "\n---\n".join(output)

@tool
def fetch_page(url: str) -> str:
    """Fetch the full content of a webpage."""
    result = tavily.extract(urls=[url])
    if result and result.get("results"):
        return result["results"][0].get("raw_content", "Could not fetch page.")
    return "Could not fetch page."