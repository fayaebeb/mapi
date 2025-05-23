import os
from typing import List, Optional, Dict, Any
import httpx

TAVILY_API_KEY = os.getenv("TAVILY_API_KEY")

def tavily_search(
    query: str,
    search_depth: str = "advanced",  # "basic" or "advanced"
    chunks_per_source: Optional[int] = 3,
    topic: str = "general",          # e.g., "general", "news", etc.
    days: Optional[int] = 7,
    max_results: int = 5,
    include_answer: bool = True,
    include_images: bool = True,
    time_range: Optional[str] = None,
    include_domains: Optional[List[str]] = None,
    exclude_domains: Optional[List[str]] = None,
    include_raw_content: bool = False
) -> Dict[str, Any]:
    payload = {
        "api_key": TAVILY_API_KEY,
        "query": query,
        "search_depth": search_depth,
        "topic": topic,
        "max_results": max_results,
        "include_images": include_images,
        "include_answer": include_answer,
        "include_raw_content": include_raw_content,
    }

    # optional parameters
    if search_depth == "advanced" and chunks_per_source is not None:
        payload["chunks_per_source"] = chunks_per_source
    if topic == "news" and days is not None:
        payload["days"] = days
    if time_range:
        payload["time_range"] = time_range
    if include_domains:
        payload["include_domains"] = include_domains
    if exclude_domains:
        payload["exclude_domains"] = exclude_domains

    try:
        with httpx.Client(timeout=90.0) as client:
            response = client.post("https://api.tavily.com/search", json=payload)
            response.raise_for_status()
            return response.json()

    except httpx.HTTPError as e:
        return {
            "error": str(e),
            "details": getattr(e.response, "text", "")
        }
