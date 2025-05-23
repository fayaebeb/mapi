import os
from langchain_google_community import GoogleSearchAPIWrapper

def google_search(query: str, k: int = 3):
    try:
        wrapper = GoogleSearchAPIWrapper(
            google_api_key=os.getenv("GOOGLE_API_KEY"),
            google_cse_id=os.getenv("GOOGLE_CSE_ID"),
            k=k
        )
        results = wrapper.results(query=query, num_results=k)

        return [
            f"🔗[{r.get('title')}]({r.get('link')})\n{r.get('snippet', '').strip()}\n---"
            for r in results
        ]

    except Exception as e:
        # Format the error into a user-friendly string
        return [f"❗️Google Search Error: {str(e)}"]
