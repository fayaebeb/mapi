""" from langchain_community.utilities.wikipedia import WikipediaAPIWrapper

def wikipedia_search(query: str, lang: str = "en", k: int = 3):
    wiki = WikipediaAPIWrapper(top_k_results=k, lang=lang)
    results = wiki.run(query)
    return results """
