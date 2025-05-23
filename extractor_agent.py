from langchain.tools import tool
from langchain.agents import initialize_agent, AgentType
from langchain_openai import ChatOpenAI
import requests
from bs4 import BeautifulSoup
import os

# 1. Define the scraping tool
@tool
def scrape_url(url: str) -> str:
    """Scrape and return clean text content from the given URL."""
    try:
        response = requests.get(url, timeout=10)
        soup = BeautifulSoup(response.text, "html.parser")
        return soup.get_text(separator=" ", strip=True)
    except Exception as e:
        return f"Error scraping {url}: {str(e)}"

# 2. Initialize the agent with the tool
llm = ChatOpenAI(
    model="gpt-4o",
    temperature=0,
    openai_api_key=os.getenv("OPENAI_API_KEY")
)

tools = [scrape_url]

agent_executor = initialize_agent(
    tools=tools,
    llm=llm,
    agent=AgentType.OPENAI_FUNCTIONS,
    verbose=True,
    handle_parsing_errors=True,
    agent_kwargs={
        "system_message": (
            "You are an information extractor. Your job is to extract structured information from the user's input. "
            "If the input includes a URL, use the url tool to scrape information. Return only information, not explanations."
        )
    }
)

# 3. Function for FastAPI to call
def run_extraction_agent(user_input: str) -> str:
    result = agent_executor.invoke({"input": user_input})

    # If agent returns a dict (as expected with OPENAI_FUNCTIONS), extract the output only
    if isinstance(result, dict) and "output" in result:
        return result["output"]

    # Fallback: stringify anything else
    return str(result)