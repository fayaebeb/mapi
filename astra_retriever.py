import os
from dotenv import load_dotenv
from langchain_astradb import AstraDBVectorStore
import warnings

load_dotenv()

# Load credentials from environment
ASTRA_DB_APPLICATION_TOKEN = os.getenv("ASTRA_DB_APPLICATION_TOKEN")
ASTRA_DB_API_ENDPOINT = os.getenv("ASTRA_DB_API_ENDPOINT")
ASTRA_DB_NAMESPACE = os.getenv("ASTRA_DB_NAMESPACE")
ASTRA_DB_COLLECTION = os.getenv("ASTRA_DB_COLLECTION", "data")  # fallback

# Instantiate the vector store with error handling
try:
    vector_store = AstraDBVectorStore(
        token=ASTRA_DB_APPLICATION_TOKEN,
        api_endpoint=ASTRA_DB_API_ENDPOINT,
        namespace=ASTRA_DB_NAMESPACE,
        collection_name=ASTRA_DB_COLLECTION,
        autodetect_collection=True
    )
    # retriever = vector_store.as_retriever(search_kwargs={"k": 3})

except Exception as e:
    warnings.warn(f"❌ Failed to connect to existing collection: {e}")
    retriever = None