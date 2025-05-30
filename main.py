from fastapi import FastAPI
from pydantic import BaseModel
from typing import List, Optional
from langchain_openai import ChatOpenAI
from langchain_core.messages import SystemMessage, HumanMessage
from dotenv import load_dotenv
import os

from astra_retriever import retriever
from message_formatter import format_as_message
from tavily_search import tavily_search
from google_search import google_search

load_dotenv()

app = FastAPI()

llm = ChatOpenAI(
    model="gpt-4o",
    temperature=1.0,
    openai_api_key=os.getenv("OPENAI_API_KEY")
)

class ChatInput(BaseModel):
    message: str
    useweb: Optional[bool] = False
    usedb: Optional[bool] = False

def deduplicate_docs(docs):
    unique = []
    seen = set()
    for doc in docs:
        identifier = (doc.page_content, frozenset(doc.metadata.items()) if hasattr(doc, 'metadata') else None)
        if identifier not in seen:
            unique.append(doc)
            seen.add(identifier)
    return unique

@app.post("/chat")
async def chat(input: ChatInput):
    internal_docs_text = ""
    tavily_text = ""
    formatted_output_docs = ""

    # 1. Retrieve internal documents via Astra (if enabled)
    if input.usedb:
        retrieved_docs = retriever.invoke(input.message)
        unique_docs = deduplicate_docs(retrieved_docs)
        internal_docs_text = format_as_message(unique_docs, mode="openai")
        formatted_output_docs = format_as_message(unique_docs, mode="output")

    # 2. Optionally retrieve Tavily web results
    if input.useweb:
        tavily_result = tavily_search(
            query=input.message,
            search_depth="basic",
            chunks_per_source=3,
            topic="general",
            max_results=3,
            include_answer=True,
            include_images=False,
            include_raw_content=False
        )
        tavily_text = tavily_result.get("answer") or ""

    # 3. Retrieve Google search results (only used in output, not prompt)
    google_results = []
    if input.useweb:
        try:
            google_results = google_search(query=input.message, k=3)
        except Exception as e:
            google_results = [{"error": str(e)}]

    # 4. Prompt to LLM(internal docs + tavily context (not Google))
    prompt_parts = []

    if internal_docs_text:
        prompt_parts.append("[Internal Documents]\n" + internal_docs_text)
    if tavily_text:
        prompt_parts.append("[Web Results]\n" + tavily_text)

    prompt_parts.append(f"[User Question]\n{input.message}\n\n[Answer]")

    final_prompt = "\n\n".join(prompt_parts)

    # 5. Send to LLM
    gpt_reply = llm.invoke([
        SystemMessage(content=("You are ミライAI, the AI of パシフィックコンサルタンツ株式会社")),
        HumanMessage(content=final_prompt)
    ])

    # 6. Final output
    final_output = gpt_reply.content.strip()

    # Contact line
    contact_line = "\nご不明な点がございましたら、以下のアドレスまでお気軽にお問い合わせください。\n" \
                   "[future-service-devlopment@tk.pacific.co.jp](mailto:future-service-devlopment@tk.pacific.co.jp)\n"

    # Append internal docs and contact info
    if input.usedb and formatted_output_docs:
        final_output += "\n\n### 社内文書情報:\n\n" + formatted_output_docs
        final_output += contact_line

    # Append web info
    if input.useweb and (google_results or tavily_text):
        final_output += "\n\n### オンラインWeb情報:\n"

        if google_results:
            final_output += "\n" + "\n".join(google_results)

        if tavily_text:
            final_output += "\n" + tavily_text.strip()

    return {
        "reply": final_output
    }


"""
return {
    "reply": gpt_reply,
    "retrieved_docs": docs_content if input.usedb else [],
    "formatted_output_docs": formatted_output_docs if input.usedb else None,
    "used_tavily": input.useweb,
    "tavily_text": tavily_text if input.useweb else None,
    "google_results": google_results if input.useweb else None
}
"""

from extractor_agent import run_extraction_agent
from astra_retriever import vector_store
from uuid import uuid4
from datetime import datetime
import json
import re

class ExtractInput(BaseModel):
    input: str
    session_id: str

def extract_url(text: str) -> str | None:
    """Extract the first URL from the text if present."""
    match = re.search(r'https?://\S+', text)
    return match.group(0) if match else None

@app.post("/mimod")
async def extract(input: ExtractInput):
    # Step 1: Run the extraction agent
    structured_result = run_extraction_agent(input.input)

    # ✅ Ensure it's a string
    if not isinstance(structured_result, str):
        try:
            structured_result = json.dumps(structured_result, ensure_ascii=False, indent=2)
        except Exception:
            structured_result = str(structured_result)

    # Step 2: Generate a unique message ID
    msgid = str(uuid4())

    # Step 3: Check if input has a URL
    url_in_input = extract_url(input.input)

    # Step 4: Build metadata with URL or session_id
    metadata = {
        "msgid": msgid,
        "timestamp": datetime.utcnow().isoformat(),
    }
    if url_in_input:
        metadata["url"] = url_in_input
    else:
        metadata["session_id"] = input.session_id

    # Step 5: Attempt to save to Astra DB
    try:
        if vector_store:
            vector_store.add_texts(
                texts=[structured_result],
                metadatas=[metadata]
            )
        else:
            return {"error": "❌ Astra DB connection is not initialized."}
    except Exception as e:
        print("❌ Error during Astra DB save:", e)
        return {"error": f"❌ Astra DB save failed: {str(e)}"}

    # Step 6: Return formatted confirmation response
    return {
        "reply": (
            " ✅ メッセージを正常に保存しました。\n\n"
            " 🚀 データの抽出が完了しました。\n\n"
            "📂 抽出データ:\n\n"
            f"{structured_result}\n\n"
            "⚠️🗑️ ご注意ください： このメッセージを削除すると、上記の抽出データも同時に削除されます。\n\n"
            f"MSGID: {msgid}"
        )
    }