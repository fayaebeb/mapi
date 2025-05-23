def format_as_message(
    docs: list,
    mode: str = "openai",  # "openai" or "output"
    sep: str = "\n"
) -> str:
    def format_single(doc):
        metadata = doc.metadata if hasattr(doc, 'metadata') else {}
        text = doc.page_content if hasattr(doc, 'page_content') else str(doc)

        if mode == "openai":
            if "session_id" in metadata:
                return f"{text}\n\nデータ作成者: {metadata['session_id']}\n---"
            elif "url" in metadata:
                return f"[URL] {metadata['url']}\n{text}\n---"
            elif "filename" in metadata:
                return f"[DOC] {metadata['filename']}\n{text}\n---"
            else:
                return text

        elif mode == "output":
            if "session_id" in metadata:
                return f"{text}\n\n✍️ データ作成者: {metadata['session_id']}\n---"
            elif "url" in metadata:
                return f"🔗{metadata['url']}\n\n{text}\n---"
            elif "filename" in metadata:
                return f"📄[{metadata['filename']}]({metadata.get('filelink', '#')})\n\n{text}\n---"
            else:
                return text

        return text

    return sep.join([format_single(doc) for doc in docs])

def format_as_message_list(docs):
    def format_single(doc):
        metadata = doc.metadata if hasattr(doc, 'metadata') else {}
        text = doc.page_content if hasattr(doc, 'page_content') else str(doc)

        if "session_id" in metadata:
            return f"Session ID: {metadata['session_id']}\n{text}"
        elif "url" in metadata:
            return f"URL: {metadata['url']}\n{text}\n---"
        elif "filename" in metadata:
            filelink = metadata.get("filelink", "#")
            return f"Filename: {metadata['filename']}\nFilelink: {filelink}\n{text}\n---"
        else:
            return text

    return [format_single(doc) for doc in docs]
