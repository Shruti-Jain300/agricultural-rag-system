import os
import time
from typing import List, Dict
from dotenv import load_dotenv
from google import genai

load_dotenv()

# DEFAULT_MODEL = "gemini-2.5-flash"
DEFAULT_MODEL = "gemini-3-flash-preview"

def get_api_key():
    try:
        import streamlit as st
        if "GEMINI_API_KEY" in st.secrets:
            return st.secrets["GEMINI_API_KEY"]
    except Exception:
        pass

    return os.getenv("GEMINI_API_KEY")

def get_gemini_client():
    api_key = get_api_key()
    if not api_key:
        raise ValueError(
            "GEMINI_API_KEY not found. Add it to Streamlit secrets or local .env."
        )
    return genai.Client(api_key=api_key)

def build_context(docs: List[str], metas: List[Dict], max_chunks: int = 3) -> str:
    parts = []
    limit = min(max_chunks, len(docs))

    for i in range(limit):
        meta = metas[i] if i < len(metas) else {}
        source_type = meta.get("source_type", "unknown")
        source_name = meta.get("source_name", "unknown")
        file_name = meta.get("file_name", "unknown")
        chunk_index = meta.get("chunk_index_in_file", "N/A")

        block = (
            f"[Source {i+1}]\n"
            f"source_type: {source_type}\n"
            f"source_name: {source_name}\n"
            f"file_name: {file_name}\n"
            f"chunk_index_in_file: {chunk_index}\n"
            f"text:\n{docs[i]}"
        )
        parts.append(block)

    return "\n\n".join(parts).strip()

def build_prompt(query: str, context: str) -> str:
    return f"""
You are a domain-specific agricultural knowledge assistant.

Your task is to answer the user's question ONLY from the retrieved context below.

Rules:
1. Use only the provided retrieved context.
2. Do not add outside knowledge.
3. If the answer is not clearly supported by the context, say:
   "The retrieved documents do not contain enough information to answer this confidently."
4. Keep the answer clear, factual, and concise.
5. After the answer, include a short section called "Evidence Basis".
6. Do not invent scheme names, statistics, or policy details.

User Question:
{query}

Retrieved Context:
{context}

Return format:
Answer:
<clear grounded answer>

Evidence Basis:
- <short point 1>
- <short point 2>
- <short point 3 if available>
""".strip()

def generate_grounded_answer(
    query: str,
    docs: List[str],
    metas: List[Dict],
    model_name: str = DEFAULT_MODEL,
    max_chunks: int = 3,
) -> str:
    client = get_gemini_client()
    context = build_context(docs, metas, max_chunks=max_chunks)
    prompt = build_prompt(query, context)

    last_error = None

    for attempt in range(3):
        try:
            response = client.models.generate_content(
                model=model_name,
                contents=prompt,
            )

            text = getattr(response, "text", None)
            if text and text.strip():
                return text.strip()

        except Exception as e:
            last_error = e
            time.sleep(2 * (attempt + 1))

    raise RuntimeError(f"Gemini request failed after retries: {last_error}")