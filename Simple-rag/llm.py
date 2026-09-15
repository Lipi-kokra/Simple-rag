import streamlit as st
from groq import Groq

CHAT_MODEL = "openai/gpt-oss-20b"

_client = None


def get_client() -> Groq:
    global _client
    if _client is None:
        _client = Groq(api_key=st.secrets["GROQ_API_KEY"])
    return _client

def stream_answer(question: str, context: str):
    system_prompt = (
        "Answer the question using only the context below. "
        "If the context doesn't contain the answer, say you don't know. "
        "Cite sources by filename in brackets, e.g. [manual.docx]."
    )
    stream = get_client().chat.completions.create(
        model=CHAT_MODEL,
        messages=[
            {"role": "system", "content": f"{system_prompt}\n\nContext:\n{context}"},
            {"role": "user", "content": question},
        ],
        stream=True,
    )
    for chunk in stream:
        delta = chunk.choices[0].delta.content
        if delta:
            yield delta
