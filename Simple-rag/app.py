from pathlib import Path

import chromadb
import streamlit as st

from chunker import chunk_text
from embeddings import embed
from llm import stream_answer
from loader import load_documents

DOCS_DIR = Path(__file__).parent / "docs_dir"
DB_DIR = Path(__file__).parent / "chroma_db"
COLLECTION = "documents"
TOP_K = 4

st.set_page_config(page_title="RAG Chatbot")
st.title("RAG Chatbot")


@st.cache_resource
def get_collection():
    client = chromadb.PersistentClient(path=str(DB_DIR))
    return client.get_or_create_collection(COLLECTION)


def ingest() -> int:
    docs = load_documents(DOCS_DIR)
    
    collection = get_collection()
    ids, texts, metadatas = [], [], []
    for source, text in docs:
        for i, chunk in enumerate(chunk_text(text)):
            ids.append(f"{source}::{i}")
            texts.append(chunk)
            metadatas.append({"source": source, "chunk": i})

    if texts:
        collection.upsert(ids=ids, documents=texts, metadatas=metadatas, embeddings=embed(texts))
    return len(texts)


def retrieve(question: str):
    collection = get_collection()
    if collection.count() == 0:
        return [], []
    results = collection.query(query_embeddings=embed([question]), n_results=TOP_K)
    return results["documents"][0], [m["source"] for m in results["metadatas"][0]]


with st.sidebar:
    st.header("Documents")
    uploaded = st.file_uploader(
        "Upload files (pdf, docx, txt, md)", type=["pdf", "docx", "txt", "md"], accept_multiple_files=True
    )
    if st.button("Re-ingest documents"):
        with st.spinner("Chunking + embedding..."):
            n = ingest()
        st.success(f"Indexed {n} chunks.")
    st.caption(f"{get_collection().count()} chunks currently indexed.")

if "messages" not in st.session_state:
    st.session_state.messages = []

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

if question := st.chat_input("Ask about your documents..."):
    st.session_state.messages.append({"role": "user", "content": question})
    with st.chat_message("user"):
        st.markdown(question)

    chunks, sources = retrieve(question)
    if not chunks:
        answer_placeholder_text = "No documents indexed yet — upload some in the sidebar first."
        with st.chat_message("assistant"):
            st.markdown(answer_placeholder_text)
        st.session_state.messages.append({"role": "assistant", "content": answer_placeholder_text})
    else:
        context = "\n\n".join(f"[{s}]\n{c}" for c, s in zip(chunks, sources))
        with st.chat_message("assistant"):
            answer = st.write_stream(stream_answer(question, context))
        st.session_state.messages.append({"role": "assistant", "content": answer})
