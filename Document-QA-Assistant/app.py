"""Streamlit entry point for a PDF question answering demo."""
from __future__ import annotations

import hashlib
from pathlib import Path

import streamlit as st
from dotenv import dotenv_values
from groq import AuthenticationError

from rag import build_index, answer_question

st.set_page_config(page_title="Document Q&A", page_icon="📄", layout="centered")
st.title("Document Q&A Assistant")
st.caption("Upload a searchable or scanned PDF and ask questions. Answers include page references.")

settings = dotenv_values(Path(__file__).with_name(".env"))
key = (settings.get("GROQ_API_KEY") or "").strip()
model = (settings.get("GROQ_MODEL") or "").strip() or None
if not key:
    st.info("Set GROQ_API_KEY in a local .env file before asking a question.")

uploaded = st.file_uploader("Choose a PDF", type=["pdf"], max_upload_size=10)
if uploaded is not None:
    pdf_bytes = uploaded.getvalue()
    digest = hashlib.sha256(pdf_bytes).hexdigest()
    if st.session_state.get("document_digest") != digest:
        try:
            with st.spinner("Reading pages (including OCR for scans) and building the search index..."):
                index, page_count, chunk_count = build_index(pdf_bytes)
        except Exception as exc:
            st.error(f"Could not process this PDF: {exc}")
            st.stop()
        st.session_state.document_digest = digest
        st.session_state.index = index
        st.session_state.history = []
        st.session_state.page_count = page_count
        st.session_state.chunk_count = chunk_count

    st.success(f"Ready: {uploaded.name} · {st.session_state.page_count} pages · {st.session_state.chunk_count} searchable passages")
    for message in st.session_state.get("history", []):
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
            if message.get("sources"):
                with st.expander("Retrieved passages"):
                    for source in message["sources"]:
                        st.markdown(f"**[{source['id']}] Page {source['page']}**")
                        st.caption(source["text"])

    question = st.chat_input("Ask about this PDF", disabled=not bool(key))
    if question:
        previous_messages = st.session_state.history[-6:]
        st.session_state.history.append({"role": "user", "content": question})
        with st.chat_message("user"):
            st.markdown(question)
        try:
            with st.spinner("Searching the PDF and generating an answer..."):
                answer, sources = answer_question(
                    st.session_state.index, question, key,
                    history=previous_messages, filename=uploaded.name, model=model,
                )
        except AuthenticationError:
            st.error("Groq rejected the API key in this project's .env file.")
        except Exception as exc:
            # Do not display raw API exceptions: they can contain request details.
            st.error(f"The request failed ({type(exc).__name__}). Check the key, model access, and connection.")
        else:
            st.session_state.history.append({"role": "assistant", "content": answer, "sources": sources})
            with st.chat_message("assistant"):
                st.markdown(answer)
                with st.expander("Retrieved passages"):
                    for source in sources:
                        st.markdown(f"**[{source['id']}] Page {source['page']}**")
                        st.caption(source["text"])
