"""Streamlit app for document Q&A using LangChain and Google GenAI."""

import streamlit as st
from dotenv import load_dotenv

from document_processor import process_document
from qa_system import answer_question
from utils import clear_vectorstore, init_session_state

load_dotenv()
init_session_state()

st.set_page_config(page_title="DocQA | Ask the docs")
st.title("DocQA")

st.subheader("Upload a Document")
file = st.file_uploader("Choose a PDF file", type=["pdf"], on_change=clear_vectorstore)

if file and not st.session_state.vectorstore:
    with st.spinner("Processing document..."):
        st.session_state.vectorstore = process_document(file)

if st.session_state.vectorstore:
    st.subheader("Ask a Question")
    question = st.text_input("Enter your question:")

    if question:
        with st.spinner("Thinking..."):
            answer = answer_question(
                question,
                st.session_state.vectorstore,
                st.session_state.combine_docs_chain,
            )
        st.info(answer)
