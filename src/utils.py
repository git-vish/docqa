"""Utility functions."""

import streamlit as st
from langchain import hub
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_google_genai import GoogleGenerativeAI

LLM_MODEL = "gemini-pro"
PROMPT = "langchain-ai/retrieval-qa-chat"


def init_session_state():
    """Initialize the session state."""
    if "vectorstore" not in st.session_state:
        st.session_state.vectorstore = None

        llm = GoogleGenerativeAI(model=LLM_MODEL)
        prompt = hub.pull(PROMPT)
        st.session_state.combine_docs_chain = create_stuff_documents_chain(llm, prompt)


def clear_vectorstore():
    """Clear the vector store from the session state."""
    st.session_state.vectorstore = None
