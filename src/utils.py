"""Utility functions."""

import streamlit as st
from langchain import hub
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.messages import AIMessage, HumanMessage
from langchain_google_genai import GoogleGenerativeAI

LLM_MODEL = "gemini-pro"
QA_PROMPT = "langchain-ai/retrieval-qa-chat"
REPHRASE_PROMPT = "langchain-ai/chat-langchain-rephrase"


def init_session_state():
    """Initialize the session state."""
    if "vectorstore" not in st.session_state:
        st.session_state.vectorstore = None

        st.session_state.llm = GoogleGenerativeAI(model=LLM_MODEL)
        st.session_state.qa_prompt = hub.pull(QA_PROMPT)
        st.session_state.rephrase_prompt = hub.pull(REPHRASE_PROMPT)

        st.session_state.combine_docs_chain = create_stuff_documents_chain(
            llm=st.session_state.llm,
            prompt=st.session_state.qa_prompt,
        )

        st.session_state.chat_history = []


def clear_vectorstore():
    """Clear the vector store and chat history from the session state."""
    st.session_state["vectorstore"] = None
    st.session_state["chat_history"] = []


def update_chat_history(question: str, answer: str):
    """
    Update the chat history with a question and answer.

    The chat history is stored in the Streamlit session state.

    Args:
        question: The question that was asked.
        answer: The answer that was given.
    """
    st.session_state.chat_history.extend(
        [
            HumanMessage(content=question),
            AIMessage(content=answer),
        ]
    )
