"""Streamlit app for document Q&A using LangChain and Google GenAI."""

import streamlit as st
from dotenv import load_dotenv
from langchain.chains.history_aware_retriever import create_history_aware_retriever
from langchain.chains.retrieval import create_retrieval_chain

from document_processor import process_document
from utils import clear_session_state, init_session_state, update_chat_history

st.set_page_config(page_title="DocQA | Ask the docs", layout="wide", page_icon="📚")

with st.spinner("Initializing..."):
    load_dotenv()
    init_session_state()

st.title("DocQA")

# Sidebar for document upload
with st.sidebar:
    st.header("Document Upload")
    file = st.file_uploader(
        "Choose a file",
        type=["pdf", "doc", "docx", "txt"],
        on_change=clear_session_state,
    )

    if file and not st.session_state.vectorstore:
        with st.spinner("Processing document..."):
            st.session_state.vectorstore = process_document(file)

            history_aware_retriever = create_history_aware_retriever(
                llm=st.session_state.llm,
                retriever=st.session_state.vectorstore.as_retriever(),
                prompt=st.session_state.rephrase_prompt,
            )
            st.session_state.retrieval_chain = create_retrieval_chain(
                retriever=history_aware_retriever,
                combine_docs_chain=st.session_state.combine_docs_chain,
            )
        st.success("Document processed successfully!")

# Main content area
st.subheader("Ask a Question")
if st.session_state.vectorstore:
    # Display the chat history
    for message in st.session_state.chat_history:
        with st.chat_message(message.type):
            st.markdown(message.content)

    if question := st.chat_input("Enter your question:"):
        with st.chat_message("human"):
            st.markdown(question)

        with st.spinner("Thinking..."):
            output = st.session_state.retrieval_chain.invoke(
                {"input": question, "chat_history": st.session_state.chat_history}
            )
            answer = output["answer"]

        with st.chat_message("ai"):
            st.markdown(answer)

        update_chat_history(question, answer)
else:
    st.write("Upload a document to start asking questions.")
