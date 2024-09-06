# DocQA

This project implements a document-based knowledge base and Q&A system using Streamlit, LangChain, Google GenAI, and FAISS.

## Features

- Text extraction and chunking
- In-memory vector storage using FAISS
- Question answering using LangChain and Google GenAI

## Setup

1. Clone the repository
2. Install dependencies: `uv install`
3. Set up environment variables in `.env` file
4. Run the application: `streamlit run app/main.py`

## References
- [uv](https://docs.astral.sh/uv/): Python package manager
- [LangChain Tutorials](https://python.langchain.com/v0.2/docs/tutorials/rag/#built-in-chains)
- [FAISS](https://python.langchain.com/v0.2/docs/integrations/vectorstores/faiss/#saving-and-loading)
