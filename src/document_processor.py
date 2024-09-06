"""Module for processing PDF documents."""

import tempfile
from io import BytesIO

from langchain_community.document_loaders import PyPDFLoader
from langchain_community.vectorstores import FAISS
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter

EMBEDDINGS_MODEL = "models/embedding-001"
CHUNK_SIZE = 1000
CHUNK_OVERLAP = 200


def process_document(file: BytesIO) -> FAISS:
    """
    Process a PDF document and create a FAISS vector store.

    Args:
        file (BytesIO): The uploaded PDF file.

    Returns:
        FAISS: The created vector store.
    """
    embeddings = GoogleGenerativeAIEmbeddings(model=EMBEDDINGS_MODEL)
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE, chunk_overlap=CHUNK_OVERLAP
    )

    with tempfile.NamedTemporaryFile("wb") as temp_file:
        temp_file.write(file.getvalue())
        loader = PyPDFLoader(temp_file.name)
        documents = loader.load()
        chunks = text_splitter.split_documents(documents)
        return FAISS.from_documents(chunks, embeddings)
