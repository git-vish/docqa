"""Module for processing PDF documents."""

import tempfile
from io import BytesIO

from langchain_community.document_loaders import Docx2txtLoader, PyPDFLoader, TextLoader
from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter

EMBEDDINGS_MODEL = "models/embedding-001"
CHUNK_SIZE = 1000
CHUNK_OVERLAP = 200


def _load_document(file: BytesIO, filename: str) -> list[Document]:
    """
    Load the document based on its file extension.

    Args:
        file: A BytesIO object containing the file content.
        filename: The name of the uploaded file.

    Returns:
        A list of Document objects.

    Raises:
        ValueError: If the file format is not supported.
    """
    file_extension = filename.split(".")[-1].lower()

    with tempfile.NamedTemporaryFile("wb") as temp_file:
        temp_file.write(file.getvalue())

        if file_extension == "pdf":
            loader = PyPDFLoader(temp_file.name)
        elif file_extension in ["docx", "doc"]:
            loader = Docx2txtLoader(temp_file.name)
        elif file_extension == "txt":
            loader = TextLoader(temp_file.name)
        else:
            raise ValueError(f"Unsupported file format: {file_extension}")

        documents = loader.load()

    return documents


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

    documents = _load_document(file, file.name)
    chunks = text_splitter.split_documents(documents)
    return FAISS.from_documents(chunks, embeddings)
