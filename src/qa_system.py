"""Module for the Question Answering system."""

from langchain.chains.retrieval import create_retrieval_chain
from langchain_community.vectorstores import FAISS
from langchain_core.runnables import Runnable


def answer_question(
    question: str, vectorstore: FAISS, combine_docs_chain: Runnable
) -> str:
    """
    Answer a given question based on the document context.

    Args:
        question (str): The question to be answered.
        vectorstore (FAISS): The FAISS vector store containing document embeddings.
        combine_docs_chain (Runnable[StuffDocumentsChain]):
            The chain for combining documents.

    Returns:
        str: The generated answer.
    """
    retrieval_chain = create_retrieval_chain(
        retriever=vectorstore.as_retriever(), combine_docs_chain=combine_docs_chain
    )
    response = retrieval_chain.invoke({"input": question})
    return response["answer"]
