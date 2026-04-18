import os
from langchain_community.document_loaders import PyPDFLoader, Docx2txtLoader, CSVLoader, TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain.tools import tool

CHROMA_PATH = "chroma_db"

def get_embeddings():
    return HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

def load_document(file_path):
    ext = file_path.split(".")[-1].lower()
    if ext == "pdf":
        loader = PyPDFLoader(file_path)
    elif ext == "docx":
        loader = Docx2txtLoader(file_path)
    elif ext == "csv":
        loader = CSVLoader(file_path)
    else:
        loader = TextLoader(file_path)
    return loader.load()

def add_document(file_path):
    docs = load_document(file_path)
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=50
    )
    chunks = splitter.split_documents(docs)
    db = Chroma(
        persist_directory=CHROMA_PATH,
        embedding_function=get_embeddings()
    )
    db.add_documents(chunks)
    return len(chunks)

@tool
def search_documents(query: str) -> str:
    """Search uploaded documents for relevant information about the query."""
    try:
        db = Chroma(
            persist_directory=CHROMA_PATH,
            embedding_function=get_embeddings()
        )
        results = db.similarity_search(query, k=4)
        if not results:
            return "No relevant information found in uploaded documents."
        output = []
        for r in results:
            source = r.metadata.get("source", "uploaded document")
            output.append(f"Source: {source}\nContent: {r.page_content}")
        return "\n---\n".join(output)
    except Exception:
        return "No documents uploaded yet."