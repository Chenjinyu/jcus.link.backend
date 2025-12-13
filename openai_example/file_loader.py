# file_loaders.py
import os
from langchain_community.document_loaders import (
    PyPDFLoader, Docx2txtLoader, TextLoader,
    JSONLoader, UnstructuredMarkdownLoader
)

def load_file_content(file_path: str) -> str:
    ext = os.path.splitext(file_path)[1].lower()

    if ext == ".pdf":
        docs = PyPDFLoader(file_path).load()
    elif ext == ".docx":
        docs = Docx2txtLoader(file_path).load()
    elif ext in [".txt"]:
        docs = TextLoader(file_path).load()
    elif ext in [".md", ".markdown"]:
        docs = UnstructuredMarkdownLoader(file_path).load()
    elif ext == ".json":
        docs = JSONLoader(file_path).load()
    else:
        raise ValueError(f"Unsupported file type: {ext}")

    return "\n\n".join([d.page_content for d in docs])
