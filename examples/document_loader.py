import os
import tempfile
from typing import Dict, List, Type

# 1. Supabase and Embeddings
from supabase.client import Client, create_client
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import SupabaseVectorStore

# 2. Document Loading & Splitting
from langchain_community.document_loaders import (
    PyPDFLoader, TextLoader, WebBaseLoader, JSONLoader, 
    UnstructuredMarkdownLoader, Docx2txtLoader
)
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document

# --- Initialization ---
# Ensure environment variables are set before running
supabase_url = os.environ.get("SUPABASE_URL")
supabase_key = os.environ.get("SUPABASE_SERVICE_KEY")

if not all([supabase_url, supabase_key]):
    raise ValueError("SUPABASE_URL and SUPABASE_SERVICE_KEY must be set.")

supabase: Client = create_client(supabase_url, supabase_key)
embeddings = OpenAIEmbeddings()

# --- Configuration ---
# Map file extensions/types to the appropriate LangChain Loader
LOADER_MAPPING: Dict[str, Type] = {
    ".pdf": PyPDFLoader,
    ".docx": Docx2txtLoader, # Use Docx2txtLoader for .docx files
    ".doc": Docx2txtLoader,
    ".md": UnstructuredMarkdownLoader,
    ".txt": TextLoader,
    ".json": JSONLoader, # NOTE: JSONLoader requires a 'jq_schema' to extract content
    "weblink": WebBaseLoader # Special case for web links
}

def get_loader(file_path: str, file_type: str):
    """Dynamically selects the correct document loader."""
    if file_type == "weblink":
        # WebBaseLoader takes the URL directly
        return WebBaseLoader(file_path)
    
    # Handle JSON specifically due to the required jq_schema
    if file_path.lower().endswith(".json"):
        # For a simple case, use '.' to load the whole JSON structure 
        # as content, or adjust the schema for your specific JSON format.
        return JSONLoader(
            file_path=file_path, 
            jq_schema='.', 
            text_content=True
        )

    ext = os.path.splitext(file_path.lower())[1]
    loader_class = LOADER_MAPPING.get(ext)
    
    if loader_class:
        return loader_class(file_path)
    else:
        raise ValueError(f"Unsupported file type: {ext}")

def ingest_document_to_vector_store(file_stream, filename: str, content_type: str) -> List[Document]:
    """
    Handles file upload, loads, splits, embeds, and stores the document chunks.
    
    Args:
        file_stream: A file-like object from the client upload.
        filename: The original name of the file (e.g., "job_description.pdf").
        content_type: The file's MIME type or a custom type like "weblink".

    Returns:
        The list of split documents that were ingested.
    """
    
    # 1. Handle File Upload (Save to a temporary file for LangChain Loaders)
    # LangChain Loaders generally require a file path.
    with tempfile.NamedTemporaryFile(delete=False, suffix=f"_{filename}") as tmp_file:
        tmp_file.write(file_stream.read())
        tmp_path = tmp_file.name

    try:
        # 2. Load the document
        loader = get_loader(tmp_path, content_type)
        documents = loader.load()
        
        # 3. Split the document into chunks
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000, 
            chunk_overlap=200 # Overlap helps maintain context between chunks
        )
        docs = text_splitter.split_documents(documents)
        
        # Add original filename to all chunks' metadata for retrieval filtering
        for doc in docs:
            doc.metadata['source_filename'] = filename
        
        # 4. Embed and Store in Supabase
        SupabaseVectorStore.from_documents(
            docs, 
            embeddings, 
            client=supabase, 
            table_name="documents",
            query_name="match_documents"
        )
        
        print(f"Successfully ingested {len(docs)} chunks from {filename} into Supabase.")
        return docs

    except Exception as e:
        print(f"Error processing {filename}: {e}")
        return []
    finally:
        # Clean up the temporary file
        os.remove(tmp_path)

# --- Example of use (Conceptual, depends on your web framework) ---
# Assuming a FastAPI/Flask endpoint receives the file:
"""
# Example for a PDF file stream
# with open("path/to/my/resume.pdf", "rb") as f:
#     ingested_documents = ingest_document_to_vector_store(
#         f, "my_resume.pdf", "application/pdf"
#     )

# Example for a Web Link (no file stream needed, only a temporary text file path)
# The file_stream would be a simple stream with the URL, and the get_loader logic
# for 'weblink' would need to be adapted to read the URL path instead of file path.
# For simplicity, if the client is uploading a file, you'd use the file path logic.
# If the client is entering a URL, you'd call WebBaseLoader directly:
# web_loader = WebBaseLoader("https://example.com/job_post")
# web_docs = web_loader.load()
# split_web_docs = text_splitter.split_documents(web_docs)
# SupabaseVectorStore.from_documents(split_web_docs, embeddings, ...)

"""