import os
import tempfile
from typing import Dict, Type, List, Union
from concurrent.futures import ThreadPoolExecutor

# LangChain Imports
from langchain_community.document_loaders import (
    PyPDFLoader, TextLoader, WebBaseLoader, JSONLoader, 
    UnstructuredMarkdownLoader, Docx2txtLoader
)
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document

# Custom Configuration Import
from .config import supabase_client, embeddings # Assuming config.py has initialized these
from langchain_community.vectorstores import SupabaseVectorStore

# Using a ThreadPoolExecutor for blocking I/O operations (like file reading and sync LangChain loaders)
# This prevents the blocking of FastAPI's main event loop.
executor = ThreadPoolExecutor() 

# --- Configuration: Loader Mapping ---
LOADER_MAPPING: Dict[str, Type] = {
    ".pdf": PyPDFLoader,
    ".docx": Docx2txtLoader,
    ".doc": Docx2txtLoader,
    ".md": UnstructuredMarkdownLoader,
    ".txt": TextLoader,
    ".json": JSONLoader,
}

def get_loader(file_path: str, file_extension: str):
    """Dynamically selects the correct document loader based on file extension."""
    if file_extension == ".json":
        # JSONLoader requires a 'jq_schema' to extract content
        # Use '.' to load the entire JSON structure for embedding
        return JSONLoader(file_path=file_path, jq_schema='.', text_content=True)

    loader_class = LOADER_MAPPING.get(file_extension)
    
    if loader_class:
        return loader_class(file_path)
    else:
        raise ValueError(f"Unsupported file type: {file_extension}")

def _sync_ingestion_task(temp_path: str, filename: str) -> List[Document]:
    """
    Synchronous task that performs the core LangChain loading, splitting, and storing.
    This runs in the executor thread pool.
    """
    file_extension = os.path.splitext(filename.lower())[1]

    # 1. Load the document
    loader = get_loader(temp_path, file_extension)
    documents = loader.load()
    
    # 2. Split the document into chunks
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000, 
        chunk_overlap=200
    )
    docs = text_splitter.split_documents(documents)
    
    # Add metadata
    for doc in docs:
        doc.metadata['source_filename'] = filename
        
    # 3. Embed and Store in Supabase (This is often a synchronous call)
    SupabaseVectorStore.from_documents(
        docs, 
        embeddings, 
        client=supabase_client, 
        table_name="documents", 
        query_name="match_documents"
    )
    
    print(f"Successfully ingested {len(docs)} chunks from {filename}.")
    return docs


async def ingest_document_to_vector_store_async(
    file_bytes: bytes, filename: str, content_type: str
) -> List[Document]:
    """
    Async function that handles file bytes from the API, saves them temporarily, 
    and runs the synchronous ingestion in a background thread.
    """
    
    # 1. Save file bytes to a temporary location
    with tempfile.NamedTemporaryFile(delete=False, suffix=f"_{filename}") as tmp_file:
        tmp_file.write(file_bytes)
        tmp_path = tmp_file.name

    try:
        # 2. Run the synchronous ingestion task in the background thread pool
        # This is the "best practice" way to handle blocking I/O (like LangChain loaders) in FastAPI
        loop = os.get_event_loop()
        ingested_docs = await loop.run_in_executor(
            executor, 
            _sync_ingestion_task, 
            tmp_path, 
            filename
        )
        return ingested_docs

    except Exception as e:
        print(f"Error processing {filename}: {e}")
        # Re-raise the exception to be caught by the API endpoint
        raise e 
        
    finally:
        # 3. Clean up the temporary file
        os.remove(tmp_path)

# You would need a separate function for web links if they don't involve an upload.
# For example, an async function using WebBaseLoader and then running the splitting/storing sync part.