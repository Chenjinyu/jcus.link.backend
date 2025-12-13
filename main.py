import tempfile
from fastapi import FastAPI, UploadFile, File, HTTPException
from pydantic import BaseModel
from typing import List

from .ingestion_service import ingest_document_to_vector_store
from .rag_service import analyze_job_fit_async

# --- FastAPI Initialization ---
app = FastAPI(title="MCP RAG Server API", version="1.0")

# --- Pydantic Models for Input/Output Validation ---
class IngestResponse(BaseModel):
    filename: str
    status: str
    chunks_ingested: int
    
class JobFitRequest(BaseModel):
    job_description: str

class JobFitResponse(BaseModel):
    analysis_result: str
    

# --- 1. File Ingestion Endpoint ---
@app.post("/ingest/upload", response_model=IngestResponse, summary="Uploads and indexes documents into the Vector Store.")
async def upload_document(file: UploadFile = File(...)):
    """
    Accepts various file types (PDF, Word, Text, etc.) and indexes them
    into the Supabase vector store for RAG retrieval.
    """
    try:
        # FastAPI's UploadFile reads file data asynchronously
        file_bytes = await file.read()
        
        # Use a temporary file for LangChain loaders that require a path
        with tempfile.NamedTemporaryFile(delete=False, suffix=f"_{file.filename}") as tmp_file:
            tmp_file.write(file_bytes)
            tmp_path = tmp_file.name

        # The core ingestion logic is synchronous in LangChain/Supabase, 
        # so we run it in a thread pool to avoid blocking the main event loop.
        # If your ingestion_service.py was fully async, you could use await directly.
        
        # NOTE: For simplicity, the sync ingest_document_to_vector_store 
        # needs to be adapted or wrapped to run in the background thread.
        # For a truly best-practice FastAPI app, you should make your 
        # ingest_document_to_vector_store fully async (using async loaders/clients).
        
        # Assuming you've converted your ingestion_service to be fully async:
        ingested_docs = await ingest_document_to_vector_store_async(
            file_bytes, file.filename, file.content_type
        )
        
        return IngestResponse(
            filename=file.filename,
            status="Success",
            chunks_ingested=len(ingested_docs)
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Ingestion failed: {e}")
    finally:
        # Cleanup of the temp file should also happen here if a path was used
        pass 

# --- 2. Job Fit Analysis Endpoint (The core Use Case) ---
@app.post("/analyze/job-fit", response_model=JobFitResponse, summary="Analyzes job fit using RAG on indexed documents.")
async def analyze_fit(request: JobFitRequest):
    """
    Analyzes the provided job description against the web app owner's
    indexed documents (CV, portfolio, etc.) and returns a structured analysis.
    """
    try:
        # This calls the asynchronous RAG chain function
        analysis = await analyze_job_fit_async(request.job_description)
        return JobFitResponse(analysis_result=analysis)
    except Exception as e:
        # Log the error (best practice)
        print(f"RAG Analysis Error: {e}")
        raise HTTPException(status_code=500, detail="Failed to run job fit analysis.")