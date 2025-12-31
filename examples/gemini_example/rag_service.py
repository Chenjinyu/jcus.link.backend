# rag_service.py (Updated Streaming Logic)
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnableLambda, RunnablePassthrough

from .config import llm, retriever  # Import initialized components


# ... (Include format_docs helper function here) ...
def format_docs(docs):
    """Formats the retrieved documents into a single string for the LLM's context."""
    # This structure can help the LLM better cite its sources within the response
    return "\n\n".join(
        f"--- Document Source: {doc.metadata.get('source_filename', 'N/A')} ---\n{doc.page_content}"
        for doc in docs
    )


# Define the Prompt Template (Same as before)
template = """
You are an expert Career Fit Analyst. Your task is to evaluate a candidate's profile against a provided job description.
The candidate's profile is provided in the 'Context' section below, sourced from their embedded documents... (full prompt from step 3)...

--- JOB DESCRIPTION ---
{job_description}

--- CONTEXT (Candidate's Relevant Info) ---
{context}
"""
prompt = ChatPromptTemplate.from_template(template)

# RAG Chain (Note: Output Parser is removed here to stream raw tokens)
rag_chain = (
    {
        "context": retriever | RunnableLambda(format_docs),
        "job_description": RunnablePassthrough(),
    }
    | prompt
    | llm
)


async def stream_job_fit_analysis(job_description_text: str):
    """
    Executes the RAG chain and streams the LLM tokens as an async generator.
    """
    # Use .astream() for asynchronous token streaming in LCEL
    async for chunk in rag_chain.astream(job_description_text):
        # LangChain's astream yields chunks (AIMessageChunk objects);
        # we extract the content string for streaming to the client.
        if chunk.content:
            yield chunk.content
