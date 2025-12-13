# rag_service.py
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough, RunnableLambda
from langchain_core.output_parsers import StrOutputParser
from .config import llm, retriever # Import initialized components

# ... (Include the format_docs helper function here) ...
def format_docs(docs):
    """Formats the retrieved documents into a single string for the LLM's context."""
    return "\n\n".join(f"--- Document Source: {doc.metadata.get('source_filename', 'N/A')} ---\n{doc.page_content}" for doc in docs)


# Define the Prompt Template (Same as before)
template = """
You are an expert Career Fit Analyst... (full prompt from step 3)...
--- JOB DESCRIPTION ---
{job_description}

--- CONTEXT (Candidate's Relevant Info) ---
{context}
"""
prompt = ChatPromptTemplate.from_template(template)

# RAG Chain
rag_chain = (
    {"context": retriever | RunnableLambda(format_docs), "job_description": RunnablePassthrough()}
    | prompt
    | llm
    | StrOutputParser()
)

async def analyze_job_fit_async(job_description_text: str) -> str:
    """
    Executes the RAG chain asynchronously for API performance.
    """
    # Use .ainvoke() for asynchronous chain execution
    analysis_result = await rag_chain.ainvoke(job_description_text)
    return analysis_result