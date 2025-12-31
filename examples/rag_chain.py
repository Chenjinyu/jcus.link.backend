import os

from langchain_community.vectorstores import SupabaseVectorStore
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnableLambda, RunnablePassthrough
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from supabase.client import Client, create_client

# --- Initialization (Reuse from Ingestion Step) ---
# Ensure environment variables are set before running
supabase_url = os.environ.get("SUPABASE_URL")
supabase_key = os.environ.get("SUPABASE_SERVICE_KEY")

if not all([supabase_url, supabase_key]):
    raise ValueError("SUPABASE_URL and SUPABASE_SERVICE_KEY must be set.")

supabase: Client = create_client(supabase_url, supabase_key)
embeddings = OpenAIEmbeddings()

# 1. Instantiate the LLM (Using a modern model like GPT-4 or Gemini for complex reasoning)
llm = ChatOpenAI(
    model="gpt-4o", temperature=0.2
)  # Use temperature > 0 for creative analysis

# 2. Instantiate the Vector Store and Retriever
# Note: You instantiate the vector store directly here since the data is already in the DB.
vector_store = SupabaseVectorStore(
    embedding=embeddings,
    client=supabase,
    table_name="documents",
    query_name="match_documents",
)

# Convert the vector store to a retriever. Using 'mmr' (Maximal Marginal Relevance)
# is often better than 'similarity' as it retrieves chunks that are both relevant AND diverse,
# which is good when comparing a job description to a comprehensive resume/portfolio.
retriever = vector_store.as_retriever(
    search_type="mmr", k=4
)  # k=4 retrieves 4 relevant chunks

# 3. Define the Prompt Template (The heart of the analysis)
# We use a ChatPromptTemplate to define a clear System Message, which is best practice.
template = """
You are an expert Career Fit Analyst. Your task is to evaluate a candidate's profile against a provided job description.
The candidate's profile is provided in the 'Context' section below, sourced from their embedded documents (resume, portfolio, etc.).

Analyze the Job Description and the Context to determine the candidate's suitability.

--- GUIDELINES ---
1.  **Suitability**: Directly state if the candidate is a Strong Fit, Good Fit, Moderate Fit, or Poor Fit.
2.  **Related Info (If Yes)**: If the candidate is a Good or Strong Fit, list the *top 3* pieces of evidence from the 'Context' that directly support the fit (e.g., specific projects, years of experience, key skills).
3.  **Gap Analysis (If Not/Moderate)**: If there are gaps, list the *top 3* most significant missing skills, experiences, or certifications required by the Job Description that were *not* found in the Context.
4.  **Recommendations**: Provide 2-3 actionable, high-impact suggestions for the candidate to close the identified gaps or further strengthen their application.

--- OUTPUT FORMAT (MUST FOLLOW THIS) ---
**FIT RATING:** [Strong/Good/Moderate/Poor Fit]

**Related Information/Strengths:**
* [Evidence 1]
* [Evidence 2]
* [Evidence 3]

**Gaps/Missing Requirements:**
* [Gap 1]
* [Gap 2]
* [Gap 3]

**Actionable Recommendations:**
* [Recommendation 1]
* [Recommendation 2]
* [Recommendation 3 (Optional)]

--- JOB DESCRIPTION ---
{job_description}

--- CONTEXT (Candidate's Relevant Info) ---
{context}
"""

prompt = ChatPromptTemplate.from_template(template)


# 4. Construct the RAG Chain using LCEL
def format_docs(docs):
    """Formats the retrieved documents into a single string for the LLM's context."""
    return "\n\n".join(doc.page_content for doc in docs)


# LCEL allows you to define a fluid data flow:
rag_chain = (
    # Step 1: Pass the job_description query to the retriever
    {
        "context": retriever | RunnableLambda(format_docs),
        "job_description": RunnablePassthrough(),
    }
    # Step 2: Combine context and query with the prompt
    | prompt
    # Step 3: Call the LLM with the structured prompt
    | llm
    # Step 4: Parse the output back to a simple string
    | StrOutputParser()
)


# --- Function to call from your MCP Server API ---
def analyze_job_fit(job_description_text: str) -> str:
    """
    Executes the RAG chain to analyze the job fit.
    """
    # The job_description_text is passed as the input to the RAG chain.
    analysis_result = rag_chain.invoke(job_description_text)
    return analysis_result


# --- Example Usage (Conceptual) ---
# job_query = "Upload the job description text here, e.g., 'Senior Python Engineer with 5+ years of experience in Django, AWS, and vector databases like Supabase/pgvector.'"
# print(analyze_job_fit(job_query))
