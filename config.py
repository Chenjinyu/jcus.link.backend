import os
from supabase.client import Client, create_client
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import SupabaseVectorStore
from langchain_openai import ChatOpenAI

# Load environment variables (ensure they are set in your OS/deployment environment)
SUPABASE_URL = os.environ.get("SUPABASE_URL")
SUPABASE_KEY = os.environ.get("SUPABASE_SERVICE_KEY")
OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY")

if not all([SUPABASE_URL, SUPABASE_KEY, OPENAI_API_KEY]):
    raise EnvironmentError("Missing required environment variables (SUPABASE_URL, SUPABASE_SERVICE_KEY, OPENAI_API_KEY).")

# Initialize shared components
supabase_client: Client = create_client(SUPABASE_URL, SUPABASE_KEY)
embeddings = OpenAIEmbeddings()
llm = ChatOpenAI(model="gpt-4o", temperature=0.2)

# Instantiate the vector store for retrieval (assuming data is already indexed)
vector_store = SupabaseVectorStore(
    embedding=embeddings,
    client=supabase_client,
    table_name="documents",
    query_name="match_documents",
)

# Shared RAG components
retriever = vector_store.as_retriever(search_type="mmr", k=4)