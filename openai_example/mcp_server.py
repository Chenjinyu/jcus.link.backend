# mcp_server.py
from fastmcp import FastMCP
from file_loaders import load_file_content
from embeddings import get_embedding
from vector_search import query_supabase
from summarizer import summarize

app = FastMCP()

@app.tool()
async def process_file(file_path: str, model: str = "openai"):
    """
    1. Extract file content
    2. Generate embedding
    3. Query supabase vector DB
    4. Summarize results
    """
    # 1. extract
    content = load_file_content(file_path)
    # 2. embed
    embedding = get_embedding(content, model)
    # 3. vector search
    related_docs = query_supabase(embedding)
    # 4. summarize
    summary = summarize(content, related_docs)

    return summary

if __name__ == "__main__":
    app.run()
