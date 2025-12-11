"""
Example usage of the Document Analysis MCP Server

This script demonstrates how to use the MCP server programmatically
"""

import asyncio
import json

# Mock MCP tool call function (replace with actual MCP client)
async def call_mcp_tool(tool_name: str, params: dict) -> dict:
    """
    This is a placeholder function. In production, you would use
    an actual MCP client to communicate with the server.
    """
    print(f"\n{'='*60}")
    print(f"Calling tool: {tool_name}")
    print(f"Parameters: {json.dumps(params, indent=2)}")
    print(f"{'='*60}\n")
    
    # In production, this would send the request to the MCP server
    # and return the actual response
    return {"status": "example_response"}


async def example_1_extract_from_web():
    """Example 1: Extract job description from a web page"""
    print("Example 1: Extract job description from web")
    
    result = await call_mcp_tool("extract_content", {
        "file_type": "web",
        "url": "https://example.com/job-posting",
        "max_length": 5000
    })
    
    print(f"Result: {result}")
    return result


async def example_2_create_profile_embeddings():
    """Example 2: Create embeddings from resume text"""
    print("\nExample 2: Create embeddings from profile")
    
    resume_text = """
    John Doe
    Senior Python Developer
    
    Experience:
    - 5 years of Python development
    - Expert in Django and FastAPI frameworks
    - Built scalable microservices handling 1M+ requests/day
    - Led team of 4 developers
    - Strong background in machine learning with TensorFlow and PyTorch
    
    Skills:
    - Python, JavaScript, TypeScript
    - Docker, Kubernetes, AWS
    - PostgreSQL, MongoDB, Redis
    - CI/CD with GitHub Actions
    - Agile/Scrum methodologies
    
    Education:
    - BS Computer Science, MIT (2018)
    - MS Artificial Intelligence, Stanford (2020)
    """
    
    result = await call_mcp_tool("create_embeddings", {
        "content": resume_text,
        "collection_name": "profile",
        "chunk_size": 500,
        "chunk_overlap": 100,
        "embedding_provider": "ollama",
        "embedding_model": "nomic-embed-text",
        "metadata": {
            "source": "resume",
            "name": "John Doe",
            "date": "2024-01-15"
        }
    })
    
    print(f"Result: {result}")
    return result


async def example_3_query_relevant_experience():
    """Example 3: Query for relevant experience"""
    print("\nExample 3: Query for machine learning experience")
    
    result = await call_mcp_tool("query_embeddings", {
        "query": "machine learning and AI experience with Python",
        "collection_name": "profile",
        "top_k": 3,
        "embedding_provider": "ollama",
        "embedding_model": "nomic-embed-text",
        "response_format": "markdown"
    })
    
    print(f"Result: {result}")
    return result


async def example_4_analyze_job_fit():
    """Example 4: Analyze job description fit"""
    print("\nExample 4: Analyze job description fit")
    
    job_description = """
    Senior Python Developer
    
    We are looking for an experienced Python developer to join our team.
    
    Required Qualifications:
    - 5+ years of Python development experience
    - Strong experience with Django or FastAPI
    - Experience with Docker and Kubernetes
    - Knowledge of AWS cloud services
    - Must have experience with microservices architecture
    - Strong understanding of RESTful APIs
    
    Preferred Qualifications:
    - Experience with machine learning frameworks
    - Knowledge of CI/CD pipelines
    - Leadership experience
    - Master's degree in Computer Science or related field
    
    Responsibilities:
    - Design and implement scalable backend services
    - Lead technical discussions and code reviews
    - Mentor junior developers
    - Collaborate with product and design teams
    """
    
    result = await call_mcp_tool("analyze_job_description", {
        "job_description": job_description,
        "profile_collection": "profile",
        "top_k_matches": 10,
        "embedding_provider": "ollama",
        "embedding_model": "nomic-embed-text",
        "response_format": "markdown"
    })
    
    print(f"Result: {result}")
    return result


async def example_5_extract_pdf():
    """Example 5: Extract content from PDF"""
    print("\nExample 5: Extract content from PDF")
    
    # In production, you would read actual file and encode to base64
    result = await call_mcp_tool("extract_content", {
        "file_type": "pdf",
        "content": "/path/to/resume.pdf"  # or use base64_content
    })
    
    print(f"Result: {result}")
    return result


async def example_6_complete_workflow():
    """Example 6: Complete workflow from file upload to analysis"""
    print("\nExample 6: Complete workflow")
    print("="*60)
    
    # Step 1: Extract resume from uploaded file
    print("\nStep 1: Extract resume content")
    extracted = await call_mcp_tool("extract_content", {
        "file_type": "pdf",
        "content": "/path/to/resume.pdf"
    })
    
    # Step 2: Create embeddings
    print("\nStep 2: Create profile embeddings")
    embeddings = await call_mcp_tool("create_embeddings", {
        "content": "extracted_resume_text_here",
        "collection_name": "profile",
        "metadata": {"source": "resume_upload"}
    })
    
    # Step 3: Extract job description
    print("\nStep 3: Extract job description")
    job_desc = await call_mcp_tool("extract_content", {
        "file_type": "web",
        "url": "https://example.com/job"
    })
    
    # Step 4: Analyze fit
    print("\nStep 4: Analyze job fit")
    analysis = await call_mcp_tool("analyze_job_description", {
        "job_description": "extracted_job_description_here",
        "profile_collection": "profile"
    })
    
    print("\n" + "="*60)
    print("Workflow complete!")
    return analysis


async def example_7_with_metadata_filtering():
    """Example 7: Query with metadata filtering"""
    print("\nExample 7: Query with metadata filtering")
    
    result = await call_mcp_tool("query_embeddings", {
        "query": "leadership and team management",
        "collection_name": "profile",
        "top_k": 5,
        "filter_metadata": {
            "source": "resume",
            "section": "experience"
        },
        "response_format": "json"
    })
    
    print(f"Result: {result}")
    return result


async def example_8_multiple_providers():
    """Example 8: Using different embedding providers"""
    print("\nExample 8: Comparing Ollama vs OpenAI embeddings")
    
    content = "Python developer with ML expertise"
    
    # Using Ollama (local)
    print("\nCreating embeddings with Ollama...")
    ollama_result = await call_mcp_tool("create_embeddings", {
        "content": content,
        "collection_name": "test_ollama",
        "embedding_provider": "ollama",
        "embedding_model": "nomic-embed-text"
    })
    
    # Using OpenAI (cloud) - requires API key
    print("\nCreating embeddings with OpenAI...")
    openai_result = await call_mcp_tool("create_embeddings", {
        "content": content,
        "collection_name": "test_openai",
        "embedding_provider": "openai",
        "embedding_model": "text-embedding-3-small"
    })
    
    print("Note: These two collections are NOT compatible!")
    print("Always use the same provider/model for index and query.")
    
    return {"ollama": ollama_result, "openai": openai_result}


async def main():
    """Run all examples"""
    print("Document Analysis MCP Server - Usage Examples")
    print("="*60)
    print("\nNote: These are example calls. In production, use actual MCP client.")
    print()
    
    # Run examples
    await example_1_extract_from_web()
    await example_2_create_profile_embeddings()
    await example_3_query_relevant_experience()
    await example_4_analyze_job_fit()
    await example_5_extract_pdf()
    await example_6_complete_workflow()
    await example_7_with_metadata_filtering()
    await example_8_multiple_providers()
    
    print("\n" + "="*60)
    print("All examples completed!")
    print("\nTo use these with actual MCP server:")
    print("1. Start the server: python document_analysis_mcp.py")
    print("2. Connect with MCP client")
    print("3. Call tools using the examples above")


if __name__ == "__main__":
    asyncio.run(main())