"""
Easy Insert Example - Simple way to insert records with automatic model selection

This example shows how to easily insert records using the enhanced VectorDatabase
with automatic model selection for OpenAI, Google Gemini, or Ollama.
"""

import asyncio
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from libs.vector_database import VectorDatabase


async def main():
    """Example usage with easy model selection"""

    # Initialize with all API keys (only needed for the providers you want to use)
    db = VectorDatabase(
        supabase_url=os.environ.get("SUPABASE_URL", "your-supabase-url"),
        supabase_key=os.environ.get("SUPABASE_SERVICE_KEY", "your-supabase-key"),
        openai_key=os.environ.get("OPENAI_API_KEY"),  # Optional
        google_key=os.environ.get("GOOGLE_API_KEY"),  # Optional
        ollama_url=os.environ.get("OLLAMA_URL", "http://localhost:11434"),
    )

    user_id = "user-123"

    print("=" * 70)
    print("Available Models by Provider")
    print("=" * 70)

    # List all available models
    all_models = db.list_available_models()
    print(f"\nAll models ({len(all_models)}):")
    for model in all_models:
        print(f"  - {model['name']} ({model['provider']}, {model['dimensions']}D)")

    # List by provider
    print("\n📊 OpenAI models:")
    openai_models = db.get_models_by_provider("openai")
    for name in openai_models:
        print(f"  - {name}")

    print("\n🦙 Ollama models:")
    ollama_models = db.get_models_by_provider("ollama")
    for name in ollama_models:
        print(f"  - {name}")

    print("\n🔷 Google models:")
    google_models = db.get_models_by_provider("google")
    for name in google_models:
        print(f"  - {name}")

    print("\n" + "=" * 70)
    print("Easy Insert Examples")
    print("=" * 70)

    # Example 1: Insert document with Ollama (default)
    print("\n1️⃣ Inserting document with Ollama (automatic model selection)...")
    doc_id = await db.add_document_with_model(
        user_id=user_id,
        title="Python Tutorial",
        content="Python is a powerful programming language...",
        provider="ollama",  # Automatically uses first available Ollama model
    )
    print(f"   ✅ Document ID: {doc_id}")

    # Example 2: Insert document with OpenAI
    print("\n2️⃣ Inserting document with OpenAI...")
    try:
        doc_id = await db.add_document_with_model(
            user_id=user_id,
            title="AI Guide",
            content="Artificial Intelligence is transforming...",
            provider="openai",  # Automatically uses first available OpenAI model
        )
        print(f"   ✅ Document ID: {doc_id}")
    except Exception as e:
        print(f"   ⚠️  OpenAI not available: {e}")

    # Example 3: Insert document with Google Gemini
    print("\n3️⃣ Inserting document with Google Gemini...")
    try:
        doc_id = await db.add_document_with_model(
            user_id=user_id,
            title="Machine Learning Basics",
            content="Machine learning is a subset of AI...",
            provider="google",  # Automatically uses first available Google model
        )
        print(f"   ✅ Document ID: {doc_id}")
    except Exception as e:
        print(f"   ⚠️  Google not available: {e}")

    # Example 4: Insert with specific model name
    print("\n4️⃣ Inserting document with specific model...")
    doc_id = await db.add_document_with_model(
        user_id=user_id,
        title="Custom Model Document",
        content="This uses a specific model...",
        provider="ollama",
        model_name="nomic-embed-text-768",  # Specify exact model
    )
    print(f"   ✅ Document ID: {doc_id}")

    # Example 5: Insert work experience
    print("\n5️⃣ Inserting work experience...")
    result = await db.add_profile_data(
        user_id=user_id,
        category="work_experience",
        data={
            "title": "Senior Developer",
            "company": "Tech Corp",
            "description": "Built amazing products with Python and FastAPI",
            "start_date": "2022-01-01",
            "current": True,
            "technologies": ["Python", "FastAPI", "PostgreSQL"],
        },
        searchable=True,
        model_names=[db.get_default_model("ollama")],  # Use default Ollama model
    )
    print(f"   ✅ Profile ID: {result['profile_id']}")

    # Example 6: Search with automatic model selection
    print("\n6️⃣ Searching with Ollama model...")
    results = await db.search_with_model(
        query="Python developer experience",
        user_id=user_id,
        provider="ollama",  # Automatically selects Ollama model
        limit=5,
    )
    print(f"   ✅ Found {len(results)} results")
    for r in results[:3]:
        print(f"      - {r['title']} (similarity: {r['similarity']:.2f})")

    # Example 7: Using SQL function wrapper
    print("\n7️⃣ Using SQL function wrapper for personal attribute...")
    try:
        attr_id = await db.add_personal_attribute_sql(
            user_id=user_id,
            attribute_type="value",
            title="Continuous Learning",
            description="I believe in constantly improving my skills",
            examples=["Completed online courses", "Attend meetups"],
            importance_score=9,
            embedding_model_name=db.get_default_model("ollama")
            or "nomic-embed-text-768",
        )
        print(f"   ✅ Attribute ID: {attr_id}")
    except Exception as e:
        print(f"   ⚠️  Error: {e}")

    print("\n" + "=" * 70)
    print("✅ All examples completed!")
    print("=" * 70)


if __name__ == "__main__":
    asyncio.run(main())
