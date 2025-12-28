
import os
import helper.insert_records_helper as db_helper
import asyncio
import dotenv

dotenv.load_dotenv()
# ============================================================================
# CONFIGURATION
# ============================================================================

# Load from environment variables or set directly
SUPABASE_URL = os.environ.get("SUPABASE_URL", "your-supabase-url")
print("="* 20 + SUPABASE_URL + "="*20)
SUPABASE_SERVICE_KEY = os.environ.get("SUPABASE_SERVICE_ROLE_KEY", "your-supabase-key")
print("="* 20 + SUPABASE_SERVICE_KEY + "="*20)
OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY", None)  # Optional if using Ollama
OLLAMA_URL = os.environ.get("OLLAMA_URL", "http://localhost:11434")

# Test user ID (replace with actual user ID)
JC_USER_ID = "jinyu.chen"

# ============================================================================
# MAIN FUNCTION
# ============================================================================

async def main():
    """Main function to run examples"""
    print("=" * 70)
    print("Vector Database - Easy Record Insertion Examples")
    print("=" * 70)
    # Check configuration
    if SUPABASE_URL == "your-supabase-url" or SUPABASE_SERVICE_KEY == "your-supabase-key":
        print("\n⚠️  WARNING: Please set SUPABASE_URL and SUPABASE_SERVICE_KEY environment variables")
        print("   or update them in this script.")
        print("\n   Example:")
        print("   export SUPABASE_URL='https://your-project.supabase.co'")
        print("   export SUPABASE_SERVICE_KEY='your-service-key'")
        return
    
    # Initialize database
    print("\n🔌 Connecting to database...")
    try:
        db = db_helper.create_db(
            supabase_url=SUPABASE_URL,
            supabase_key=SUPABASE_SERVICE_KEY,
            ollama_url=OLLAMA_URL
        )
        print("✅ Connected successfully!")
    except Exception as e:
        print(f"❌ Connection failed: {e}")
        return
    
    # Run examples
    try:
        # Basic examples
        await db_helper.insert_document_example(db, JC_USER_ID)
        # await db_helper.insert_article_example(db, JC_USER_ID)
        
        # # Profile data examples
        # await db_helper.insert_work_experience_example(db, JC_USER_ID)
        # await db_helper.insert_education_example(db, JC_USER_ID)
        # await db_helper.insert_skill_example(db, JC_USER_ID)
        # await db_helper.insert_certification_example(db, JC_USER_ID)
        
        # # Personal attributes examples
        # await db_helper.insert_value_example(db, JC_USER_ID)
        # await db_helper.insert_principle_example(db, JC_USER_ID)
        # await db_helper.insert_aspiration_example(db, JC_USER_ID)
        
        # Batch examples (uncomment to use)
        # await db_helper.insert_multiple_work_experiences(db, JC_USER_ID)
        # await db_helper.insert_multiple_skills(db, JC_USER_ID)
        
        print("\n" + "=" * 70)
        print("✅ All examples completed successfully!")
        print("=" * 70)
        
    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    # Run async main function
    asyncio.run(main())

