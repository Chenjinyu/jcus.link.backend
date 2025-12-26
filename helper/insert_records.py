
from helper.insert_records_helper import *


# ============================================================================
# MAIN FUNCTION
# ============================================================================

async def main():
    """Main function to run examples"""
    print("=" * 70)
    print("Vector Database - Easy Record Insertion Examples")
    print("=" * 70)
    
    # Check configuration
    if SUPABASE_URL == "your-supabase-url" or SUPABASE_KEY == "your-supabase-key":
        print("\n⚠️  WARNING: Please set SUPABASE_URL and SUPABASE_SERVICE_KEY environment variables")
        print("   or update them in this script.")
        print("\n   Example:")
        print("   export SUPABASE_URL='https://your-project.supabase.co'")
        print("   export SUPABASE_SERVICE_KEY='your-service-key'")
        return
    
    # Initialize database
    print("\n🔌 Connecting to database...")
    try:
        db = create_db()
        print("✅ Connected successfully!")
    except Exception as e:
        print(f"❌ Connection failed: {e}")
        return
    
    # Run examples
    try:
        # Basic examples
        await insert_document_example(db, JC_USER_ID)
        await insert_article_example(db, JC_USER_ID)
        
        # Profile data examples
        await insert_work_experience_example(db, JC_USER_ID)
        await insert_education_example(db, JC_USER_ID)
        await insert_skill_example(db, JC_USER_ID)
        await insert_certification_example(db, JC_USER_ID)
        
        # Personal attributes examples
        await insert_value_example(db, JC_USER_ID)
        await insert_principle_example(db, JC_USER_ID)
        await insert_aspiration_example(db, JC_USER_ID)
        
        # Batch examples (uncomment to use)
        # await insert_multiple_work_experiences(db, JC_USER_ID)
        # await insert_multiple_skills(db, JC_USER_ID)
        
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

