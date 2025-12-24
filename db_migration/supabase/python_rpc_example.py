from supabase import create_client
import openai

# Initialize
supabase = create_client(SUPABASE_URL, SUPABASE_KEY)
openai_client = openai.OpenAI(api_key=OPENAI_KEY)

async def search_my_content(query: str, user_id: str):
    # 1. Create embedding for query
    response = openai_client.embeddings.create(
        model='text-embedding-3-small',
        input=query
    )
    query_embedding = response.data[0].embedding
    
    # 2. Call the PostgreSQL function
    results = supabase.rpc('search_all_content', {
        'query_embedding': query_embedding,  # vector parameter
        'user_id_filter': user_id,            # TEXT parameter
        'match_threshold': 0.75,              # FLOAT parameter (optional)
        'match_count': 10                     # INT parameter (optional)
    }).execute()
    
    # 3. Process results
    for row in results.data:
        print(f"Type: {row['content_type']}")
        print(f"Title: {row['title']}")
        print(f"Similarity: {row['similarity']:.2f}")
        
        if row['article_id']:
            print(f"Article ID: {row['article_id']}")
        elif row['profile_id']:
            print(f"Profile ID: {row['profile_id']}")
        
        print(f"Excerpt: {row['chunk_text'][:100]}...")
        print()
    
    return results.data

# Usage
results = await search_my_content(
    query="What is JC's current job?",
    user_id="jc123"
)