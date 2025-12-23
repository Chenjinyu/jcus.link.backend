import asyncio
from vector_database import VectorDatabase

async def main():
    # Initialize
    db = VectorDatabase(
        supabase_url='YOUR_SUPABASE_URL',
        supabase_key='YOUR_SUPABASE_KEY',
        openai_key='YOUR_OPENAI_KEY'
    )
    
    # ========================================================================
    # Example 1: Add work experience (multi-model)
    # ========================================================================
    await db.add_profile_data(
        user_id='jc123',
        category='work_experience',
        data={
            'id': 'work_1',
            'company': 'Anthropic',
            'title': 'Senior Software Engineer',
            'description': 'Working on AI safety and alignment. Building scalable ML systems.',
            'start_date': '2024-01',
            'current': True,
            'skills': ['Python', 'React', 'AI/ML', 'Vector Databases']
        },
        model_names=['openai-small', 'ollama-nomic']  # Create embeddings with both!
    )
    
    # ========================================================================
    # Example 2: Add article (production + local models)
    # ========================================================================
    article_id = await db.add_article(
        user_id='jc123',
        title='Understanding Vector Databases',
        content='''
        Vector databases are essential for modern AI applications...
        [Your full article content]
        ''',
        tags=['AI', 'Databases', 'Machine Learning'],
        category='Technical',
        status='published',
        model_names=['openai-small', 'ollama-nomic']  # Dual embeddings!
    )
    
    # ========================================================================
    # Example 3: Add soft skill
    # ========================================================================
    await db.add_profile_data(
        user_id='jc123',
        category='value',
        data={
            'id': 'value_1',
            'title': 'Continuous Learning',
            'description': 'I believe in constantly learning and adapting to new technologies',
            'examples': [
                'Learned Next.js 16 with Turbopack in 2 weeks',
                'Built MCP server from scratch to understand the protocol'
            ],
            'importance': 9
        }
    )
    
    # ========================================================================
    # Example 4: Search (production - OpenAI)
    # ========================================================================
    results = await db.search(
        query='What is JC\'s experience with AI and vector databases?',
        user_id='jc123',
        model_name='openai-small',  # Use OpenAI for production
        content_types=['work_experience', 'article'],
        limit=5
    )
    
    for result in results:
        print(f"Title: {result['title']}")
        print(f"Similarity: {result['similarity']:.2f}")
        print(f"Content: {result['content'][:200]}...")
        print()
    
    # ========================================================================
    # Example 5: Search (local - Ollama for development)
    # ========================================================================
    local_results = await db.search(
        query='What are JC\'s values?',
        user_id='jc123',
        model_name='ollama-nomic',  # Use local model for testing
        content_types=['value', 'worldview'],
        limit=5
    )
    
    # ========================================================================
    # Example 6: Update work experience
    # ========================================================================
    await db.update_document(
        document_id='existing_doc_id',
        content='Updated description with new achievements...',
        recreate_embeddings=True  # Regenerate all embeddings
    )
    
    # ========================================================================
    # Example 7: List all articles
    # ========================================================================
    articles = db.supabase.table('articles').select('*').eq(
        'user_id', 'jc123'
    ).eq('status', 'published').order('published_at', desc=True).execute()
    
    for article in articles.data:
        print(f"{article['title']} - {article['published_at']}")

if __name__ == '__main__':
    asyncio.run(main())