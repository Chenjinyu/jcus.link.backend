# Easy Record Insertion Guide

This guide shows you how to easily insert records into your vector database.

## Quick Start

### 1. Set Environment Variables

```bash
export SUPABASE_URL="https://your-project.supabase.co"
export SUPABASE_SERVICE_KEY="your-service-key"
export OPENAI_API_KEY="your-openai-key"  # Optional if using Ollama
```

### 2. Run the Example Script

```bash
# Full examples with all record types
python examples/insert_records.py

# Or use the quick insert helper
python examples/quick_insert.py
```

## Available Record Types

### 1. Documents

Basic document insertion:

```python
from libs.vector_database import VectorDatabase

db = VectorDatabase(supabase_url, supabase_key, openai_key)

document_id = await db.add_document(
    user_id="user-123",
    content_type="article",
    title="My Document",
    content="Document content here...",
    tags=["tag1", "tag2"]
)
```

### 2. Articles

Articles with publishing status:

```python
result = await db.add_article(
    user_id="user-123",
    title="My Article",
    content="Article content...",
    subtitle="Subtitle here",
    status="published",  # or "draft"
    tags=["tech", "ai"]
)
```

### 3. Work Experience

```python
result = await db.add_profile_data(
    user_id="user-123",
    category="work_experience",
    data={
        "title": "Senior Developer",
        "company": "Tech Corp",
        "description": "Built amazing products",
        "start_date": "2022-01-01",
        "end_date": None,  # None for current position
        "current": True,
        "technologies": ["Python", "FastAPI"]
    },
    searchable=True
)
```

### 4. Education

```python
result = await db.add_profile_data(
    user_id="user-123",
    category="education",
    data={
        "institution": "University Name",
        "degree": "Bachelor of Science",
        "field_of_study": "Computer Science",
        "start_date": "2018-09-01",
        "end_date": "2022-05-15"
    },
    searchable=True
)
```

### 5. Skills

```python
result = await db.add_profile_data(
    user_id="user-123",
    category="skill",
    data={
        "name": "Python",
        "level": "expert",
        "years_of_experience": 5,
        "description": "Expert in Python programming"
    },
    searchable=True
)
```

### 6. Certifications

```python
result = await db.add_profile_data(
    user_id="user-123",
    category="certification",
    data={
        "name": "AWS Certified",
        "issuer": "Amazon",
        "issue_date": "2023-06-01",
        "expiry_date": "2026-06-01"
    },
    searchable=True
)
```

### 7. Personal Values

```python
result = await db.add_personal_attribute(
    user_id="user-123",
    attribute_type="value",
    title="Continuous Learning",
    description="I believe in constantly improving...",
    examples=["Example 1", "Example 2"],
    importance_score=9
)
```

### 8. Principles

```python
result = await db.add_personal_attribute(
    user_id="user-123",
    attribute_type="principle",
    title="Code Quality",
    description="I prioritize clean code...",
    importance_score=8
)
```

### 9. Aspirations

```python
result = await db.add_personal_attribute(
    user_id="user-123",
    attribute_type="aspiration",
    title="Become Tech Lead",
    description="I want to lead a team...",
    importance_score=10
)
```

## Using QuickInsert Helper

For even simpler usage:

```python
from examples.quick_insert import QuickInsert

quick = QuickInsert(supabase_url, supabase_key, openai_key)

# Insert a job
await quick.job(
    user_id="user-123",
    title="Developer",
    company="Tech Corp",
    description="Built products",
    start_date="2022-01-01",
    technologies=["Python"]
)

# Insert a skill
await quick.skill(
    user_id="user-123",
    name="Python",
    level="expert",
    years=5
)
```

## Batch Insertion

Insert multiple records at once:

```python
# Multiple work experiences
experiences = [
    {"title": "Job 1", "company": "Company A", ...},
    {"title": "Job 2", "company": "Company B", ...}
]

for exp in experiences:
    await db.add_profile_data(
        user_id=user_id,
        category="work_experience",
        data=exp,
        searchable=True
    )
```

## Important Notes

1. **Content Types**: Must exist in `content_types` table. Common types:
   - `article`
   - `work_experience`
   - `education`
   - `skill`
   - `certification`
   - `value`
   - `principle`
   - `aspiration`

2. **Embedding Models**: The database automatically uses all active embedding models unless you specify `model_names`.

3. **Searchable**: Set `searchable=True` to create embeddings for semantic search.

4. **User ID**: Replace `"user-123"` with actual user IDs from your system.

5. **Async**: All insert methods are async, so use `await` or run in an async context.

## Error Handling

```python
try:
    result = await db.add_document(...)
    print(f"Success: {result}")
except ValueError as e:
    print(f"Validation error: {e}")
except Exception as e:
    print(f"Error: {e}")
```

## Next Steps

After inserting records, you can:
- Search using `db.search(query, user_id)`
- Update using `db.update_document(document_id, ...)`
- Delete using `db.delete_document(document_id)`

See `vector_database.py` for all available methods.

