# VectorDatabase Enhancements

## Summary

Enhanced `vector_database.py` with:
1. **Multi-provider embedding support** (OpenAI, Google Gemini, Ollama)
2. **SQL function wrappers** for efficient database operations
3. **Easy model selection** helpers for simplified usage

## New Features

### 1. Google Gemini Support

Added support for Google Gemini embeddings:

```python
db = VectorDatabase(
    supabase_url=url,
    supabase_key=key,
    google_key="your-google-api-key"  # New parameter
)

# Automatically uses Google if model provider is 'google'
embedding = await db.create_embedding(text, model_name="google-text-embedding-004")
```

**Installation:**
```bash
pip install google-generativeai
```

### 2. Model Selection Helpers

Easy methods to work with different providers:

```python
# List all available models
all_models = db.list_available_models()

# List models by provider
openai_models = db.get_models_by_provider('openai')
ollama_models = db.get_models_by_provider('ollama')
google_models = db.get_models_by_provider('google')

# Get default model for a provider
default_ollama = db.get_default_model('ollama')

# Validate a model
is_valid = db.validate_model('nomic-embed-text-768')
```

### 3. Convenience Methods

Simplified methods that automatically select models:

```python
# Add document with automatic model selection
doc_id = await db.add_document_with_model(
    user_id="user-123",
    title="My Document",
    content="Content here...",
    provider='ollama'  # Automatically uses first available Ollama model
)

# Search with automatic model selection
results = await db.search_with_model(
    query="Python programming",
    user_id="user-123",
    provider='openai'  # Automatically uses first available OpenAI model
)
```

### 4. SQL Function Wrappers

Efficient wrappers for SQL functions defined in `functions.sql`:

```python
# Upsert document using SQL function
doc_id = await db.upsert_document_with_embedding_sql(
    user_id="user-123",
    title="Title",
    content="Content",
    embedding_model_name="nomic-embed-text-768"
)

# Add personal attribute using SQL function (uses triggers)
attr_id = await db.add_personal_attribute_sql(
    user_id="user-123",
    attribute_type="value",
    title="Continuous Learning",
    description="I believe in learning...",
    embedding_model_name="nomic-embed-text-768"
)

# Update personal attribute using SQL function
await db.update_personal_attribute_sql(
    attribute_id=attr_id,
    description="Updated description",
    recreate_embedding=True
)
```

## Supported Providers

### OpenAI
- **Models**: `text-embedding-3-small`, `text-embedding-3-large`, etc.
- **Dimensions**: 1536 (small), 3072 (large), or custom
- **Setup**: Provide `openai_key` in constructor

### Google Gemini
- **Models**: `text-embedding-004`
- **Dimensions**: 768
- **Setup**: Provide `google_key` in constructor or set `GOOGLE_API_KEY` env var
- **Installation**: `pip install google-generativeai`

### Ollama (Local)
- **Models**: `nomic-embed-text`, `mxbai-embed-large`, etc.
- **Dimensions**: Varies (768, 1024, etc.)
- **Setup**: Run Ollama locally, default URL: `http://localhost:11434`

## Usage Examples

### Basic Usage with Provider Selection

```python
from libs.vector_database import VectorDatabase

# Initialize with all providers
db = VectorDatabase(
    supabase_url=os.environ["SUPABASE_URL"],
    supabase_key=os.environ["SUPABASE_SERVICE_KEY"],
    openai_key=os.environ.get("OPENAI_API_KEY"),
    google_key=os.environ.get("GOOGLE_API_KEY"),
    ollama_url="http://localhost:11434"
)

# Use Ollama (local, free)
doc_id = await db.add_document_with_model(
    user_id="user-123",
    title="Python Guide",
    content="Python is great...",
    provider='ollama'
)

# Use OpenAI (cloud, paid)
doc_id = await db.add_document_with_model(
    user_id="user-123",
    title="AI Guide",
    content="AI is transforming...",
    provider='openai'
)

# Use Google Gemini (cloud, paid)
doc_id = await db.add_document_with_model(
    user_id="user-123",
    title="ML Guide",
    content="Machine learning...",
    provider='google'
)
```

### Advanced Usage with Specific Models

```python
# Use specific model
doc_id = await db.add_document(
    user_id="user-123",
    title="Document",
    content="Content",
    model_names=["nomic-embed-text-768", "openai-small"]  # Multiple models
)

# Search with specific model
results = await db.search(
    query="Python programming",
    user_id="user-123",
    model_name="nomic-embed-text-768"
)
```

## Database Setup

Ensure your `embedding_models` table has entries for all providers:

```sql
-- OpenAI model
INSERT INTO embedding_models (name, provider, model_identifier, dimensions, is_active)
VALUES ('openai-small', 'openai', 'text-embedding-3-small', 1536, true);

-- Google model
INSERT INTO embedding_models (name, provider, model_identifier, dimensions, is_active)
VALUES ('google-text-embedding-004', 'google', 'text-embedding-004', 768, true);

-- Ollama model
INSERT INTO embedding_models (name, provider, model_identifier, dimensions, is_active, is_local)
VALUES ('nomic-embed-text-768', 'ollama', 'nomic-embed-text', 768, true, true);
```

## Error Handling

The library gracefully handles missing providers:

- If Google package not installed: Warning message, Google methods will raise errors
- If API key missing: Clear error message indicating what's needed
- If model not found: Lists available models in error message

## Migration Notes

### Breaking Changes
None - all new features are additive.

### New Dependencies
- `google-generativeai` (optional, only if using Google Gemini)

### Environment Variables
- `GOOGLE_API_KEY` (optional, for Google Gemini)

## Performance Tips

1. **Use SQL function wrappers** for batch operations (more efficient)
2. **Use local Ollama** for development (free, fast)
3. **Use cloud providers** (OpenAI/Google) for production (better quality)
4. **Cache embeddings** when possible to reduce API calls

## See Also

- `examples/easy_insert_example.py` - Complete usage examples
- `db_migration/supabase/functions.sql` - SQL function definitions
- `db_migration/supabase/create_tables.sql` - Database schema

