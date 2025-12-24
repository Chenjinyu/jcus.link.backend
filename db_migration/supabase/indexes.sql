-- ============================================================================
-- VECTOR INDEXES (One per embedding dimension)
-- ivfflat -> inverted File with Flat Compression. 
-- it's a special index type for vector similarity search provided by the pgvector extension in PostgreSQL.
-- hnsw - Hierarchical Navigable Small World. It's a graph-based index for vector similarity search.
-- ivfflat - Fast build, good for most cases.

-- ============================================================================
-- Step 1: Get the model IDs
SELECT id, name, dimensions FROM embedding_models;

-- Step 2: Use the actual UUIDs in the index
-- Replace 'your-model-id-here' with actual UUID from step 1

-- For 768 dimensions (Ollama)
CREATE INDEX embeddings_768_idx ON embeddings 
USING ivfflat (embedding vector_cosine_ops)
WITH (lists = 50)
WHERE embedding_model_id = 'abc-123-def-456'::UUID;  -- ← Actual UUID

-- For 1536 dimensions (OpenAI small)
CREATE INDEX embeddings_1536_idx ON embeddings 
USING ivfflat (embedding vector_cosine_ops)
WITH (lists = 100)
WHERE embedding_model_id = 'xyz-789-ghi-012'::UUID;  -- ← Actual UUID

-- For 3072 dimensions (OpenAI large)
CREATE INDEX embeddings_3072_idx ON embeddings 
USING ivfflat (embedding vector_cosine_ops)
WITH (lists = 150)
WHERE embedding_model_id = 'uvw-345-rst-678'::UUID;  -- ← Actual UUID

-- ============================================================================
-- REGULAR INDEXES
-- ============================================================================

-- Documents
CREATE INDEX documents_user_id_idx ON documents(user_id);
CREATE INDEX documents_content_type_idx ON documents(content_type_id);
CREATE INDEX documents_status_idx ON documents(status) WHERE deleted_at IS NULL;
CREATE INDEX documents_current_idx ON documents(is_current) WHERE is_current = TRUE;
CREATE INDEX documents_tags_idx ON documents USING GIN(tags);
CREATE INDEX documents_created_at_idx ON documents(created_at DESC);

-- Embeddings
CREATE INDEX embeddings_document_id_idx ON embeddings(document_id);
CREATE INDEX embeddings_model_id_idx ON embeddings(embedding_model_id);


-- Create unique index for entries with 'id' in data
CREATE UNIQUE INDEX profile_data_unique_entry_idx 
ON profile_data(user_id, category, (data->>'id'))
WHERE data->>'id' IS NOT NULL;
-- Profile Data
CREATE INDEX profile_data_user_id_idx ON profile_data(user_id);
CREATE INDEX profile_data_category_idx ON profile_data(category);
CREATE INDEX profile_data_is_current_idx ON profile_data(is_current) WHERE is_current = TRUE;

-- Articles
CREATE INDEX articles_user_id_idx ON articles(user_id);
CREATE INDEX articles_status_idx ON articles(status);
CREATE INDEX articles_published_at_idx ON articles(published_at DESC);
CREATE INDEX articles_slug_idx ON articles(slug);
CREATE INDEX articles_tags_idx ON articles USING GIN(tags);

-- Personal Attributes
CREATE INDEX personal_attributes_user_id_idx ON personal_attributes(user_id);
CREATE INDEX personal_attributes_type_idx ON personal_attributes(attribute_type);
CREATE INDEX personal_attributes_document_id_idx ON personal_attributes(document_id);