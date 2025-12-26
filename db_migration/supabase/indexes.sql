-- ============================================================================
-- VECTOR INDEXES (One per embedding dimension)
-- ivfflat -> inverted File with Flat Compression. 
-- it's a special index type for vector similarity search provided by the pgvector extension in PostgreSQL.
-- hnsw - Hierarchical Navigable Small World. It's a graph-based index for vector similarity search.
-- ivfflat - Fast build, good for most cases.
----------------------------------------------------------------------------
-- | Index Type         | Use Case                                         |
-- | GIN                | Full-Text Search, JSONB containment.             |
-- | GiST               | Geospatial Data, Range Queries                   |
-- | BRIN               | Very Large Tables with Natural Order             |
-- | ivfflat            | Vector Similarity Search (Large Datasets)        |
-- | hnsw               | Vector Similarity Search (High-Dimensional Data).|
-- | btree(clower(col)) | Case-Insensitive Text Search                     |
----------------------------------------------------------------------------
-- ============================================================================
-- Step 1: Get the model IDs
SELECT id, name, dimensions FROM embedding_models;

-- Step 2: Use the actual UUIDs in the index
-- Replace 'your-model-id-here' with actual UUID from step 1

-- Create HNSW type of indexes (work with vector without dimensions)
-- For 768 dimensions (Ollama nomic-embed-text)
CREATE INDEX embeddings_768_idx ON embeddings 
USING hnsw ((embedding::vector(768)) vector_cosine_ops)
WITH (m = 16, ef_construction = 64)
WHERE embedding_model_id = '0bf77eef-8a60-433e-85e8-0e5ed9fd12a7'::UUID;

-- For 1536 dimensions (OpenAI text-embedding-3-small)
CREATE INDEX embeddings_1536_idx ON embeddings 
USING hnsw ((embedding::vector(1536)) vector_cosine_ops)
WITH (m = 16, ef_construction = 64)
WHERE embedding_model_id = 'e21adccd-dc5e-4e1f-a0d6-ac137a93607c'::UUID;

-- ⚠️ BOTH HNSW and ivfflat have a 2000-dimension limit in pgvector. its a hard limitation. CANNOT create index for dimension 3072
-- For 3072 dimensions (OpenAI text-embedding-3-large)
CREATE INDEX embeddings_3072_idx ON embeddings 
USING hnsw ((embedding::vector(3072)) vector_cosine_ops)
WITH (m = 16, ef_construction = 64)
WHERE embedding_model_id = '8946dc3d-f227-4d04-916f-be1bafaf66ea'::UUID;

-- ============================================================================
-- REGULAR INDEXES, Postgres creates the B-Tree index by default.
-- ============================================================================

-- Documents
DROP INDEX IF EXISTS documents_user_id_idx;
DROP INDEX IF EXISTS documents_current_idx;
DROP INDEX IF EXISTS documents_tags_idx;
CREATE INDEX documents_user_id_idx ON documents(user_id);
CREATE INDEX documents_current_idx ON documents(is_current) WHERE is_current = TRUE;
CREATE INDEX documents_tags_idx ON documents USING GIN(tags);

-- Embeddings
DROP INDEX IF EXISTS embeddings_document_id_idx;
DROP INDEX IF EXISTS embeddings_model_id_idx;
CREATE INDEX embeddings_document_id_idx ON embeddings(document_id);
CREATE INDEX embeddings_model_id_idx ON embeddings(embedding_model_id);


-- Create unique index for entries with 'id' in data
DROP INDEX IF EXISTS profile_data_unique_entry_idx;
CREATE UNIQUE INDEX profile_data_unique_entry_idx 
ON profile_data(user_id, category, (data->>'id'))
WHERE data->>'id' IS NOT NULL;
-- Profile Data
DROP INDEX IF EXISTS profile_data_user_id_idx;
DROP INDEX IF EXISTS profile_data_category_idx;
DROP INDEX IF EXISTS profile_data_document_id_idx;
CREATE INDEX profile_data_user_id_idx ON profile_data(user_id);
CREATE INDEX profile_data_category_idx ON profile_data(category);
CREATE INDEX profile_data_document_id_idx ON profile_data(document_id);

-- Articles
DROP INDEX IF EXISTS articles_user_id_idx;
DROP INDEX IF EXISTS articles_status_idx;
DROP INDEX IF EXISTS articles_document_id_idx;
DROP INDEX IF EXISTS articles_slug_idx;
DROP INDEX IF EXISTS articles_tags_idx;
CREATE INDEX articles_user_id_idx ON articles(user_id);
CREATE INDEX articles_status_idx ON articles(status);
CREATE INDEX articles_document_id_idx ON articles(document_id);
CREATE INDEX articles_slug_idx ON articles(slug);
CREATE INDEX articles_tags_idx ON articles USING GIN(tags);

-- Personal Attributes
DROP INDEX IF EXISTS personal_attributes_user_id_idx;
DROP INDEX IF EXISTS personal_attributes_type_idx;
DROP INDEX IF EXISTS personal_attributes_document_id_idx;
CREATE INDEX personal_attributes_user_id_idx ON personal_attributes(user_id);
CREATE INDEX personal_attributes_type_idx ON personal_attributes(attribute_type);
CREATE INDEX personal_attributes_document_id_idx ON personal_attributes(document_id);