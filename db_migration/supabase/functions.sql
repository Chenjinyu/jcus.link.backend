-- ============================================================================
-- Search stored function that returns article or profile info
-- ============================================================================
CREATE OR REPLACE FUNCTION search_all_content(
  query_embedding vector,
  user_id_filter TEXT,
  match_threshold FLOAT DEFAULT 0.7,
  match_count INT DEFAULT 10
)
-- Defines the structure of row returned
-- it can return text, integer, uuid, etc.
RETURNS TABLE ( 
  embedding_id UUID,
  document_id UUID,
  article_id UUID,
  profile_id UUID,
  content_type TEXT,
  title TEXT,
  chunk_text TEXT,
  similarity FLOAT
)
LANGUAGE plpgsql
AS $$
BEGIN
  RETURN QUERY
  SELECT 
    e.id AS embedding_id,
    d.id AS document_id,
    a.id AS article_id,
    p.id AS profile_id,
    ct.name AS content_type,
    COALESCE(d.title, p.data->>'title', 'Untitled') AS title,
    e.chunk_text,
    1 - (e.embedding <=> query_embedding) AS similarity
  FROM embeddings e
  JOIN documents d ON e.document_id = d.id
  JOIN content_types ct ON d.content_type_id = ct.id
  LEFT JOIN articles a ON a.document_id = d.id      -- May be article
  LEFT JOIN profile_data p ON p.document_id = d.id  -- May be profile
  WHERE 
    d.user_id = user_id_filter
    AND (1 - (e.embedding <=> query_embedding)) > match_threshold
  ORDER BY similarity DESC
  LIMIT match_count;
END;
$$;

-- ============================================================================
-- SEARCH DOCUMENTS WITH ARTICLE, PROFILE DATA, AND PERSONAL ATTRIBUTES
-- ============================================================================
CREATE OR REPLACE FUNCTION search_documents(
  query_embedding vector,
  model_id UUID,
  match_threshold FLOAT DEFAULT 0.7,
  match_count INT DEFAULT 10,
  filter_user_id TEXT DEFAULT NULL,
  filter_content_types TEXT[] DEFAULT NULL,
  filter_tags TEXT[] DEFAULT NULL
)
RETURNS TABLE (
  embedding_id UUID,
  document_id UUID,
  article_id UUID,
  profile_data_id UUID,
  personal_attribute_id UUID,
  title TEXT,
  content TEXT,
  chunk_text TEXT,
  chunk_index INTEGER,
  total_chunks INTEGER,
  content_type TEXT,
  attribute_type TEXT,
  similarity FLOAT,
  metadata JSONB,
  tags TEXT[],
  created_at TIMESTAMPTZ
)
LANGUAGE plpgsql
AS $$
BEGIN
  RETURN QUERY
  SELECT 
    e.id AS embedding_id,
    d.id AS document_id,
    a.id AS article_id,
    p.id AS profile_data_id,
    pa.id AS personal_attribute_id,
    COALESCE(
      d.title, 
      a.title, 
      p.data->>'title',
      pa.title,
      'Untitled'
    ) AS title,
    d.content,
    e.chunk_text,
    e.chunk_index,
    e.total_chunks,
    ct.name AS content_type,
    pa.attribute_type,
    1 - (e.embedding <=> query_embedding) AS similarity,
    d.metadata,
    d.tags,
    d.created_at
  FROM embeddings e
  JOIN documents d ON e.document_id = d.id
  JOIN content_types ct ON d.content_type_id = ct.id
  LEFT JOIN articles a ON a.document_id = d.id
  LEFT JOIN profile_data p ON p.document_id = d.id
  LEFT JOIN personal_attributes pa ON pa.document_id = d.id
  WHERE 
    e.embedding_model_id = model_id
    AND (1 - (e.embedding <=> query_embedding)) > match_threshold
    AND d.is_current = TRUE
    AND d.deleted_at IS NULL
    AND (filter_user_id IS NULL OR d.user_id = filter_user_id)
    AND (filter_content_types IS NULL OR ct.name = ANY(filter_content_types))
    AND (filter_tags IS NULL OR d.tags && filter_tags)
  ORDER BY e.embedding <=> query_embedding
  LIMIT match_count;
END;
$$;

-- ============================================================================
-- Upsert stored function that inserts a document with embeddings
-- ============================================================================
CREATE OR REPLACE FUNCTION upsert_document_with_embedding(
  p_user_id TEXT,
  p_content_type TEXT,
  p_title TEXT,
  p_content TEXT,
  p_metadata JSONB DEFAULT '{}'::jsonb,
  p_tags TEXT[] DEFAULT '{}',
  p_embedding_model_name TEXT DEFAULT 'openai-small',
  p_chunks JSONB DEFAULT '[]'::jsonb  -- Array of {text, embedding, chunk_index}
)
RETURNS UUID
LANGUAGE plpgsql
AS $$
DECLARE
  v_content_type_id UUID;
  v_document_id UUID;
  v_embedding_model_id UUID;
  v_chunk JSONB;
  v_total_chunks INTEGER;
BEGIN
  -- Get content type ID
  SELECT id INTO v_content_type_id
  FROM content_types
  WHERE name = p_content_type;
  
  IF v_content_type_id IS NULL THEN
    RAISE EXCEPTION 'Content type % not found', p_content_type;
  END IF;
  
  -- Get embedding model ID
  SELECT id INTO v_embedding_model_id
  FROM embedding_models
  WHERE name = p_embedding_model_name;
  
  IF v_embedding_model_id IS NULL THEN
    RAISE EXCEPTION 'Embedding model % not found', p_embedding_model_name;
  END IF;
  
  -- Insert document
  INSERT INTO documents (
    user_id, content_type_id, title, content, metadata, tags
  ) VALUES (
    p_user_id, v_content_type_id, p_title, p_content, p_metadata, p_tags
  )
  RETURNING id INTO v_document_id;
  
  -- Get total chunks count
  v_total_chunks := jsonb_array_length(p_chunks);
  
  -- If no chunks provided, create single chunk from full content
  IF v_total_chunks = 0 THEN
    INSERT INTO embeddings (
      document_id, 
      embedding_model_id, 
      embedding, 
      chunk_text,
      chunk_index,
      total_chunks
    ) VALUES (
      v_document_id, 
      v_embedding_model_id, 
      NULL,  -- Will be filled by caller
      p_content,
      0,
      1
    );
  ELSE
    -- Insert embeddings for each chunk
    FOR v_chunk IN SELECT * FROM jsonb_array_elements(p_chunks)
    LOOP
      INSERT INTO embeddings (
        document_id, 
        embedding_model_id, 
        embedding, 
        chunk_text,
        chunk_index,
        total_chunks
      ) VALUES (
        v_document_id, 
        v_embedding_model_id,
        (v_chunk->>'embedding')::vector,
        v_chunk->>'text',
        (v_chunk->>'chunk_index')::INTEGER,
        v_total_chunks
      );
    END LOOP;
  END IF;
  
  RETURN v_document_id;
END;
$$;

-- ============================================================================
-- Update triggers to update the updated_at column for all tables
-- ============================================================================
CREATE OR REPLACE FUNCTION update_updated_at_column()
RETURNS TRIGGER AS $$
BEGIN
  NEW.updated_at = NOW();
  RETURN NEW;
END;
$$ LANGUAGE plpgsql;

-- Drop triggers if they exist (to avoid errors on re-run)
DROP TRIGGER IF EXISTS update_documents_updated_at ON documents;
DROP TRIGGER IF EXISTS update_profile_data_updated_at ON profile_data;
DROP TRIGGER IF EXISTS update_articles_updated_at ON articles;
DROP TRIGGER IF EXISTS update_embedding_models_updated_at ON embedding_models;
DROP TRIGGER IF EXISTS update_personal_attributes_updated_at ON personal_attributes;

-- Create triggers
CREATE TRIGGER update_documents_updated_at
  BEFORE UPDATE ON documents
  FOR EACH ROW
  EXECUTE FUNCTION update_updated_at_column();

CREATE TRIGGER update_profile_data_updated_at
  BEFORE UPDATE ON profile_data
  FOR EACH ROW
  EXECUTE FUNCTION update_updated_at_column();

CREATE TRIGGER update_articles_updated_at
  BEFORE UPDATE ON articles
  FOR EACH ROW
  EXECUTE FUNCTION update_updated_at_column();

CREATE TRIGGER update_embedding_models_updated_at
  BEFORE UPDATE ON embedding_models
  FOR EACH ROW
  EXECUTE FUNCTION update_updated_at_column();

CREATE TRIGGER update_personal_attributes_updated_at
  BEFORE UPDATE ON personal_attributes
  FOR EACH ROW
  EXECUTE FUNCTION update_updated_at_column();

CREATE TRIGGER update_personal_attributes_updated_at
  BEFORE UPDATE ON personal_attributes
  FOR EACH ROW
  EXECUTE FUNCTION update_updated_at_column();