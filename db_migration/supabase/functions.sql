-- ============================================================================
-- SEARCH FUNCTION (Multi-model support)
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
  id UUID,
  document_id UUID,
  title TEXT,
  content TEXT,
  content_type TEXT,
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
    e.id,
    d.id AS document_id,
    d.title,
    d.content,
    ct.name AS content_type,
    1 - (e.embedding <=> query_embedding) AS similarity,
    d.metadata,
    d.tags,
    d.created_at
  FROM embeddings e
  JOIN documents d ON e.document_id = d.id
  JOIN content_types ct ON d.content_type_id = ct.id
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
-- UPSERT DOCUMENT WITH EMBEDDING
-- ============================================================================
CREATE OR REPLACE FUNCTION upsert_document_with_embedding(
  p_user_id TEXT,
  p_content_type TEXT,
  p_title TEXT,
  p_content TEXT,
  p_metadata JSONB,
  p_tags TEXT[],
  p_embedding_model_name TEXT,
  p_embedding vector
)
RETURNS UUID
LANGUAGE plpgsql
AS $$
DECLARE
  v_content_type_id UUID;
  v_document_id UUID;
  v_embedding_model_id UUID;
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
  
  -- Insert embedding
  INSERT INTO embeddings (
    document_id, embedding_model_id, embedding, chunk_text
  ) VALUES (
    v_document_id, v_embedding_model_id, p_embedding, p_content
  );
  
  RETURN v_document_id;
END;
$$;

-- ============================================================================
-- UPDATE TRIGGERS
-- ============================================================================
CREATE OR REPLACE FUNCTION update_updated_at_column()
RETURNS TRIGGER AS $$
BEGIN
  NEW.updated_at = NOW();
  RETURN NEW;
END;
$$ LANGUAGE plpgsql;

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