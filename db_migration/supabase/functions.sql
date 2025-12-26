-- ============================================================================
-- Search stored function that returns article or profile info
-- ============================================================================
DROP FUNCTION IF EXISTS search_all_content(vector, text, float, int);

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
SET search_path = public  -- explicitly set search path to prevent search path hijacking attachs.
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
DROP FUNCTION IF EXISTS search_documents(vector, uuid, float, int, text, text[], text[]);

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
SET search_path = public
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
DROP FUNCTION IF EXISTS upsert_document_with_embedding(text, text, text, text, jsonb, text[], text, jsonb);

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
SET search_path = public
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
-- ADD PERSONAL ATTRIBUTE WITH DOCUMENT AND EMBEDDING
-- ============================================================================
DROP FUNCTION IF EXISTS add_personal_attribute(text, text, text, text, text[], int, int, uuid[], uuid[], text, boolean);

CREATE OR REPLACE FUNCTION add_personal_attribute(
  p_user_id TEXT,
  p_attribute_type TEXT,
  p_title TEXT,
  p_description TEXT,
  p_examples TEXT[] DEFAULT '{}',
  p_importance_score INTEGER DEFAULT NULL,
  p_confidence_level INTEGER DEFAULT NULL,
  p_related_articles UUID[] DEFAULT '{}',
  p_related_experiences UUID[] DEFAULT '{}',
  p_embedding_model_name TEXT DEFAULT 'openai-small',
  p_create_searchable BOOLEAN DEFAULT TRUE
)
RETURNS UUID
LANGUAGE plpgsql
SET search_path = public
AS $$
DECLARE
  v_content_type_id UUID;
  v_document_id UUID;
  v_attribute_id UUID;
  v_searchable_text TEXT;
  v_embedding_model_id UUID;
BEGIN
  -- Only create document if searchable
  IF p_create_searchable THEN
    -- Get content type ID for the attribute type
    SELECT id INTO v_content_type_id
    FROM content_types
    WHERE name = p_attribute_type;
    
    IF v_content_type_id IS NULL THEN
      RAISE EXCEPTION 'Content type % not found. Please add it to content_types table.', p_attribute_type;
    END IF;
    
    -- Build searchable text
    v_searchable_text := p_title || '. ' || p_description;
    IF array_length(p_examples, 1) > 0 THEN
      v_searchable_text := v_searchable_text || '. Examples: ' || array_to_string(p_examples, '. ');
    END IF;
    
    -- Get embedding model ID
    SELECT id INTO v_embedding_model_id
    FROM embedding_models
    WHERE name = p_embedding_model_name AND is_active = TRUE;
    
    IF v_embedding_model_id IS NULL THEN
      RAISE EXCEPTION 'Embedding model % not found or not active', p_embedding_model_name;
    END IF;
    
    -- Insert document
    INSERT INTO documents (
      user_id, 
      content_type_id, 
      title, 
      content,
      metadata
    ) VALUES (
      p_user_id, 
      v_content_type_id, 
      p_title, 
      v_searchable_text,
      jsonb_build_object(
        'attribute_type', p_attribute_type,
        'importance_score', p_importance_score,
        'confidence_level', p_confidence_level
      )
    )
    RETURNING id INTO v_document_id;
    
    -- Insert placeholder embedding (to be updated by caller)
    INSERT INTO embeddings (
      document_id, 
      embedding_model_id, 
      chunk_text,
      chunk_index,
      total_chunks
    ) VALUES (
      v_document_id, 
      v_embedding_model_id,
      v_searchable_text,
      0,
      1
    );
  END IF;
  
  -- Insert personal attribute
  INSERT INTO personal_attributes (
    user_id,
    document_id,
    attribute_type,
    title,
    description,
    examples,
    searchable_text,
    importance_score,
    confidence_level,
    related_articles,
    related_experiences
  ) VALUES (
    p_user_id,
    v_document_id,
    p_attribute_type,
    p_title,
    p_description,
    p_examples,
    v_searchable_text,
    p_importance_score,
    p_confidence_level,
    p_related_articles,
    p_related_experiences
  )
  RETURNING id INTO v_attribute_id;
  
  RETURN v_attribute_id;
END;
$$;

-- ============================================================================
-- UPDATE PERSONAL ATTRIBUTE (AND OPTIONALLY RECREATE EMBEDDING)
-- ============================================================================
DROP FUNCTION IF EXISTS update_personal_attribute(uuid, text, text, text[], int, int, uuid[], uuid[], boolean);

CREATE OR REPLACE FUNCTION update_personal_attribute(
  p_attribute_id UUID,
  p_title TEXT DEFAULT NULL,
  p_description TEXT DEFAULT NULL,
  p_examples TEXT[] DEFAULT NULL,
  p_importance_score INTEGER DEFAULT NULL,
  p_confidence_level INTEGER DEFAULT NULL,
  p_related_articles UUID[] DEFAULT NULL,
  p_related_experiences UUID[] DEFAULT NULL,
  p_recreate_embedding BOOLEAN DEFAULT TRUE
)
RETURNS BOOLEAN
LANGUAGE plpgsql
SET search_path = public
AS $$
DECLARE
  v_document_id UUID;
  v_searchable_text TEXT;
  v_current_attribute RECORD;
BEGIN
  -- Get current attribute
  SELECT * INTO v_current_attribute
  FROM personal_attributes
  WHERE id = p_attribute_id;
  
  IF NOT FOUND THEN
    RAISE EXCEPTION 'Personal attribute % not found', p_attribute_id;
  END IF;
  
  -- Update personal_attributes table
  UPDATE personal_attributes
  SET
    title = COALESCE(p_title, title),
    description = COALESCE(p_description, description),
    examples = COALESCE(p_examples, examples),
    importance_score = COALESCE(p_importance_score, importance_score),
    confidence_level = COALESCE(p_confidence_level, confidence_level),
    related_articles = COALESCE(p_related_articles, related_articles),
    related_experiences = COALESCE(p_related_experiences, related_experiences),
    updated_at = NOW()
  WHERE id = p_attribute_id
  RETURNING document_id INTO v_document_id;
  
  -- If has document and should recreate embedding
  IF v_document_id IS NOT NULL AND p_recreate_embedding THEN
    -- Get updated values
    SELECT * INTO v_current_attribute
    FROM personal_attributes
    WHERE id = p_attribute_id;
    
    -- Build new searchable text
    v_searchable_text := v_current_attribute.title || '. ' || v_current_attribute.description;
    IF array_length(v_current_attribute.examples, 1) > 0 THEN
      v_searchable_text := v_searchable_text || '. Examples: ' || array_to_string(v_current_attribute.examples, '. ');
    END IF;
    
    -- Update document
    UPDATE documents
    SET
      title = v_current_attribute.title,
      content = v_searchable_text,
      metadata = jsonb_build_object(
        'attribute_type', v_current_attribute.attribute_type,
        'importance_score', v_current_attribute.importance_score,
        'confidence_level', v_current_attribute.confidence_level
      ),
      updated_at = NOW()
    WHERE id = v_document_id;
    
    -- Update embedding chunk_text (embedding vector itself updated by caller)
    UPDATE embeddings
    SET chunk_text = v_searchable_text
    WHERE document_id = v_document_id;
    
    -- Update searchable_text in personal_attributes
    UPDATE personal_attributes
    SET searchable_text = v_searchable_text
    WHERE id = p_attribute_id;
  END IF;
  
  RETURN TRUE;
END;
$$;

-- ============================================================================
-- Create vector indexes for active embedding models
-- ============================================================================
DROP FUNCTION IF EXISTS create_vector_indexes();

CREATE OR REPLACE FUNCTION create_vector_indexes()
RETURNS TEXT
LANGUAGE plpgsql
SET search_path = public  
AS $$
DECLARE 
    model_record RECORD; 
    index_name TEXT; 
    lists_count INTEGER;
    result_message TEXT := '';
BEGIN 
    FOR model_record IN 
        SELECT id, name, dimensions 
        FROM embedding_models 
        WHERE is_active = TRUE 
    LOOP 
        index_name := 'embeddings_' || replace(model_record.name, '-', '_') || '_idx';
        lists_count := GREATEST(10, FLOOR(SQRT(model_record.dimensions)));
        
        EXECUTE format(
            'CREATE INDEX IF NOT EXISTS %I ON embeddings 
             USING ivfflat (embedding vector_cosine_ops)
             WITH (lists = %s)
             WHERE embedding_model_id = %L',
            index_name,
            lists_count,
            model_record.id
        );
        
        result_message := result_message || format(
            'Created index %s for model %s (%s dimensions, %s lists)' || E'\n',
            index_name,
            model_record.name,
            model_record.dimensions,
            lists_count
        );
    END LOOP;
    
    RETURN result_message;
END;
$$;

-- ============================================================================
-- Update triggers to update the updated_at column for all tables
-- ============================================================================
-- Drop triggers if they exist (to avoid errors on re-run)
DROP TRIGGER IF EXISTS update_documents_updated_at ON documents;
DROP TRIGGER IF EXISTS update_profile_data_updated_at ON profile_data;
DROP TRIGGER IF EXISTS update_articles_updated_at ON articles;
DROP TRIGGER IF EXISTS update_embedding_models_updated_at ON embedding_models;
DROP TRIGGER IF EXISTS update_personal_attributes_updated_at ON personal_attributes;

DROP FUNCTION IF EXISTS update_updated_at_column();

CREATE OR REPLACE FUNCTION update_updated_at_column()
RETURNS TRIGGER AS $$
BEGIN
  NEW.updated_at = NOW();
  RETURN NEW;
END;
$$ LANGUAGE plpgsql;

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

