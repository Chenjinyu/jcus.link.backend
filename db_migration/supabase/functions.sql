-- ============================================================================
-- SEARCH ALL CONTENT
-- ============================================================================
DROP FUNCTION IF EXISTS search_all_content(vector, text, float, int);

CREATE OR REPLACE FUNCTION search_all_content(
  query_embedding vector,
  user_id_filter TEXT,
  match_threshold FLOAT DEFAULT 0.7,
  match_count INT DEFAULT 10
)
RETURNS TABLE ( 
  embedding_id UUID,
  document_id UUID,
  article_id UUID,
  profile_id UUID,
  personal_attribute_id UUID,
  content_type TEXT,  -- Derived from which table has data
  title TEXT,
  chunk_text TEXT,
  similarity FLOAT
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
    p.id AS profile_id,
    pa.id AS personal_attribute_id,
    -- Derive content_type from which table has data
    CASE 
      WHEN a.id IS NOT NULL THEN 'article'
      WHEN p.id IS NOT NULL THEN p.category
      WHEN pa.id IS NOT NULL THEN pa.attribute_type
      ELSE 'document'
    END AS content_type,
    COALESCE(d.title, a.title, p.data->>'title', pa.title, 'Untitled') AS title,
    e.chunk_text,
    1 - (e.embedding <=> query_embedding) AS similarity
  FROM embeddings e
  JOIN documents d ON e.document_id = d.id
  LEFT JOIN articles a ON a.document_id = d.id
  LEFT JOIN profile_data p ON p.document_id = d.id
  LEFT JOIN personal_attributes pa ON pa.document_id = d.id
  WHERE 
    d.user_id = user_id_filter
    AND (1 - (e.embedding <=> query_embedding)) > match_threshold
    AND d.deleted_at IS NULL
  ORDER BY similarity DESC
  LIMIT match_count;
END;
$$;

-- ============================================================================
-- SEARCH DOCUMENTS (Simplified - no content_types)
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
  category TEXT,  -- For profile_data
  attribute_type TEXT,  -- For personal_attributes
  similarity FLOAT,
  metadata JSONB,
  tags TEXT[],
  created_at TIMESTAMPTZ
)
LANGUAGE plpgsql
SET search_path = public
AS $$
DECLARE
  query_dims INTEGER;
BEGIN
  -- Get query vector dimensions
  query_dims := vector_dims(query_embedding);
  
  -- Execute query with dimension-specific casting
  IF query_dims = 768 THEN
    RETURN QUERY
    SELECT 
      e.id AS embedding_id,
      d.id AS document_id,
      a.id AS article_id,
      p.id AS profile_data_id,
      pa.id AS personal_attribute_id,
      COALESCE(d.title, a.title, p.data->>'title', pa.title, 'Untitled') AS title,
      d.content,
      e.chunk_text,
      e.chunk_index,
      e.total_chunks,
      -- Derive content_type from which table has data
      CASE 
        WHEN a.id IS NOT NULL THEN 'article'
        WHEN p.id IS NOT NULL THEN p.category
        WHEN pa.id IS NOT NULL THEN pa.attribute_type
        ELSE 'document'
      END AS content_type,
      p.category,
      pa.attribute_type,
      1 - (e.embedding::vector(768) <=> query_embedding) AS similarity,
      d.metadata,
      d.tags,
      d.created_at
    FROM embeddings e
    JOIN documents d ON e.document_id = d.id
    LEFT JOIN articles a ON a.document_id = d.id
    LEFT JOIN profile_data p ON p.document_id = d.id
    LEFT JOIN personal_attributes pa ON pa.document_id = d.id
    WHERE 
      e.embedding_model_id = model_id
      AND (1 - (e.embedding::vector(768) <=> query_embedding)) > match_threshold
      AND d.is_current = TRUE
      AND d.deleted_at IS NULL
      AND (filter_user_id IS NULL OR d.user_id = filter_user_id)
      AND (filter_content_types IS NULL OR 
        CASE 
          WHEN a.id IS NOT NULL THEN 'article'
          WHEN p.id IS NOT NULL THEN p.category
          WHEN pa.id IS NOT NULL THEN pa.attribute_type
          ELSE 'document'
        END = ANY(filter_content_types))
      AND (filter_tags IS NULL OR d.tags && filter_tags)
    ORDER BY e.embedding::vector(768) <=> query_embedding
    LIMIT match_count;
    
  ELSIF query_dims = 1536 THEN
    RETURN QUERY
    SELECT 
      e.id AS embedding_id,
      d.id AS document_id,
      a.id AS article_id,
      p.id AS profile_data_id,
      pa.id AS personal_attribute_id,
      COALESCE(d.title, a.title, p.data->>'title', pa.title, 'Untitled') AS title,
      d.content,
      e.chunk_text,
      e.chunk_index,
      e.total_chunks,
      CASE 
        WHEN a.id IS NOT NULL THEN 'article'
        WHEN p.id IS NOT NULL THEN p.category
        WHEN pa.id IS NOT NULL THEN pa.attribute_type
        ELSE 'document'
      END AS content_type,
      p.category,
      pa.attribute_type,
      1 - (e.embedding::vector(1536) <=> query_embedding) AS similarity,
      d.metadata,
      d.tags,
      d.created_at
    FROM embeddings e
    JOIN documents d ON e.document_id = d.id
    LEFT JOIN articles a ON a.document_id = d.id
    LEFT JOIN profile_data p ON p.document_id = d.id
    LEFT JOIN personal_attributes pa ON pa.document_id = d.id
    WHERE 
      e.embedding_model_id = model_id
      AND (1 - (e.embedding::vector(1536) <=> query_embedding)) > match_threshold
      AND d.is_current = TRUE
      AND d.deleted_at IS NULL
      AND (filter_user_id IS NULL OR d.user_id = filter_user_id)
      AND (filter_content_types IS NULL OR 
        CASE 
          WHEN a.id IS NOT NULL THEN 'article'
          WHEN p.id IS NOT NULL THEN p.category
          WHEN pa.id IS NOT NULL THEN pa.attribute_type
          ELSE 'document'
        END = ANY(filter_content_types))
      AND (filter_tags IS NULL OR d.tags && filter_tags)
    ORDER BY e.embedding::vector(1536) <=> query_embedding
    LIMIT match_count;
    
  ELSE
    RAISE EXCEPTION 'Unsupported vector dimension: %. Supported: 768, 1536', query_dims;
  END IF;
END;
$$;

-- ============================================================================
-- UPSERT DOCUMENT (Simplified - no content_types)
-- ============================================================================
DROP FUNCTION IF EXISTS upsert_document_with_embedding(text, text, text, text, jsonb, text[], text, jsonb);

CREATE OR REPLACE FUNCTION upsert_document_with_embedding(
  p_user_id TEXT,
  p_title TEXT,
  p_content TEXT,
  p_metadata JSONB DEFAULT '{}'::jsonb,
  p_tags TEXT[] DEFAULT '{}'::TEXT[], -- [] is an array in Python, '{}' is an array in Postgres
  p_embedding_model_name TEXT DEFAULT 'openai-small',
  p_chunks JSONB DEFAULT '[]'::jsonb
)
RETURNS UUID
LANGUAGE plpgsql
SET search_path = public
AS $$
DECLARE
  v_document_id UUID;
  v_embedding_model_id UUID;
  v_chunk JSONB;
  v_total_chunks INTEGER;
  v_existing_document_id UUID;
BEGIN
  -- Get embedding model ID
  SELECT id INTO v_embedding_model_id
  FROM embedding_models
  WHERE name = p_embedding_model_name AND is_active = TRUE;
  
  IF v_embedding_model_id IS NULL THEN
    RAISE EXCEPTION 'Embedding model % not found or not active', p_embedding_model_name;
  END IF;
  
  -- Check for existing document (latest current, non-deleted)
  SELECT id INTO v_existing_document_id
  FROM documents
  WHERE user_id = p_user_id
    AND title = p_title
    AND is_current = TRUE
    AND deleted_at IS NULL
  ORDER BY updated_at DESC
  LIMIT 1;
  
  IF v_existing_document_id IS NULL THEN
    -- Insert document
    INSERT INTO documents (
      user_id, title, content, metadata, tags
    ) VALUES (
      p_user_id, p_title, p_content, p_metadata, p_tags
    )
    RETURNING id INTO v_document_id;
  ELSE
    -- Update existing document
    UPDATE documents
    SET
      title = p_title,
      content = p_content,
      metadata = p_metadata,
      tags = p_tags,
      updated_at = NOW()
    WHERE id = v_existing_document_id
    RETURNING id INTO v_document_id;
  END IF;
  
  -- Get total chunks count
  v_total_chunks := jsonb_array_length(p_chunks);
  
  IF v_total_chunks = 0 THEN
    RAISE EXCEPTION 'p_chunks must contain at least one chunk';
  END IF;
  
  -- Replace embeddings for this document/model with provided chunks
  DELETE FROM embeddings
  WHERE document_id = v_document_id
    AND embedding_model_id = v_embedding_model_id;
  
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
    )
    ON CONFLICT (document_id, embedding_model_id, chunk_index)
    DO UPDATE SET
      embedding = EXCLUDED.embedding,
      chunk_text = EXCLUDED.chunk_text,
      total_chunks = EXCLUDED.total_chunks;
  END LOOP;
  
  RETURN v_document_id;
END;
$$;

-- ============================================================================
-- ADD PERSONAL ATTRIBUTE (Simplified - uses trigger for searchable_text)
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
  v_document_id UUID;
  v_attribute_id UUID;
  v_searchable_text TEXT;
  v_embedding_model_id UUID;
BEGIN
  -- Only create document if searchable
  IF p_create_searchable THEN
    -- Get embedding model ID
    SELECT id INTO v_embedding_model_id
    FROM embedding_models
    WHERE name = p_embedding_model_name AND is_active = TRUE;
    
    IF v_embedding_model_id IS NULL THEN
      RAISE EXCEPTION 'Embedding model % not found or not active', p_embedding_model_name;
    END IF;
    
    -- Build searchable text (same logic as trigger)
    v_searchable_text := p_title || '. ' || p_description;
    IF array_length(p_examples, 1) > 0 THEN
      v_searchable_text := v_searchable_text || '. Examples: ' || array_to_string(p_examples, '. ');
    END IF;
    
    -- Insert document
    INSERT INTO documents (
      user_id, 
      title, 
      content,
      metadata,
      tags
    ) VALUES (
      p_user_id, 
      p_title, 
      v_searchable_text,
      jsonb_build_object(
        'source', 'personal_attribute',
        'attribute_type', p_attribute_type,
        'importance_score', p_importance_score,
        'confidence_level', p_confidence_level
      ),
      ARRAY[p_attribute_type]
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
  -- Note: searchable_text will be auto-generated by trigger
  INSERT INTO personal_attributes (
    user_id,
    document_id,
    attribute_type,
    title,
    description,
    examples,
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
-- UPDATE PERSONAL ATTRIBUTE (Trigger auto-updates searchable_text)
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
  -- Note: Trigger will auto-update searchable_text
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
  RETURNING document_id, searchable_text INTO v_document_id, v_searchable_text;
  
  -- If has document and should recreate embedding
  IF v_document_id IS NOT NULL AND p_recreate_embedding THEN
    -- Update document with new searchable text
    UPDATE documents
    SET
      title = (SELECT title FROM personal_attributes WHERE id = p_attribute_id),
      content = v_searchable_text,
      metadata = jsonb_build_object(
        'source', 'personal_attribute',
        'attribute_type', (SELECT attribute_type FROM personal_attributes WHERE id = p_attribute_id),
        'importance_score', (SELECT importance_score FROM personal_attributes WHERE id = p_attribute_id),
        'confidence_level', (SELECT confidence_level FROM personal_attributes WHERE id = p_attribute_id)
      ),
      updated_at = NOW()
    WHERE id = v_document_id;
    
    -- Update embedding chunk_text (embedding vector itself updated by caller)
    UPDATE embeddings
    SET chunk_text = v_searchable_text
    WHERE document_id = v_document_id;
  END IF;
  
  RETURN TRUE;
END;
$$;

-- ============================================================================
-- AUTO-UPDATE TRIGGERS FOR SEARCHABLE_TEXT
-- ============================================================================

-- Personal Attributes
DROP TRIGGER IF EXISTS update_personal_attributes_searchable_text_trigger ON personal_attributes;
DROP FUNCTION IF EXISTS update_personal_attributes_searchable_text();

CREATE OR REPLACE FUNCTION update_personal_attributes_searchable_text()
RETURNS TRIGGER
LANGUAGE plpgsql
SET search_path = public
AS $$
BEGIN
  -- Build searchable text from title + description + examples
  NEW.searchable_text := NEW.title || '. ' || NEW.description;
  
  IF array_length(NEW.examples, 1) > 0 THEN
    NEW.searchable_text := NEW.searchable_text || '. Examples: ' || array_to_string(NEW.examples, '. ');
  END IF;
  
  RETURN NEW;
END;
$$;

CREATE TRIGGER update_personal_attributes_searchable_text_trigger
  BEFORE INSERT OR UPDATE ON personal_attributes
  FOR EACH ROW
  EXECUTE FUNCTION update_personal_attributes_searchable_text();

-- Profile Data
DROP TRIGGER IF EXISTS update_profile_data_searchable_text_trigger ON profile_data;
DROP FUNCTION IF EXISTS update_profile_data_searchable_text();

-- purpose: take structured JSONB data(NEW.data), extract common fields, and
-- combine them into a single searchable_text field for better searchability.
CREATE OR REPLACE FUNCTION update_profile_data_searchable_text()
RETURNS TRIGGER -- the func will be called by trigger, which receives NEW
LANGUAGE plpgsql -- support variable, conditions, loop/logic
SET search_path = public -- best practice and safety measure, to ensure all table/function refereneces to public
AS $$
DECLARE
  text_parts TEXT[] := '{}';
BEGIN
  -- Extract searchable fields from JSONB based on common patterns
  -- NEW is the row beling inserted or updated.
  IF NEW.data ? 'title' THEN -- if NEW.data has the key of title
    text_parts := array_append(text_parts, NEW.data->>'title');
  END IF;
  
  IF NEW.data ? 'company' THEN
    text_parts := array_append(text_parts, NEW.data->>'company');
  END IF;
  
  IF NEW.data ? 'position' THEN
    text_parts := array_append(text_parts, NEW.data->>'position');
  END IF;
  
  IF NEW.data ? 'description' THEN
    text_parts := array_append(text_parts, NEW.data->>'description');
  END IF;
  -- ->> etract data, -> extract JSON
  IF NEW.data ? 'skills' THEN
    -- Handle both string and array
    IF jsonb_typeof(NEW.data->'skills') = 'array' THEN
      text_parts := array_append(text_parts, 'Skills: ' || 
        (SELECT string_agg(value::text, ', ') FROM jsonb_array_elements_text(NEW.data->'skills')));
    ELSE
      text_parts := array_append(text_parts, 'Skills: ' || (NEW.data->>'skills'));
    END IF;
  END IF;
  
  IF NEW.data ? 'achievements' THEN
    text_parts := array_append(text_parts, NEW.data->>'achievements');
  END IF;
  
  IF NEW.data ? 'responsibilities' THEN
    text_parts := array_append(text_parts, NEW.data->>'responsibilities');
  END IF;
  
  -- Join all parts
  NEW.searchable_text := array_to_string(text_parts, '. ');
  
  RETURN NEW;
END;
$$;

-- create trigger when insert or update the profile_data, then call 
-- the function of update_profile_data_searchable_text
CREATE TRIGGER update_profile_data_searchable_text_trigger
  BEFORE INSERT OR UPDATE ON profile_data
  FOR EACH ROW
  EXECUTE FUNCTION update_profile_data_searchable_text();

-- ============================================================================
-- UPDATED TRIGGER: update_updated_at_column
-- ============================================================================
DROP TRIGGER IF EXISTS update_documents_updated_at ON documents;
DROP TRIGGER IF EXISTS update_profile_data_updated_at ON profile_data;
DROP TRIGGER IF EXISTS update_articles_updated_at ON articles;
DROP TRIGGER IF EXISTS update_embedding_models_updated_at ON embedding_models;
DROP TRIGGER IF EXISTS update_personal_attributes_updated_at ON personal_attributes;

DROP FUNCTION IF EXISTS update_updated_at_column();

CREATE OR REPLACE FUNCTION update_updated_at_column()
RETURNS TRIGGER 
LANGUAGE plpgsql
SET search_path = public
AS $$
BEGIN
  NEW.updated_at = NOW();
  RETURN NEW;
END;
$$;

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

-- ============================================================================
-- DIMENSION VALIDATION TRIGGER
-- ============================================================================
DROP TRIGGER IF EXISTS validate_embedding_dimension ON embeddings;
DROP FUNCTION IF EXISTS check_embedding_dimension();

CREATE OR REPLACE FUNCTION check_embedding_dimension()
RETURNS TRIGGER
LANGUAGE plpgsql
SET search_path = public
AS $$
DECLARE
  expected_dims INTEGER;
  actual_dims INTEGER;
BEGIN
  IF NEW.embedding IS NULL THEN
    RETURN NEW;
  END IF;
  
  SELECT dimensions INTO expected_dims
  FROM embedding_models
  WHERE id = NEW.embedding_model_id;
  
  IF expected_dims IS NULL THEN
    RAISE EXCEPTION 'Embedding model % not found', NEW.embedding_model_id;
  END IF;
  
  actual_dims := vector_dims(NEW.embedding);
  
  IF actual_dims != expected_dims THEN
    RAISE EXCEPTION 'Embedding dimension mismatch for model %: expected %, got %',
      NEW.embedding_model_id, expected_dims, actual_dims;
  END IF;
  
  RETURN NEW;
END;
$$;

CREATE TRIGGER validate_embedding_dimension
  BEFORE INSERT OR UPDATE ON embeddings
  FOR EACH ROW
  WHEN (NEW.embedding IS NOT NULL)
  EXECUTE FUNCTION check_embedding_dimension();
