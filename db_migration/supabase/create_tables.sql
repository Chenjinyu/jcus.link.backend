-- ============================================================================
-- 1. EMBEDDING_MODELS (Reference data)
-- Tracks different embedding models and their configurations
-- ============================================================================
CREATE TABLE embedding_models (
  id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
  name TEXT NOT NULL UNIQUE, -- 'openai-small', 'openai-large', 'ollama-local'
  provider TEXT NOT NULL, -- 'openai', 'ollama', 'cohere', 'google'
  model_identifier TEXT NOT NULL, -- 'text-embedding-3-small', 'nomic-embed-text'
  dimensions INTEGER NOT NULL, -- 1536, 3072, 768, etc.
  is_active BOOLEAN DEFAULT TRUE,
  is_local BOOLEAN DEFAULT FALSE, -- TRUE for ollama models
  cost_per_token DECIMAL(10, 8), -- Track costs
  created_at TIMESTAMPTZ DEFAULT NOW(),
  updated_at TIMESTAMPTZ DEFAULT NOW()
);


-- ============================================================================
-- 2. DOCUMENTS (Searchable text only - for embeddings)
-- ============================================================================
CREATE TABLE documents (
  id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
  user_id TEXT NOT NULL, -- Your user identifier
  
  -- Content
  title TEXT,
  content TEXT NOT NULL,
  summary TEXT, -- Auto-generated summary
  
  -- Metadata
  metadata JSONB, -- Flexible storage for type-specific data
  tags TEXT[], -- Array of tags for filtering
  
  -- Versioning
  version INTEGER DEFAULT 1,
  is_current BOOLEAN DEFAULT TRUE,
  parent_id UUID REFERENCES documents(id), -- For version history
  
  -- Soft delete
  deleted_at TIMESTAMPTZ,
  
  -- Timestamps
  created_at TIMESTAMPTZ DEFAULT NOW(),
  updated_at TIMESTAMPTZ DEFAULT NOW()
);

-- ============================================================================
-- 3. EMBEDDINGS TABLE (Vector storage)
-- Multi-model vector storage with model tracking
-- ============================================================================
CREATE TABLE embeddings (
  id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
  document_id UUID NOT NULL REFERENCES documents(id) ON DELETE CASCADE,
  embedding_model_id UUID NOT NULL REFERENCES embedding_models(id),
  
  -- The actual vector (dimension varies by model)
  embedding vector, -- Can be 768, 1536, etc. 3072 is out the 2000 limitation in pyvector.
  
  -- Metadata
  chunk_index INTEGER DEFAULT 0, -- For chunked documents
  total_chunks INTEGER DEFAULT 1,
  chunk_text TEXT, -- Store the specific text that was embedded
  
  created_at TIMESTAMPTZ DEFAULT NOW(),
  
  -- Composite unique constraint: one embedding per document per model per chunk
  UNIQUE(document_id, embedding_model_id, chunk_index)
);

-- ============================================================================
-- 4. PROFILE DATA TABLE
-- Structured storage for LinkedIn-style profile information
-- ============================================================================
CREATE TABLE profile_data (
  id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
  user_id TEXT NOT NULL,
  
  -- Optional embedding search, Keep the child rows, but remove the reference
  document_id UUID REFERENCES documents(id) ON DELETE SET NULL,
  -- Category: 'work_experience', 'education', 'certification', 'skill', 'value', 'goal'
  category TEXT NOT NULL CHECK (
    category IN (
      'work_experience', 
      'education', 
      'certification', 
      'skill', 
      'value', 
      'goal',
      'project',
      'volunteering',
      'event'
    )
  ),
  
  -- Structured data
  data JSONB NOT NULL, -- Flexible schema per category
  
  -- Time range (for experiences)
  is_current BOOLEAN DEFAULT FALSE,
  start_date DATE,
  end_date DATE,
  
  -- Display
  display_order INTEGER,
  is_featured BOOLEAN DEFAULT FALSE,
  
  created_at TIMESTAMPTZ DEFAULT NOW(),
  updated_at TIMESTAMPTZ DEFAULT NOW()
);

-- ============================================================================
-- 5. ARTICLES TABLE
-- Specialized table for your articles
-- ============================================================================
CREATE TABLE articles (
  id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
  user_id TEXT NOT NULL,

  -- Always has embedding. Delete the child rows if article is deleted.
  document_id UUID REFERENCES documents(id) ON DELETE CASCADE,
  
  -- Article metadata
  slug TEXT UNIQUE NOT NULL, -- URL-friendly identifier
  title TEXT NOT NULL,
  subtitle TEXT,
  content TEXT NOT NULL,
  excerpt TEXT,
  
  -- Publishing
  status TEXT DEFAULT 'draft' CHECK (
    status IN ('draft', 'published', 'archived')
  ),
  published_at TIMESTAMPTZ,
  
  -- SEO
  seo_title TEXT,
  seo_description TEXT,
  og_image TEXT,
  
  -- Engagement
  view_count INTEGER DEFAULT 0,
  like_count INTEGER DEFAULT 0,
  
  -- Categorization
  tags TEXT[],
  category TEXT,
  
  -- Timestamps
  created_at TIMESTAMPTZ DEFAULT NOW(),
  updated_at TIMESTAMPTZ DEFAULT NOW()
);

-- ============================================================================
-- 6. PERSONAL_ATTRIBUTES - SOFT SKILLS & VALUES TABLE
-- Store personal philosophy, soft skills, aspirations
-- ============================================================================
CREATE TABLE personal_attributes (
  id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
  user_id TEXT NOT NULL,
  
  -- Link to documents for vector search
  document_id UUID REFERENCES documents(id) ON DELETE SET NULL,
  
  -- Type: 'soft_skill', 'value', 'worldview', 'aspiration', 'principle'
  attribute_type TEXT NOT NULL CHECK (
    attribute_type IN (
      'soft_skill', 
      'value', 
      'worldview', 
      'aspiration', 
      'principle'
    )
  ),
  
  -- Content
  title TEXT NOT NULL,
  description TEXT NOT NULL,
  examples TEXT[], -- Real-world examples
  
  -- Searchable text (generated from title + description + examples)
  searchable_text TEXT,
  
  -- Importance/Confidence
  importance_score INTEGER CHECK (importance_score BETWEEN 1 AND 10),
  confidence_level INTEGER CHECK (confidence_level BETWEEN 1 AND 10),
  
  -- Related references
  related_articles UUID[], -- References to your articles
  related_experiences UUID[], -- References to work experiences
  
  created_at TIMESTAMPTZ DEFAULT NOW(),
  updated_at TIMESTAMPTZ DEFAULT NOW()
);