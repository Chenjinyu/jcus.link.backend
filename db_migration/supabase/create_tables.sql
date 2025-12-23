-- ============================================================================
-- EMBEDDING MODELS TABLE
-- Tracks different embedding models and their configurations
-- ============================================================================
CREATE TABLE embedding_models (
  id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
  name TEXT NOT NULL UNIQUE, -- 'openai-small', 'openai-large', 'ollama-local'
  provider TEXT NOT NULL, -- 'openai', 'ollama', 'cohere'
  model_identifier TEXT NOT NULL, -- 'text-embedding-3-small', 'nomic-embed-text'
  dimensions INTEGER NOT NULL, -- 1536, 3072, 768, etc.
  is_active BOOLEAN DEFAULT TRUE,
  is_local BOOLEAN DEFAULT FALSE, -- TRUE for ollama models
  cost_per_token DECIMAL(10, 8), -- Track costs
  metadata JSONB, -- Additional config
  created_at TIMESTAMPTZ DEFAULT NOW(),
  updated_at TIMESTAMPTZ DEFAULT NOW()
);

-- ============================================================================
-- CONTENT TYPES TABLE
-- Categories for different types of content
-- ============================================================================
CREATE TABLE content_types (
  id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
  name TEXT NOT NULL UNIQUE, -- 'profile', 'article', 'work_experience', etc.
  description TEXT,
  schema_definition JSONB, -- Define expected fields for this content type
  created_at TIMESTAMPTZ DEFAULT NOW()
);

-- ============================================================================
-- DOCUMENTS TABLE (Main content storage)
-- Stores all your content with metadata
-- ============================================================================
CREATE TABLE documents (
  id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
  user_id TEXT NOT NULL, -- Your user identifier
  content_type_id UUID REFERENCES content_types(id),
  
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
  
  -- Status
  status TEXT DEFAULT 'published', -- 'draft', 'published', 'archived'
  published_at TIMESTAMPTZ,
  
  -- Timestamps
  created_at TIMESTAMPTZ DEFAULT NOW(),
  updated_at TIMESTAMPTZ DEFAULT NOW(),
  deleted_at TIMESTAMPTZ -- Soft delete
);

-- ============================================================================
-- EMBEDDINGS TABLE (Vector storage)
-- Multi-model vector storage with model tracking
-- ============================================================================
CREATE TABLE embeddings (
  id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
  document_id UUID NOT NULL REFERENCES documents(id) ON DELETE CASCADE,
  embedding_model_id UUID NOT NULL REFERENCES embedding_models(id),
  
  -- The actual vector (dimension varies by model)
  embedding vector, -- Can be 768, 1536, 3072, etc.
  
  -- Metadata
  chunk_index INTEGER DEFAULT 0, -- For chunked documents
  total_chunks INTEGER DEFAULT 1,
  chunk_text TEXT, -- Store the specific text that was embedded
  
  -- Performance tracking
  embedding_time_ms INTEGER, -- Track how long embedding took
  
  created_at TIMESTAMPTZ DEFAULT NOW(),
  
  -- Composite unique constraint: one embedding per document per model per chunk
  UNIQUE(document_id, embedding_model_id, chunk_index)
);

-- ============================================================================
-- PROFILE DATA TABLE
-- Structured storage for LinkedIn-style profile information
-- ============================================================================
CREATE TABLE profile_data (
  id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
  user_id TEXT NOT NULL,
  
  -- Category: 'work_experience', 'education', 'certification', 'skill', 'value', 'goal'
  category TEXT NOT NULL,
  
  -- Structured data
  data JSONB NOT NULL, -- Flexible schema per category
  
  -- Searchable summary text
  searchable_text TEXT, -- Flattened text for embedding
  
  -- Ordering and display
  display_order INTEGER,
  is_featured BOOLEAN DEFAULT FALSE,
  is_current BOOLEAN DEFAULT TRUE, -- For work experience
  
  -- Time ranges
  start_date DATE,
  end_date DATE,
  
  -- Timestamps
  created_at TIMESTAMPTZ DEFAULT NOW(),
  updated_at TIMESTAMPTZ DEFAULT NOW()
);

-- ============================================================================
-- ARTICLES TABLE
-- Specialized table for your articles
-- ============================================================================
CREATE TABLE articles (
  id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
  user_id TEXT NOT NULL,
  document_id UUID REFERENCES documents(id) ON DELETE CASCADE,
  
  -- Article metadata
  slug TEXT UNIQUE, -- URL-friendly identifier
  title TEXT NOT NULL,
  subtitle TEXT,
  content TEXT NOT NULL,
  excerpt TEXT,
  
  -- Publishing
  status TEXT DEFAULT 'draft', -- 'draft', 'published', 'archived'
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
-- SOFT SKILLS & VALUES TABLE
-- Store personal philosophy, soft skills, aspirations
-- ============================================================================
CREATE TABLE personal_attributes (
  id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
  user_id TEXT NOT NULL,
  
  -- Type: 'soft_skill', 'value', 'worldview', 'aspiration', 'principle'
  attribute_type TEXT NOT NULL,
  
  -- Content
  title TEXT NOT NULL,
  description TEXT NOT NULL,
  examples TEXT[], -- Real-world examples
  
  -- Importance/Confidence
  importance_score INTEGER CHECK (importance_score BETWEEN 1 AND 10),
  confidence_level INTEGER CHECK (confidence_level BETWEEN 1 AND 10),
  
  -- Related references
  related_articles UUID[], -- References to your articles
  related_experiences UUID[], -- References to work experiences
  
  created_at TIMESTAMPTZ DEFAULT NOW(),
  updated_at TIMESTAMPTZ DEFAULT NOW()
);