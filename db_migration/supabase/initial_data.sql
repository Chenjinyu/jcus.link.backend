-- ============================================================================
-- INSERT EMBEDDING MODELS
-- ============================================================================
INSERT INTO embedding_models (name, provider, model_identifier, dimensions, is_local) VALUES
('openai-small', 'openai', 'text-embedding-3-small', 1536, FALSE),
('openai-large', 'openai', 'text-embedding-3-large', 3072, FALSE),
('ollama-nomic', 'ollama', 'nomic-embed-text', 768, TRUE),
('ollama-mxbai', 'ollama', 'mxbai-embed-large', 1024, TRUE);

-- ============================================================================
-- INSERT CONTENT TYPES
-- ============================================================================
INSERT INTO content_types (name, description, schema_definition) VALUES
('profile', 'General profile information', '{"fields": ["summary", "bio"]}'::jsonb),
('work_experience', 'Professional work history', '{"fields": ["company", "title", "description", "start_date", "end_date"]}'::jsonb),
('education', 'Educational background', '{"fields": ["school", "degree", "field", "start_date", "end_date"]}'::jsonb),
('certification', 'Professional certifications', '{"fields": ["name", "issuer", "date", "credential_id"]}'::jsonb),
('skill', 'Technical and soft skills', '{"fields": ["name", "category", "proficiency"]}'::jsonb),
('article', 'Written articles and blog posts', '{"fields": ["title", "content", "tags", "category"]}'::jsonb),
('value', 'Personal values and principles', '{"fields": ["title", "description", "examples"]}'::jsonb),
('worldview', 'Understanding and philosophy', '{"fields": ["topic", "perspective", "reasoning"]}'::jsonb),
('aspiration', 'Goals and aspirations', '{"fields": ["goal", "why", "timeline"]}'::jsonb);