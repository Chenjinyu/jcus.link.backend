create table embeddings (
  id bigserial primary key,
  original_text text not null,
  embedding vector(1536),       -- default for OpenAI; flexible
  model_name text not null,
  created_at timestamp default now()
);
