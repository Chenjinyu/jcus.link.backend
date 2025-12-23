## 

### 1. Enable pgvector extension.
go to Supabase Dashboard -> SQL Editor -> New Query
```sql
create extension if not exists vector;
```


### 2. Choose and Decide the embedding dimensions
**Common ones:**
- OpenAI text-embedding-3-small → 1536
- OpenAI text-embedding-3-large → 3072
- Google embedding (text-embedding-004) → 768
- Ollama (nomic-embed-text) → 768
- Ollama (all-MiniLM) → 384

> I want to use Ollama(nomic-embed-text)  one for local deploying. and either OpenAI(text-embedding-3-small) or Gemini(text-embedding-004) text-embedding model
**Multi-Model Recommanded by OpenAI**
```bash
documents_1536   → OpenAI
documents_768    → Google / Ollama
documents_384    → small local models
```

### 3. Create Vector Tables
example:
```sql
create table documents_1536 (
  id uuid primary key default gen_random_uuid(),
  content text not null,
  embedding vector(1536) not null,
  embedding_model text not null,
  source text,
  metadata jsonb,
  created_at timestamptz default now()
);

```
### 4. Create a vector index
```sql
create index on documents_1536
using ivfflat (embedding vector_cosine_ops)
with (lists = 100);
```


### 5. (Optional but powerful) Create a search function
This is what **LangChain / RAG** system love.
```sql
create or replace function match_documents_1536 (
  query_embedding vector(1536),
  match_count int default 5
)
returns table (
  id uuid,
  content text,
  similarity float
)
language sql stable
as $$
  select
    id,
    content,
    1 - (embedding <=> query_embedding) as similarity
  from documents_1536
  order by embedding <=> query_embedding
  limit match_count;
$$;

```

then call it like:
```sql
select * from match_documents_1536('[...]', 5);

```

### 6. RLS (For security)
**For Supabase auth**
```sql
alter table documents_1536 enable row level security;

create policy "Allow read"
on documents_1536
for select
using (true);
```