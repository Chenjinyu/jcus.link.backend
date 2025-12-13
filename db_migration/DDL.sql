-- Run the SQL in Supabase SQL Eidtor.

-- Enable the pgvector extension
create extension if not exists vector;


-- Create the table to store documents. it's for JSON - JC
create table documents (
    id uuid primary key,
    content text,           -- Document.page_content
    metadata jsonb,         -- Document.metadata
    embedding vector (1536) -- 1536 is standard for OpenAI embeddings
);

-- Create the function to search for documents (using cosine similarity)
create function match_documents(
    query_embedding vector (1536),
    filter jsonb default '{}'
) returns table (
    id uuid,
    context text,
    metadata jsonb,
    similarity float
) language plpgsql as $$
#vairable_conflict use_column
begin 
    return query
    select
        id,             
        content,
        metadata,
        1 - (documents.embedding <=> query_embedding) as similarity
    from documents
    where metadata @> filter
    order by documents.embedding <=> query_embedding;
end;
$$;