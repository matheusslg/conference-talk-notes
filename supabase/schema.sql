-- Conference Talk Notes Schema
-- Run this in your Supabase SQL editor

-- Enable the pgvector extension (if not already enabled)
create extension if not exists vector;

-- Talks table: Parent entity for organizing content
create table if not exists talks (
  id uuid primary key default gen_random_uuid(),
  title text not null,
  speaker text,
  event text default 'AWS re:Invent 2025',
  created_at timestamp with time zone default now(),
  updated_at timestamp with time zone default now()
);

-- Talk chunks: Unified content storage for audio transcripts and slide content
create table if not exists talk_chunks (
  id uuid primary key default gen_random_uuid(),
  talk_id uuid references talks(id) on delete cascade,
  content text not null,
  content_type text not null check (content_type in ('audio_transcript', 'slide_ocr', 'slide_vision')),
  source_file text not null,
  chunk_index int default 0,
  slide_number int,  -- null for audio, 1-indexed for slides
  embedding vector(768),  -- Gemini text-embedding-004 outputs 768 dimensions
  created_at timestamp with time zone default now()
);

-- Index for vector similarity search
create index if not exists talk_chunks_embedding_idx
  on talk_chunks
  using ivfflat (embedding vector_cosine_ops)
  with (lists = 100);

-- Index for talk lookups
create index if not exists talk_chunks_talk_id_idx on talk_chunks(talk_id);
create index if not exists talk_chunks_content_type_idx on talk_chunks(content_type);

-- Search function for talk chunks
create or replace function search_talk_chunks(
  query_embedding vector(768),
  match_threshold float default 0.3,
  match_count int default 10,
  filter_talk_id uuid default null,
  filter_content_types text[] default null
)
returns table (
  id uuid,
  talk_id uuid,
  content text,
  content_type text,
  source_file text,
  slide_number int,
  similarity float
)
language plpgsql
as $$
begin
  return query
  select
    tc.id,
    tc.talk_id,
    tc.content,
    tc.content_type,
    tc.source_file,
    tc.slide_number,
    1 - (tc.embedding <=> query_embedding) as similarity
  from talk_chunks tc
  where 1 - (tc.embedding <=> query_embedding) > match_threshold
    and (filter_talk_id is null or tc.talk_id = filter_talk_id)
    and (filter_content_types is null or tc.content_type = any(filter_content_types))
  order by tc.embedding <=> query_embedding
  limit match_count;
end;
$$;

-- AI-generated content storage (summaries, insights, chat history)
create table if not exists talk_ai_content (
  id uuid primary key default gen_random_uuid(),
  talk_id uuid references talks(id) on delete cascade,
  content_type text not null check (content_type in ('summary', 'quotes', 'actions', 'chat')),
  content text not null,
  model_used text,
  created_at timestamp with time zone default now(),
  updated_at timestamp with time zone default now()
);

create index if not exists talk_ai_content_talk_id_idx on talk_ai_content(talk_id);

-- Migration: To support multiple AI generations per content type, run:
-- DROP INDEX IF EXISTS talk_ai_content_unique_idx;
-- This allows storing history of AI-generated summaries, quotes, and actions
