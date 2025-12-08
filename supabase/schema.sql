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
  audio_url text,  -- URL to audio file in Supabase Storage for playback
  audio_start_timestamp timestamp with time zone,  -- When audio recording started (for alignment)
  created_at timestamp with time zone default now(),
  updated_at timestamp with time zone default now()
);

-- Talk chunks: Unified content storage for audio transcripts and slide content
create table if not exists talk_chunks (
  id uuid primary key default gen_random_uuid(),
  talk_id uuid references talks(id) on delete cascade,
  content text not null,
  content_type text not null check (content_type in ('audio_transcript', 'slide_ocr', 'slide_vision', 'aligned_segment')),
  source_file text not null,
  chunk_index int default 0,
  slide_number int,  -- null for audio, 1-indexed for slides
  start_time_seconds float,  -- relative timestamp from recording start
  end_time_seconds float,    -- end timestamp for aligned segments
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
  start_time_seconds float,
  end_time_seconds float,
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
    tc.start_time_seconds,
    tc.end_time_seconds,
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

-- Migration: Add timestamp columns for aligned audio-slide segments
-- ALTER TABLE talk_chunks ADD COLUMN start_time_seconds float;
-- ALTER TABLE talk_chunks ADD COLUMN end_time_seconds float;
-- ALTER TABLE talk_chunks DROP CONSTRAINT talk_chunks_content_type_check;
-- ALTER TABLE talk_chunks ADD CONSTRAINT talk_chunks_content_type_check
--   CHECK (content_type in ('audio_transcript', 'slide_ocr', 'slide_vision', 'aligned_segment'));

-- Migration: Add slide thumbnail for Timeline preview
-- ALTER TABLE talk_chunks ADD COLUMN slide_thumbnail text;

-- Migration: Add audio URL and start timestamp for playback and alignment
-- ALTER TABLE talks ADD COLUMN audio_url text;
-- ALTER TABLE talks ADD COLUMN audio_start_timestamp timestamp with time zone;

-- Migration: Add slide_urls for multimodal summary/insights generation
-- Stores URLs to original slide images in Supabase Storage
-- ALTER TABLE talks ADD COLUMN slide_urls jsonb DEFAULT '[]';

-- Migration: slide_urls format changed from array of strings to array of objects
-- Old format: ["https://storage.../slide_001.jpg", "https://storage.../slide_002.jpg"]
-- New format: [{"url": "https://...", "filename": "IMG_1234.jpg", "taken_at": "2025-01-01T11:02:15+00:00"}, ...]
-- This stores original filename and EXIF timestamp for chronological context in multimodal prompts
-- No schema change needed - jsonb column handles both formats (backwards compatible)

-- Storage bucket: talk-slides
-- Create in Supabase dashboard: Storage > New bucket > "talk-slides" (public access)

-- Migration: Add author_name for talk attribution
-- ALTER TABLE talks ADD COLUMN author_name text;
