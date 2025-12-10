# Talk Notes

**Capture, transcribe, and extract insights from conference talks using AI.**

[![Streamlit](https://img.shields.io/badge/Streamlit-FF4B4B?style=flat&logo=streamlit&logoColor=white)](https://streamlit.io)
[![Supabase](https://img.shields.io/badge/Supabase-3FCF8E?style=flat&logo=supabase&logoColor=white)](https://supabase.com)
[![Google Gemini](https://img.shields.io/badge/Google%20Gemini-4285F4?style=flat&logo=google&logoColor=white)](https://ai.google.dev)

---

## Features

- **Audio Transcription** - Upload audio recordings (MP3, M4A, WAV, etc.) and get AI-powered transcriptions with timestamps
- **Multiple Audio Fragments** - Automatically merge split recordings with gap detection
- **Slide Capture** - Upload slide photos with EXIF timestamp extraction for alignment
- **Slide-Audio Alignment** - AI matches slides to the corresponding audio segments
- **AI Insights** - Generate summaries, extract key quotes, and identify action items
- **Semantic Search** - Search across all talks using vector embeddings
- **Q&A Chat** - Ask questions about talk content using RAG
- **Markdown Export** - Download formatted notes

---

## Tech Stack

| Layer | Technology |
|-------|-----------|
| Frontend | Streamlit |
| Backend | Supabase (PostgreSQL + pgvector) |
| AI/LLM | Google Gemini (primary), OpenAI, Anthropic (optional) |
| Audio | ffmpeg, ffprobe |
| Images | Pillow, pillow-heif |

---

## Prerequisites

- Python 3.10+
- [ffmpeg](https://ffmpeg.org/download.html) and ffprobe installed
- [Supabase](https://supabase.com) account
- [Google Gemini API key](https://ai.google.dev)

---

## Installation

```bash
git clone https://github.com/gnarlysoft-ai/talk-notes.git
cd talk-notes
pip install -r requirements.txt
```

---

## Configuration

Create `.streamlit/secrets.toml`:

```toml
# Required
GEMINI_API_KEY = "your-gemini-api-key"
SUPABASE_URL = "https://your-project.supabase.co"
SUPABASE_SERVICE_KEY = "your-service-key"
APP_PASSWORD = "your-app-password"

# Optional (for additional model support)
OPENAI_API_KEY = "your-openai-key"
ANTHROPIC_API_KEY = "your-anthropic-key"
```

---

## Database Setup

1. Create a new Supabase project
2. Enable the `pgvector` extension:
   ```sql
   CREATE EXTENSION IF NOT EXISTS vector;
   ```
3. Run the SQL scripts in the `/supabase` folder to create required tables:
   - `talks` - Talk metadata
   - `talk_chunks` - Content segments with embeddings
   - `talk_ai_content` - AI-generated summaries, quotes, actions
   - `chat_history` - Q&A conversations

---

## Running the App

```bash
streamlit run app.py
```

The app will be available at `http://localhost:8501`

---

## Usage

### Creating a New Talk

1. Click **"+ New Talk"**
2. Optionally upload a title slide to auto-extract talk info
3. Enter talk title and speaker name

### Uploading Content

1. Select a talk from the list
2. Go to the **Upload** tab
3. Upload audio files and/or slide photos
4. Choose a processing mode:
   - **Audio Only** - Transcription without slides
   - **Audio + Slides** - Full alignment pipeline
   - **Multimodal** - Unified Gemini processing (recommended)

### Generating Insights

1. Go to the **Summary** tab to generate an AI summary
2. Use the **Insights** tab for key quotes and action items
3. Chat with the talk content using the Q&A interface

### Searching

Use the **Search** tab to find content across all talks using semantic search.

---

## Project Structure

```
talk-notes/
├── app.py                 # Main Streamlit application
├── requirements.txt       # Python dependencies
├── src/
│   ├── config.py          # Configuration constants
│   ├── auth.py            # Password authentication
│   ├── utils.py           # Utility functions
│   ├── search.py          # Vector search
│   ├── database/
│   │   ├── client.py      # Supabase client
│   │   ├── talks.py       # Talk CRUD operations
│   │   ├── chunks.py      # Content storage
│   │   ├── ai_content.py  # AI content storage
│   │   └── storage.py     # File storage
│   ├── llm/
│   │   ├── clients.py     # AI client initialization
│   │   ├── generation.py  # LLM generation
│   │   └── embeddings.py  # Vector embeddings
│   ├── processing/
│   │   ├── pipeline.py    # Processing orchestration
│   │   ├── audio.py       # Audio transcription
│   │   ├── images.py      # Image processing
│   │   ├── alignment.py   # Slide-audio alignment
│   │   └── multimodal.py  # Unified processing
│   └── insights/
│       ├── summary.py     # Summary generation
│       ├── quotes.py      # Quote extraction
│       ├── actions.py     # Action items
│       └── chat.py        # Q&A chat
└── supabase/              # Database migrations
```

---

## Supported Formats

**Audio**: `.mp3`, `.m4a`, `.wav`, `.mp4`, `.mpeg`, `.mpga`, `.webm`, `.qta`

**Images**: `.png`, `.jpg`, `.jpeg`, `.webp`, `.heic`, `.heif`

---

## License

MIT License - see [LICENSE](LICENSE) for details.
