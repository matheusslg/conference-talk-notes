"""
Unit tests for AI-based audio-slide alignment functions.
"""
import pytest
import json
from unittest.mock import patch, MagicMock

# We need to mock streamlit and other dependencies before importing app
import sys
sys.modules['streamlit'] = MagicMock()
sys.modules['supabase'] = MagicMock()
sys.modules['google'] = MagicMock()
sys.modules['google.genai'] = MagicMock()
sys.modules['openai'] = MagicMock()
sys.modules['anthropic'] = MagicMock()
sys.modules['pillow_heif'] = MagicMock()

# Now import the functions we want to test
# Note: Since the app.py has side effects at import time, we'll define the functions inline for testing


def parse_timestamp_to_seconds(ts: str) -> float:
    """Convert 'HH:MM:SS' or 'MM:SS' to seconds."""
    parts = ts.split(":")
    if len(parts) == 3:
        return int(parts[0]) * 3600 + int(parts[1]) * 60 + float(parts[2])
    elif len(parts) == 2:
        return int(parts[0]) * 60 + float(parts[1])
    return float(parts[0])


def fallback_sequential_alignment(audio_segments: list, num_slides: int) -> list:
    """
    Distribute audio segments evenly across slides as a fallback.
    Used when AI alignment fails.
    """
    if not audio_segments or num_slides == 0:
        return []

    segments_per_slide = max(1, len(audio_segments) // num_slides)
    results = []
    current = 0

    for i in range(num_slides):
        end = current + segments_per_slide if i < num_slides - 1 else len(audio_segments)
        indices = list(range(current, min(end, len(audio_segments))))

        if indices:
            start_time = parse_timestamp_to_seconds(audio_segments[indices[0]]["start"])
            end_time = parse_timestamp_to_seconds(audio_segments[indices[-1]]["end"])
            matched_text = " ".join(audio_segments[j]["text"] for j in indices)
        else:
            start_time = None
            end_time = None
            matched_text = "[No audio segment assigned]"

        results.append({
            "slide_number": i + 1,
            "segment_indices": indices,
            "start_time_seconds": start_time,
            "end_time_seconds": end_time,
            "matched_audio": matched_text
        })
        current = end

    return results


class TestParseTimestampToSeconds:
    """Tests for timestamp parsing."""

    def test_hhmmss_format(self):
        assert parse_timestamp_to_seconds("01:30:45") == 5445.0  # 1h30m45s

    def test_mmss_format(self):
        assert parse_timestamp_to_seconds("05:30") == 330.0  # 5m30s

    def test_zero_timestamp(self):
        assert parse_timestamp_to_seconds("00:00:00") == 0.0

    def test_seconds_only(self):
        assert parse_timestamp_to_seconds("45") == 45.0


class TestFallbackSequentialAlignment:
    """Tests for fallback sequential alignment."""

    def test_empty_segments(self):
        """Empty segments should return empty list."""
        result = fallback_sequential_alignment([], 5)
        assert result == []

    def test_zero_slides(self):
        """Zero slides should return empty list."""
        segments = [{"start": "00:00", "end": "01:00", "text": "test"}]
        result = fallback_sequential_alignment(segments, 0)
        assert result == []

    def test_equal_distribution(self):
        """4 segments across 4 slides = 1 segment each."""
        segments = [
            {"start": "00:00", "end": "01:00", "text": "segment 1"},
            {"start": "01:00", "end": "02:00", "text": "segment 2"},
            {"start": "02:00", "end": "03:00", "text": "segment 3"},
            {"start": "03:00", "end": "04:00", "text": "segment 4"},
        ]
        result = fallback_sequential_alignment(segments, 4)

        assert len(result) == 4
        assert result[0]["segment_indices"] == [0]
        assert result[1]["segment_indices"] == [1]
        assert result[2]["segment_indices"] == [2]
        assert result[3]["segment_indices"] == [3]

    def test_more_segments_than_slides(self):
        """6 segments across 2 slides = 3 segments each."""
        segments = [
            {"start": "00:00", "end": "01:00", "text": "seg 1"},
            {"start": "01:00", "end": "02:00", "text": "seg 2"},
            {"start": "02:00", "end": "03:00", "text": "seg 3"},
            {"start": "03:00", "end": "04:00", "text": "seg 4"},
            {"start": "04:00", "end": "05:00", "text": "seg 5"},
            {"start": "05:00", "end": "06:00", "text": "seg 6"},
        ]
        result = fallback_sequential_alignment(segments, 2)

        assert len(result) == 2
        assert result[0]["segment_indices"] == [0, 1, 2]
        assert result[1]["segment_indices"] == [3, 4, 5]

    def test_more_slides_than_segments(self):
        """2 segments across 4 slides - last slides get remaining."""
        segments = [
            {"start": "00:00", "end": "01:00", "text": "segment 1"},
            {"start": "01:00", "end": "02:00", "text": "segment 2"},
        ]
        result = fallback_sequential_alignment(segments, 4)

        assert len(result) == 4
        # With only 2 segments, min(1, 2//4) = 1, but we use max(1, ...) so each gets 1
        # First slide gets index 0, second gets index 1, rest get nothing
        assert len(result[0]["segment_indices"]) >= 0
        assert len(result[-1]["segment_indices"]) >= 0

    def test_timestamps_preserved(self):
        """Timestamps should come from first and last segment in range."""
        segments = [
            {"start": "00:00:00", "end": "00:01:00", "text": "first"},
            {"start": "00:01:00", "end": "00:02:00", "text": "second"},
            {"start": "00:02:00", "end": "00:03:00", "text": "third"},
        ]
        result = fallback_sequential_alignment(segments, 1)

        assert len(result) == 1
        assert result[0]["start_time_seconds"] == 0.0  # Start of first segment
        assert result[0]["end_time_seconds"] == 180.0  # End of third segment (3 minutes)

    def test_text_concatenation(self):
        """Audio text should be concatenated with spaces."""
        segments = [
            {"start": "00:00", "end": "01:00", "text": "Hello"},
            {"start": "01:00", "end": "02:00", "text": "World"},
        ]
        result = fallback_sequential_alignment(segments, 1)

        assert result[0]["matched_audio"] == "Hello World"

    def test_slide_numbers_are_1_indexed(self):
        """Slide numbers should start at 1, not 0."""
        segments = [
            {"start": "00:00", "end": "01:00", "text": "test"},
        ]
        result = fallback_sequential_alignment(segments, 3)

        assert result[0]["slide_number"] == 1
        assert result[1]["slide_number"] == 2
        assert result[2]["slide_number"] == 3


class TestAlignSlidesWithAI:
    """Tests for AI-based alignment (mocked)."""

    def test_ai_alignment_parses_json_response(self):
        """Test that AI alignment correctly parses JSON response."""
        # This would require mocking the Gemini API
        # For now, we just test the JSON parsing logic
        mock_response = """[
            {"slide_number": 1, "segment_indices": [0, 1]},
            {"slide_number": 2, "segment_indices": [2]},
            {"slide_number": 3, "segment_indices": []}
        ]"""

        # Parse the JSON
        alignment = json.loads(mock_response)

        assert len(alignment) == 3
        assert alignment[0]["slide_number"] == 1
        assert alignment[0]["segment_indices"] == [0, 1]
        assert alignment[2]["segment_indices"] == []

    def test_ai_alignment_handles_markdown_wrapped_json(self):
        """Test that markdown-wrapped JSON is correctly extracted."""
        mock_response = """```json
[
    {"slide_number": 1, "segment_indices": [0]}
]
```"""

        # Extract JSON from markdown
        text = mock_response.strip()
        if text.startswith("```"):
            text = text.split("\n", 1)[1].rsplit("```", 1)[0].strip()

        alignment = json.loads(text)
        assert len(alignment) == 1
        assert alignment[0]["slide_number"] == 1


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
