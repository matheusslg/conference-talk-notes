# Processing layer - audio transcription, image processing, alignment

from src.processing.audio import (
    convert_audio_to_mp3,
    get_audio_creation_time,
    get_audio_duration,
    get_audio_metadata,
    get_gemini_model_for_multimodal,
    sort_audio_fragments,
    merge_transcripts,
    normalize_transcript_timestamps,
    transcribe_audio_with_timestamps,
)

from src.processing.images import (
    create_thumbnail_base64,
    get_image_timestamp,
    extract_talk_info_from_slide,
    extract_slide_ocr,
    describe_slide_vision,
)

from src.processing.alignment import (
    build_slide_timing_context,
    match_audio_to_slide,
    align_slides_with_ai,
    fallback_sequential_alignment,
)

from src.processing.multimodal import (
    build_multimodal_prompt,
    normalize_image_for_gemini,
    execute_multimodal_request,
)

from src.processing.pipeline import (
    step_transcribe_audio,
    step_transcribe_audio_multi,
    step_prepare_slides,
    step_upload_audio,
    step_align_slides_to_audio,
    step_store_segments,
    step_process_multimodal,
    step_store_multimodal_results,
    ingest_audio,
    ingest_images,
    parse_aligned_content,
)

__all__ = [
    # Audio
    "convert_audio_to_mp3",
    "get_audio_creation_time",
    "get_audio_duration",
    "get_audio_metadata",
    "get_gemini_model_for_multimodal",
    "sort_audio_fragments",
    "merge_transcripts",
    "normalize_transcript_timestamps",
    "transcribe_audio_with_timestamps",
    # Images
    "create_thumbnail_base64",
    "get_image_timestamp",
    "extract_talk_info_from_slide",
    "extract_slide_ocr",
    "describe_slide_vision",
    # Alignment
    "build_slide_timing_context",
    "match_audio_to_slide",
    "align_slides_with_ai",
    "fallback_sequential_alignment",
    # Multimodal
    "build_multimodal_prompt",
    "normalize_image_for_gemini",
    "execute_multimodal_request",
    # Pipeline
    "step_transcribe_audio",
    "step_transcribe_audio_multi",
    "step_prepare_slides",
    "step_upload_audio",
    "step_align_slides_to_audio",
    "step_store_segments",
    "step_process_multimodal",
    "step_store_multimodal_results",
    "ingest_audio",
    "ingest_images",
    "parse_aligned_content",
]
