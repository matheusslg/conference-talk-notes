"""Processing pipeline - orchestrates audio/slide processing workflows."""

import io
import os
import tempfile
import time
from datetime import datetime

from PIL import Image

from src.config import CHUNK_SIZE, CHUNK_OVERLAP, PHOTO_CAPTURE_DELAY_SECONDS
from src.utils import (
    chunk_text,
    parse_timestamp_to_seconds,
    format_seconds_to_timestamp,
)
from src.database import supabase
from src.llm import ai, generate_embedding
from src.processing.audio import (
    get_audio_creation_time,
    get_audio_duration,
    get_audio_metadata,
    get_gemini_model_for_multimodal,
    sort_audio_fragments,
    merge_transcripts,
    transcribe_audio_with_timestamps,
)
from src.processing.images import (
    create_thumbnail_base64,
    get_image_timestamp,
    extract_slide_ocr,
    describe_slide_vision,
)
from src.processing.alignment import (
    match_audio_to_slide,
    align_slides_with_ai,
    fallback_sequential_alignment,
)
from src.processing.multimodal import execute_multimodal_request


def step_transcribe_audio(audio_bytes: bytes, audio_name: str, model: str = "gemini-2.5-pro") -> dict:
    """Step 1: Transcribe audio using Gemini.

    Returns dict with:
        - segments: list of {start, end, text}
        - messages: list of status messages
    """
    result = {"segments": [], "messages": []}

    # Write to temp file
    suffix = os.path.splitext(audio_name)[1]
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
        tmp.write(audio_bytes)
        tmp_path = tmp.name

    try:
        # Transcribe using existing function
        segments = transcribe_audio_with_timestamps(tmp_path, model)
        result["segments"] = segments
        result["messages"].append(f"Transcribed {len(segments)} segments")

        # Calculate total duration
        if segments:
            last_end = parse_timestamp_to_seconds(segments[-1]["end"])
            result["messages"].append(f"Total duration: {format_seconds_to_timestamp(last_end)}")
    finally:
        os.unlink(tmp_path)

    return result


def step_transcribe_audio_multi(audio_files: list[dict], model: str = "gemini-2.5-pro") -> dict:
    """Transcribe multiple audio fragments and merge with adjusted timestamps.

    Args:
        audio_files: List of {"bytes": bytes, "name": str}
        model: Model to use for transcription

    Returns:
        dict with:
            - segments: merged list of {start, end, text, source_file}
            - audio_files: sorted list with metadata
            - gaps: list of {after_file, gap_seconds}
            - total_duration: float
            - messages: list of status messages
    """
    result = {
        "segments": [],
        "audio_files": [],
        "gaps": [],
        "total_duration": 0.0,
        "messages": []
    }

    # Handle single file case - use existing function
    if len(audio_files) == 1:
        single_result = step_transcribe_audio(
            audio_files[0]["bytes"],
            audio_files[0]["name"],
            model
        )
        result["segments"] = single_result["segments"]
        result["messages"] = single_result["messages"]
        result["audio_files"] = [{
            "bytes": audio_files[0]["bytes"],
            "name": audio_files[0]["name"],
            "creation_time": None,
            "duration_seconds": None,
            "offset_seconds": 0.0,
            "gap_seconds": None
        }]
        return result

    # Multiple files - sort by creation time
    result["messages"].append(f"Processing {len(audio_files)} audio fragments...")
    sorted_files = sort_audio_fragments(audio_files)
    result["audio_files"] = sorted_files

    # Log sorting results
    files_with_time = sum(1 for f in sorted_files if f["creation_time"] is not None)
    if files_with_time == 0:
        result["messages"].append("Warning: No creation timestamps found. Using filename order.")
    else:
        result["messages"].append(f"Sorted by creation time ({files_with_time}/{len(sorted_files)} files have timestamps)")

    # Detect and log gaps
    for i, f in enumerate(sorted_files):
        if f["gap_seconds"] is not None and f["gap_seconds"] > 0:
            prev_file = sorted_files[i - 1]["name"] if i > 0 else None
            result["gaps"].append({
                "after_file": prev_file,
                "before_file": f["name"],
                "gap_seconds": f["gap_seconds"]
            })
            # Format gap duration
            gap_min = int(f["gap_seconds"] // 60)
            gap_sec = int(f["gap_seconds"] % 60)
            result["messages"].append(f"Gap detected: {gap_min}m {gap_sec}s before {f['name']}")

    # Transcribe each file
    transcripts_for_merge = []
    for i, audio_file in enumerate(sorted_files):
        result["messages"].append(f"Transcribing {audio_file['name']}...")

        single_result = step_transcribe_audio(
            audio_file["bytes"],
            audio_file["name"],
            model
        )

        transcripts_for_merge.append({
            "segments": single_result["segments"],
            "offset_seconds": audio_file["offset_seconds"],
            "filename": audio_file["name"]
        })

        result["messages"].extend(single_result["messages"])

    # Merge all transcripts
    result["messages"].append("Merging transcripts with adjusted timestamps...")
    merged_segments = merge_transcripts(transcripts_for_merge)
    result["segments"] = merged_segments
    result["messages"].append(f"Total: {len(merged_segments)} segments across {len(sorted_files)} files")

    # Calculate total duration
    if merged_segments:
        last_end = parse_timestamp_to_seconds(merged_segments[-1]["end"])
        result["total_duration"] = last_end
        result["messages"].append(f"Total duration: {format_seconds_to_timestamp(last_end)}")

    return result


def step_prepare_slides(slide_files: list) -> dict:
    """Step 2a: Extract EXIF timestamps from slides and sort.

    Args:
        slide_files: list of dicts with {bytes, name}

    Returns dict with:
        - slides_with_time: sorted list of slide dicts
        - use_exif_alignment: bool
        - messages: list of status messages
    """
    result = {"slides_with_time": [], "use_exif_alignment": False, "messages": []}

    for slide in slide_files:
        img_bytes = slide["bytes"]
        timestamp = get_image_timestamp(img_bytes)
        result["slides_with_time"].append({
            "bytes": img_bytes,
            "timestamp": timestamp,
            "name": slide["name"],
            "relative_seconds": None
        })

    # Sort by timestamp (None timestamps go to end, then by filename)
    result["slides_with_time"].sort(
        key=lambda x: (x["timestamp"] is None, x["timestamp"] or datetime.max, x["name"])
    )

    # Calculate relative timestamps for slides with EXIF
    recording_start = result["slides_with_time"][0]["timestamp"] if result["slides_with_time"][0]["timestamp"] else None
    for slide in result["slides_with_time"]:
        if slide["timestamp"] and recording_start:
            delta = slide["timestamp"] - recording_start
            slide["relative_seconds"] = delta.total_seconds()

    # Decide alignment strategy: EXIF if ≥50% have timestamps
    slides_with_exif = sum(1 for s in result["slides_with_time"] if s["relative_seconds"] is not None)
    result["use_exif_alignment"] = slides_with_exif >= len(result["slides_with_time"]) * 0.5

    # Debug logging
    first_slide_ts = result["slides_with_time"][0]["timestamp"]
    if first_slide_ts:
        result["messages"].append(f"First slide taken at: {first_slide_ts.strftime('%Y-%m-%d %H:%M:%S')}")
    else:
        result["messages"].append("No EXIF timestamp found in first slide")

    result["messages"].append(f"Slides with EXIF: {slides_with_exif}/{len(result['slides_with_time'])}")
    result["messages"].append(f"Alignment mode: {'EXIF-based' if result['use_exif_alignment'] else 'AI-based'}")

    return result


def step_upload_audio(audio_bytes: bytes, audio_name: str, talk_id: str) -> dict:
    """Step 2b: Upload audio to Supabase storage and extract creation time.

    Returns dict with:
        - audio_url: str or None
        - audio_start_time: datetime or None
        - messages: list of status messages
    """
    result = {"audio_url": None, "audio_start_time": None, "messages": []}

    # Write to temp file for metadata extraction
    suffix = os.path.splitext(audio_name)[1]
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
        tmp.write(audio_bytes)
        tmp_path = tmp.name

    try:
        # Extract audio creation time
        result["audio_start_time"] = get_audio_creation_time(tmp_path)
        if result["audio_start_time"]:
            result["messages"].append(f"Audio recording started at: {result['audio_start_time'].strftime('%Y-%m-%d %H:%M:%S')}")
        else:
            result["messages"].append("No audio creation time found in metadata")
    finally:
        os.unlink(tmp_path)

    # Upload to Supabase Storage
    try:
        audio_filename = f"{talk_id}/{audio_name}"
        supabase.storage.from_("talk-audio").upload(
            audio_filename,
            audio_bytes,
            {"content-type": "audio/mpeg"}
        )
        result["audio_url"] = supabase.storage.from_("talk-audio").get_public_url(audio_filename)
        result["messages"].append("Audio uploaded for playback")

        # Update talk record
        update_data = {"audio_url": result["audio_url"]}
        if result["audio_start_time"]:
            update_data["audio_start_timestamp"] = result["audio_start_time"].isoformat()
        supabase.from_("talks").update(update_data).eq("id", talk_id).execute()

    except Exception as upload_error:
        result["messages"].append(f"Audio upload skipped: {str(upload_error)}")

    return result


def step_align_slides_to_audio(
    slides_with_time: list,
    audio_segments: list,
    audio_start_time,
    use_exif: bool,
    model: str = "gemini-2.5-pro"
) -> dict:
    """Step 3: Align slides to audio segments.

    Returns dict with:
        - alignment: list of aligned slide dicts
        - messages: list of status messages
    """
    result = {"alignment": [], "messages": []}

    # Recalculate relative timestamps using audio start time (if available)
    if audio_start_time and slides_with_time:
        for slide in slides_with_time:
            if slide["timestamp"]:
                delta = slide["timestamp"] - audio_start_time
                slide["relative_seconds"] = delta.total_seconds()
            else:
                slide["relative_seconds"] = None

        # Re-check EXIF alignment availability
        slides_with_exif = sum(1 for s in slides_with_time if s["relative_seconds"] is not None)
        use_exif = slides_with_exif >= len(slides_with_time) * 0.5

        # Debug: log alignment offset
        first_slide = slides_with_time[0]
        if first_slide["relative_seconds"] is not None:
            offset = first_slide["relative_seconds"]
            result["messages"].append(f"First slide offset from audio start: {offset:.1f}s ({format_seconds_to_timestamp(offset)})")

    if use_exif:
        # EXIF-based alignment
        result["messages"].append("Using EXIF-based alignment")
        result["messages"].append(f"Photo capture delay adjustment: -{PHOTO_CAPTURE_DELAY_SECONDS}s")

        for idx, slide in enumerate(slides_with_time):
            img = Image.open(io.BytesIO(slide["bytes"]))
            ocr_text = extract_slide_ocr(img)
            vision_desc = describe_slide_vision(img)
            thumbnail = create_thumbnail_base64(slide["bytes"])

            next_slide_time = slides_with_time[idx + 1]["relative_seconds"] if idx + 1 < len(slides_with_time) else None
            matched_audio = match_audio_to_slide(audio_segments, slide["relative_seconds"], next_slide_time)

            # Calculate adjusted times for storage (matching what match_audio_to_slide uses)
            adjusted_start = max(0, slide["relative_seconds"] - PHOTO_CAPTURE_DELAY_SECONDS) if slide["relative_seconds"] is not None else None
            adjusted_end = (next_slide_time - PHOTO_CAPTURE_DELAY_SECONDS) if next_slide_time is not None else None

            result["alignment"].append({
                "slide_number": idx + 1,
                "slide_name": slide["name"],
                "thumbnail": thumbnail,
                "start_time_seconds": adjusted_start,
                "end_time_seconds": adjusted_end,
                "matched_audio": matched_audio,
                "ocr_text": ocr_text,
                "vision_description": vision_desc
            })
    else:
        # AI-based alignment
        result["messages"].append("Using AI-based alignment (insufficient EXIF timestamps)")

        # First process all slides
        processed_slides = []
        for idx, slide in enumerate(slides_with_time):
            img = Image.open(io.BytesIO(slide["bytes"]))
            ocr_text = extract_slide_ocr(img, model)
            vision_desc = describe_slide_vision(img, model)
            thumbnail = create_thumbnail_base64(slide["bytes"])

            processed_slides.append({
                "slide_number": idx + 1,
                "ocr_text": ocr_text,
                "vision_description": vision_desc,
                "thumbnail": thumbnail,
                "filename": slide["name"]
            })

        # Run AI alignment
        try:
            alignments = align_slides_with_ai(audio_segments, processed_slides, model)
            result["messages"].append("AI alignment successful")
        except Exception as align_error:
            result["messages"].append(f"AI alignment failed, using sequential fallback: {str(align_error)}")
            alignments = fallback_sequential_alignment(audio_segments, len(processed_slides))

        # Build alignment result
        for alignment in alignments:
            slide_idx = alignment["slide_number"] - 1
            slide = processed_slides[slide_idx]

            result["alignment"].append({
                "slide_number": alignment["slide_number"],
                "slide_name": slide["filename"],
                "thumbnail": slide["thumbnail"],
                "start_time_seconds": alignment["start_time_seconds"],
                "end_time_seconds": alignment["end_time_seconds"],
                "matched_audio": alignment["matched_audio"],
                "ocr_text": slide["ocr_text"],
                "vision_description": slide["vision_description"]
            })

    result["messages"].append(f"Aligned {len(result['alignment'])} slides")
    return result


def step_store_segments(talk_id: str, alignment: list, audio_name: str = "") -> dict:
    """Step 4: Generate embeddings and store aligned segments.

    Returns dict with:
        - segment_count: int
        - messages: list of status messages
    """
    result = {"segment_count": 0, "messages": []}

    for item in alignment:
        time_str = format_seconds_to_timestamp(item["start_time_seconds"])
        aligned_content = f"""## Slide {item['slide_number']} [{time_str}]

### Slide Text
{item['ocr_text'] if item['ocr_text'] else "[No text detected]"}

### Visual Description
{item['vision_description']}

### Speaker Said
{item['matched_audio']}
"""

        embedding = generate_embedding(aligned_content)

        source_file = f"{audio_name}+{item['slide_name']}" if audio_name else item['slide_name']

        supabase.from_("talk_chunks").insert({
            "talk_id": talk_id,
            "content": aligned_content,
            "content_type": "aligned_segment",
            "source_file": source_file,
            "slide_number": item["slide_number"],
            "chunk_index": 0,
            "start_time_seconds": item["start_time_seconds"],
            "end_time_seconds": item["end_time_seconds"],
            "embedding": embedding,
            "slide_thumbnail": item["thumbnail"]
        }).execute()

        result["segment_count"] += 1

    result["messages"].append(f"Stored {result['segment_count']} aligned segments")
    return result


def step_process_multimodal(
    audio_file: dict,
    slide_files: list,
    talk_id: str,
    model: str = "gemini-2.5-pro"
) -> dict:
    """Unified processing step - sends audio + all slides to Gemini in one request.

    Args:
        audio_file: dict with {bytes, name}
        slide_files: list of dicts with {bytes, name}
        talk_id: Talk ID for audio upload
        model: Gemini model to use (must be a Gemini model)

    Returns:
        dict with:
            - alignment: list of slide alignment dicts
            - messages: list of status messages
            - audio_url: URL of uploaded audio (for playback)
            - model_used: The model that was used for processing
    """
    # Ensure we have a valid Gemini model
    gemini_model = get_gemini_model_for_multimodal(model)

    result = {"alignment": [], "messages": [], "audio_url": None, "model_used": gemini_model}

    # 1. Extract EXIF timestamps from slides and sort
    slides_with_time = []
    for slide in slide_files:
        timestamp = get_image_timestamp(slide["bytes"])
        slides_with_time.append({
            "bytes": slide["bytes"],
            "timestamp": timestamp,
            "name": slide["name"],
            "relative_seconds": None
        })

    # Sort by timestamp (None timestamps go to end, then by filename)
    slides_with_time.sort(
        key=lambda x: (x["timestamp"] is None, x["timestamp"] or datetime.max, x["name"])
    )

    # Calculate relative timestamps
    if slides_with_time and slides_with_time[0]["timestamp"]:
        recording_start = slides_with_time[0]["timestamp"]
        for slide in slides_with_time:
            if slide["timestamp"]:
                delta = slide["timestamp"] - recording_start
                slide["relative_seconds"] = delta.total_seconds()

    slides_with_exif = sum(1 for s in slides_with_time if s["timestamp"] is not None)
    result["messages"].append(f"Slides with EXIF timestamps: {slides_with_exif}/{len(slides_with_time)}")

    # 2. Upload audio to Gemini for processing
    suffix = os.path.splitext(audio_file['name'])[1]
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
        tmp.write(audio_file['bytes'])
        tmp_path = tmp.name

    try:
        result["messages"].append("Uploading audio to Gemini...")
        uploaded_audio = ai.files.upload(file=tmp_path)

        # Wait for processing
        while uploaded_audio.state.name == "PROCESSING":
            time.sleep(2)
            uploaded_audio = ai.files.get(name=uploaded_audio.name)

        if uploaded_audio.state.name == "FAILED":
            raise Exception("Audio processing failed in Gemini")

        result["messages"].append("Audio processed, running unified analysis...")

        # 3. Prepare slides metadata for prompt
        slides_metadata = [
            {
                "number": i + 1,
                "name": s["name"],
                "relative_seconds": s.get("relative_seconds")
            }
            for i, s in enumerate(slides_with_time)
        ]

        # 4. Execute multimodal request
        slide_images = [s["bytes"] for s in slides_with_time]
        alignment_result = execute_multimodal_request(
            uploaded_audio,
            slide_images,
            slides_metadata,
            model=gemini_model
        )

        result["messages"].append(f"Gemini returned {len(alignment_result)} slide analyses")

        # 5. Add thumbnails and slide names to results
        for i, item in enumerate(alignment_result):
            item["thumbnail"] = create_thumbnail_base64(slides_with_time[i]["bytes"])
            item["slide_name"] = slides_with_time[i]["name"]

        result["alignment"] = alignment_result

    finally:
        os.unlink(tmp_path)

    # 6. Upload audio to Supabase Storage for playback
    try:
        audio_filename = f"{talk_id}/{audio_file['name']}"
        supabase.storage.from_("talk-audio").upload(
            audio_filename,
            audio_file['bytes'],
            {"content-type": "audio/mpeg"}
        )
        result["audio_url"] = supabase.storage.from_("talk-audio").get_public_url(audio_filename)
        result["messages"].append("Audio uploaded for playback")

        # Update talk record
        supabase.from_("talks").update({"audio_url": result["audio_url"]}).eq("id", talk_id).execute()

    except Exception as upload_error:
        result["messages"].append(f"Audio storage skipped: {str(upload_error)}")

    # 7. Upload slides to Supabase Storage for multimodal summary/insights
    try:
        slide_urls = []
        for i, slide in enumerate(slides_with_time):
            slide_filename = f"{talk_id}/slide_{i+1:03d}.jpg"

            # Normalize image to JPEG for storage
            img = Image.open(io.BytesIO(slide["bytes"]))
            if img.mode in ("RGBA", "LA", "P"):
                img = img.convert("RGB")
            elif img.mode != "RGB":
                img = img.convert("RGB")

            buffer = io.BytesIO()
            img.save(buffer, format="JPEG", quality=85)
            normalized_bytes = buffer.getvalue()

            supabase.storage.from_("talk-slides").upload(
                slide_filename,
                normalized_bytes,
                {"content-type": "image/jpeg"}
            )

            url = supabase.storage.from_("talk-slides").get_public_url(slide_filename)
            # Store metadata alongside URL for multimodal prompts
            slide_urls.append({
                "url": url,
                "filename": slide["name"],
                "taken_at": slide["timestamp"].isoformat() if slide["timestamp"] else None
            })

        # Store slide URLs in talk record
        supabase.from_("talks").update({"slide_urls": slide_urls}).eq("id", talk_id).execute()
        result["slide_urls"] = slide_urls
        result["messages"].append(f"Uploaded {len(slide_urls)} slides for multimodal")

    except Exception as slide_error:
        result["messages"].append(f"Slide storage skipped: {str(slide_error)}")

    return result


def step_store_multimodal_results(talk_id: str, alignment: list, audio_name: str = "") -> dict:
    """Store multimodal processing results to database.

    Args:
        talk_id: Talk ID
        alignment: List of slide alignment dicts from Gemini
        audio_name: Original audio filename

    Returns:
        dict with:
            - segment_count: number of segments stored
            - messages: list of status messages
    """
    result = {"segment_count": 0, "messages": []}

    for item in alignment:
        # Combine content for storage and embedding
        key_points_text = "\n".join(f"- {p}" for p in item.get("key_points", []))

        combined_content = f"""## Slide {item['slide_number']} [{format_seconds_to_timestamp(item.get('start_time_seconds'))}]

### Slide Text
{item.get('ocr_text') or '[No text detected]'}

### Visual Description
{item.get('visual_description') or '[Text-only slide]'}

### Speaker Said
{item.get('transcript_text') or '[No transcript]'}

### Key Points
{key_points_text or '[No key points]'}
"""

        embedding = generate_embedding(combined_content)

        source_file = f"{audio_name}+{item['slide_name']}" if audio_name else item['slide_name']

        supabase.from_("talk_chunks").insert({
            "talk_id": talk_id,
            "content": combined_content,
            "content_type": "aligned_segment",
            "source_file": source_file,
            "slide_number": item["slide_number"],
            "chunk_index": 0,
            "start_time_seconds": item.get("start_time_seconds"),
            "end_time_seconds": item.get("end_time_seconds"),
            "embedding": embedding,
            "slide_thumbnail": item.get("thumbnail")
        }).execute()

        result["segment_count"] += 1

    result["messages"].append(f"Stored {result['segment_count']} aligned segments")
    return result


def ingest_audio(file_path: str, file_name: str, talk_id: str, model: str, progress_callback=None) -> dict:
    """Ingest audio file: transcribe, chunk, embed, and store.

    Args:
        file_path: Path to audio file
        file_name: Original filename
        talk_id: Talk ID
        model: Gemini model to use
        progress_callback: Optional callback(progress, message)

    Returns:
        dict with status, messages, chunks count
    """
    results = {"status": "success", "messages": [], "chunks": 0}

    try:
        if progress_callback:
            progress_callback(0.1, "Uploading audio to Gemini...")

        uploaded_file = ai.files.upload(file=file_path)

        if progress_callback:
            progress_callback(0.2, "Processing audio file...")

        file = ai.files.get(name=uploaded_file.name)
        while file.state.name == "PROCESSING":
            time.sleep(2)
            file = ai.files.get(name=uploaded_file.name)

        if file.state.name == "FAILED":
            results["status"] = "error"
            results["messages"].append("File processing failed")
            return results

        if progress_callback:
            progress_callback(0.4, f"Transcribing with {model}...")

        transcription_result = ai.models.generate_content(
            model=model,
            contents=[
                file,
                "Transcribe this audio file. Output only the transcription text, nothing else."
            ]
        )

        transcript_text = transcription_result.text or ""
        results["messages"].append(f"Transcription complete ({len(transcript_text):,} characters)")

        ai.files.delete(name=file.name)

        if progress_callback:
            progress_callback(0.6, "Chunking transcript...")

        chunks = chunk_text(transcript_text, CHUNK_SIZE, CHUNK_OVERLAP)
        results["messages"].append(f"Split into {len(chunks)} chunks")

        if progress_callback:
            progress_callback(0.7, "Generating embeddings and storing...")

        stored_count = 0
        for i, chunk in enumerate(chunks):
            embedding = generate_embedding(chunk)

            supabase.from_("talk_chunks").insert({
                "talk_id": talk_id,
                "content": chunk,
                "content_type": "audio_transcript",
                "source_file": file_name,
                "chunk_index": i,
                "embedding": embedding,
            }).execute()

            stored_count += 1
            if progress_callback and i % 5 == 0:
                progress = 0.7 + (0.25 * (i / len(chunks)))
                progress_callback(progress, f"Storing chunk {i+1}/{len(chunks)}...")

        results["chunks"] = stored_count
        results["messages"].append(f"Stored {stored_count} chunks")

        if progress_callback:
            progress_callback(1.0, "Done!")

    except Exception as e:
        results["status"] = "error"
        results["messages"].append(f"Error: {str(e)}")

    return results


def ingest_images(images: list, talk_id: str, progress_callback=None) -> dict:
    """Ingest slide images: OCR, vision describe, embed, and store.

    Args:
        images: List of (PIL.Image, filename) tuples
        talk_id: Talk ID
        progress_callback: Optional callback(progress, message)

    Returns:
        dict with status, messages, slides_processed count
    """
    results = {"status": "success", "messages": [], "slides_processed": 0}

    try:
        for idx, (image, filename) in enumerate(images):
            if progress_callback:
                progress_callback(idx / len(images), f"Processing slide {idx + 1}/{len(images)}...")

            # Extract OCR text
            ocr_text = extract_slide_ocr(image)
            if ocr_text:
                chunks = chunk_text(ocr_text, 1500, 0) if len(ocr_text) > 1500 else [ocr_text]
                for chunk_idx, chunk in enumerate(chunks):
                    embedding = generate_embedding(chunk)
                    supabase.from_("talk_chunks").insert({
                        "talk_id": talk_id,
                        "content": chunk,
                        "content_type": "slide_ocr",
                        "source_file": filename,
                        "slide_number": idx + 1,
                        "chunk_index": chunk_idx,
                        "embedding": embedding,
                    }).execute()

            # Get vision description
            vision_desc = describe_slide_vision(image)
            if vision_desc:
                embedding = generate_embedding(vision_desc)
                supabase.from_("talk_chunks").insert({
                    "talk_id": talk_id,
                    "content": vision_desc,
                    "content_type": "slide_vision",
                    "source_file": filename,
                    "slide_number": idx + 1,
                    "chunk_index": 0,
                    "embedding": embedding,
                }).execute()

            results["slides_processed"] += 1

        results["messages"].append(f"Processed {results['slides_processed']} slides")

        if progress_callback:
            progress_callback(1.0, "Done!")

    except Exception as e:
        results["status"] = "error"
        results["messages"].append(f"Error: {str(e)}")

    return results


def parse_aligned_content(content: str) -> dict:
    """Parse the aligned segment content into sections.

    Args:
        content: Raw aligned segment content string

    Returns:
        dict with slide_text, visual_desc, audio sections
    """
    sections = {
        'slide_text': '',
        'visual_desc': '',
        'audio': ''
    }

    current_section = None
    lines = content.split('\n')

    for line in lines:
        if '### Slide Text' in line:
            current_section = 'slide_text'
        elif '### Visual Description' in line:
            current_section = 'visual_desc'
        elif '### Speaker Audio' in line or '### Speaker Said' in line:
            current_section = 'audio'
        elif current_section and line.strip() and not line.startswith('##'):
            sections[current_section] += line + '\n'

    # Patterns that indicate no audio content
    NO_AUDIO_PATTERNS = [
        '[No matching audio in this time range]',
        '[No timestamp - audio not aligned]',
        '[Quick transition slide - no extended discussion]',
        '[No audio segment assigned]',
    ]

    # Clean up
    for key in sections:
        sections[key] = sections[key].strip()
        if sections[key] == '[No text detected]':
            sections[key] = ''
        # Normalize all no-audio patterns to a single marker
        for pattern in NO_AUDIO_PATTERNS:
            if pattern in sections[key]:
                sections[key] = '_No matching audio_'
                break

    return sections
