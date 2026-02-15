"""
Mobily Creative Studio — Video Generation API.

A branded video generator powered by Veo (Google Vertex AI).
Builds structured prompts from user inputs following cinematic best practices,
with optional TTS audio overlay and source image animation.

Run:  uvicorn video_main:app --host 0.0.0.0 --port 8081 --reload
Docs: http://localhost:8081/docs
"""

import logging
from typing import Optional

from fastapi import FastAPI, File, UploadFile, Form, HTTPException
from fastapi.responses import JSONResponse

from src.config import (
    VIDEO_DEFAULTS,
    DEFAULT_BRAND,
    VIDEO_MODEL_ID,
    VIDEO_FAST_MODEL_ID,
    VIDEO_PREVIEW_MODEL_ID,
    VIDEO_ALLOWED_ASPECT_RATIOS,
    ALLOWED_DURATIONS,
    ALLOWED_AUDIO_MODES,
    ALLOWED_TTS_LANGUAGES,
    CAMERA_ANGLE_MAP,
    CAMERA_MOVEMENT_MAP,
    LENS_EFFECT_MAP,
    VIDEO_STYLE_MAP,
    TEMPORAL_MAP,
    SOUND_AMBIENCE_MAP,
    PROJECT_ID,
    BUCKET_NAME,
    PORT,
)
from src.core.video_prompt_generator import VideoPromptInputs, build_video_prompt
from src.helpers.video_utils import (
    generate_video,
    generate_video_output_uri,
    synthesize_speech,
    merge_audio_video,
    upload_bytes_to_gcs,
    download_from_gcs,
)

logger = logging.getLogger("creative_studio")

# ---------------------------------------------------------------------------
# FastAPI App
# ---------------------------------------------------------------------------
app = FastAPI(
    title="Mobily Creative Studio — Video",
    description=(
        "Generate branded marketing videos for **any campaign or topic** "
        "using Veo on Google Vertex AI.\n\n"
        "Supports text-to-video, image-to-video, TTS audio overlay, "
        "and custom audio upload. The API structures user inputs into "
        "cinematic prompts with camera, lens, style, and pacing controls."
    ),
    version="1.0.0",
)


# ---------------------------------------------------------------------------
# Discovery — list all dropdown options
# ---------------------------------------------------------------------------
@app.get("/video-options", summary="List all available video parameter choices")
async def list_video_options():
    """Return every valid value for dropdowns. Use for frontend integration."""
    return {
        "camera_angles": list(CAMERA_ANGLE_MAP.keys()),
        "camera_movements": list(CAMERA_MOVEMENT_MAP.keys()),
        "lens_effects": list(LENS_EFFECT_MAP.keys()),
        "visual_styles": list(VIDEO_STYLE_MAP.keys()),
        "temporal_elements": list(TEMPORAL_MAP.keys()),
        "sound_ambiences": list(SOUND_AMBIENCE_MAP.keys()),
        "aspect_ratios": VIDEO_ALLOWED_ASPECT_RATIOS,
        "durations": ALLOWED_DURATIONS,
        "audio_modes": ALLOWED_AUDIO_MODES,
        "tts_languages": ALLOWED_TTS_LANGUAGES,
        "defaults": VIDEO_DEFAULTS,
        "default_brand": DEFAULT_BRAND,
        "models": {
            "standard": VIDEO_MODEL_ID,
            "fast": VIDEO_FAST_MODEL_ID,
            "preview": VIDEO_PREVIEW_MODEL_ID,
        },
    }


# ---------------------------------------------------------------------------
# Health Check
# ---------------------------------------------------------------------------
@app.get("/health")
async def health():
    return {"status": "ok", "model": VIDEO_MODEL_ID, "project": PROJECT_ID}


# ---------------------------------------------------------------------------
# Video Generation
# ---------------------------------------------------------------------------
@app.post("/generate-video", summary="Generate a branded marketing video")
async def generate_video_endpoint(
    # ── CORE ─────────────────────────────────────────────────────────────
    prompt: str = Form(
        ...,
        description=(
            "Description of what happens in the video. This is the CORE input. "
            "Describe the visual scene, actions, and mood. "
            "Example: 'A family gathered around a festive dinner table, "
            "laughing and sharing food, warm golden light'"
        ),
    ),
    source_image: Optional[UploadFile] = File(
        None,
        description=(
            "Image to animate as the first frame. If provided, Veo will animate "
            "from this image. Skip to generate entirely from text. "
            "Recommended: 720p+ resolution, 16:9 or 9:16 aspect ratio."
        ),
    ),
    # ── BRANDING ─────────────────────────────────────────────────────────
    company_website_url: str = Form(
        None,
        description=(
            "Brand website URL to display as subtle watermark in the video. "
            "Example: 'mobily.com.sa'"
        ),
    ),
    use_default_brand: bool = Form(
        False,
        description=(
            "If true, uses the default Mobily brand info from config "
            f"(website: {DEFAULT_BRAND['website_url']}). "
            "Overridden if company_website_url is provided."
        ),
    ),
    # ── SUBJECT & SCENE ──────────────────────────────────────────────────
    subject: str = Form(
        None,
        description=(
            "The main subject of the video. Merged with prompt. "
            "Example: 'a CEO in a tailored suit', 'a product rotating on a pedestal'"
        ),
    ),
    action: str = Form(
        None,
        description=(
            "What the subject is doing. Merged with subject. "
            "Example: 'saying congratulations for Ramadan', 'explaining 5G benefits'"
        ),
    ),
    scene_context: str = Form(
        None,
        description=(
            "Where the scene takes place. "
            "Example: 'in a modern glass office', 'outdoor garden with fairy lights'"
        ),
    ),
    # ── CAMERA ───────────────────────────────────────────────────────────
    camera_angle: str = Form(
        VIDEO_DEFAULTS["camera_angle"],
        description=(
            "Camera angle preset. "
            "Options: " + ", ".join(CAMERA_ANGLE_MAP.keys())
        ),
    ),
    camera_movement: str = Form(
        VIDEO_DEFAULTS["camera_movement"],
        description=(
            "Camera movement preset. "
            "Options: " + ", ".join(CAMERA_MOVEMENT_MAP.keys())
        ),
    ),
    # ── LENS ─────────────────────────────────────────────────────────────
    lens_effect: str = Form(
        VIDEO_DEFAULTS["lens_effect"],
        description=(
            "Lens / depth-of-field effect. "
            "Options: " + ", ".join(LENS_EFFECT_MAP.keys())
        ),
    ),
    # ── STYLE & PACING ──────────────────────────────────────────────────
    visual_style: str = Form(
        VIDEO_DEFAULTS["visual_style"],
        description=(
            "Visual style preset. "
            "Options: " + ", ".join(VIDEO_STYLE_MAP.keys())
        ),
    ),
    temporal_elements: str = Form(
        VIDEO_DEFAULTS["temporal_elements"],
        description=(
            "Pacing / temporal effect. "
            "Options: " + ", ".join(TEMPORAL_MAP.keys())
        ),
    ),
    sound_ambience: str = Form(
        VIDEO_DEFAULTS["sound_ambience"],
        description=(
            "Background sound cue (embedded in prompt for Veo 3 native audio). "
            "Options: " + ", ".join(SOUND_AMBIENCE_MAP.keys())
        ),
    ),
    dialogue: str = Form(
        None,
        description=(
            "Spoken text for lip-sync. Helps Veo generate matching lip movements. "
            "Also auto-feeds TTS if tts_text is empty. "
            "Example: 'رمضان كريم من موبايلي', 'Happy National Day!'"
        ),
    ),
    # ── VIDEO OUTPUT ─────────────────────────────────────────────────────
    duration_seconds: int = Form(
        VIDEO_DEFAULTS["duration_seconds"],
        description="Video duration in seconds: 5, 6, 7, or 8.",
    ),
    aspect_ratio: str = Form(
        VIDEO_DEFAULTS["aspect_ratio"],
        description="Output aspect ratio: 16:9 (landscape) or 9:16 (portrait/stories).",
    ),
    # ── AUDIO ────────────────────────────────────────────────────────────
    audio_mode: str = Form(
        VIDEO_DEFAULTS["audio_mode"],
        description=(
            "Audio strategy: "
            "'none' = silent / Veo native audio only, "
            "'tts' = generate speech via Google Cloud TTS, "
            "'upload' = use the audio_file you provide."
        ),
    ),
    tts_text: str = Form(
        None,
        description=(
            "Text-to-speech script. If empty and audio_mode=tts, "
            "falls back to the dialogue field."
        ),
    ),
    tts_language: str = Form(
        VIDEO_DEFAULTS["tts_language"],
        description=(
            "TTS language code: " + ", ".join(ALLOWED_TTS_LANGUAGES)
        ),
    ),
    tts_voice_name: str = Form(
        None,
        description=(
            "Specific GCP voice ID (e.g. 'ar-XA-Wavenet-A'). "
            "If empty, auto-selects best voice for the language."
        ),
    ),
    audio_file: Optional[UploadFile] = File(
        None,
        description=(
            "Pre-recorded audio file when audio_mode=upload. "
            "Accepts MP3 or WAV. Merged onto the generated video."
        ),
    ),
    # ── MODEL SELECTION ──────────────────────────────────────────────────
    veo_variant: str = Form(
        "standard",
        description=(
            "Model variant: "
            "'standard' = best quality, "
            "'fast' = faster + cheaper, "
            "'preview' = latest features (Veo 3.1)"
        ),
    ),
    # ── CONSTRAINTS ──────────────────────────────────────────────────────
    negative_prompt: str = Form(
        None,
        description=(
            "What to avoid in the video. Sent as a separate Veo parameter. "
            "Example: 'cartoon, low quality, blurry, watermark, text overlay'"
        ),
    ),
):
    """
    Generate a branded marketing video for **any** campaign or topic.

    **Workflow:**
    1. Assembles your inputs into a cinematic prompt
    2. Sends to Veo (long-running ~1-5 min)
    3. Optionally generates TTS or merges uploaded audio
    4. Returns GCS URI of the final video

    **Prompt hierarchy** (applied automatically):
    `Style → Camera → Lens → Movement → Subject+Action → Scene → Pacing → Audio → Brand`
    """

    # ── Sanitise Swagger placeholders ─────────────────────────────────────
    # Swagger UI sends the literal "string" for unfilled text fields.
    def _clean(val: str | None) -> str | None:
        if val is None or val.strip() == "" or val.strip().lower() == "string":
            return None
        return val.strip()

    subject = _clean(subject)
    action = _clean(action)
    scene_context = _clean(scene_context)
    company_website_url = _clean(company_website_url)
    dialogue = _clean(dialogue)
    tts_text = _clean(tts_text)
    tts_voice_name = _clean(tts_voice_name)
    negative_prompt = _clean(negative_prompt)

    # ── Validation ────────────────────────────────────────────────────────
    if aspect_ratio not in VIDEO_ALLOWED_ASPECT_RATIOS:
        raise HTTPException(400, f"Invalid aspect_ratio. Allowed: {VIDEO_ALLOWED_ASPECT_RATIOS}")
    if duration_seconds not in ALLOWED_DURATIONS:
        raise HTTPException(400, f"Invalid duration. Allowed: {ALLOWED_DURATIONS}")
    if audio_mode not in ALLOWED_AUDIO_MODES:
        raise HTTPException(400, f"Invalid audio_mode. Allowed: {ALLOWED_AUDIO_MODES}")
    if camera_angle not in CAMERA_ANGLE_MAP:
        raise HTTPException(400, f"Invalid camera_angle. Allowed: {list(CAMERA_ANGLE_MAP.keys())}")
    if camera_movement not in CAMERA_MOVEMENT_MAP:
        raise HTTPException(400, f"Invalid camera_movement. Allowed: {list(CAMERA_MOVEMENT_MAP.keys())}")
    if lens_effect not in LENS_EFFECT_MAP:
        raise HTTPException(400, f"Invalid lens_effect. Allowed: {list(LENS_EFFECT_MAP.keys())}")
    if visual_style not in VIDEO_STYLE_MAP:
        raise HTTPException(400, f"Invalid visual_style. Allowed: {list(VIDEO_STYLE_MAP.keys())}")
    if temporal_elements not in TEMPORAL_MAP:
        raise HTTPException(400, f"Invalid temporal_elements. Allowed: {list(TEMPORAL_MAP.keys())}")

    # ── Read source image (optional) ─────────────────────────────────────
    source_image_bytes = None
    source_image_mime = "image/png"
    if source_image is not None and isinstance(source_image, UploadFile) and source_image.filename:
        source_image_bytes = await source_image.read()
        if len(source_image_bytes) > 0:
            source_image_mime = source_image.content_type or "image/png"
            logger.info(
                "Source image: %s (%s, %d bytes)",
                source_image.filename, source_image_mime, len(source_image_bytes),
            )
        else:
            source_image_bytes = None

    # ── Build prompt ─────────────────────────────────────────────────────
    prompt_inputs = VideoPromptInputs(
        prompt=prompt,
        subject=subject or "",
        action=action or "",
        scene_context=scene_context or "",
        camera_angle=camera_angle,
        camera_movement=camera_movement,
        lens_effect=lens_effect,
        visual_style=visual_style,
        temporal_elements=temporal_elements,
        sound_ambience=sound_ambience,
        dialogue=dialogue or "",
        company_website_url=company_website_url or "",
        use_default_brand=use_default_brand,
        negative_prompt=negative_prompt or "",
    )
    final_prompt = build_video_prompt(prompt_inputs)
    logger.info("Final video prompt: %s", final_prompt)

    # ── Select model ─────────────────────────────────────────────────────
    model_map = {
        "standard": VIDEO_MODEL_ID,
        "fast": VIDEO_FAST_MODEL_ID,
        "preview": VIDEO_PREVIEW_MODEL_ID,
    }
    selected_model = model_map.get(veo_variant, VIDEO_MODEL_ID)

    # ── Determine if Veo should generate native audio ────────────────────
    # If user wants TTS or upload, we'll overlay audio separately,
    # so disable Veo's native audio to avoid double-audio.
    generate_native_audio = audio_mode == "none"

    # ── Generate GCS output path ─────────────────────────────────────────
    subject_hint = subject or prompt[:60]
    output_uri = generate_video_output_uri(subject_hint)

    # ── Generate video ───────────────────────────────────────────────────
    try:
        video_result = generate_video(
            prompt=final_prompt,
            aspect_ratio=aspect_ratio,
            duration_seconds=duration_seconds,
            negative_prompt=negative_prompt or None,
            generate_audio=generate_native_audio,
            source_image_bytes=source_image_bytes,
            source_image_mime=source_image_mime,
            output_gcs_uri=output_uri,
            model_id=selected_model,
        )
    except RuntimeError as e:
        raise HTTPException(500, str(e))
    except Exception as e:
        logger.exception("Veo API call failed")
        raise HTTPException(502, f"Video generation failed: {e}")

    # ── Audio overlay (TTS or uploaded file) ─────────────────────────────
    final_gcs_uri = video_result.gcs_uri
    audio_info = {"mode": audio_mode, "detail": None}

    if audio_mode in ("tts", "upload"):
        try:
            # Get the generated video bytes from GCS
            video_bytes = download_from_gcs(video_result.gcs_uri)

            audio_bytes = None
            audio_mime = "audio/mpeg"

            if audio_mode == "tts":
                speech_text = tts_text or dialogue or ""
                if not speech_text:
                    raise HTTPException(400, "audio_mode=tts but no tts_text or dialogue provided.")

                tts_result = synthesize_speech(
                    text=speech_text,
                    language_code=tts_language,
                    voice_name=tts_voice_name or None,
                )
                audio_bytes = tts_result.audio_bytes
                audio_mime = tts_result.mime_type
                audio_info["detail"] = f"TTS: {tts_language}, {len(speech_text)} chars"

            elif audio_mode == "upload":
                if audio_file is None or not isinstance(audio_file, UploadFile) or not audio_file.filename:
                    raise HTTPException(400, "audio_mode=upload but no audio_file provided.")
                audio_bytes = await audio_file.read()
                audio_mime = audio_file.content_type or "audio/mpeg"
                audio_info["detail"] = f"Upload: {audio_file.filename}"

            if audio_bytes:
                merged = merge_audio_video(video_bytes, audio_bytes, audio_mime)
                # Upload merged video back to GCS
                merged_path = video_result.gcs_uri.replace("gs://", "").split("/", 1)[1]
                merged_path = merged_path.rstrip("/") + "_with_audio.mp4"
                merged_result = upload_bytes_to_gcs(
                    merged, merged_path, content_type="video/mp4"
                )
                final_gcs_uri = merged_result.gcs_uri
                logger.info("Audio merged, uploaded to %s", final_gcs_uri)

        except HTTPException:
            raise
        except Exception as e:
            logger.exception("Audio processing failed")
            audio_info["detail"] = f"Audio merge failed: {e}. Video available without audio."

    # ── Build public URL ─────────────────────────────────────────────────
    public_url = final_gcs_uri.replace(
        f"gs://{BUCKET_NAME}/",
        f"https://storage.googleapis.com/{BUCKET_NAME}/",
    )

    # ── Respond ──────────────────────────────────────────────────────────
    return JSONResponse(
        status_code=200,
        content={
            "status": "success",
            "gcs_uri": final_gcs_uri,
            "public_url": public_url,
            "model": selected_model,
            "prompt_used": final_prompt,
            "aspect_ratio": aspect_ratio,
            "duration_seconds": duration_seconds,
            "audio": audio_info,
            "has_source_image": source_image_bytes is not None,
            "operation": video_result.operation_name,
        },
    )


# ---------------------------------------------------------------------------
# Entrypoint
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    import uvicorn

    uvicorn.run(
        "video_main:app",
        host="0.0.0.0",
        port=PORT + 1,  # 8081 by default to not clash with banner API
        reload=True,
    )
