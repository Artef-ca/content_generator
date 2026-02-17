"""
Video Generation Helpers — shared logic for all video endpoints.

Called from app.py. Keeps endpoints thin.
"""

import logging

from fastapi import HTTPException

from src.config import (
    BUCKET_NAME,
    VIDEO_DEFAULTS,
    CAMERA_ANGLE_MAP,
    CAMERA_MOVEMENT_MAP,
    LENS_EFFECT_MAP,
    VIDEO_STYLE_MAP,
    TEMPORAL_MAP,
    SOUND_AMBIENCE_MAP,
    VIDEO_ALLOWED_ASPECT_RATIOS,
    ALLOWED_DURATIONS,
    ALLOWED_RESOLUTIONS,
    ALLOWED_TTS_LANGUAGES,
)
from src.core.video_prompt_generator import (
    VideoPromptInputs,
    build_video_prompt,
    get_brand_negative_prompt,
)
from src.helpers.video_utils import (
    generate_video,
    generate_video_output_uri,
    synthesize_speech,
    merge_audio_video,
    upload_bytes_to_gcs,
    download_from_gcs,
    ImageInput,
    _upload_image_to_gcs,
)

logger = logging.getLogger("creative_studio")


# ---------------------------------------------------------------------------
# Upload reading
# ---------------------------------------------------------------------------
async def read_upload(upload, field_name: str = "file") -> ImageInput | None:
    """Read a FastAPI UploadFile into ImageInput."""
    has_read = hasattr(upload, "read") and hasattr(upload, "filename")
    fname = getattr(upload, "filename", None)
    if not has_read or not fname:
        return None
    data = await upload.read()
    if len(data) == 0:
        return None
    ct = getattr(upload, "content_type", None) or "image/png"
    logger.info("%s OK: %s (%s, %d bytes)", field_name, fname, ct, len(data))
    return ImageInput(data=data, mime_type=ct, filename=fname)


def clean(val: str | None) -> str:
    """Strip and normalise form values."""
    if val is None:
        return ""
    val = val.strip()
    return "" if val.lower() in ("", "string") else val


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------
def validate_video_params(
    aspect_ratio, duration, resolution, num_videos,
    camera_angle, camera_movement, lens_effect, visual_style, temporal,
):
    errors = []
    if aspect_ratio not in VIDEO_ALLOWED_ASPECT_RATIOS:
        errors.append(f"aspect_ratio: {VIDEO_ALLOWED_ASPECT_RATIOS}")
    if duration not in ALLOWED_DURATIONS:
        errors.append(f"duration_seconds: {ALLOWED_DURATIONS}")
    if resolution not in ALLOWED_RESOLUTIONS:
        errors.append(f"resolution: {ALLOWED_RESOLUTIONS}")
    if not 1 <= num_videos <= 4:
        errors.append("number_of_videos: 1–4")
    if camera_angle and camera_angle not in CAMERA_ANGLE_MAP:
        errors.append(f"camera_angle: {list(CAMERA_ANGLE_MAP.keys())}")
    if camera_movement and camera_movement not in CAMERA_MOVEMENT_MAP:
        errors.append(f"camera_movement: {list(CAMERA_MOVEMENT_MAP.keys())}")
    if lens_effect and lens_effect not in LENS_EFFECT_MAP:
        errors.append(f"lens_effect: {list(LENS_EFFECT_MAP.keys())}")
    if visual_style and visual_style not in VIDEO_STYLE_MAP:
        errors.append(f"visual_style: {list(VIDEO_STYLE_MAP.keys())}")
    if temporal and temporal not in TEMPORAL_MAP:
        errors.append(f"temporal_elements: {list(TEMPORAL_MAP.keys())}")
    if errors:
        raise HTTPException(400, "; ".join(errors))


# ---------------------------------------------------------------------------
# Prompt
# ---------------------------------------------------------------------------
def build_veo_prompt(
    subject, action, scene_context, prompt, dialogue,
    camera_angle, camera_movement, lens_effect, visual_style,
    temporal_elements, sound_ambience, negative_prompt,
    inject_dialogue=False,
) -> str:
    inputs = VideoPromptInputs(
        prompt=prompt or "",
        subject=subject, action=action, scene_context=scene_context,
        dialogue=dialogue if inject_dialogue else "",
        camera_angle=camera_angle or VIDEO_DEFAULTS["camera_angle"],
        camera_movement=camera_movement or VIDEO_DEFAULTS["camera_movement"],
        lens_effect=lens_effect or VIDEO_DEFAULTS["lens_effect"],
        visual_style=visual_style or VIDEO_DEFAULTS["visual_style"],
        temporal_elements=temporal_elements or VIDEO_DEFAULTS["temporal_elements"],
        sound_ambience=sound_ambience or VIDEO_DEFAULTS["sound_ambience"],
        negative_prompt=negative_prompt,
    )
    return build_video_prompt(inputs)


def effective_negative(user_neg: str) -> str | None:
    parts = [p for p in [get_brand_negative_prompt(), user_neg] if p]
    return ", ".join(parts) if parts else None


# ---------------------------------------------------------------------------
# End frame
# ---------------------------------------------------------------------------
async def prepare_end_frame(end_frame_image) -> tuple[str | None, str | None, dict]:
    img = await read_upload(end_frame_image, "end_frame_image")
    if not img:
        return None, None, {"applied": False}
    try:
        uri = _upload_image_to_gcs(img, "end_frame")
        logger.info("End frame uploaded: %s", uri)
        return uri, img.mime_type, {"applied": True, "filename": img.filename}
    except Exception as e:
        logger.exception("End frame upload failed")
        return None, None, {"applied": False, "error": str(e)}


# ---------------------------------------------------------------------------
# Audio
# ---------------------------------------------------------------------------
async def handle_audio(audio_mode, tts_text, tts_language, audio_file, dialogue, video_uri):
    info: dict = {"mode": audio_mode}
    if audio_mode == "none":
        return video_uri, info
    try:
        audio_bytes, audio_mime = None, "audio/mpeg"
        if audio_mode == "tts":
            script = tts_text or dialogue
            if not script:
                info["detail"] = "No text provided"
                return video_uri, info
            if tts_language not in ALLOWED_TTS_LANGUAGES:
                raise HTTPException(400, f"tts_language: {ALLOWED_TTS_LANGUAGES}")
            r = synthesize_speech(text=script, language_code=tts_language)
            audio_bytes, audio_mime = r.audio_bytes, r.mime_type
            info.update(tts_language=tts_language, tts_chars=len(script))
        elif audio_mode == "upload":
            u = await read_upload(audio_file, "audio_file")
            if not u:
                info["detail"] = "No audio file"
                return video_uri, info
            audio_bytes, audio_mime = u.data, u.mime_type
            info["uploaded_file"] = u.filename
        if audio_bytes:
            vb = download_from_gcs(video_uri)
            merged = merge_audio_video(vb, audio_bytes, audio_mime)
            blob = video_uri.replace("gs://", "").split("/", 1)[1].rstrip("/")
            blob = (blob[:-4] + "_audio.mp4") if blob.endswith(".mp4") else f"{blob}/merged.mp4"
            r = upload_bytes_to_gcs(merged, blob, content_type="video/mp4")
            video_uri = r.gcs_uri
            info["merged"] = True
    except HTTPException:
        raise
    except Exception as e:
        logger.exception("Audio failed")
        info["error"] = str(e)
    return video_uri, info


# ---------------------------------------------------------------------------
# Response builder
# ---------------------------------------------------------------------------
def video_response(
    uri, model, mode, prompt, aspect, res, dur, n, seed,
    audio_info, end_frame_info, ref_fnames, op,
):
    return {
        "status": "success",
        "gcs_uri": uri,
        "public_url": uri.replace(f"gs://{BUCKET_NAME}/", f"https://storage.googleapis.com/{BUCKET_NAME}/"),
        "model": model, "generation_mode": mode, "prompt_used": prompt,
        "aspect_ratio": aspect, "resolution": res, "duration_seconds": dur,
        "number_of_videos": n, "seed": seed,
        "audio": audio_info, "end_frame": end_frame_info,
        "reference_images": ref_fnames, "operation": op,
    }
