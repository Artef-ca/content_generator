"""
Mobily Creative Studio — Combined API (v3.0)
Run: python app.py
"""

import logging
from contextlib import asynccontextmanager
from typing import Optional

from fastapi import FastAPI, Form, File, UploadFile, HTTPException
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware

from src.config import (
    PORT, BUCKET_NAME, MODEL_ID, PROJECT_ID, DEFAULTS,
    ALLOWED_ASPECT_RATIOS, ALLOWED_IMAGE_SIZES,
    VISUAL_STYLE_MAP, CAMERA_ANGLE_IMG_MAP, FRAMING_MAP,
    LIGHTING_MAP, TONE_MAP, COLOR_GRADING_MAP,
    VIDEO_MODEL_ID, VIDEO_FAST_MODEL_ID, VIDEO_PREVIEW_MODEL_ID,
    VIDEO_FAST_PREVIEW_MODEL_ID, VIDEO_DEFAULTS, BRAND_CONFIG,
    CAMERA_ANGLE_MAP, CAMERA_MOVEMENT_MAP, LENS_EFFECT_MAP,
    VIDEO_STYLE_MAP, TEMPORAL_MAP, SOUND_AMBIENCE_MAP,
    VIDEO_ALLOWED_ASPECT_RATIOS, ALLOWED_DURATIONS, ALLOWED_RESOLUTIONS,
    ALLOWED_AUDIO_MODES, ALLOWED_TTS_LANGUAGES, logger,
)
from src.helpers.utils import POSITION_MAP
from src.helpers.image_helpers import (
    validate_image_params, read_logo, generate_and_process_image,
)
from src.helpers.video_helpers import (
    read_upload, clean, validate_video_params, build_veo_prompt,
    effective_negative, prepare_end_frame, handle_audio, video_response,
)
from src.helpers.video_utils import (
    generate_video, generate_video_output_uri, ImageInput, _upload_image_to_gcs,
)
from src.schemas import VideoGenerationResponse, HealthResponse

VIDEO_MODEL_MAP = {
    "standard": VIDEO_MODEL_ID, "fast": VIDEO_FAST_MODEL_ID,
    "preview": VIDEO_PREVIEW_MODEL_ID, "fast_preview": VIDEO_FAST_PREVIEW_MODEL_ID,
}


# ═══════════════════════════════════════════════════════════════════════════
# App
# ═══════════════════════════════════════════════════════════════════════════

@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("Creative Studio v3.0 — image + video endpoints ready")
    yield

app = FastAPI(
    title="Mobily Creative Studio", version="3.0.0",
    description="Image + Video generation. All prompts enriched with brand context.",
    lifespan=lifespan,
)
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])


# ═══════════════════════════════════════════════════════════════════════════
#  IMAGE
# ═══════════════════════════════════════════════════════════════════════════

@app.post("/generate-image", tags=["Image Generation"], summary="Generate branded marketing image")
async def generate_image_endpoint(
    subject: str = Form(..., description="Main subject."),
    action: str = Form("", description="What the subject is doing."),
    setting: str = Form("", description="Where & when."),
    items_in_scene: str = Form("", description="Additional objects."),
    visual_style: str = Form("", description="Art direction."),
    camera_angle: str = Form("", description="Camera perspective."),
    framing: str = Form("", description="Shot type / composition."),
    lighting: str = Form("", description="Lighting preset."),
    tone: str = Form("", description="Emotional mood."),
    color_grading: str = Form("", description="Color palette."),
    custom_prompt: str = Form("", description="Free-text fine-tuning."),
    logo_file: UploadFile | None = File(default=None, description="Brand logo PNG. Upload file here."),
    logo_position: str = Form("bottom_right", description="Logo placement (9 positions)."),
    logo_scale: float = Form(0.15, description="Logo width fraction (0.05–0.5)."),
    logo_opacity: float = Form(1.0, description="Logo opacity (0.0–1.0)."),
    campaign_text: str = Form("", description="Text to render on image."),
    text_style: str = Form("", description="Typography style."),
    website_url: str = Form("", description="URL to display."),
    negative_prompt: str = Form("", description="What to avoid."),
    aspect_ratio: str = Form(DEFAULTS["aspect_ratio"]),
    image_size: str = Form(DEFAULTS["image_size"]),
):
    # Validate
    errors = validate_image_params(
        aspect_ratio, image_size, visual_style, camera_angle,
        framing, lighting, tone, color_grading, logo_position,
    )
    if errors:
        raise HTTPException(400, "; ".join(errors))

    # Read logo
    logger.info("Logo file param: %s (type=%s)", logo_file, type(logo_file).__name__)
    has_logo, logo_bytes = await read_logo(logo_file)

    # Generate → overlay → upload (all in helper)
    try:
        result = await generate_and_process_image(
            subject=subject, action=action, setting=setting,
            items_in_scene=items_in_scene, visual_style=visual_style,
            camera_angle=camera_angle, framing=framing, lighting=lighting,
            tone=tone, color_grading=color_grading, custom_prompt=custom_prompt,
            has_logo=has_logo, logo_bytes=logo_bytes,
            logo_position=logo_position, logo_scale=logo_scale, logo_opacity=logo_opacity,
            campaign_text=campaign_text, text_style=text_style, website_url=website_url,
            negative_prompt=negative_prompt, aspect_ratio=aspect_ratio, image_size=image_size,
        )
    except RuntimeError as e:
        raise HTTPException(500, str(e))
    except Exception as e:
        logger.exception("Image generation failed")
        raise HTTPException(502, f"Image generation failed: {e}")

    return JSONResponse(content={
        "status": "success",
        "gcs_uri": result.gcs_uri,
        "public_url": result.public_url,
        "model": MODEL_ID,
        "prompt_used": result.prompt_used,
        "aspect_ratio": aspect_ratio,
        "image_size": image_size,
        "logo": result.logo_info,
        "model_commentary": result.model_commentary,
    })


# ═══════════════════════════════════════════════════════════════════════════
#  VIDEO — Text-to-Video
# ═══════════════════════════════════════════════════════════════════════════

@app.post("/generate-video/text", tags=["Video Generation"],
          response_model=VideoGenerationResponse, summary="Text-to-Video")
async def text_to_video(
    subject: str = Form(...), action: str = Form(""),
    scene_context: str = Form(""),
    camera_angle: str = Form(""), camera_movement: str = Form(""),
    lens_effect: str = Form(""), visual_style: str = Form(""),
    temporal_elements: str = Form(""), sound_ambience: str = Form(""),
    dialogue: str = Form("", description="Voiceover — TTS only."),
    prompt: str = Form(""), negative_prompt: str = Form(""),
    audio_mode: str = Form("none"), tts_text: str = Form(""),
    tts_language: str = Form(VIDEO_DEFAULTS["tts_language"]),
    audio_file: UploadFile | str | None = File(None),
    duration_seconds: int = Form(VIDEO_DEFAULTS["duration_seconds"]),
    aspect_ratio: str = Form(VIDEO_DEFAULTS["aspect_ratio"]),
    resolution: str = Form(VIDEO_DEFAULTS["resolution"]),
    number_of_videos: int = Form(1), seed: int | None = Form(None),
    veo_variant: str = Form("standard"),
):
    subject, action, scene_context = clean(subject), clean(action), clean(scene_context)
    dialogue, negative_prompt = clean(dialogue), clean(negative_prompt)
    tts_text, prompt = clean(tts_text), clean(prompt)

    if not subject and not prompt:
        raise HTTPException(400, "subject or prompt required.")
    validate_video_params(aspect_ratio, duration_seconds, resolution, number_of_videos,
                          camera_angle, camera_movement, lens_effect, visual_style, temporal_elements)
    if veo_variant not in VIDEO_MODEL_MAP:
        raise HTTPException(400, f"veo_variant: {list(VIDEO_MODEL_MAP.keys())}")

    fp = build_veo_prompt(subject, action, scene_context, prompt, dialogue,
                          camera_angle, camera_movement, lens_effect, visual_style,
                          temporal_elements, sound_ambience, negative_prompt, inject_dialogue=False)

    if audio_mode == "none" and dialogue:
        audio_mode = "tts"

    model = VIDEO_MODEL_MAP[veo_variant]
    result = generate_video(
        prompt=fp, aspect_ratio=aspect_ratio, duration_seconds=duration_seconds,
        negative_prompt=effective_negative(negative_prompt), generate_audio=True,
        resolution=resolution, number_of_videos=number_of_videos, seed=seed,
        generation_mode="text_to_video",
        output_gcs_uri=generate_video_output_uri(subject or prompt[:50]), model_id=model,
    )
    uri, audio_info = await handle_audio(audio_mode, tts_text, tts_language, audio_file, dialogue, result.gcs_uri)
    return video_response(uri, model, "text_to_video", fp, aspect_ratio, resolution,
                          duration_seconds, number_of_videos, seed, audio_info,
                          {"applied": False}, [], result.operation_name)


# ═══════════════════════════════════════════════════════════════════════════
#  VIDEO — Image-to-Video
# ═══════════════════════════════════════════════════════════════════════════

@app.post("/generate-video/image", tags=["Video Generation"],
          response_model=VideoGenerationResponse, summary="Image-to-Video (+ end frame)")
async def image_to_video(
    source_image: UploadFile = File(..., description="First frame."),
    action: str = Form(""), scene_context: str = Form(""),
    camera_angle: str = Form(""), camera_movement: str = Form(""),
    lens_effect: str = Form(""), visual_style: str = Form(""),
    temporal_elements: str = Form(""), sound_ambience: str = Form(""),
    dialogue: str = Form("", description="Lip-sync text."),
    end_frame_image: UploadFile | str | None = File(None, description="End frame (as-is)."),
    prompt: str = Form(""), negative_prompt: str = Form(""),
    audio_mode: str = Form("none"), tts_text: str = Form(""),
    tts_language: str = Form(VIDEO_DEFAULTS["tts_language"]),
    audio_file: UploadFile | str | None = File(None),
    duration_seconds: int = Form(VIDEO_DEFAULTS["duration_seconds"]),
    aspect_ratio: str = Form(VIDEO_DEFAULTS["aspect_ratio"]),
    resolution: str = Form(VIDEO_DEFAULTS["resolution"]),
    number_of_videos: int = Form(1), seed: int | None = Form(None),
    veo_variant: str = Form("standard"),
):
    img = await read_upload(source_image, "source_image")
    if not img:
        raise HTTPException(400, "source_image required.")

    action, scene_context = clean(action), clean(scene_context)
    dialogue, negative_prompt = clean(dialogue), clean(negative_prompt)
    tts_text, prompt = clean(tts_text), clean(prompt)

    validate_video_params(aspect_ratio, duration_seconds, resolution, number_of_videos,
                          camera_angle, camera_movement, lens_effect, visual_style, temporal_elements)
    if veo_variant not in VIDEO_MODEL_MAP:
        raise HTTPException(400, f"veo_variant: {list(VIDEO_MODEL_MAP.keys())}")

    fp = build_veo_prompt("", action, scene_context, prompt, dialogue,
                          camera_angle, camera_movement, lens_effect, visual_style,
                          temporal_elements, sound_ambience, negative_prompt, inject_dialogue=True)

    image_uri = _upload_image_to_gcs(img, "source")
    ef_uri, ef_mime, ef_info = await prepare_end_frame(end_frame_image)

    if ef_uri and veo_variant not in {"preview", "fast_preview"}:
        old = veo_variant
        veo_variant = "preview"
        ef_info["auto_upgraded_model"] = f"{old} → {veo_variant}"

    model = VIDEO_MODEL_MAP[veo_variant]
    result = generate_video(
        prompt=fp, aspect_ratio=aspect_ratio, duration_seconds=duration_seconds,
        negative_prompt=effective_negative(negative_prompt), generate_audio=True,
        resolution=resolution, number_of_videos=number_of_videos, seed=seed,
        generation_mode="image_to_video", source_image_gcs_uri=image_uri,
        source_image_mime=img.mime_type, last_frame_gcs_uri=ef_uri, last_frame_mime=ef_mime,
        output_gcs_uri=generate_video_output_uri(action or prompt[:50] or "animated"),
        model_id=model,
    )
    uri, audio_info = await handle_audio(audio_mode, tts_text, tts_language, audio_file, dialogue, result.gcs_uri)
    return video_response(uri, model, "image_to_video", fp, aspect_ratio, resolution,
                          duration_seconds, number_of_videos, seed, audio_info,
                          ef_info, [img.filename], result.operation_name)


# ═══════════════════════════════════════════════════════════════════════════
#  VIDEO — Reference-to-Video
# ═══════════════════════════════════════════════════════════════════════════

@app.post("/generate-video/reference", tags=["Video Generation"],
          response_model=VideoGenerationResponse, summary="Reference-to-Video (Veo 3.1)")
async def reference_to_video(
    source_image: UploadFile = File(...),
    ref_image_2: UploadFile | str | None = File(None),
    ref_image_3: UploadFile | str | None = File(None),
    subject: str = Form(...), action: str = Form(""),
    scene_context: str = Form(""), dialogue: str = Form(""),
    camera_angle: str = Form(""), camera_movement: str = Form(""),
    lens_effect: str = Form(""), visual_style: str = Form(""),
    temporal_elements: str = Form(""), sound_ambience: str = Form(""),
    prompt: str = Form(""), negative_prompt: str = Form(""),
    audio_mode: str = Form("none"), tts_text: str = Form(""),
    tts_language: str = Form(VIDEO_DEFAULTS["tts_language"]),
    audio_file: UploadFile | str | None = File(None),
    aspect_ratio: str = Form(VIDEO_DEFAULTS["aspect_ratio"]),
    resolution: str = Form(VIDEO_DEFAULTS["resolution"]),
    number_of_videos: int = Form(1), seed: int | None = Form(None),
    veo_variant: str = Form("preview"),
):
    dur = 8
    img1 = await read_upload(source_image, "source_image")
    if not img1:
        raise HTTPException(400, "source_image required.")
    img2 = await read_upload(ref_image_2, "ref_image_2")
    img3 = await read_upload(ref_image_3, "ref_image_3")
    imgs = [i for i in [img1, img2, img3] if i]

    subject, action, scene_context = clean(subject), clean(action), clean(scene_context)
    dialogue, negative_prompt, prompt = clean(dialogue), clean(negative_prompt), clean(prompt)

    validate_video_params(aspect_ratio, dur, resolution, number_of_videos,
                          camera_angle, camera_movement, lens_effect, visual_style, temporal_elements)
    if veo_variant not in ["preview", "fast_preview"]:
        raise HTTPException(400, "Reference: preview or fast_preview only.")

    fp = build_veo_prompt(subject, action, scene_context, prompt, dialogue,
                          camera_angle, camera_movement, lens_effect, visual_style,
                          temporal_elements, sound_ambience, negative_prompt, inject_dialogue=True)

    model = VIDEO_MODEL_MAP[veo_variant]
    result = generate_video(
        prompt=fp, aspect_ratio=aspect_ratio, duration_seconds=dur,
        negative_prompt=effective_negative(negative_prompt), generate_audio=True,
        resolution=resolution, number_of_videos=number_of_videos, seed=seed,
        generation_mode="reference_to_video", reference_images=imgs,
        output_gcs_uri=generate_video_output_uri(subject or "reference"), model_id=model,
    )
    uri, audio_info = await handle_audio(audio_mode, tts_text, tts_language, audio_file, dialogue, result.gcs_uri)
    return video_response(uri, model, "reference_to_video", fp, aspect_ratio, resolution,
                          dur, number_of_videos, seed, audio_info,
                          {"applied": False}, [i.filename for i in imgs], result.operation_name)


# ═══════════════════════════════════════════════════════════════════════════
#  System
# ═══════════════════════════════════════════════════════════════════════════

@app.get("/health", tags=["System"])
async def health():
    return {"status": "ok", "version": "3.0.0", "image_model": MODEL_ID, "video_models": VIDEO_MODEL_MAP}

@app.get("/options", tags=["System"], summary="Image dropdown values")
async def image_options():
    return {
        "visual_styles": list(VISUAL_STYLE_MAP.keys()),
        "camera_angles": list(CAMERA_ANGLE_IMG_MAP.keys()),
        "framings": list(FRAMING_MAP.keys()),
        "lightings": list(LIGHTING_MAP.keys()),
        "tones": list(TONE_MAP.keys()),
        "color_gradings": list(COLOR_GRADING_MAP.keys()),
        "logo_positions": list(POSITION_MAP.keys()),
        "aspect_ratios": ALLOWED_ASPECT_RATIOS,
        "image_sizes": ALLOWED_IMAGE_SIZES,
    }

@app.get("/video-options", tags=["System"], summary="Video dropdown values")
async def video_options():
    return {
        "camera_angles": list(CAMERA_ANGLE_MAP.keys()),
        "camera_movements": list(CAMERA_MOVEMENT_MAP.keys()),
        "lens_effects": list(LENS_EFFECT_MAP.keys()),
        "visual_styles": list(VIDEO_STYLE_MAP.keys()),
        "temporal_elements": list(TEMPORAL_MAP.keys()),
        "sound_ambiences": list(SOUND_AMBIENCE_MAP.keys()),
        "veo_variants": list(VIDEO_MODEL_MAP.keys()),
        "durations": ALLOWED_DURATIONS,
        "resolutions": ALLOWED_RESOLUTIONS,
    }

@app.get("/brand-assets", tags=["System"])
async def brand_assets():
    prefix = BRAND_CONFIG.get("assets_gcs_prefix", "gs://content_creation_data/brand_assets/mobily/")
    return {"gcs_prefix": prefix, "folders": {
        "logos": f"{prefix}logos/", "end_frames": f"{prefix}end_frames/",
        "style_refs": f"{prefix}style_refs/", "people": f"{prefix}people/",
    }}


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("app:app", host="0.0.0.0", port=PORT, reload=True)
