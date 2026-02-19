"""
Mobily Creative Studio — Combined API
VERSION: 4.1.1

Field name mapping (Swagger title → internal variable → downstream):
  "Image Size"       → image_size    → Gemini aspect_ratio param (16:9, 9:16 …)
  "Resolution"       → resolution    → Gemini image_size param (1K, 2K, 4K)
  "Visual Text"      → visual_text   → PromptInputs.campaign_text
  "Visual Text Style" → visual_text_style → PromptInputs.text_style
  "Framing" (video)  → framing       → build_veo_prompt lens_effect param
  "Video Motion"     → video_motion  → build_veo_prompt temporal_elements param
"""

import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI, Form, File, UploadFile, HTTPException
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware

from src.config import (
    PORT, BUCKET_NAME, MODEL_ID, DEFAULTS, GCS_PREFIXES,
    ALLOWED_ASPECT_RATIOS, ALLOWED_IMAGE_SIZES,
    VISUAL_STYLE_MAP, CAMERA_ANGLE_IMG_MAP, FRAMING_MAP,
    LIGHTING_MAP, TONE_MAP, COLOR_GRADING_MAP,
    MOBILY_PALETTE,
    VIDEO_DEFAULTS, VIDEO_MODEL_DEFAULTS, VIDEO_TONE_MAP,
    CAMERA_ANGLE_MAP, CAMERA_MOVEMENT_MAP, LENS_EFFECT_MAP,
    VIDEO_STYLE_MAP, TEMPORAL_MAP, SOUND_AMBIENCE_MAP,
    VIDEO_ALLOWED_ASPECT_RATIOS, ALLOWED_DURATIONS, ALLOWED_RESOLUTIONS,
    ALLOWED_TTS_LANGUAGES,
    gcs_client, logger,
)
from src.core.prompt_generator import PromptInputs, build_prompt
from src.helpers.utils import (
    generate_image, generate_output_path, upload_to_gcs,
    overlay_logo, POSITION_MAP, LOGO_SIZE_SCALE, list_gcs_logos, download_gcs_logo,
)
from src.helpers.video_helpers import (
    read_upload, clean, validate_video_params, build_veo_prompt,
    effective_negative, prepare_end_frame, handle_audio, video_response,
)
from src.helpers.video_utils import (
    generate_video, generate_video_output_uri, ImageInput, _upload_image_to_gcs,
)
from src.schema.responses import (                           # ← FIX #1: was src.schema.responses
    ImageGenerationResponse, VideoGenerationResponse,
    RefineImageResponse, RefineVideoResponse, HealthResponse,
)

_VERSION = "4.1.1"


@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("Creative Studio %s — READY", _VERSION)
    yield

app = FastAPI(title="Mobily Creative Studio", version=_VERSION, lifespan=lifespan)
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])


# ═══════════════════════════════════════════════════════════════════════════
#  TEXT-TO-IMAGE
#
#  Swagger field order:
#    Subject* → Action → Items In Scene → Setting
#    Logo File (PNG) → Logo Size → Logo Position
#    Visual Text → Visual Text Style
#    Framing → Color Grading → Lighting → Tone / Mood → Visual Style
#    Image Size → Resolution → Negative Prompt (Avoid)
# ═══════════════════════════════════════════════════════════════════════════

@app.post("/generate-image", tags=["Image Generation"],
          summary="Generate Branded Marketing Image")
async def generate_image_endpoint(
    # ── Core ──────────────────────────────────────────────────────────
    subject: str = Form(..., title="Subject", description="Main subject of the image."),
    action: str = Form("", title="Action", description="What the subject is doing."),
    items_in_scene: str = Form("", title="Items In Scene", description="Additional objects in scene."),
    setting: str = Form("", title="Setting", description="Environment and context."),
    # ── Logo ──────────────────────────────────────────────────────────
    logo_file: UploadFile | None = File(default=None, title="Logo File (PNG)", description="Upload logo PNG. Or pick from library (GET /logos)."),
    logo_size: str = Form(DEFAULTS["logo_size"], title="Logo Size", description="small (8%), medium (15%), large (25%)."),
    logo_position: str = Form(DEFAULTS["logo_position"], title="Logo Position", description="top_left, top_center, top_right, center_left, center, center_right, bottom_left, bottom_center, bottom_right."),
    logo_name: str = Form("", title="Logo (From Library)", description="Pick a pre-uploaded logo name (see GET /logos)."),
    # ── Visual Text ───────────────────────────────────────────────────
    visual_text: str = Form("", title="Visual Text", description="Text to render on the image."),
    visual_text_style: str = Form(DEFAULTS["text_style"], title="Visual Text Style", description="Typography style for visual text."),
    # ── Creative Controls ─────────────────────────────────────────────
    framing: str = Form(DEFAULTS["framing"], title="Framing"),
    color_grading: str = Form(DEFAULTS["color_grading"], title="Color Grading"),
    lighting: str = Form(DEFAULTS["lighting"], title="Lighting"),
    tone: str = Form(DEFAULTS["tone"], title="Tone / Mood"),
    visual_style: str = Form(DEFAULTS["visual_style"], title="Visual Style"),
    # ── Output ────────────────────────────────────────────────────────
    image_size: str = Form(DEFAULTS["aspect_ratio"], title="Image Size", description="Aspect ratio: 1:1, 16:9, 9:16, 4:3, etc."),
    resolution: str = Form(DEFAULTS["image_size"], title="Resolution", description="1K, 2K, or 4K."),
    negative_prompt: str = Form("", title="Negative Prompt (Avoid)"),
):
    # ── Validate ─────────────────────────────────────────────────────
    errors = []
    if visual_style and visual_style not in VISUAL_STYLE_MAP:
        errors.append(f"Visual Style: {list(VISUAL_STYLE_MAP.keys())}")
    if framing and framing not in FRAMING_MAP:
        errors.append(f"Framing: {list(FRAMING_MAP.keys())}")
    if lighting and lighting not in LIGHTING_MAP:
        errors.append(f"Lighting: {list(LIGHTING_MAP.keys())}")
    if tone and tone not in TONE_MAP:
        errors.append(f"Tone / Mood: {list(TONE_MAP.keys())}")
    if color_grading and color_grading not in COLOR_GRADING_MAP:
        errors.append(f"Color Grading: {list(COLOR_GRADING_MAP.keys())}")
    if image_size not in ALLOWED_ASPECT_RATIOS:                 # ← FIX #4: validate ratio only
        errors.append(f"Image Size: {ALLOWED_ASPECT_RATIOS}")
    if resolution not in ALLOWED_IMAGE_SIZES:                   # ← FIX #4: validate resolution only
        errors.append(f"Resolution: {ALLOWED_IMAGE_SIZES}")
    if logo_position and logo_position not in POSITION_MAP:
        errors.append(f"Logo Position: {list(POSITION_MAP.keys())}")
    if logo_size and logo_size not in LOGO_SIZE_SCALE:
        errors.append(f"Logo Size: {list(LOGO_SIZE_SCALE.keys())}")
    if errors:
        raise HTTPException(400, "; ".join(errors))

    # ── Read logo (upload > GCS name) ────────────────────────────────
    has_logo, logo_bytes, logo_source = False, None, "none"
    if logo_file is not None and not isinstance(logo_file, str):
        try:
            data = await logo_file.read()
            if data and len(data) > 0:
                has_logo, logo_bytes = True, data
                logo_source = f"upload:{getattr(logo_file, 'filename', 'logo.png')}"
        except Exception as e:
            logger.exception("Logo read error: %s", e)
    if not has_logo and logo_name and logo_name.strip():
        try:
            logo_bytes = download_gcs_logo(logo_name.strip())
            if logo_bytes:
                has_logo = True
                logo_source = f"gcs:{logo_name.strip()}"
        except Exception as e:
            logger.exception("Logo GCS error: %s", e)

    # ── Build prompt ─────────────────────────────────────────────────
    # Map Swagger names → PromptInputs names
    inputs = PromptInputs(
        subject=subject,
        action=action,
        setting=setting,
        items_in_scene=items_in_scene,
        visual_style=visual_style,
        framing=framing,
        lighting=lighting,
        tone=tone,
        color_grading=color_grading,
        campaign_text=visual_text or None,          # ← FIX #2: Swagger "Visual Text" → internal campaign_text
        text_style=visual_text_style or None,       # ← FIX #2: Swagger "Visual Text Style" → internal text_style
        custom_prompt="",
        negative_prompt=negative_prompt or None,
    )
    final_prompt = build_prompt(inputs)
    logger.info("Image prompt [%s]: %s", _VERSION, final_prompt[:500])

    # ── Generate ─────────────────────────────────────────────────────
    try:
        result = generate_image(
            prompt=final_prompt,
            aspect_ratio=image_size,    # ← FIX #3: Swagger "Image Size" → Gemini aspect_ratio
            image_size=resolution,      # ← Swagger "Resolution" → Gemini image_size (1K/2K/4K)
            reference_images=None,
        )
    except RuntimeError as e:
        raise HTTPException(500, str(e))
    except Exception as e:
        logger.exception("Gemini API failed")
        raise HTTPException(502, f"Image generation failed: {e}")

    # ── Overlay logo ─────────────────────────────────────────────────
    final_image = result.image_bytes
    logo_info = {"applied": False}
    if has_logo and logo_bytes:
        try:
            final_image = overlay_logo(
                image_bytes=result.image_bytes, logo_bytes=logo_bytes,
                position=logo_position, size=logo_size,
            )
            logo_info = {"applied": True, "source": logo_source,
                         "position": logo_position, "size": logo_size}
        except Exception as e:
            logger.exception("Logo overlay failed")
            logo_info = {"applied": False, "error": str(e)}

    # ── Upload ───────────────────────────────────────────────────────
    output_path = generate_output_path(subject, prefix=GCS_PREFIXES.get("banners", "banners"))
    gcs_result = upload_to_gcs(final_image, output_path)

    return JSONResponse(content={
        "status": "success",
        "gcs_uri": gcs_result.gcs_uri,
        "public_url": gcs_result.public_url,
        "model": MODEL_ID,
        "prompt_used": final_prompt,
        "image_size": image_size,
        "resolution": resolution,
        "logo": logo_info,
        "model_commentary": result.model_text,
        "api_version": _VERSION,
    })


# ═══════════════════════════════════════════════════════════════════════════
#  IMAGE — Refine
# ═══════════════════════════════════════════════════════════════════════════

@app.post("/refine-image", tags=["Image Generation"],
          summary="Refine / Edit A Generated Image")
async def refine_image_endpoint(
    source_image: UploadFile = File(..., title="Source Image", description="Image to refine."),
    edit_prompt: str = Form(..., title="Edit Instructions", description="What to change, add, or fix."),
    image_size: str = Form(DEFAULTS["aspect_ratio"], title="Image Size"),
    resolution: str = Form(DEFAULTS["image_size"], title="Resolution"),
):
    """Send an existing image + edit instructions to Gemini for refinement."""
    img_data = await source_image.read()
    if not img_data:
        raise HTTPException(400, "Source Image is empty.")

    mime = getattr(source_image, "content_type", "image/png") or "image/png"

    try:
        result = generate_image(
            prompt=edit_prompt,
            aspect_ratio=image_size,
            image_size=resolution,
            reference_images=[(img_data, mime)],
        )
    except RuntimeError as e:
        raise HTTPException(500, str(e))
    except Exception as e:
        logger.exception("Refine image failed")
        raise HTTPException(502, f"Image refinement failed: {e}")

    output_path = generate_output_path("refined", prefix=GCS_PREFIXES.get("banners", "banners"))
    gcs_result = upload_to_gcs(result.image_bytes, output_path)

    return JSONResponse(content={
        "status": "success", "gcs_uri": gcs_result.gcs_uri, "public_url": gcs_result.public_url,
        "model": MODEL_ID, "prompt_used": edit_prompt,
        "original_image_source": getattr(source_image, "filename", ""),
        "model_commentary": result.model_text, "api_version": _VERSION,
    })


# ═══════════════════════════════════════════════════════════════════════════
#  TEXT-TO-VIDEO
#
#  Swagger field order:
#    Subject* → Action → Dialogue (Lip-Sync) → Scene Context
#    Camera Angle → Camera Movement → Framing → Video Motion
#    Visual Style → Tone / Mood
#    Audio File (Upload) → Sound Ambience → Language
#    Video Size → Duration (Seconds) → Variations → Resolution
#    Negative Prompt (Avoid)
# ═══════════════════════════════════════════════════════════════════════════

@app.post("/generate-video/text", tags=["Video Generation"],
          response_model=VideoGenerationResponse, summary="Text To Video")
async def text_to_video(
    # ── Core ──────────────────────────────────────────────────────────
    subject: str = Form(..., title="Subject", description="Main subject."),
    action: str = Form("", title="Action", description="What the subject is doing."),
    dialogue: str = Form("", title="Dialogue (Lip-Sync)", description="Voiceover script (auto-TTS)."),
    scene_context: str = Form("", title="Scene Context", description="Environment, location, time."),
    # ── Creative Controls ─────────────────────────────────────────────
    camera_angle: str = Form(VIDEO_DEFAULTS["camera_angle"], title="Camera Angle"),
    camera_movement: str = Form(VIDEO_DEFAULTS["camera_movement"], title="Camera Movement"),
    framing: str = Form(VIDEO_DEFAULTS["lens_effect"], title="Framing", description="Lens / framing (e.g., standard_50mm, wide_angle_24mm)."),
    video_motion: str = Form(VIDEO_DEFAULTS["temporal_elements"], title="Video Motion", description="Time / motion pacing (e.g., normal, slow_motion)."),
    visual_style: str = Form(VIDEO_DEFAULTS["visual_style"], title="Visual Style"),
    tone: str = Form(VIDEO_DEFAULTS.get("tone", "warm"), title="Tone / Mood"),
    # ── Audio ─────────────────────────────────────────────────────────
    audio_file: UploadFile | str | None = File(None, title="Audio File (Upload)", description="Upload custom audio."),
    sound_ambience: str = Form(VIDEO_DEFAULTS["sound_ambience"], title="Sound Ambience"),
    Language: str = Form(VIDEO_DEFAULTS["tts_language"], title="Language"),
    # ── Output ────────────────────────────────────────────────────────
    video_size: str = Form(VIDEO_DEFAULTS["aspect_ratio"], title="Video Size", description="16:9 or 9:16."),
    duration_seconds: int = Form(VIDEO_DEFAULTS["duration_seconds"], title="Duration (Seconds)"),
    number_of_videos: int = Form(1, title="Variations", description="How many variants to generate (1-4)."),
    resolution: str = Form(VIDEO_DEFAULTS["resolution"], title="Resolution"),
    negative_prompt: str = Form("", title="Negative Prompt (Avoid)"),
):
    subject, action, scene_context = clean(subject), clean(action), clean(scene_context)
    dialogue, negative_prompt = clean(dialogue), clean(negative_prompt)
    if not subject:
        raise HTTPException(400, "Subject is required.")

    # FIX #5: pass framing→lens_effect and video_motion→temporal_elements
    validate_video_params(video_size, duration_seconds, resolution, number_of_videos,
                          camera_angle, camera_movement, framing, visual_style, tone, video_motion)

    fp = build_veo_prompt(subject, action, scene_context, "", dialogue,
                          camera_angle, camera_movement,
                          framing,              # → lens_effect param
                          visual_style, tone,
                          video_motion,         # → temporal_elements param
                          sound_ambience, negative_prompt,
                          inject_dialogue=False)

    model = VIDEO_MODEL_DEFAULTS["text_to_video"]
    result = generate_video(
        prompt=fp, aspect_ratio=video_size, duration_seconds=duration_seconds,
        negative_prompt=effective_negative(negative_prompt), generate_audio=True,
        resolution=resolution, number_of_videos=number_of_videos, seed=None,
        generation_mode="text_to_video",
        output_gcs_uri=generate_video_output_uri(subject), model_id=model,
    )
    uri, audio_info = await handle_audio(dialogue, Language, audio_file, dialogue, result.gcs_uri)
    return video_response(uri, model, "text_to_video", fp, video_size, resolution,
                          duration_seconds, number_of_videos, audio_info,
                          {"applied": False}, [], result.operation_name)


# ═══════════════════════════════════════════════════════════════════════════
#  IMAGE-TO-VIDEO
#
#  Swagger field order:
#    Source Image* → End Frame Image
#    Action → Dialogue (Lip-Sync) → Scene Context
#    Camera Angle → Camera Movement → Framing → Video Motion
#    Visual Style → Tone / Mood
#    Audio File (Upload) → Sound Ambience → Language
#    Video Size → Duration (Seconds) → Resolution → Variations
#    Negative Prompt (Avoid)
# ═══════════════════════════════════════════════════════════════════════════

@app.post("/generate-video/image", tags=["Video Generation"],
          response_model=VideoGenerationResponse, summary="Image To Video")
async def image_to_video(
    # ── Source ─────────────────────────────────────────────────────────
    source_image: UploadFile = File(..., title="Source Image", description="Primary reference / source image (required)."),
    end_frame_image: UploadFile | str | None = File(None, title="End Frame Image", description="End frame image appears at the end of video."),
    # ── Core ──────────────────────────────────────────────────────────
    action: str = Form("", title="Action", description="What the subject is doing."),
    dialogue: str = Form("", title="Dialogue (Lip-Sync)", description="Dialogue / lip-sync text (if applicable)."),
    scene_context: str = Form("", title="Scene Context", description="Context / location / time of scene."),
    # ── Creative Controls ─────────────────────────────────────────────
    camera_angle: str = Form(VIDEO_DEFAULTS["camera_angle"], title="Camera Angle", description="Camera perspective."),
    camera_movement: str = Form(VIDEO_DEFAULTS["camera_movement"], title="Camera Movement", description="Camera movement (e.g., static, pan)."),
    framing: str = Form(VIDEO_DEFAULTS["lens_effect"], title="Framing", description="Lens choice / effect (e.g., standard_50mm, wide_angle_24mm)."),
    video_motion: str = Form(VIDEO_DEFAULTS["temporal_elements"], title="Video Motion", description="Time / motion pacing (e.g., normal, slow_motion)."),
    visual_style: str = Form(VIDEO_DEFAULTS["visual_style"], title="Visual Style", description="Overall visual style (e.g., photorealistic, cinematic)."),
    tone: str = Form(VIDEO_DEFAULTS.get("tone", "warm"), title="Tone / Mood", description="Emotional mood (e.g., serene)."),
    # ── Audio ─────────────────────────────────────────────────────────
    audio_file: UploadFile | str | None = File(None, title="Audio File (Upload)", description="Uploaded audio file (optional)."),
    sound_ambience: str = Form(VIDEO_DEFAULTS["sound_ambience"], title="Sound Ambience", description="Sound bed / ambience description."),
    Language: str = Form(VIDEO_DEFAULTS["tts_language"], title="Language", description="TTS language (e.g., ar-XA)."),
    # ── Output ────────────────────────────────────────────────────────
    video_size: str = Form(VIDEO_DEFAULTS["aspect_ratio"], title="Video Size", description="Output aspect ratio (e.g., 16:9)."),
    duration_seconds: int = Form(VIDEO_DEFAULTS["duration_seconds"], title="Duration (Seconds)", description="Video duration in seconds."),
    resolution: str = Form(VIDEO_DEFAULTS["resolution"], title="Resolution", description="Output resolution (e.g., 720p)."),
    number_of_videos: int = Form(1, title="Variations", description="How many variants to generate."),
    negative_prompt: str = Form("", title="Negative Prompt (Avoid)", description="What to avoid."),
):
    img = await read_upload(source_image, "source_image")
    if not img:
        raise HTTPException(400, "Source Image is required.")
    action, scene_context = clean(action), clean(scene_context)
    dialogue, negative_prompt = clean(dialogue), clean(negative_prompt)
    validate_video_params(video_size, duration_seconds, resolution, number_of_videos,
                          camera_angle, camera_movement, framing, visual_style, tone, video_motion)

    fp = build_veo_prompt("", action, scene_context, "", dialogue,
                          camera_angle, camera_movement, framing, visual_style, tone,
                          video_motion, sound_ambience, negative_prompt,
                          inject_dialogue=True)
    image_uri = _upload_image_to_gcs(img, "source")
    ef_uri, ef_mime, ef_info = await prepare_end_frame(end_frame_image)

    model_key = "image_to_video_endframe" if ef_uri else "image_to_video"
    model = VIDEO_MODEL_DEFAULTS[model_key]

    result = generate_video(
        prompt=fp, aspect_ratio=video_size, duration_seconds=duration_seconds,
        negative_prompt=effective_negative(negative_prompt), generate_audio=True,
        resolution=resolution, number_of_videos=number_of_videos, seed=None,
        generation_mode="image_to_video", source_image_gcs_uri=image_uri,
        source_image_mime=img.mime_type, last_frame_gcs_uri=ef_uri, last_frame_mime=ef_mime,
        output_gcs_uri=generate_video_output_uri(action or "animated"), model_id=model,
    )
    uri, audio_info = await handle_audio(dialogue, Language, audio_file, dialogue, result.gcs_uri)
    return video_response(uri, model, "image_to_video", fp, video_size, resolution,
                          duration_seconds, number_of_videos, audio_info,
                          ef_info, [img.filename], result.operation_name)


# ═══════════════════════════════════════════════════════════════════════════
#  REFERENCE-TO-VIDEO
# ═══════════════════════════════════════════════════════════════════════════

@app.post("/generate-video/reference", tags=["Video Generation"],
          response_model=VideoGenerationResponse, summary="Reference To Video (Identity Preservation)")
async def reference_to_video(
    # ── Source ─────────────────────────────────────────────────────────
    source_image: UploadFile = File(..., title="Source Image", description="Primary reference image (required)."),
    ref_image_2: UploadFile | str | None = File(None, title="Reference Image 2"),
    ref_image_3: UploadFile | str | None = File(None, title="Reference Image 3"),
    # ── Core ──────────────────────────────────────────────────────────
    subject: str = Form(..., title="Subject", description="Subject description matching the images."),
    action: str = Form("", title="Action", description="What the subject is doing."),
    dialogue: str = Form("", title="Dialogue (Lip-Sync)", description="Lip-sync text."),
    scene_context: str = Form("", title="Scene Context", description="Context / location / time."),
    # ── Creative Controls ─────────────────────────────────────────────
    camera_angle: str = Form(VIDEO_DEFAULTS["camera_angle"], title="Camera Angle"),
    camera_movement: str = Form(VIDEO_DEFAULTS["camera_movement"], title="Camera Movement"),
    framing: str = Form(VIDEO_DEFAULTS["lens_effect"], title="Framing"),
    video_motion: str = Form(VIDEO_DEFAULTS["temporal_elements"], title="Video Motion"),
    visual_style: str = Form(VIDEO_DEFAULTS["visual_style"], title="Visual Style"),
    tone: str = Form(VIDEO_DEFAULTS.get("tone", "warm"), title="Tone / Mood"),
    # ── Audio ─────────────────────────────────────────────────────────
    audio_file: UploadFile | str | None = File(None, title="Audio File (Upload)"),
    sound_ambience: str = Form(VIDEO_DEFAULTS["sound_ambience"], title="Sound Ambience"),
    Language: str = Form(VIDEO_DEFAULTS["tts_language"], title="Language"),
    # ── Output ────────────────────────────────────────────────────────
    video_size: str = Form(VIDEO_DEFAULTS["aspect_ratio"], title="Video Size"),
    duration_seconds: int = Form(8, title="Duration (Seconds)"),
    resolution: str = Form(VIDEO_DEFAULTS["resolution"], title="Resolution"),
    Variants: int = Form(1, title="Variations"),
    negative_prompt: str = Form("", title="Negative Prompt (Avoid)"),
):
    dur = 8
    img1 = await read_upload(source_image, "source_image")
    if not img1:
        raise HTTPException(400, "Source Image is required.")
    img2 = await read_upload(ref_image_2, "ref_image_2")
    img3 = await read_upload(ref_image_3, "ref_image_3")
    imgs = [i for i in [img1, img2, img3] if i]
    subject, action, scene_context = clean(subject), clean(action), clean(scene_context)
    dialogue, negative_prompt = clean(dialogue), clean(negative_prompt)
    validate_video_params(video_size, dur, resolution, Variants,
                          camera_angle, camera_movement, framing, visual_style, tone, video_motion)

    fp = build_veo_prompt(subject, action, scene_context, "", dialogue,
                          camera_angle, camera_movement, framing, visual_style, tone,
                          video_motion, sound_ambience, negative_prompt,
                          inject_dialogue=True)

    model = VIDEO_MODEL_DEFAULTS["reference_to_video"]
    result = generate_video(
        prompt=fp, aspect_ratio=video_size, duration_seconds=dur,
        negative_prompt=effective_negative(negative_prompt), generate_audio=True,
        resolution=resolution, number_of_videos=Variants, seed=None,
        generation_mode="reference_to_video", reference_images=imgs,
        output_gcs_uri=generate_video_output_uri(subject or "reference"), model_id=model,
    )
    uri, audio_info = await handle_audio(dialogue, Language, audio_file, dialogue, result.gcs_uri)
    return video_response(uri, model, "reference_to_video", fp, video_size, resolution,
                          dur, Variants, audio_info,
                          {"applied": False}, [i.filename for i in imgs], result.operation_name)


# ═══════════════════════════════════════════════════════════════════════════
#  VIDEO — Refine
# ═══════════════════════════════════════════════════════════════════════════

@app.post("/refine-video", tags=["Video Generation"],
          summary="Refine / Extend A Generated Video")
async def refine_video_endpoint(
    source_video: UploadFile = File(..., title="Source Video", description="Video to refine or extend."),
    edit_prompt: str = Form(..., title="Edit Instructions", description="What to change or extend."),
    duration_seconds: int = Form(VIDEO_DEFAULTS["duration_seconds"], title="Duration (Seconds)"),
    video_size: str = Form(VIDEO_DEFAULTS["aspect_ratio"], title="Video Size"),
    resolution: str = Form(VIDEO_DEFAULTS["resolution"], title="Resolution"),
):
    vid_data = await source_video.read()
    if not vid_data:
        raise HTTPException(400, "Source Video is empty.")

    from src.helpers.video_utils import upload_bytes_to_gcs
    vid_blob = f"{GCS_PREFIXES.get('video_inputs', 'video_inputs')}/refine/{getattr(source_video, 'filename', 'input.mp4')}"
    vid_upload = upload_bytes_to_gcs(vid_data, vid_blob, content_type="video/mp4")

    model = VIDEO_MODEL_DEFAULTS["refine_video"]
    result = generate_video(
        prompt=edit_prompt, aspect_ratio=video_size,
        duration_seconds=duration_seconds,
        negative_prompt=effective_negative(""), generate_audio=True,
        resolution=resolution, number_of_videos=1, seed=None,
        generation_mode="extend_video",
        source_video_gcs_uri=vid_upload.gcs_uri,
        output_gcs_uri=generate_video_output_uri("refined"), model_id=model,
    )

    return JSONResponse(content={
        "status": "success", "gcs_uri": result.gcs_uri,
        "public_url": result.gcs_uri.replace(f"gs://{BUCKET_NAME}/", f"https://storage.googleapis.com/{BUCKET_NAME}/"),
        "model": model, "prompt_used": edit_prompt,
        "original_video_source": getattr(source_video, "filename", ""),
        "operation": result.operation_name, "api_version": _VERSION,
    })


# ═══════════════════════════════════════════════════════════════════════════
#  System / Data Endpoints
# ═══════════════════════════════════════════════════════════════════════════

@app.get("/health", tags=["System"])
async def health():
    return {"status": "ok", "version": _VERSION, "image_model": MODEL_ID,
            "video_models": VIDEO_MODEL_DEFAULTS}


@app.get("/colors", tags=["Brand Assets"],
         summary="Mobily Brand Color Palette (For Frontend Dropdown + Preview)")
async def get_colors():
    """Returns Mobily palette grouped by category with hex for swatches."""
    grouped = {"primary": [], "bright_accent": [], "dark_secondary": [], "gradient": []}
    for key, info in MOBILY_PALETTE.items():
        entry = {
            "key": key, "name": info["name"], "hex": info["hex"],
            "rgb": info.get("rgb"), "group": info.get("group", "primary"),
        }
        if "hex_end" in info:
            entry["hex_end"] = info["hex_end"]
        group = info.get("group", "primary")
        grouped.setdefault(group, []).append(entry)
    return {"palette": grouped, "all_keys": list(MOBILY_PALETTE.keys())}


@app.get("/logos", tags=["Brand Assets"],
         summary="List Available Logos (For Frontend Dropdown + Preview)")
async def list_logos():
    """Returns logo names + public URLs for preview thumbnails."""
    try:
        return {"logos": list_gcs_logos()}
    except Exception as e:
        return {"logos": [], "error": str(e)}


@app.get("/options", tags=["System"], summary="Image Dropdown Values")
async def image_options():
    return {
        "visual_styles": list(VISUAL_STYLE_MAP.keys()),
        "framings": list(FRAMING_MAP.keys()),
        "lightings": list(LIGHTING_MAP.keys()),
        "tones": list(TONE_MAP.keys()),
        "color_gradings": list(COLOR_GRADING_MAP.keys()),
        "logo_sizes": list(LOGO_SIZE_SCALE.keys()),
        "logo_positions": list(POSITION_MAP.keys()),
        "image_sizes": ALLOWED_ASPECT_RATIOS,
        "resolutions": ALLOWED_IMAGE_SIZES,
    }


@app.get("/video-options", tags=["System"], summary="Video Dropdown Values")
async def video_options():
    return {
        "camera_angles": list(CAMERA_ANGLE_MAP.keys()),
        "camera_movements": list(CAMERA_MOVEMENT_MAP.keys()),
        "framings": list(LENS_EFFECT_MAP.keys()),
        "video_motions": list(TEMPORAL_MAP.keys()),
        "visual_styles": list(VIDEO_STYLE_MAP.keys()),
        "tones": list(VIDEO_TONE_MAP.keys()),
        "sound_ambiences": list(SOUND_AMBIENCE_MAP.keys()),
        "video_sizes": VIDEO_ALLOWED_ASPECT_RATIOS,
        "durations": ALLOWED_DURATIONS,
        "resolutions": ALLOWED_RESOLUTIONS,
        "languages": ALLOWED_TTS_LANGUAGES,
    }


@app.get("/brand-assets", tags=["Brand Assets"])
async def brand_assets():
    return {"bucket": BUCKET_NAME, "prefixes": GCS_PREFIXES}


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("app:app", host="0.0.0.0", port=PORT, reload=True)
