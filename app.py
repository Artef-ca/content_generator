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

from contextlib import asynccontextmanager
import asyncio

from fastapi import FastAPI, Form, File, UploadFile, HTTPException, Request
from fastapi.responses import JSONResponse , Response
from fastapi.middleware.cors import CORSMiddleware
from typing import Optional, List
from google.genai import types
import time
from datetime import timedelta
import json
from src.schemas import (TrendingEvent, TrendingEventsResponse)
from src.helpers.utils import (get_trending_events, get_trending_events_cached)



from src.config import (
    PORT, BUCKET_NAME, MODEL_ID, DEFAULTS, GCS_PREFIXES,
    ALLOWED_ASPECT_RATIOS, ALLOWED_IMAGE_SIZES,
    VISUAL_STYLE_MAP, CAMERA_ANGLE_IMG_MAP, FRAMING_MAP,
    LIGHTING_MAP, TONE_MAP, COLOR_GRADING_MAP,
    MOBILY_PALETTE, FONTS,
    TEXT_SIZES, TEXT_WEIGHTS, TEXT_POSITIONS, TEXT_LANGUAGES,
    VIDEO_DEFAULTS, VIDEO_MODEL_DEFAULTS, VIDEO_TONE_MAP,
    CAMERA_ANGLE_MAP, CAMERA_MOVEMENT_MAP, LENS_EFFECT_MAP,
    VIDEO_STYLE_MAP, TEMPORAL_MAP, SOUND_AMBIENCE_MAP,
    VIDEO_ALLOWED_ASPECT_RATIOS, ALLOWED_DURATIONS, ALLOWED_RESOLUTIONS,
    ALLOWED_TTS_LANGUAGES,
    gcs_client, logger,genai_client,PROJECT_ID
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
    overlay_logo_on_video, download_from_gcs,
)
from src.helpers.utils import download_gcs_logo
from src.schema.responses import (                           # ← FIX #1: was src.schema.responses
    ImageGenerationResponse, VideoGenerationResponse,
    RefineImageResponse, RefineVideoResponse, HealthResponse,
)
from pydantic import BaseModel

from datetime import timedelta
from fastapi import HTTPException, Depends, status
from sqlalchemy.ext.asyncio import AsyncSession
from src.core.auth import (
    UserCreate, UserLogin, UserResponse, Token,
    get_user_by_username, create_user, authenticate_user,
    create_access_token, get_current_user,
    ACCESS_TOKEN_EXPIRE_MINUTES
)
from src.core.database import get_db, User

_VERSION = "4.1.1"


@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("Creative Studio %s — READY", _VERSION)
    yield

app = FastAPI(title="Mobily Creative Studio", version=_VERSION, lifespan=lifespan)
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])

###auth
# @app.post("/api/auth/register", response_model=UserResponse, status_code=status.HTTP_201_CREATED)
# async def register(user_data: UserCreate, db: AsyncSession = Depends(get_db)):
#     existing_user = await get_user_by_username(db, user_data.username)
#     if existing_user:
#         raise HTTPException(
#             status_code=status.HTTP_400_BAD_REQUEST,
#             detail="Username already registered"
#         )
#     user = await create_user(db, user_data)
#     return user


# @app.post("/api/auth/login", response_model=Token)
# async def login(user_data: UserLogin, db: AsyncSession = Depends(get_db)):
#     user = await authenticate_user(db, user_data.username, user_data.password)
#     if not user:
#         raise HTTPException(
#             status_code=status.HTTP_401_UNAUTHORIZED,
#             detail="Incorrect username or password",
#             headers={"WWW-Authenticate": "Bearer"},
#         )
#     access_token_expires = timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES)
#     access_token = create_access_token(
#         data={"sub": str(user.id), "username": user.username},
#         expires_delta=access_token_expires
#     )
#     return Token(access_token=access_token)


# @app.get("/api/auth/me", response_model=UserResponse)
# async def get_me(current_user: User = Depends(get_current_user)):
#     return current_user

###trending events

@app.get("/api/trending-events", response_model=TrendingEventsResponse)
async def trending_events_endpoint():
    try:
        events = get_trending_events_cached()
        return TrendingEventsResponse(events=events)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to fetch trending events: {str(e)}")

# ═══════════════════════════════════════════════════════════════════════════
#  TEXT-TO-IMAGE
#
#  Swagger field order:
#    Subject* → Action → Items In Scene → Setting
#    Logo File (PNG) → Logo Name → Logo Size → Logo Position → Logo Comments
#    Text Content → Text Font → Text Size → Text Color → Text Weight →
#      Text Position → Text Language → Text Notes
#    Visual Style → Framing → Lighting → Tone → Color Palette
#    Image Size → Resolution → Variations → Negative Prompt
# ═══════════════════════════════════════════════════════════════════════════

# Pre-built description strings from config (keeps Form() calls readable)
_D_VISUAL_STYLE  = f"Overall visual style. Options: {', '.join(VISUAL_STYLE_MAP.keys())}."
_D_FRAMING       = f"Framing / composition. Options: {', '.join(FRAMING_MAP.keys())}."
_D_LIGHTING      = f"Lighting style. Options: {', '.join(LIGHTING_MAP.keys())}."
_D_TONE          = f"Tone / mood. Options: {', '.join(TONE_MAP.keys())}."
_D_COLOR         = (
    f"Color palette key(s), comma-separated for multiple (first = dominant). "
    f"Options: {', '.join(COLOR_GRADING_MAP.keys())}."
)
_D_LOGO_SIZE     = f"Logo size. Options: {', '.join(LOGO_SIZE_SCALE.keys())}."
_D_LOGO_POS      = f"Logo placement. Options: {', '.join(POSITION_MAP.keys())}."
_D_IMAGE_SIZE    = f"Aspect ratio. Options: {', '.join(ALLOWED_ASPECT_RATIOS)}."
_D_RESOLUTION    = f"Output resolution. Options: {', '.join(ALLOWED_IMAGE_SIZES)}."
_D_TEXT_FONT     = f"Font for the text block. Options: {', '.join(f['value'] for f in FONTS)}."
_D_TEXT_SIZE     = f"Text size. Options: {', '.join(TEXT_SIZES)} (S=Small, M=Medium, L=Large)."
_D_TEXT_WEIGHT   = f"Font weight. Options: {', '.join(TEXT_WEIGHTS)}."
_D_TEXT_POS      = f"Text placement on image. Options: {', '.join(TEXT_POSITIONS)}."
_D_TEXT_LANG     = f"Text language. Options: {', '.join(f'{k} ({v})' for k, v in TEXT_LANGUAGES.items())}."
_D_TEXT_COLOR    = f"Text color (Mobily palette key). Options: {', '.join(COLOR_GRADING_MAP.keys())}."

@app.post("/generate-image", tags=["Image Generation"],
          summary="Generate Branded Marketing Image")
async def generate_image_endpoint(
    request: Request,
    # ── Core ──────────────────────────────────────────────────────────
    subject: str = Form(...,  title="Subject",        description="Main subject of the image (required)."),
    action:  str = Form("",   title="Action",         description="What the subject is doing."),
    items_in_scene: str = Form("", title="Items In Scene", description="Additional objects or elements to include in the scene."),
    setting: str = Form("",   title="Setting",        description="Environment, location and time of day."),
    # ── Logo ──────────────────────────────────────────────────────────
    logo_file: UploadFile | None = File(default=None, title="Logo File (PNG)", description="Upload a logo PNG directly."),
    logo_name: str = Form("", title="Logo (From Library)", description="Name of a pre-uploaded logo (see GET /logos). Used if no logo file is uploaded."),
    logo_size: str = Form(DEFAULTS["logo_size"],     title="Logo Size",     description=_D_LOGO_SIZE),
    logo_position: str = Form(DEFAULTS["logo_position"], title="Logo Position", description=_D_LOGO_POS),
    logo_comments: str = Form("", title="Logo Comments", description="Extra instructions for logo placement, e.g. 'place inside white circle'."),
    # ── Text Block (individual fields — assembled into one text block) ─
    text_content:  str = Form("",       title="Text Content",  description="Text to display on the image (e.g. رمضان كريم)."),
    text_font:     str = Form("Arial",  title="Text Font",     description=_D_TEXT_FONT),
    text_size:     str = Form("M",      title="Text Size",     description=_D_TEXT_SIZE),
    text_color:    str = Form("white",  title="Text Color",    description=_D_TEXT_COLOR),
    text_weight:   str = Form("bold",   title="Text Weight",   description=_D_TEXT_WEIGHT),
    text_position: str = Form("bottom_center", title="Text Position", description=_D_TEXT_POS),
    text_language: str = Form("ar",     title="Text Language", description=_D_TEXT_LANG),
    text_notes:    str = Form("",       title="Text Notes",    description="Extra styling instructions for this text block."),
    # ── Creative Controls ─────────────────────────────────────────────
    visual_style:  str = Form(DEFAULTS["visual_style"], title="Visual Style",  description=_D_VISUAL_STYLE),
    framing:       str = Form(DEFAULTS["framing"],      title="Framing",       description=_D_FRAMING),
    lighting:      str = Form(DEFAULTS["lighting"],     title="Lighting",      description=_D_LIGHTING),
    tone:          str = Form(DEFAULTS["tone"],         title="Tone / Mood",   description=_D_TONE),
    color_grading: str = Form(DEFAULTS["color_grading"], title="Color Palette", description=_D_COLOR),
    # ── Output ────────────────────────────────────────────────────────
    image_size:      str = Form(DEFAULTS["aspect_ratio"], title="Image Size",  description=_D_IMAGE_SIZE),
    resolution:      str = Form(DEFAULTS["image_size"],   title="Resolution",  description=_D_RESOLUTION),
    variations:      int = Form(1, title="Variations", description="Number of images to generate (1–4)."),
    negative_prompt: str = Form("", title="Negative Prompt (Avoid)", description="Describe what to exclude from the image."),
):
    # ── Validate ─────────────────────────────────────────────────────
    errors = []
    if visual_style and visual_style not in VISUAL_STYLE_MAP:
        errors.append(f"Visual Style must be one of: {', '.join(VISUAL_STYLE_MAP.keys())}")
    if framing and framing not in FRAMING_MAP:
        errors.append(f"Framing must be one of: {', '.join(FRAMING_MAP.keys())}")
    if lighting and lighting not in LIGHTING_MAP:
        errors.append(f"Lighting must be one of: {', '.join(LIGHTING_MAP.keys())}")
    if tone and tone not in TONE_MAP:
        errors.append(f"Tone must be one of: {', '.join(TONE_MAP.keys())}")
    if image_size not in ALLOWED_ASPECT_RATIOS:
        errors.append(f"Image Size must be one of: {', '.join(ALLOWED_ASPECT_RATIOS)}")
    if resolution not in ALLOWED_IMAGE_SIZES:
        errors.append(f"Resolution must be one of: {', '.join(ALLOWED_IMAGE_SIZES)}")
    if logo_position and logo_position not in POSITION_MAP:
        errors.append(f"Logo Position must be one of: {', '.join(POSITION_MAP.keys())}")
    if logo_size and logo_size not in LOGO_SIZE_SCALE:
        errors.append(f"Logo Size must be one of: {', '.join(LOGO_SIZE_SCALE.keys())}")
    # Validate each color key
    color_grading_list = [c.strip() for c in color_grading.split(",") if c.strip()]
    invalid_colors = [c for c in color_grading_list if c not in COLOR_GRADING_MAP]
    if invalid_colors:
        errors.append(f"Color Palette invalid key(s) {invalid_colors}. Valid: {', '.join(COLOR_GRADING_MAP.keys())}")
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

    # ── Parse text blocks ─────────────────────────────────────────────
    # Frontend sends text_blocks as a JSON array (not a Swagger param).
    # Swagger testers use the individual text_* fields instead.
    parsed_blocks: list = []
    raw_form = await request.form()
    text_blocks_json = raw_form.get("text_blocks", "")
    if text_blocks_json:
        try:
            parsed_blocks = json.loads(text_blocks_json)
        except Exception:
            parsed_blocks = []

    # If no JSON blocks (Swagger tester), build one block from individual fields
    if not parsed_blocks and text_content.strip():
        parsed_blocks = [{
            "text":     text_content.strip(),
            "font":     text_font,
            "size":     text_size,
            "color":    text_color,
            "weight":   text_weight,
            "position": text_position,
            "language": text_language,
            "comments": text_notes,
        }]

    # ── Build prompt ─────────────────────────────────────────────────
    inputs = PromptInputs(
        subject=subject,
        action=action,
        setting=setting,
        items_in_scene=items_in_scene,
        visual_style=visual_style,
        framing=framing,
        lighting=lighting,
        tone=tone,
        color_grading=color_grading_list,
        text_blocks=parsed_blocks,
        logo_comments=logo_comments,
        logo_position=logo_position,
        logo_size=logo_size,
        custom_prompt="",
        negative_prompt=negative_prompt or None,
    )
    final_prompt = build_prompt(inputs)
    logger.info("Image prompt [%s]: %s", _VERSION, final_prompt[:500])

    # ── Generate all variations sequentially ─────────────────────────
    import asyncio

    num_variations = max(1, min(4, variations))
    ref_images = [(logo_bytes, "image/png")] if (has_logo and logo_bytes and logo_comments) else None

    _IMAGE_GEN_TIMEOUT = 120  # seconds per attempt
    _MAX_RETRIES = 3

    async def _gen_one(i: int) -> dict:
        result = None
        for attempt in range(_MAX_RETRIES):
            try:
                result = await asyncio.wait_for(
                    asyncio.to_thread(
                        generate_image,
                        prompt=final_prompt,
                        aspect_ratio=image_size,
                        image_size=resolution,
                        reference_images=ref_images,
                    ),
                    timeout=_IMAGE_GEN_TIMEOUT,
                )
                break  # success — exit retry loop
            except asyncio.TimeoutError:
                raise
            except Exception as e:
                if ("429" in str(e) or "RESOURCE_EXHAUSTED" in str(e)) and attempt < _MAX_RETRIES - 1:
                    wait = (2 ** attempt) * 10  # 10s, then 20s
                    logger.warning("Rate limited on variation %d (attempt %d/%d), retrying in %ds", i + 1, attempt + 1, _MAX_RETRIES, wait)
                    await asyncio.sleep(wait)
                else:
                    raise
        img = result.image_bytes
        if has_logo and logo_bytes and not logo_comments:
            img = await asyncio.to_thread(
                overlay_logo,
                image_bytes=img, logo_bytes=logo_bytes,
                position=logo_position, size=logo_size,
            )
        output_path = generate_output_path(subject, prefix=GCS_PREFIXES.get("banners", "banners"))
        gcs = await asyncio.to_thread(upload_to_gcs, img, output_path)
        return {"gcs_uri": gcs.gcs_uri, "public_url": gcs.public_url, "model_commentary": result.model_text}

    try:
        images_out = []
        for idx in range(num_variations):
            logger.info("Generating variation %d/%d", idx + 1, num_variations)
            images_out.append(await _gen_one(idx))
    except asyncio.TimeoutError:
        raise HTTPException(504, "Image generation timed out after 120s. Please try again.")
    except RuntimeError as e:
        raise HTTPException(500, str(e))
    except Exception as e:
        logger.exception("Gemini API failed")
        raise HTTPException(502, f"Image generation failed: {e}")

    # ── Logo info (same for all variations) ──────────────────────────
    if has_logo and logo_bytes and not logo_comments:
        logo_info = {"applied": True, "source": logo_source, "position": logo_position, "size": logo_size}
    elif has_logo and logo_comments:
        logo_info = {"applied": True, "method": "llm_placed", "source": logo_source}
    else:
        logo_info = {"applied": False}

    return JSONResponse(content={
        "status": "success",
        "images": images_out,
        "model": MODEL_ID,
        "prompt_used": final_prompt,
        "image_size": image_size,
        "resolution": resolution,
        "logo": logo_info,
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

    # Wrap user instructions with logo-preservation template so the model
    # only changes what was explicitly requested and leaves everything else intact.
    final_edit_prompt = (
        f"{edit_prompt.strip()} "
        "Preserve the Mobily logo exactly as placed — only change what is explicitly requested. "
        "Do not alter any other elements of the image that were not mentioned above."
    )

    try:
        result = generate_image(
            prompt=final_edit_prompt,
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
        "model": MODEL_ID, "prompt_used": final_edit_prompt,
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

async def _read_video_logo(logo_file, logo_name: str) -> bytes | None:
    """Read logo bytes from upload or GCS, same pattern as image endpoint."""
    if logo_file is not None and not isinstance(logo_file, str):
        try:
            data = await logo_file.read()
            if data:
                return data
        except Exception:
            pass
    if logo_name and logo_name.strip():
        try:
            return download_gcs_logo(logo_name.strip())
        except Exception:
            pass
    return None


async def _apply_video_logo(
    uris: list[str], logo_bytes: bytes | None,
    logo_position: str, logo_size: str,
) -> list[str]:
    """Overlay logo on all video URIs and return new GCS URIs.

    Falls back to original URIs if overlay fails, so the video is always returned.
    """
    if not logo_bytes:
        return uris
    from src.helpers.video_utils import upload_bytes_to_gcs
    new_uris = []
    for uri in uris:
        try:
            # download_from_gcs handles both direct file URIs and GCS prefix URIs
            video_bytes = await asyncio.to_thread(download_from_gcs, uri)
            composited = await asyncio.to_thread(
                overlay_logo_on_video, video_bytes, logo_bytes, logo_position, logo_size
            )
            blob_path = uri.replace(f"gs://{BUCKET_NAME}/", "").rstrip("/")
            if not blob_path.endswith(".mp4"):
                blob_path += "_logo.mp4"
            else:
                blob_path = blob_path.replace(".mp4", "_logo.mp4")
            result = await asyncio.to_thread(
                upload_bytes_to_gcs, composited, blob_path, "video/mp4"
            )
            new_uris.append(result.gcs_uri)
            logger.info("Logo overlay applied: %s → %s", uri, result.gcs_uri)
        except Exception as e:
            logger.exception("Logo overlay failed for %s — returning original URI: %s", uri, e)
            new_uris.append(uri)
    return new_uris


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
    # ── Logo ──────────────────────────────────────────────────────────
    logo_file: UploadFile | None = File(default=None, title="Logo File (PNG)"),
    logo_name: str = Form("", title="Logo (From Library)"),
    logo_position: str = Form(DEFAULTS["logo_position"], title="Logo Position"),
    logo_size: str = Form(DEFAULTS["logo_size"], title="Logo Size"),
):
    subject, action, scene_context = clean(subject), clean(action), clean(scene_context)
    dialogue, negative_prompt = clean(dialogue), clean(negative_prompt)
    if not subject:
        raise HTTPException(400, "Subject is required.")

    validate_video_params(video_size, duration_seconds, resolution, number_of_videos,
                          camera_angle, camera_movement, framing, visual_style, tone, video_motion)

    fp = build_veo_prompt(subject, action, scene_context, "", dialogue,
                          camera_angle, camera_movement,
                          framing, visual_style, tone,
                          video_motion, sound_ambience, negative_prompt,
                          inject_dialogue=False)

    logo_bytes = await _read_video_logo(logo_file, logo_name)
    model = VIDEO_MODEL_DEFAULTS["text_to_video"]
    try:
        result = generate_video(
            prompt=fp, aspect_ratio=video_size, duration_seconds=duration_seconds,
            negative_prompt=effective_negative(negative_prompt), generate_audio=True,
            resolution=resolution, number_of_videos=number_of_videos, seed=None,
            generation_mode="text_to_video",
            output_gcs_uri=generate_video_output_uri(subject), model_id=model,
        )
        final_uris = await _apply_video_logo(result.gcs_uris, logo_bytes, logo_position, logo_size)
        primary, audio_info = await handle_audio(dialogue, Language, audio_file, dialogue, final_uris[0])
        final_uris[0] = primary
        logo_info = {"applied": bool(logo_bytes), "position": logo_position, "size": logo_size} if logo_bytes else {"applied": False}
        return video_response(final_uris, model, "text_to_video", fp, video_size, resolution,
                              duration_seconds, number_of_videos, audio_info,
                              logo_info, [], result.operation_name)
    except RuntimeError as e:
        msg = str(e)
        code = 422 if any(k in msg for k in ("code 3", "sensitive words", "Responsible AI", "allowlisting", "safety filter")) else 500
        raise HTTPException(status_code=code, detail=msg)
    except Exception as e:
        logger.exception("text_to_video failed")
        raise HTTPException(status_code=500, detail=f"Video generation failed: {e}")


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
    # ── Logo ──────────────────────────────────────────────────────────
    logo_file: UploadFile | None = File(default=None, title="Logo File (PNG)"),
    logo_name: str = Form("", title="Logo (From Library)"),
    logo_position: str = Form(DEFAULTS["logo_position"], title="Logo Position"),
    logo_size: str = Form(DEFAULTS["logo_size"], title="Logo Size"),
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
    logo_bytes = await _read_video_logo(logo_file, logo_name)

    model_key = "image_to_video_endframe" if ef_uri else "image_to_video"
    model = VIDEO_MODEL_DEFAULTS[model_key]

    try:
        result = generate_video(
            prompt=fp, aspect_ratio=video_size, duration_seconds=duration_seconds,
            negative_prompt=effective_negative(negative_prompt), generate_audio=True,
            resolution=resolution, number_of_videos=number_of_videos, seed=None,
            generation_mode="image_to_video", source_image_gcs_uri=image_uri,
            source_image_mime=img.mime_type, last_frame_gcs_uri=ef_uri, last_frame_mime=ef_mime,
            output_gcs_uri=generate_video_output_uri(action or "animated"), model_id=model,
        )
        final_uris = await _apply_video_logo(result.gcs_uris, logo_bytes, logo_position, logo_size)
        primary, audio_info = await handle_audio(dialogue, Language, audio_file, dialogue, final_uris[0])
        final_uris[0] = primary
        logo_info = {"applied": bool(logo_bytes), "position": logo_position, "size": logo_size} if logo_bytes else ef_info
        return video_response(final_uris, model, "image_to_video", fp, video_size, resolution,
                              duration_seconds, number_of_videos, audio_info,
                              logo_info, [img.filename], result.operation_name)
    except RuntimeError as e:
        msg = str(e)
        code = 422 if any(k in msg for k in ("code 3", "sensitive words", "Responsible AI", "allowlisting", "safety filter")) else 500
        raise HTTPException(status_code=code, detail=msg)
    except Exception as e:
        logger.exception("image_to_video failed")
        raise HTTPException(status_code=500, detail=f"Video generation failed: {e}")


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
    # ── Logo ──────────────────────────────────────────────────────────
    logo_file: UploadFile | None = File(default=None, title="Logo File (PNG)"),
    logo_name: str = Form("", title="Logo (From Library)"),
    logo_position: str = Form(DEFAULTS["logo_position"], title="Logo Position"),
    logo_size: str = Form(DEFAULTS["logo_size"], title="Logo Size"),
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

    logo_bytes = await _read_video_logo(logo_file, logo_name)
    model = VIDEO_MODEL_DEFAULTS["reference_to_video"]
    try:
        result = generate_video(
            prompt=fp, aspect_ratio=video_size, duration_seconds=dur,
            negative_prompt=effective_negative(negative_prompt), generate_audio=True,
            resolution=resolution, number_of_videos=Variants, seed=None,
            generation_mode="reference_to_video", reference_images=imgs,
            output_gcs_uri=generate_video_output_uri(subject or "reference"), model_id=model,
        )
        final_uris = await _apply_video_logo(result.gcs_uris, logo_bytes, logo_position, logo_size)
        primary, audio_info = await handle_audio(dialogue, Language, audio_file, dialogue, final_uris[0])
        final_uris[0] = primary
        logo_info = {"applied": bool(logo_bytes), "position": logo_position, "size": logo_size} if logo_bytes else {"applied": False}
        return video_response(final_uris, model, "reference_to_video", fp, video_size, resolution,
                              dur, Variants, audio_info,
                              logo_info, [i.filename for i in imgs], result.operation_name)
    except RuntimeError as e:
        msg = str(e)
        code = 422 if any(k in msg for k in ("code 3", "sensitive words", "Responsible AI", "allowlisting", "safety filter")) else 500
        raise HTTPException(status_code=code, detail=msg)
    except Exception as e:
        logger.exception("reference_to_video failed")
        raise HTTPException(status_code=500, detail=f"Video generation failed: {e}")


# ═══════════════════════════════════════════════════════════════════════════
#  VIDEO — Refine
# ═══════════════════════════════════════════════════════════════════════════

@app.post("/refine-video", tags=["Video Generation"],
          summary="Refine / Extend A Generated Video")
async def refine_video_endpoint(
    source_video: UploadFile = File(..., title="Source Video", description="Video to refine or extend."),
    edit_prompt: str = Form(..., title="Edit Instructions", description="What to change or extend."),
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
    try:
        result = generate_video(
            prompt=edit_prompt, aspect_ratio=video_size,
            duration_seconds=7,  # extend_video only supports 7s
            negative_prompt=effective_negative(""), generate_audio=True,
            resolution=resolution, number_of_videos=1, seed=None,
            generation_mode="extend_video",
            source_video_gcs_uri=vid_upload.gcs_uri,
            output_gcs_uri=generate_video_output_uri("refined"), model_id=model,
        )
    except RuntimeError as e:
        msg = str(e)
        code = 422 if any(k in msg for k in ("code 3", "sensitive words", "Responsible AI", "allowlisting", "safety filter")) else 500
        raise HTTPException(status_code=code, detail=msg)
    except Exception as e:
        logger.exception("refine_video failed")
        raise HTTPException(status_code=500, detail=f"Video refinement failed: {e}")

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
    
@app.get("/logos/{name}", tags=["Brand Assets"],
         summary="Serve Logo Image From GCS")
async def get_logo_image(name: str):
    """Proxy-serve a logo image directly from GCS so the frontend can display it."""
    try:
        logo_bytes = download_gcs_logo(name)
        if not logo_bytes:
            raise HTTPException(404, f"Logo '{name}' not found.")
        lower = name.lower()
        if lower.endswith(".png"):
            ct = "image/png"
        elif lower.endswith(".svg"):
            ct = "image/svg+xml"
        elif lower.endswith(".webp"):
            ct = "image/webp"
        else:
            ct = "image/jpeg"
        return Response(content=logo_bytes, media_type=ct,
                        headers={"Cache-Control": "public, max-age=3600"})
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(500, str(e))

@app.get("/content", tags=["System"], summary="Proxy GCS Content (Images & Videos)")
async def serve_gcs_content(gcs_uri: str):
    """Fetch any private GCS object and stream it to the frontend (images, videos)."""
    if not gcs_uri.startswith("gs://"):
        raise HTTPException(400, "Invalid GCS URI — must start with gs://")
    remainder = gcs_uri[5:]
    if "/" not in remainder:
        raise HTTPException(400, "Invalid GCS URI — missing bucket/path separator")
    bucket_name, blob_path = remainder.split("/", 1)
    try:
        data = gcs_client.bucket(bucket_name).blob(blob_path).download_as_bytes()
    except Exception as e:
        raise HTTPException(404, f"GCS object not found: {e}")
    ext = blob_path.rsplit(".", 1)[-1].lower() if "." in blob_path else ""
    ct = {
        "mp4": "video/mp4", "webm": "video/webm",
        "png": "image/png", "jpg": "image/jpeg", "jpeg": "image/jpeg",
        "webp": "image/webp", "svg": "image/svg+xml",
    }.get(ext, "application/octet-stream")
    return Response(content=data, media_type=ct,
                    headers={"Cache-Control": "public, max-age=3600"})



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
        "fonts": FONTS,
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


