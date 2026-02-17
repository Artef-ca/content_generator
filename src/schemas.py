"""
Pydantic schemas for Video Generation API v3.0.

Matches the Prompt Generator UI:
  - Text-to-Video:  Subject, Action, Scene, 6 dropdowns, Dialogue (TTS only)
  - Image-to-Video: Upload Image, Action, Scene, 6 dropdowns, Dialogue (lip-sync)
"""

from __future__ import annotations

from enum import Enum
from typing import Optional

from pydantic import BaseModel, Field

from src.config import VIDEO_DEFAULTS


# ═══════════════════════════════════════════════════════════════════════════
# Enums — dropdown values (UI: "-- None --" = empty string in API)
# ═══════════════════════════════════════════════════════════════════════════

class CameraAngle(str, Enum):
    eye_level = "eye_level"
    low_angle = "low_angle"
    high_angle = "high_angle"
    birds_eye = "birds_eye"
    dutch_angle = "dutch_angle"
    close_up = "close_up"
    extreme_close_up = "extreme_close_up"
    wide_shot = "wide_shot"
    medium_shot = "medium_shot"
    over_the_shoulder = "over_the_shoulder"


class CameraMovement(str, Enum):
    static = "static"
    pan = "pan"
    tilt = "tilt"
    dolly_in = "dolly_in"
    dolly_out = "dolly_out"
    tracking = "tracking"
    crane = "crane"
    handheld = "handheld"
    zoom_in = "zoom_in"
    zoom_out = "zoom_out"
    orbit = "orbit"


class LensEffect(str, Enum):
    wide_angle_24mm = "wide_angle_24mm"
    standard_50mm = "standard_50mm"
    telephoto_85mm = "telephoto_85mm"
    macro = "macro"
    fish_eye = "fish_eye"
    tilt_shift = "tilt_shift"
    anamorphic = "anamorphic"
    shallow_dof = "shallow_dof"
    bokeh = "bokeh"
    lens_flare = "lens_flare"


class VisualStyle(str, Enum):
    photorealistic = "photorealistic"
    cinematic = "cinematic"
    anime = "anime"
    watercolor = "watercolor"
    retro = "retro"
    neon = "neon"
    three_d_render = "3d_render"
    stop_motion = "stop_motion"
    minimalist = "minimalist"


class TemporalElements(str, Enum):
    slow_motion = "slow_motion"
    fast_paced = "fast_paced"
    time_lapse = "time_lapse"
    normal = "normal"
    reverse = "reverse"
    freeze_frame = "freeze_frame"
    loop = "loop"


class SoundAmbience(str, Enum):
    none = "none"
    office_hum = "office_hum"
    nature = "nature"
    city = "city"
    crowd = "crowd"
    festive = "festive"
    corporate = "corporate"
    dramatic = "dramatic"


class AudioMode(str, Enum):
    none = "none"
    tts = "tts"
    upload = "upload"


class AspectRatio(str, Enum):
    landscape = "16:9"
    portrait = "9:16"


class Resolution(str, Enum):
    r720p = "720p"
    r1080p = "1080p"
    r4k = "4k"


class VeoVariant(str, Enum):
    standard = "standard"
    fast = "fast"
    preview = "preview"
    fast_preview = "fast_preview"


class VeoVariantRef(str, Enum):
    preview = "preview"
    fast_preview = "fast_preview"


# ═══════════════════════════════════════════════════════════════════════════
# Request Schemas (documentation / validation reference)
# ═══════════════════════════════════════════════════════════════════════════

class CinematicsFields(BaseModel):
    """6 dropdown fields matching the UI."""
    camera_angle: str = Field("", description="Camera angle (empty = default).")
    camera_movement: str = Field("", description="Camera movement (empty = default).")
    lens_effect: str = Field("", description="Lens & optical effect (empty = default).")
    visual_style: str = Field("", description="Visual style (empty = default).")
    temporal_elements: str = Field("", description="Temporal pacing (empty = default).")
    sound_ambience: str = Field("", description="Sound ambience (empty = default).")


class AudioFields(BaseModel):
    """Audio overlay settings."""
    audio_mode: AudioMode = Field(AudioMode.none, description="Audio source: none | tts | upload")
    tts_text: str = Field("", description="Script for TTS (falls back to dialogue if empty).")
    tts_language: str = Field("ar-XA", description="TTS language code.")


class OutputFields(BaseModel):
    """Video output controls."""
    duration_seconds: int = Field(8, description="Video length: 4, 6, or 8 seconds.")
    aspect_ratio: AspectRatio = Field(AspectRatio.landscape, description="Video format.")
    resolution: Resolution = Field(Resolution.r720p, description="Output resolution.")
    number_of_videos: int = Field(1, ge=1, le=4, description="Number of variants (1–4).")
    seed: Optional[int] = Field(None, ge=0, le=4294967295, description="Reproducibility seed.")


class TextToVideoRequest(CinematicsFields, AudioFields, OutputFields):
    """
    Text-to-Video — matches UI Text-to-Video tab.

    Fields: Subject, Action, Scene/Context, 6 dropdowns, Dialogue.
    Dialogue is for TTS voiceover ONLY — not injected into Veo prompt.
    Prompt is optional — auto-built from subject + action + scene.
    Brand style context is injected automatically.
    """
    subject: str = Field(..., description="Main subject. E.g. 'A dog'")
    action: str = Field("", description="What the subject is doing. E.g. 'running'")
    scene_context: str = Field("", description="Environment. E.g. 'on a sunny beach'")
    dialogue: str = Field("", description="Voiceover script — for TTS audio only.")
    prompt: str = Field("", description="Optional extra scene description.")
    negative_prompt: str = Field("", description="What to avoid (brand defaults applied automatically).")
    veo_variant: VeoVariant = Field(VeoVariant.standard, description="Model variant.")

    class Config:
        json_schema_extra = {
            "example": {
                "subject": "a man in a tailored suit",
                "action": "speaking warmly and gesturing gently",
                "scene_context": "in a modern glass office with festive Ramadan lanterns",
                "camera_angle": "medium_shot",
                "visual_style": "cinematic",
            }
        }


class ImageToVideoRequest(CinematicsFields, AudioFields, OutputFields):
    """
    Image-to-Video — matches UI Image-to-Video tab.

    Fields: Upload Image, Action, Scene/Context, 6 dropdowns, Dialogue.
    Dialogue IS injected into Veo for character lip-sync.
    end_frame_image: optional ending image used as-is as video's last frame.
    """
    action: str = Field("", description="What happens. E.g. 'snow falling gently'")
    scene_context: str = Field("", description="Context. E.g. 'steam rising from a coffee cup'")
    dialogue: str = Field("", description="Character lip-sync text.")
    prompt: str = Field("", description="Optional extra scene description.")
    negative_prompt: str = Field("", description="What to avoid.")
    veo_variant: VeoVariant = Field(VeoVariant.standard, description="Model variant.")

    class Config:
        json_schema_extra = {
            "example": {
                "action": "snow falling gently",
                "scene_context": "warm cozy interior with window view",
                "visual_style": "cinematic",
            }
        }


class ReferenceToVideoRequest(CinematicsFields, AudioFields):
    """
    Reference-to-Video — identity preservation (Veo 3.1 only).

    Duration always 8 seconds. Only preview / fast_preview variants.
    Dialogue supported for lip-sync.
    """
    subject: str = Field(..., description="Subject description matching the images.")
    action: str = Field("", description="What the subject is doing.")
    scene_context: str = Field("", description="Environment.")
    dialogue: str = Field("", description="Lip-sync text.")
    prompt: str = Field("", description="Optional extra description.")
    negative_prompt: str = Field("", description="What to avoid.")
    aspect_ratio: AspectRatio = Field(AspectRatio.landscape)
    resolution: Resolution = Field(Resolution.r720p)
    number_of_videos: int = Field(1, ge=1, le=4)
    seed: Optional[int] = Field(None, ge=0, le=4294967295)
    veo_variant: VeoVariantRef = Field(VeoVariantRef.preview)


# ═══════════════════════════════════════════════════════════════════════════
# Response Schemas
# ═══════════════════════════════════════════════════════════════════════════

class AudioInfo(BaseModel):
    """Audio processing result."""
    mode: str = Field(..., description="Audio mode used: none | tts | upload")
    merged: Optional[bool] = Field(None, description="Whether audio was merged onto video.")
    tts_language: Optional[str] = Field(None, description="TTS language if used.")
    tts_chars: Optional[int] = Field(None, description="Characters synthesised.")
    uploaded_file: Optional[str] = Field(None, description="Uploaded audio filename.")
    detail: Optional[str] = Field(None, description="Additional info or warning.")
    error: Optional[str] = Field(None, description="Error message if audio processing failed.")

    class Config:
        extra = "allow"


class EndFrameInfo(BaseModel):
    """End frame result."""
    applied: bool = Field(..., description="Whether an end frame image was used.")
    filename: Optional[str] = Field(None, description="End frame filename.")
    auto_upgraded_model: Optional[str] = Field(None, description="Model upgrade if end frame triggered Veo 3.1.")
    error: Optional[str] = Field(None, description="Error if upload failed.")

    class Config:
        extra = "allow"


class VideoGenerationResponse(BaseModel):
    """Standard response for all video generation endpoints."""
    status: str = Field("success", description="Request status.")
    gcs_uri: str = Field(..., description="GCS URI of the generated video.")
    public_url: str = Field(..., description="Public HTTPS URL for the video.")
    model: str = Field(..., description="Veo model ID used.")
    generation_mode: str = Field(..., description="text_to_video | image_to_video | reference_to_video")
    prompt_used: str = Field(..., description="Final assembled prompt sent to Veo (includes brand context).")
    aspect_ratio: str = Field(...)
    resolution: str = Field(...)
    duration_seconds: int = Field(...)
    number_of_videos: int = Field(...)
    seed: Optional[int] = Field(None)
    audio: AudioInfo = Field(...)
    end_frame: EndFrameInfo = Field(...)
    reference_images: list[str] = Field(default_factory=list)
    operation: str = Field("")

    class Config:
        json_schema_extra = {
            "example": {
                "status": "success",
                "gcs_uri": "gs://content_creation_data/videos/2026-02-17/family-scene_a1b2/sample_0.mp4",
                "public_url": "https://storage.googleapis.com/content_creation_data/videos/...",
                "model": "veo-3.0-generate-001",
                "generation_mode": "text_to_video",
                "prompt_used": "Photorealistic. A family sitting together...",
                "aspect_ratio": "16:9",
                "resolution": "720p",
                "duration_seconds": 8,
                "number_of_videos": 1,
                "seed": None,
                "audio": {"mode": "none"},
                "end_frame": {"applied": False},
                "reference_images": [],
                "operation": "projects/mobily-genai/..."
            }
        }


class HealthResponse(BaseModel):
    status: str
    version: str
    endpoints: list[str]
