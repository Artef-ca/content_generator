"""
Image Generation Helpers — prompt building, generation, logo overlay, upload.

Called from app.py endpoints. Keeps the endpoint thin.
"""

import logging
from dataclasses import dataclass
from typing import Optional

from fastapi import UploadFile

from src.config import (
    DEFAULTS,
    ALLOWED_ASPECT_RATIOS,
    ALLOWED_IMAGE_SIZES,
    VISUAL_STYLE_MAP,
    CAMERA_ANGLE_IMG_MAP,
    FRAMING_MAP,
    LIGHTING_MAP,
    TONE_MAP,
    COLOR_GRADING_MAP,
    LOGO_SIZE_MAP,
)
from src.core.prompt_generator import PromptInputs, build_prompt
from src.helpers.utils import (
    generate_image,
    generate_output_path,
    upload_to_gcs,
    overlay_logo,
    POSITION_MAP,
)

logger = logging.getLogger("creative_studio")


# ---------------------------------------------------------------------------
# Data
# ---------------------------------------------------------------------------
@dataclass
class ImageGenerationResult:
    """Everything needed for the JSON response."""
    gcs_uri: str
    public_url: str
    prompt_used: str
    model_commentary: str
    logo_info: dict


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------
def validate_image_params(
    aspect_ratio: str,
    image_size: str,
    visual_style: str,
    camera_angle: str,
    framing: str,
    lighting: str,
    tone: str,
    color_grading: str,
    logo_position: str,
) -> list[str]:
    """Return list of validation errors (empty = valid)."""
    errors = []
    if aspect_ratio not in ALLOWED_ASPECT_RATIOS:
        errors.append(f"aspect_ratio must be one of {ALLOWED_ASPECT_RATIOS}")
    if image_size not in ALLOWED_IMAGE_SIZES:
        errors.append(f"image_size must be one of {ALLOWED_IMAGE_SIZES}")
    if visual_style and visual_style not in VISUAL_STYLE_MAP:
        errors.append(f"visual_style must be one of {list(VISUAL_STYLE_MAP.keys())}")
    if camera_angle and camera_angle not in CAMERA_ANGLE_IMG_MAP:
        errors.append(f"camera_angle must be one of {list(CAMERA_ANGLE_IMG_MAP.keys())}")
    if framing and framing not in FRAMING_MAP:
        errors.append(f"framing must be one of {list(FRAMING_MAP.keys())}")
    if lighting and lighting not in LIGHTING_MAP:
        errors.append(f"lighting must be one of {list(LIGHTING_MAP.keys())}")
    if tone and tone not in TONE_MAP:
        errors.append(f"tone must be one of {list(TONE_MAP.keys())}")
    if color_grading and color_grading not in COLOR_GRADING_MAP:
        errors.append(f"color_grading must be one of {list(COLOR_GRADING_MAP.keys())}")
    if logo_position and logo_position not in POSITION_MAP:
        errors.append(f"logo_position must be one of {list(POSITION_MAP.keys())}")
    return errors


# ---------------------------------------------------------------------------
# Logo reading
# ---------------------------------------------------------------------------
async def read_logo(logo_file: Optional[UploadFile]) -> tuple[bool, bytes | None]:
    """Read logo upload. Returns (has_logo, logo_bytes)."""
    if not logo_file or not isinstance(logo_file, UploadFile) or not logo_file.filename:
        return False, None
    data = await logo_file.read()
    if not data or len(data) == 0:
        return False, None
    logger.info(
        "Logo received: %s (%s, %d bytes)",
        logo_file.filename, logo_file.content_type or "image/png", len(data),
    )
    return True, data


# ---------------------------------------------------------------------------
# Full pipeline: prompt → generate → overlay logo → upload
# ---------------------------------------------------------------------------
async def generate_and_process_image(
    # Core fields
    subject: str,
    action: str = "",
    setting: str = "",
    items_in_scene: str = "",
    # Creative controls
    visual_style: str = "",
    camera_angle: str = "",
    framing: str = "",
    lighting: str = "",
    tone: str = "",
    color_grading: str = "",
    # Custom
    custom_prompt: str = "",
    # Brand
    has_logo: bool = False,
    logo_bytes: bytes | None = None,
    logo_position: str = "bottom_right",
    logo_scale: float = 0.15,
    logo_opacity: float = 1.0,
    campaign_text: str = "",
    text_style: str = "",
    website_url: str = "",
    # Constraints
    negative_prompt: str = "",
    # Output
    aspect_ratio: str = "16:9",
    image_size: str = "1K",
) -> ImageGenerationResult:
    """
    Full image generation pipeline:
      1. Build prompt (with brand context)
      2. Call Gemini (with logo as reference image)
      3. Overlay logo with Pillow (guaranteed placement)
      4. Upload to GCS
    """

    # ── 1. Build prompt ──────────────────────────────────────────────────
    inputs = PromptInputs(
        subject=subject,
        action=action,
        setting=setting,
        items_in_scene=items_in_scene,
        visual_style=visual_style,
        camera_angle=camera_angle,
        framing=framing,
        lighting=lighting,
        tone=tone,
        color_grading=color_grading,
        logo_size="medium" if has_logo else "none",
        campaign_text=campaign_text or None,
        text_style=text_style or None,
        website_url=website_url or None,
        custom_prompt=custom_prompt,
        negative_prompt=negative_prompt or None,
    )
    final_prompt = build_prompt(inputs, has_logo=has_logo)
    logger.info("Image prompt: %s", final_prompt[:500])

    # ── 2. Generate with Gemini ──────────────────────────────────────────
    # Send logo as reference image so Gemini is aware of it
    reference_images = None
    if has_logo and logo_bytes:
        reference_images = [(logo_bytes, "image/png")]

    result = generate_image(
        prompt=final_prompt,
        aspect_ratio=aspect_ratio,
        image_size=image_size,
        reference_images=reference_images,
    )

    # ── 3. Overlay logo with Pillow ──────────────────────────────────────
    final_image = result.image_bytes
    logo_info: dict = {"applied": False}

    if has_logo and logo_bytes:
        try:
            final_image = overlay_logo(
                image_bytes=result.image_bytes,
                logo_bytes=logo_bytes,
                position=logo_position,
                logo_scale=logo_scale,
                opacity=logo_opacity,
            )
            logo_info = {
                "applied": True,
                "position": logo_position,
                "scale": logo_scale,
                "opacity": logo_opacity,
            }
            logger.info("Logo overlay SUCCESS: position=%s, scale=%.2f", logo_position, logo_scale)
        except Exception as e:
            logger.exception("Logo overlay FAILED — returning image without logo")
            logo_info = {"applied": False, "error": str(e)}

    # ── 4. Upload to GCS ─────────────────────────────────────────────────
    output_path = generate_output_path(subject=subject, prefix="banners")
    gcs_result = upload_to_gcs(final_image, output_path)

    return ImageGenerationResult(
        gcs_uri=gcs_result.gcs_uri,
        public_url=gcs_result.public_url,
        prompt_used=final_prompt,
        model_commentary=result.model_text,
        logo_info=logo_info,
    )
