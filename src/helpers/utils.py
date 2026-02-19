"""
Utilities — GCS helpers, Gemini API wrapper, and response parsing.
"""

import re
import io
import uuid
import logging
from datetime import datetime, timezone, timedelta

# Saudi Arabia timezone (UTC+3)
_SAUDI_TZ = timezone(timedelta(hours=3))
from dataclasses import dataclass

from PIL import Image
from google.genai import types

from src.config import (
    BUCKET_NAME,
    MODEL_ID,
    BRAND_CONFIG,
    genai_client,
    gcs_client,
)

logger = logging.getLogger("creative_studio")

# GCS path for brand logos
_LOGO_GCS_PREFIX = BRAND_CONFIG.get("assets_gcs_prefix", "gs://content_creation_data/brand_assets/mobily/") + "logos/"
# Strip gs://bucket/ to get the blob prefix
_LOGO_BLOB_PREFIX = _LOGO_GCS_PREFIX.replace(f"gs://{BUCKET_NAME}/", "")


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------
@dataclass
class GenerationResult:
    """Structured output from an image generation call."""

    image_bytes: bytes
    model_text: str


@dataclass
class GCSUploadResult:
    """Paths returned after a successful GCS upload."""

    gcs_uri: str
    public_url: str


# ---------------------------------------------------------------------------
# GCS
# ---------------------------------------------------------------------------
def upload_to_gcs(
    image_bytes: bytes,
    filename: str,
    bucket_name: str = BUCKET_NAME,
) -> GCSUploadResult:
    """Upload raw image bytes to Google Cloud Storage."""
    bucket = gcs_client.bucket(bucket_name)
    blob = bucket.blob(filename)
    blob.upload_from_string(image_bytes, content_type="image/png")

    return GCSUploadResult(
        gcs_uri=f"gs://{bucket_name}/{filename}",
        public_url=f"https://storage.googleapis.com/{bucket_name}/{filename}",
    )


def list_gcs_logos() -> list[dict]:
    """List all logo files in the brand assets GCS folder.

    Returns list of {name, public_url} for each logo.
    """
    bucket = gcs_client.bucket(BUCKET_NAME)
    blobs = bucket.list_blobs(prefix=_LOGO_BLOB_PREFIX)
    logos = []
    for blob in blobs:
        # Skip "directory" markers
        if blob.name.endswith("/"):
            continue
        fname = blob.name.split("/")[-1]
        logos.append({
            "name": fname,
            "public_url": f"https://storage.googleapis.com/{BUCKET_NAME}/{blob.name}",
        })
    logger.info("Listed %d logos from %s", len(logos), _LOGO_BLOB_PREFIX)
    return logos


def download_gcs_logo(logo_name: str) -> bytes | None:
    """Download a logo by name from the brand assets GCS folder.

    Parameters
    ----------
    logo_name : str
        Filename of the logo (e.g. 'mobily_logo_white.png').

    Returns
    -------
    bytes or None
    """
    blob_path = f"{_LOGO_BLOB_PREFIX}{logo_name}"
    bucket = gcs_client.bucket(BUCKET_NAME)
    blob = bucket.blob(blob_path)
    if not blob.exists():
        logger.warning("Logo not found in GCS: %s", blob_path)
        return None
    data = blob.download_as_bytes()
    logger.info("Downloaded logo from GCS: %s (%d bytes)", blob_path, len(data))
    return data


def _slugify(text: str, max_words: int = 5, max_len: int = 50) -> str:
    """Turn free-text into a filename-safe slug.

    Examples
    --------
    >>> _slugify("A family enjoying an Iftar feast")
    'family-enjoying-iftar-feast'
    >>> _slugify("5G Speed!! Launch 🚀")
    '5g-speed-launch'
    """
    text = text.lower()
    text = re.sub(r"[^\w\s-]", "", text)          # strip non-word chars
    text = re.sub(r"[\s_]+", "-", text).strip("-") # spaces → hyphens
    # drop filler words
    filler = {"a", "an", "the", "of", "in", "on", "at", "for", "and", "with", "to"}
    words = [w for w in text.split("-") if w and w not in filler]
    slug = "-".join(words[:max_words])
    return slug[:max_len]


def generate_output_path(subject: str, prefix: str = "banners") -> str:
    """Create a descriptive, unique GCS object path.

    Format: banners/YYYY-MM-DD/<slug>_<short-uuid>.png

    Examples
    --------
    >>> generate_output_path("A family enjoying an Iftar feast")
    'banners/2026-02-15/family-enjoying-iftar-feast_a3b1c9d2.png'
    """
    date_str = datetime.now(_SAUDI_TZ).strftime("%Y-%m-%d")
    slug = _slugify(subject) or "banner"
    short_id = uuid.uuid4().hex[:8]
    return f"{prefix}/{date_str}/{slug}_{short_id}.png"


# ---------------------------------------------------------------------------
# Gemini API
# ---------------------------------------------------------------------------
def generate_image(
    prompt: str,
    aspect_ratio: str,
    image_size: str,
    reference_images: list[tuple[bytes, str]] | None = None,
) -> GenerationResult:
    """Call Gemini 3 Pro Image and return the generated image + model text.

    Parameters
    ----------
    prompt : str
        Fully composed prompt from prompt_generator.
    aspect_ratio : str
        e.g. "16:9", "9:16", "1:1".
    image_size : str
        "1K" | "2K" | "4K".
    reference_images : list of (bytes, mime_type) tuples, optional
        Logo, product shots, or style references to send alongside
        the text prompt. Nano Banana Pro supports up to 14 reference inputs.

    Returns
    -------
    GenerationResult

    Raises
    ------
    RuntimeError
        If the model returns no image.
    """
    # Build content parts: images first, then text (Google recommends
    # placing images before text for best results).
    contents: list[types.Part] = []

    if reference_images:
        for img_bytes, mime in reference_images:
            contents.append(types.Part.from_bytes(data=img_bytes, mime_type=mime))

    contents.append(types.Part.from_text(text=prompt))

    response = genai_client.models.generate_content(
        model=MODEL_ID,
        contents=contents,
        config=types.GenerateContentConfig(
            # IMPORTANT: Must include BOTH "TEXT" and "IMAGE".
            # Image-only output is NOT supported for Gemini image models.
            response_modalities=["TEXT", "IMAGE"],
            image_config=types.ImageConfig(
                aspect_ratio=aspect_ratio,
                image_size=image_size,
            ),
        ),
    )

    # --- Parse response parts ---
    image_data: bytes | None = None
    model_text: str = ""

    for part in response.candidates[0].content.parts:
        if part.text:
            model_text = part.text
            logger.info("Model commentary: %s", model_text)
        elif part.inline_data:
            image_data = part.inline_data.data

    if image_data is None:
        raise RuntimeError(
            f"No image returned by the model. Model response: {model_text}"
        )

    return GenerationResult(image_bytes=image_data, model_text=model_text)


# ---------------------------------------------------------------------------
# Logo Overlay (Pillow post-processing)
# ---------------------------------------------------------------------------
POSITION_MAP = {
    "top_left":      (0.0, 0.0),
    "top_center":    (0.5, 0.0),
    "top_right":     (1.0, 0.0),
    "center_left":   (0.0, 0.5),
    "center":        (0.5, 0.5),
    "center_right":  (1.0, 0.5),
    "bottom_left":   (0.0, 1.0),
    "bottom_center": (0.5, 1.0),
    "bottom_right":  (1.0, 1.0),
}

# Logo width as fraction of banner width
LOGO_SIZE_SCALE = {
    "small":  0.08,
    "medium": 0.15,
    "large":  0.25,
}

_MARGIN_FRAC = 0.03


def overlay_logo(
    image_bytes: bytes,
    logo_bytes: bytes,
    position: str = "bottom_right",
    size: str = "medium",
) -> bytes:
    """Composite a logo onto a generated image using Pillow.

    Parameters
    ----------
    image_bytes : bytes
        The generated banner image (PNG/JPEG).
    logo_bytes : bytes
        The logo image (PNG with transparency recommended).
    position : str
        Placement key — one of POSITION_MAP keys.
    size : str
        Logo size: small (8%), medium (15%), large (25%) of banner width.

    Returns
    -------
    bytes
        Composited image as PNG.
    """
    banner = Image.open(io.BytesIO(image_bytes)).convert("RGBA")
    logo = Image.open(io.BytesIO(logo_bytes)).convert("RGBA")

    bw, bh = banner.size

    # Scale logo based on size preset
    scale = LOGO_SIZE_SCALE.get(size, LOGO_SIZE_SCALE["medium"])
    target_w = int(bw * scale)
    ratio = target_w / logo.width
    target_h = int(logo.height * ratio)

    # Cap height at 40% of banner
    if target_h > bh * 0.4:
        target_h = int(bh * 0.4)
        ratio = target_h / logo.height
        target_w = int(logo.width * ratio)

    logo_resized = logo.resize((target_w, target_h), Image.LANCZOS)

    # Calculate position with padding
    pad = int(bw * _MARGIN_FRAC)
    ax, ay = POSITION_MAP.get(position, POSITION_MAP["bottom_right"])

    x = pad if ax == 0.0 else (bw - target_w - pad if ax == 1.0 else (bw - target_w) // 2)
    y = pad if ay == 0.0 else (bh - target_h - pad if ay == 1.0 else (bh - target_h) // 2)

    # Composite
    banner.paste(logo_resized, (x, y), logo_resized)

    output = io.BytesIO()
    banner.convert("RGB").save(output, format="PNG", quality=95)
    output.seek(0)

    logger.info(
        "Logo overlaid: pos=%s, size=%s (%.0f%%), logo=%dx%d → %dx%d on %dx%d",
        position, size, scale * 100, logo.width, logo.height, target_w, target_h, bw, bh,
    )
    return output.read()
