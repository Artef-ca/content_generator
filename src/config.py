"""
Configuration — loads creative presets from config/*.yaml,
manages environment variables and SDK client initialisation.
"""

import os
import logging
from pathlib import Path

import yaml
from google import genai
from google.cloud import storage

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
)
logger = logging.getLogger("creative_studio")

# ---------------------------------------------------------------------------
# Load YAML configs
# ---------------------------------------------------------------------------
_PROJECT_ROOT = Path(__file__).parent.parent
_CONFIG_DIR = _PROJECT_ROOT / "config"


def _load_yaml(filename: str) -> dict:
    path = _CONFIG_DIR / filename
    with open(path, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f)
    logger.info("Loaded config from %s", path)
    return data


_yaml = _load_yaml("config.yaml")
_video_yaml = _load_yaml("video_config.yaml")

# ---------------------------------------------------------------------------
# Environment / Project
# ---------------------------------------------------------------------------
PROJECT_ID: str = os.getenv("GOOGLE_CLOUD_PROJECT", "mobily-genai")
LOCATION: str = os.getenv("GOOGLE_CLOUD_LOCATION", "global")
BUCKET_NAME: str = os.getenv("GCS_BUCKET", "content_creation_data")
PORT: int = int(os.getenv("PORT", "8080"))

# ═══════════════════════════════════════════════════════════════════════════
# SHARED
# ═══════════════════════════════════════════════════════════════════════════
GCS_PREFIXES: dict[str, str] = _yaml.get("gcs_prefixes", {})
FIELD_LABELS: dict[str, str] = _yaml.get("field_labels", {})


def label(field_name: str) -> str:
    """Get Swagger display label for a field (Title Case fallback)."""
    return FIELD_LABELS.get(field_name, field_name.replace("_", " ").title())


# ═══════════════════════════════════════════════════════════════════════════
# IMAGE CONFIG
# ═══════════════════════════════════════════════════════════════════════════
MODEL_ID: str = _yaml["model"]["id"]
DEFAULTS: dict = _yaml["defaults"]
ALLOWED_ASPECT_RATIOS: list[str] = _yaml["allowed_aspect_ratios"]
ALLOWED_IMAGE_SIZES: list[str] = _yaml["allowed_image_sizes"]
VISUAL_STYLE_MAP: dict[str, str] = _yaml["visual_styles"]
CAMERA_ANGLE_IMG_MAP: dict[str, str] = _yaml.get("camera_angles", {})
FRAMING_MAP: dict[str, str] = _yaml["framings"]
LIGHTING_MAP: dict[str, str] = _yaml["lightings"]
TONE_MAP: dict[str, str] = _yaml["tones"]
COLOR_GRADING_MAP: dict[str, str] = _yaml["color_gradings"]

# Mobily brand palette (full details for frontend dropdowns)
MOBILY_PALETTE: dict = _yaml.get("mobily_palette", {})

# ═══════════════════════════════════════════════════════════════════════════
# VIDEO CONFIG
# ═══════════════════════════════════════════════════════════════════════════
VIDEO_DEFAULTS: dict = _video_yaml["defaults"]
BRAND_CONFIG: dict = _video_yaml.get("brand", {})

VIDEO_MODEL_DEFAULTS: dict[str, str] = {
    "text_to_video": _video_yaml["model"]["text_to_video"],
    "image_to_video": _video_yaml["model"]["image_to_video"],
    "image_to_video_endframe": _video_yaml["model"]["image_to_video_endframe"],
    "reference_to_video": _video_yaml["model"]["reference_to_video"],
    "refine_video": _video_yaml["model"]["refine_video"],
}

CAMERA_ANGLE_MAP: dict[str, str] = _video_yaml["camera_angles"]
CAMERA_MOVEMENT_MAP: dict[str, str] = _video_yaml["camera_movements"]
LENS_EFFECT_MAP: dict[str, str] = _video_yaml["lens_effects"]
VIDEO_STYLE_MAP: dict[str, str] = _video_yaml["visual_styles"]
VIDEO_TONE_MAP: dict[str, str] = _video_yaml.get("tones", {})
TEMPORAL_MAP: dict[str, str] = _video_yaml["temporal_elements"]
SOUND_AMBIENCE_MAP: dict[str, str] = _video_yaml["sound_ambiences"]

VIDEO_ALLOWED_ASPECT_RATIOS: list[str] = _video_yaml["allowed_aspect_ratios"]
ALLOWED_DURATIONS: list[int] = _video_yaml["allowed_durations"]
ALLOWED_RESOLUTIONS: list[str] = _video_yaml["allowed_resolutions"]
ALLOWED_TTS_LANGUAGES: list[str] = _video_yaml["allowed_tts_languages"]

# ---------------------------------------------------------------------------
# SDK Clients
# ---------------------------------------------------------------------------
genai_client = genai.Client(vertexai=True, project=PROJECT_ID, location=LOCATION)
gcs_client = storage.Client(project=PROJECT_ID)
