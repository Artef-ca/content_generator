"""
Configuration — loads creative presets from GCS bucket (config/*.yaml),
with local filesystem fallback for development.

GCS path: gs://{GCS_BUCKET}/{CONFIG_GCS_PREFIX}/config.yaml
          gs://{GCS_BUCKET}/{CONFIG_GCS_PREFIX}/video_config.yaml

Set CONFIG_GCS_PREFIX env var to change the folder path (default: "config").
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
# Environment / Project  (must be defined before GCS client and config load)
# ---------------------------------------------------------------------------
PROJECT_ID: str = os.getenv("GOOGLE_CLOUD_PROJECT", "mobily-genai")
LOCATION: str = os.getenv("GOOGLE_CLOUD_LOCATION", "global")
BUCKET_NAME: str = os.getenv("GCS_BUCKET", "content_creation_data")
CONFIG_GCS_PREFIX: str = os.getenv("CONFIG_GCS_PREFIX", "config")
PORT: int = int(os.getenv("PORT", "8080"))

# ---------------------------------------------------------------------------
# GCS client  (initialized early — required to load YAML configs from GCS)
# ---------------------------------------------------------------------------
gcs_client = storage.Client(project=PROJECT_ID)

# ---------------------------------------------------------------------------
# Load YAML configs — GCS first, local filesystem fallback for dev
# ---------------------------------------------------------------------------
_PROJECT_ROOT = Path(__file__).parent.parent
_CONFIG_DIR = _PROJECT_ROOT / "config"


def _load_yaml(filename: str) -> dict:
    """Load a YAML config file from GCS, falling back to local filesystem."""
    blob_name = f"{CONFIG_GCS_PREFIX}/{filename}"
    try:
        data = gcs_client.bucket(BUCKET_NAME).blob(blob_name).download_as_text(encoding="utf-8")
        result = yaml.safe_load(data)
        logger.info("Loaded config from GCS: gs://%s/%s", BUCKET_NAME, blob_name)
        return result
    except Exception as e:
        logger.warning(
            "GCS config unavailable (gs://%s/%s: %s) — falling back to local file",
            BUCKET_NAME, blob_name, e,
        )

    path = _CONFIG_DIR / filename
    with open(path, "r", encoding="utf-8") as f:
        result = yaml.safe_load(f)
    logger.info("Loaded config from local: %s", path)
    return result


_yaml = _load_yaml("config.yaml")
_video_yaml = _load_yaml("video_config.yaml")

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
VISUAL_TEXT_FONT: str = DEFAULTS.get("visual_text_font", "Arial")
FONTS: list[dict] = _yaml.get("fonts", [{"value": "Arial", "label": "Arial"}])

# Text block option lists (for Swagger descriptions + /options endpoint)
_tb_opts: dict = _yaml.get("text_block_options", {})
TEXT_SIZES: list[str]     = list(_tb_opts.get("sizes", {}).keys())
TEXT_WEIGHTS: list[str]   = _tb_opts.get("weights", ["light", "regular", "bold", "black"])
TEXT_POSITIONS: list[str] = _tb_opts.get("positions", [])
TEXT_LANGUAGES: dict      = _tb_opts.get("languages", {})

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
# GenAI client
# ---------------------------------------------------------------------------
genai_client = genai.Client(vertexai=True, project=PROJECT_ID, location=LOCATION)
