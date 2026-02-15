"""
Configuration — loads creative presets from config/*.yaml,
manages environment variables and SDK client initialisation.

Split logic:
  config/config.yaml        → banner presets, allowed values, defaults
  config/video_config.yaml  → video presets, camera, lens, pacing, audio
  src/config.py             → env vars, logging, SDK clients, YAML loading
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
# Load YAML configs  (project_root/config/*.yaml)
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
# Environment / Project Settings  (infra — stays in Python)
# ---------------------------------------------------------------------------
PROJECT_ID: str = os.getenv("GOOGLE_CLOUD_PROJECT", "mobily-genai")
LOCATION: str = os.getenv("GOOGLE_CLOUD_LOCATION", "global")
BUCKET_NAME: str = os.getenv("GCS_BUCKET", "content_creation_data")
PORT: int = int(os.getenv("PORT", "8080"))

# ═══════════════════════════════════════════════════════════════════════════
# BANNER CONFIG  (from config.yaml)
# ═══════════════════════════════════════════════════════════════════════════
MODEL_ID: str = _yaml["model"]["id"]
DEFAULTS: dict = _yaml["defaults"]
ALLOWED_ASPECT_RATIOS: list[str] = _yaml["allowed_aspect_ratios"]
ALLOWED_IMAGE_SIZES: list[str] = _yaml["allowed_image_sizes"]
VISUAL_STYLE_MAP: dict[str, str] = _yaml["visual_styles"]
LOGO_SIZE_MAP: dict[str, str] = _yaml["logo_sizes"]
COMPOSITION_MAP: dict[str, str] = _yaml["compositions"]
LIGHTING_MAP: dict[str, str] = _yaml["lightings"]

# ═══════════════════════════════════════════════════════════════════════════
# VIDEO CONFIG  (from video_config.yaml)
# ═══════════════════════════════════════════════════════════════════════════
VIDEO_MODEL_ID: str = _video_yaml["model"]["id"]
VIDEO_FAST_MODEL_ID: str = _video_yaml["model"]["fast_id"]
VIDEO_PREVIEW_MODEL_ID: str = _video_yaml["model"]["preview_id"]
VIDEO_DEFAULTS: dict = _video_yaml["defaults"]
DEFAULT_BRAND: dict = _video_yaml["default_brand"]

# Preset maps
CAMERA_ANGLE_MAP: dict[str, str] = _video_yaml["camera_angles"]
CAMERA_MOVEMENT_MAP: dict[str, str] = _video_yaml["camera_movements"]
LENS_EFFECT_MAP: dict[str, str] = _video_yaml["lens_effects"]
VIDEO_STYLE_MAP: dict[str, str] = _video_yaml["visual_styles"]
TEMPORAL_MAP: dict[str, str] = _video_yaml["temporal_elements"]
SOUND_AMBIENCE_MAP: dict[str, str] = _video_yaml["sound_ambiences"]

# Allowed values
VIDEO_ALLOWED_ASPECT_RATIOS: list[str] = _video_yaml["allowed_aspect_ratios"]
ALLOWED_DURATIONS: list[int] = _video_yaml["allowed_durations"]
ALLOWED_AUDIO_MODES: list[str] = _video_yaml["allowed_audio_modes"]
ALLOWED_TTS_LANGUAGES: list[str] = _video_yaml["allowed_tts_languages"]

# ---------------------------------------------------------------------------
# SDK Clients  (created once at import time, reused across requests)
# ---------------------------------------------------------------------------
genai_client = genai.Client(
    vertexai=True,
    project=PROJECT_ID,
    location=LOCATION,
)

gcs_client = storage.Client(project=PROJECT_ID)
