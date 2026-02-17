"""
Video Utilities — Veo API wrapper, Google Cloud TTS, audio merge, GCS helpers.

Veo is a long-running operation API:
  1. Submit generation request → get operation ID
  2. Poll operation until done
  3. Download result from GCS or response bytes
"""

import io
import re
import time
import uuid
import logging
import tempfile
import subprocess
from datetime import datetime, timezone
from dataclasses import dataclass

from google.genai import types
from google.cloud import texttospeech

from src.config import (
    BUCKET_NAME,
    VIDEO_MODEL_ID,
    genai_client,
    gcs_client,
)

logger = logging.getLogger("creative_studio")


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------
@dataclass
class VideoGenerationResult:
    """Structured output from a video generation call."""
    gcs_uri: str
    operation_name: str


@dataclass
class TTSResult:
    """Output from a TTS synthesis call."""
    audio_bytes: bytes
    mime_type: str


@dataclass
class GCSUploadResult:
    """Paths returned after a successful GCS upload."""
    gcs_uri: str
    public_url: str


# ---------------------------------------------------------------------------
# Slug / Smart Naming
# ---------------------------------------------------------------------------
def _slugify(text: str, max_words: int = 5, max_len: int = 50) -> str:
    """Turn free-text into a filename-safe slug."""
    text = text.lower()
    text = re.sub(r"[^\w\s-]", "", text)
    text = re.sub(r"[\s_]+", "-", text).strip("-")
    filler = {"a", "an", "the", "of", "in", "on", "at", "for", "and", "with", "to"}
    words = [w for w in text.split("-") if w and w not in filler]
    slug = "-".join(words[:max_words])
    return slug[:max_len]


def generate_video_output_uri(subject_hint: str, prefix: str = "videos") -> str:
    """Create a descriptive GCS output URI prefix.

    Veo writes output files into this prefix, appending its own filenames.

    Format: gs://bucket/videos/YYYY-MM-DD/<slug>_<uuid>/
    """
    date_str = datetime.now(timezone.utc).strftime("%Y-%m-%d")
    slug = _slugify(subject_hint) or "video"
    short_id = uuid.uuid4().hex[:8]
    return f"gs://{BUCKET_NAME}/{prefix}/{date_str}/{slug}_{short_id}/"


# ---------------------------------------------------------------------------
# GCS Upload (for audio files, source images, etc.)
# ---------------------------------------------------------------------------
def upload_bytes_to_gcs(
    data: bytes,
    filename: str,
    content_type: str = "application/octet-stream",
    bucket_name: str = BUCKET_NAME,
) -> GCSUploadResult:
    """Upload raw bytes to GCS and return URIs."""
    bucket = gcs_client.bucket(bucket_name)
    blob = bucket.blob(filename)
    blob.upload_from_string(data, content_type=content_type)
    return GCSUploadResult(
        gcs_uri=f"gs://{bucket_name}/{filename}",
        public_url=f"https://storage.googleapis.com/{bucket_name}/{filename}",
    )


def download_from_gcs(gcs_uri: str) -> bytes:
    """Download bytes from a gs:// URI.

    If the URI is a prefix (directory), finds the first .mp4 blob inside it.
    """
    parts = gcs_uri.replace("gs://", "").split("/", 1)
    bucket_name, blob_path = parts[0], parts[1]
    bucket = gcs_client.bucket(bucket_name)

    # Check if it's an actual file or a prefix
    blob = bucket.blob(blob_path)
    if blob.exists():
        return blob.download_as_bytes()

    # It's a prefix — list and find the .mp4
    prefix = blob_path.rstrip("/") + "/"
    blobs = list(bucket.list_blobs(prefix=prefix, max_results=20))
    for b in blobs:
        if b.name.endswith(".mp4"):
            logger.info("Downloading video from prefix: gs://%s/%s", bucket_name, b.name)
            return b.download_as_bytes()

    raise RuntimeError(f"No .mp4 file found at {gcs_uri}")


# ---------------------------------------------------------------------------
# Veo Video Generation  (supports Veo 3.0 + 3.1 features)
# ---------------------------------------------------------------------------

@dataclass
class ImageInput:
    """A single image with its bytes and MIME type."""
    data: bytes
    mime_type: str = "image/png"
    filename: str = ""


def _upload_image_to_gcs(image: ImageInput, purpose: str = "input") -> str:
    """Upload image bytes to GCS and return the gs:// URI."""
    date_str = datetime.now(timezone.utc).strftime("%Y-%m-%d")
    ext = "png" if "png" in image.mime_type else "jpg"
    slug = _slugify(image.filename or purpose, max_words=3)
    blob_path = f"video_inputs/{date_str}/{slug}_{uuid.uuid4().hex[:8]}.{ext}"
    upload_bytes_to_gcs(image.data, blob_path, content_type=image.mime_type)
    uri = f"gs://{BUCKET_NAME}/{blob_path}"
    logger.info("Uploaded %s image to %s", purpose, uri)
    return uri


def generate_video(
    prompt: str,
    aspect_ratio: str = "16:9",
    duration_seconds: int = 8,
    negative_prompt: str | None = None,
    generate_audio: bool = True,
    resolution: str = "720p",
    number_of_videos: int = 1,
    seed: int | None = None,
    generation_mode: str = "text_to_video",
    source_image_gcs_uri: str | None = None,
    source_image_mime: str | None = None,
    last_frame_gcs_uri: str | None = None,
    last_frame_mime: str | None = None,
    reference_images: list[ImageInput] | None = None,
    output_gcs_uri: str | None = None,
    model_id: str | None = None,
    poll_interval: int = 15,
    max_wait: int = 600,
) -> VideoGenerationResult:
    """Submit a Veo video generation request and poll until complete.

    Three modes (aligned with official Vertex AI Veo docs):

    - **text_to_video**: Prompt only. No images.
      SDK: generate_videos(model, prompt, config=...)

    - **image_to_video**: Single image as first frame + prompt.
      SDK: generate_videos(model, prompt, image=Image(...), config=...)
      Optionally accepts last_frame for controlled transition (e.g. logo reveal).

    - **reference_to_video**: 1-3 reference images for identity/asset
      preservation (Veo 3.1 preview only, duration always 8s).
      SDK: generate_videos(model, prompt, config=GenerateVideosConfig(reference_images=[...]))

    Parameters
    ----------
    prompt : str
        The assembled cinematic prompt.
    source_image_gcs_uri : str, optional
        GCS URI of the source image for image_to_video mode (first frame).
    source_image_mime : str, optional
        MIME type of the source image (image/jpeg, image/png, image/webp).
    last_frame_gcs_uri : str, optional
        GCS URI of the last frame image (e.g. logo card for brand reveal).
        Used with image_to_video mode to control video end state.
    last_frame_mime : str, optional
        MIME type of the last frame image.
    reference_images : list[ImageInput], optional
        Up to 3 images for identity preservation (person/product).
    resolution : str
        "720p", "1080p", or "4k".
    number_of_videos : int
        Number of variants to generate (1-4).
    seed : int, optional
        For reproducible results (0-4294967295).
    """
    model = model_id or VIDEO_MODEL_ID

    # ── Build config ─────────────────────────────────────────────────────
    config = types.GenerateVideosConfig(
        aspect_ratio=aspect_ratio,
        number_of_videos=number_of_videos,
        duration_seconds=duration_seconds,
        generate_audio=generate_audio,
        person_generation="allow_adult",
    )

    if resolution and resolution != "720p":
        config.resolution = resolution

    if negative_prompt:
        config.negative_prompt = negative_prompt

    if seed is not None:
        config.seed = seed

    if output_gcs_uri:
        config.output_gcs_uri = output_gcs_uri

    # ── Image-to-video: single source image as first frame ───────────────
    # Official SDK: image=Image(gcs_uri=..., mime_type=...)
    image_param = None
    if generation_mode == "image_to_video" and source_image_gcs_uri:
        mime = source_image_mime or "image/png"
        image_param = types.Image(gcs_uri=source_image_gcs_uri, mime_type=mime)
        logger.info("Image-to-video: using source image %s (%s)", source_image_gcs_uri, mime)

    # ── Last frame (logo card / ending image) ────────────────────────────
    # Official SDK: config.last_frame = Image(gcs_uri=..., mime_type=...)
    # ⚠ ONLY supported by Veo 3.1 models (preview/fast_preview).
    #   Veo 3.0 (standard/fast) will return 400 FAILED_PRECONDITION.
    VEO_31_MODELS = {"veo-3.1-generate-preview", "veo-3.1-fast-generate-preview", "veo-3.1-generate-001"}
    if last_frame_gcs_uri and image_param is not None:
        if model in VEO_31_MODELS:
            lf_mime = last_frame_mime or "image/png"
            config.last_frame = types.Image(gcs_uri=last_frame_gcs_uri, mime_type=lf_mime)
            logger.info("Last frame set: %s (%s) — video will transition to this", last_frame_gcs_uri, lf_mime)
        else:
            logger.warning(
                "last_frame skipped — model %s does not support it. "
                "Use veo_variant=preview or fast_preview for logo-as-last-frame.",
                model,
            )

    # ── Reference-to-video: 1-3 identity images in config ────────────────
    # Official SDK: config.reference_images = [VideoGenerationReferenceImage(...)]
    if generation_mode == "reference_to_video" and reference_images:
        ref_list = []
        for i, ref_img in enumerate(reference_images[:3]):
            ref_uri = _upload_image_to_gcs(ref_img, f"reference_{i+1}")

            # Try typed class first, fall back to dict if SDK version doesn't support it
            try:
                ref_obj = types.VideoGenerationReferenceImage(
                    image=types.Image(gcs_uri=ref_uri, mime_type=ref_img.mime_type),
                    reference_type="asset",
                )
            except (AttributeError, TypeError) as e:
                logger.warning(
                    "VideoGenerationReferenceImage not available (%s), using dict fallback", e
                )
                ref_obj = {
                    "image": {"gcs_uri": ref_uri, "mime_type": ref_img.mime_type},
                    "reference_type": "asset",
                }

            ref_list.append(ref_obj)
        config.reference_images = ref_list
        logger.info("Using %d reference image(s) for identity preservation", len(ref_list))

    # ── Submit request ───────────────────────────────────────────────────
    logger.info(
        "Submitting Veo request: model=%s, mode=%s, duration=%ds, resolution=%s, variants=%d",
        model, generation_mode, duration_seconds, resolution, number_of_videos,
    )

    operation = genai_client.models.generate_videos(
        model=model,
        prompt=prompt,
        image=image_param,
        config=config,
    )
    logger.info("Operation started: %s", getattr(operation, "name", "unknown"))

    # Poll until done
    elapsed = 0
    while not operation.done:
        if elapsed >= max_wait:
            raise RuntimeError(
                f"Video generation timed out after {max_wait}s. "
                f"Operation: {getattr(operation, 'name', 'unknown')}"
            )
        logger.info("Polling... (%ds elapsed)", elapsed)
        time.sleep(poll_interval)
        elapsed += poll_interval
        operation = genai_client.operations.get(operation)

    # Extract result — full debug dump to diagnose SDK structure
    logger.info("Operation done in %ds. Extracting result...", elapsed)

    # ── Dump everything for debugging ────────────────────────────────────
    # .result and .response may differ across SDK versions
    for attr_name in ("result", "response", "error", "metadata"):
        val = getattr(operation, attr_name, "N/A")
        if val and val != "N/A":
            logger.info("operation.%s = %s", attr_name, _safe_repr(val))
        else:
            logger.info("operation.%s = (empty/None)", attr_name)

    # Check for error
    if operation.error:
        raise RuntimeError(f"Veo generation failed: {operation.error}")

    # ── Try to extract generated videos ──────────────────────────────────
    video_uri = None
    video_bytes = None

    for source_name, source_obj in [
        ("result", operation.result),
        ("response", operation.response),
    ]:
        if source_obj is None:
            logger.info("operation.%s is None — skipping", source_name)
            continue

        # Check for RAI filtering
        rai_count = getattr(source_obj, "rai_media_filtered_count", None)
        if rai_count and int(rai_count) > 0:
            raise RuntimeError(
                f"Video was blocked by Google's safety filter "
                f"(raiMediaFilteredCount={rai_count}). "
                "Try adjusting the prompt to avoid sensitive content."
            )

        # Try to get generated_videos
        generated = getattr(source_obj, "generated_videos", None)
        if not generated:
            logger.info("operation.%s has no generated_videos. Attrs: %s",
                        source_name,
                        [a for a in dir(source_obj) if not a.startswith("_")])
            continue

        logger.info("operation.%s.generated_videos has %d items", source_name, len(generated))
        video_obj = generated[0].video
        logger.info("video object attrs: %s", [a for a in dir(video_obj) if not a.startswith("_")])

        # Try every possible URI attribute
        for uri_attr in ("uri", "gcs_uri", "video_uri"):
            uri = getattr(video_obj, uri_attr, None)
            if uri:
                video_uri = uri
                logger.info("Found video URI via %s.%s: %s", source_name, uri_attr, video_uri)
                break

        # If no URI, check for inline bytes (Veo can return bytes directly)
        if not video_uri:
            raw_data = getattr(video_obj, "video_bytes", None) or getattr(video_obj, "data", None)
            if raw_data:
                video_bytes = raw_data if isinstance(raw_data, bytes) else raw_data.encode()
                logger.info("Got inline video bytes: %d bytes", len(video_bytes))

        if video_uri or video_bytes:
            break

    # ── If we have inline bytes, upload to GCS ───────────────────────────
    if not video_uri and video_bytes:
        date_str = datetime.now(timezone.utc).strftime("%Y-%m-%d")
        slug = _slugify(prompt[:60]) or "video"
        short_id = uuid.uuid4().hex[:8]
        blob_path = f"videos/{date_str}/{slug}_{short_id}.mp4"
        result = upload_bytes_to_gcs(video_bytes, blob_path, content_type="video/mp4")
        video_uri = result.gcs_uri
        logger.info("Inline video bytes uploaded to GCS: %s", video_uri)

    # ── If still no URI, check the GCS output prefix ─────────────────────
    if not video_uri and output_gcs_uri:
        # Veo may take a moment to flush to GCS — retry a few times
        for attempt in range(4):
            found = _find_video_in_gcs_prefix(output_gcs_uri)
            if found:
                video_uri = found
                break
            logger.info("GCS prefix empty, retrying in 10s... (attempt %d/4)", attempt + 1)
            time.sleep(10)

    if not video_uri:
        raise RuntimeError(
            "Veo completed but no video was produced. "
            "This usually means the content was filtered by safety checks. "
            "Try simplifying the prompt or removing sensitive elements."
        )

    logger.info("Video generated successfully: %s (in %ds)", video_uri, elapsed)

    return VideoGenerationResult(
        gcs_uri=video_uri,
        operation_name=getattr(operation, "name", ""),
    )


def _safe_repr(obj, max_len: int = 500) -> str:
    """Safe string repr of an object, truncated for logging."""
    try:
        # Try JSON-like dump first
        if hasattr(obj, "to_json_dict"):
            s = str(obj.to_json_dict())
        elif hasattr(obj, "model_dump"):
            s = str(obj.model_dump())
        else:
            s = repr(obj)
        return s[:max_len] + ("..." if len(s) > max_len else "")
    except Exception:
        return repr(obj)[:max_len]


def _find_video_in_gcs_prefix(gcs_prefix: str) -> str | None:
    """List blobs under a GCS prefix and return the first .mp4 file URI.

    Veo writes output files into the storageUri prefix, typically as
    sample_0.mp4, sample_1.mp4, etc.
    """
    try:
        parts = gcs_prefix.replace("gs://", "").split("/", 1)
        bucket_name, prefix = parts[0], parts[1] if len(parts) > 1 else ""
        prefix = prefix.rstrip("/") + "/"

        bucket = gcs_client.bucket(bucket_name)
        blobs = list(bucket.list_blobs(prefix=prefix, max_results=20))
        logger.info("GCS prefix %s contains %d blobs: %s", gcs_prefix, len(blobs), [b.name for b in blobs])

        for blob in blobs:
            if blob.name.endswith(".mp4"):
                uri = f"gs://{bucket_name}/{blob.name}"
                logger.info("Found video file: %s", uri)
                return uri
    except Exception as e:
        logger.error("Failed to list GCS prefix %s: %s", gcs_prefix, e)
    return None


# ---------------------------------------------------------------------------
# Google Cloud Text-to-Speech
# ---------------------------------------------------------------------------
def synthesize_speech(
    text: str,
    language_code: str = "ar-XA",
    voice_name: str | None = None,
    speaking_rate: float = 1.0,
) -> TTSResult:
    """Synthesise speech using Google Cloud TTS.

    Parameters
    ----------
    text : str
        The text to speak.
    language_code : str
        BCP-47 language code, e.g. "ar-XA", "en-US".
    voice_name : str, optional
        Specific voice ID. If None, auto-selects best voice for language.
    speaking_rate : float
        Speed multiplier (0.25 – 4.0).

    Returns
    -------
    TTSResult
    """
    tts_client = texttospeech.TextToSpeechClient()

    synthesis_input = texttospeech.SynthesisInput(text=text)

    voice_params = texttospeech.VoiceSelectionParams(
        language_code=language_code,
    )
    if voice_name:
        voice_params.name = voice_name

    audio_config = texttospeech.AudioConfig(
        audio_encoding=texttospeech.AudioEncoding.MP3,
        speaking_rate=speaking_rate,
    )

    response = tts_client.synthesize_speech(
        input=synthesis_input,
        voice=voice_params,
        audio_config=audio_config,
    )

    logger.info(
        "TTS synthesised: %d bytes, lang=%s, voice=%s",
        len(response.audio_content),
        language_code,
        voice_name or "auto",
    )

    return TTSResult(
        audio_bytes=response.audio_content,
        mime_type="audio/mpeg",
    )


# ---------------------------------------------------------------------------
# FFmpeg binary discovery (cross-platform)
# ---------------------------------------------------------------------------
def _get_ffmpeg_path() -> str:
    """Find the ffmpeg binary — system PATH first, imageio-ffmpeg fallback.

    Install imageio-ffmpeg for Windows environments without system ffmpeg:
        pip install imageio-ffmpeg
    """
    import shutil

    # 1. Check system PATH
    system_ffmpeg = shutil.which("ffmpeg")
    if system_ffmpeg:
        return system_ffmpeg

    # 2. Try imageio-ffmpeg bundled binary
    try:
        import imageio_ffmpeg
        bundled = imageio_ffmpeg.get_ffmpeg_exe()
        if bundled:
            logger.info("Using imageio-ffmpeg bundled binary: %s", bundled)
            return bundled
    except ImportError:
        pass

    # 3. Give up with a helpful message
    raise FileNotFoundError(
        "ffmpeg not found. Install it via:\n"
        "  pip install imageio-ffmpeg    (recommended — bundles a static binary)\n"
        "  OR install ffmpeg system-wide: https://ffmpeg.org/download.html"
    )


# ---------------------------------------------------------------------------
# Audio Merge (ffmpeg)
# ---------------------------------------------------------------------------
def merge_audio_video(
    video_bytes: bytes,
    audio_bytes: bytes,
    audio_mime: str = "audio/mpeg",
) -> bytes:
    """Merge an audio track onto a video using ffmpeg.

    The audio is overlaid on the video. If the audio is longer than the
    video, it is trimmed. If shorter, silence fills the remainder.

    Returns the merged video as bytes (mp4).
    """
    ffmpeg_bin = _get_ffmpeg_path()
    audio_ext = "mp3" if "mpeg" in audio_mime else "wav"

    with (
        tempfile.NamedTemporaryFile(suffix=".mp4", delete=False) as vf,
        tempfile.NamedTemporaryFile(suffix=f".{audio_ext}", delete=False) as af,
        tempfile.NamedTemporaryFile(suffix=".mp4", delete=False) as out,
    ):
        vf.write(video_bytes)
        vf.flush()
        af.write(audio_bytes)
        af.flush()

        cmd = [
            ffmpeg_bin, "-y",
            "-i", vf.name,
            "-i", af.name,
            "-c:v", "copy",
            "-c:a", "aac",
            "-shortest",
            "-map", "0:v:0",
            "-map", "1:a:0",
            out.name,
        ]
        result = subprocess.run(cmd, capture_output=True, timeout=60)

        if result.returncode != 0:
            logger.error("ffmpeg error: %s", result.stderr.decode(errors="replace"))
            raise RuntimeError(f"ffmpeg merge failed: {result.stderr.decode(errors='replace')}")

        with open(out.name, "rb") as f:
            return f.read()


def create_logo_card(
    logo_bytes: bytes,
    aspect_ratio: str = "16:9",
    bg_color: str = "#FFFFFF",
    logo_scale: float = 0.35,
) -> bytes:
    """Create a branded card image with the logo centred on a solid background.

    The card is used as Veo's ``last_frame`` so the video naturally
    transitions into a brand reveal — no ffmpeg needed.

    Parameters
    ----------
    logo_bytes : bytes
        Logo image (PNG with transparency recommended).
    aspect_ratio : str
        Target video ratio ("16:9" or "9:16"). Determines card dimensions.
    bg_color : str
        Hex colour for the card background (default white).
    logo_scale : float
        Logo width as fraction of card width (0.1–0.8). Default 0.35.

    Returns
    -------
    bytes
        PNG image bytes of the branded card (720p resolution).
    """
    from PIL import Image as PILImage

    # Card dimensions to match video resolution
    if aspect_ratio == "9:16":
        card_w, card_h = 720, 1280
    else:
        card_w, card_h = 1280, 720

    # Create solid background
    card = PILImage.new("RGBA", (card_w, card_h), bg_color)

    # Open logo and resize preserving aspect ratio
    logo = PILImage.open(io.BytesIO(logo_bytes)).convert("RGBA")
    target_w = int(card_w * max(0.1, min(0.8, logo_scale)))
    ratio = target_w / logo.width
    target_h = int(logo.height * ratio)

    # Don't let the logo exceed 60% of card height
    if target_h > int(card_h * 0.6):
        target_h = int(card_h * 0.6)
        ratio = target_h / logo.height
        target_w = int(logo.width * ratio)

    logo_resized = logo.resize((target_w, target_h), PILImage.LANCZOS)

    # Centre the logo on the card
    x = (card_w - target_w) // 2
    y = (card_h - target_h) // 2
    card.paste(logo_resized, (x, y), logo_resized)  # use alpha channel as mask

    # Export as PNG bytes
    buf = io.BytesIO()
    card.save(buf, format="PNG")
    buf.seek(0)
    logger.info(
        "Created logo card: %dx%d, logo=%dx%d, bg=%s",
        card_w, card_h, target_w, target_h, bg_color,
    )
    return buf.read()
