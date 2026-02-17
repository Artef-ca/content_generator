"""
Video Prompt Generator — builds structured prompts for Veo video generation.

Follows Google's official Veo prompt guide best practices:
  https://cloud.google.com/blog/products/ai-machine-learning/ultimate-prompting-guide-for-veo-3-1
  https://docs.cloud.google.com/vertex-ai/generative-ai/docs/video/video-gen-prompt-guide

Veo prompt structure (ordered by influence on output):

    1. Visual Style + Subject + Action   (cinematic, photorealistic, etc.)
    2. Camera Angle + Composition        (close-up, medium shot, low angle)
    3. Lens & Focus                      (shallow DOF, 50mm, anamorphic)
    4. Camera Movement                   (dolly in, slow pan, static)
    5. Setting / Environment             (where/when)
    6. Lighting & Mood                   (golden hour, studio light, moody)
    7. Temporal / Pacing                 (slow-motion, time-lapse)
    8. Audio Direction                   (ambient sounds, dialogue, SFX)
    9. Negative Constraints              (handled via separate API param)

IMPORTANT — What does NOT go into the prompt:
    - Brand names, company names, real person names (triggers safety filter)
    - URLs or website addresses (Veo can't render text reliably)
    - Instructions/commands ("I want...", "Please make...")
    - The negative_prompt (separate Veo API parameter)
"""

from dataclasses import dataclass, field

from src.config import (
    VIDEO_DEFAULTS,
    BRAND_CONFIG,
    CAMERA_ANGLE_MAP,
    CAMERA_MOVEMENT_MAP,
    LENS_EFFECT_MAP,
    VIDEO_STYLE_MAP,
    TEMPORAL_MAP,
    SOUND_AMBIENCE_MAP,
)

# Brand style context — injected subtly into every prompt
_BRAND_STYLE = BRAND_CONFIG.get("style_context", "").strip()
_BRAND_NEGATIVE = BRAND_CONFIG.get("negative_prompt", "").strip()


@dataclass
class VideoPromptInputs:
    """All user-supplied creative inputs for a video generation request."""

    # --- Core prompt (required — scene description) ---
    prompt: str                                 # e.g. "A professional man warmly addressing the camera"

    # --- Subject & Scene (optional — merged into prompt if provided) ---
    subject: str = ""                           # e.g. "a man in a tailored suit"
    action: str = ""                            # e.g. "speaking warmly to the camera"
    scene_context: str = ""                     # e.g. "in a modern glass office"

    # --- Camera ---
    camera_angle: str = VIDEO_DEFAULTS["camera_angle"]
    camera_movement: str = VIDEO_DEFAULTS["camera_movement"]

    # --- Lens ---
    lens_effect: str = VIDEO_DEFAULTS["lens_effect"]

    # --- Style & Pacing ---
    visual_style: str = VIDEO_DEFAULTS["visual_style"]
    temporal_elements: str = VIDEO_DEFAULTS["temporal_elements"]

    # --- Audio cues (embedded in prompt for Veo 3 native audio) ---
    sound_ambience: str = VIDEO_DEFAULTS["sound_ambience"]
    dialogue: str = ""                          # spoken text for lip-sync (image/reference modes only)

    # --- Constraints (NOT in prompt — sent as separate Veo parameter) ---
    negative_prompt: str = ""


def build_video_prompt(inputs: VideoPromptInputs) -> str:
    """
    Assemble a Veo-optimised prompt from structured user inputs.

    The prompt follows Google's recommended structure:
      Style + Subject + Action → Camera → Lens → Movement → Setting → Pacing → Audio

    Brand names, URLs, and instructions are excluded to avoid safety filters.

    Returns
    -------
    str
        A single prompt string for the Veo API's `prompt` parameter.
    """
    sections: list[str] = []

    # ── 1. Visual Style + Subject + Action ───────────────────────────────
    # This is the MOST important part — Veo weighs early tokens heaviest.
    style_desc = VIDEO_STYLE_MAP.get(inputs.visual_style, "")

    # Build subject+action line from structured fields
    subject_action = _build_subject_action(
        inputs.subject, inputs.action, inputs.prompt
    )

    if style_desc:
        sections.append(f"{style_desc}. {subject_action}")
    else:
        sections.append(subject_action)

    # ── 1b. Brand Style Context (subtle visual direction) ─────────────
    # Injected from config/video_config.yaml → brand.style_context
    # Guides Veo toward brand-consistent visuals without brand names.
    if _BRAND_STYLE:
        sections.append(_BRAND_STYLE)

    # ── 2. Camera Angle + Composition ────────────────────────────────────
    angle_desc = CAMERA_ANGLE_MAP.get(inputs.camera_angle, "")
    if angle_desc:
        sections.append(angle_desc)

    # ── 3. Lens & Focus ─────────────────────────────────────────────────
    lens_desc = LENS_EFFECT_MAP.get(inputs.lens_effect, "")
    if lens_desc:
        sections.append(lens_desc)

    # ── 4. Camera Movement ───────────────────────────────────────────────
    movement_desc = CAMERA_MOVEMENT_MAP.get(inputs.camera_movement, "")
    if movement_desc:
        sections.append(movement_desc)

    # ── 5. Setting / Environment ─────────────────────────────────────────
    if inputs.scene_context:
        clean_scene = _sanitise_for_veo(inputs.scene_context)
        if clean_scene:
            sections.append(f"Setting: {clean_scene}.")

    # ── 6. Temporal / Pacing ─────────────────────────────────────────────
    temporal_desc = TEMPORAL_MAP.get(inputs.temporal_elements, "")
    if temporal_desc:
        sections.append(temporal_desc)

    # ── 7. Audio Direction ───────────────────────────────────────────────
    # Following Veo best practices: use "Audio:" label for ambient cues
    # and quotation marks for dialogue.
    ambience_desc = SOUND_AMBIENCE_MAP.get(inputs.sound_ambience, "")
    if ambience_desc:
        sections.append(f"Audio: {ambience_desc}.")

    if inputs.dialogue:
        # Sanitise dialogue — remove brand names that trigger safety
        clean_dialogue = _sanitise_for_veo(inputs.dialogue)
        if clean_dialogue:
            # Veo best practice: use quotation marks for specific speech
            sections.append(
                f'The subject says, "{clean_dialogue}"'
            )

    return " ".join(sections)


def get_brand_negative_prompt() -> str:
    """Return the brand-default negative prompt from config."""
    return _BRAND_NEGATIVE


def _build_subject_action(subject: str, action: str, prompt: str) -> str:
    """Merge subject, action, and prompt into a clean scene description.

    If subject+action are provided, they take priority and the prompt
    is appended as additional context.

    ALL user text is sanitised to remove brand names and instruction-style
    language that Veo doesn't understand.
    """
    # Clean all user inputs
    clean_subject = _sanitise_for_veo(subject) if subject else ""
    clean_action = _sanitise_for_veo(action) if action else ""
    clean_prompt = _clean_user_prompt(prompt)

    parts = []
    if clean_subject:
        parts.append(clean_subject)
    if clean_action:
        parts.append(clean_action)

    if parts:
        subject_line = " ".join(parts)
        if clean_prompt:
            return f"{subject_line}. {clean_prompt}"
        return f"{subject_line}."
    elif clean_prompt:
        return clean_prompt
    else:
        return "A professional scene."


def _clean_user_prompt(prompt: str) -> str:
    """Remove instruction-style phrasing from user prompts.

    Veo needs scene DESCRIPTIONS, not commands. This strips common
    instruction patterns while preserving the descriptive content.

    Examples
    --------
    >>> _clean_user_prompt("I want a video of a CEO giving a speech")
    'A professional person giving a speech'
    >>> _clean_user_prompt("Please create a sunset timelapse over the city")
    'A sunset timelapse over the city'
    >>> _clean_user_prompt("A cat sitting on a windowsill")
    'A cat sitting on a windowsill'
    """
    import re

    text = prompt.strip()

    # Strip common instruction prefixes
    instruction_patterns = [
        r"^(?:i\s+want|i\s+need|i'd\s+like)\s+(?:a\s+video\s+(?:of|for|about|showing|where)|to\s+(?:create|make|generate|see))\s*",
        r"^(?:please|can\s+you|could\s+you)\s+(?:create|make|generate|produce)\s+(?:a\s+video\s+(?:of|for|about|showing|where))?\s*",
        r"^(?:create|make|generate|produce)\s+(?:a\s+video\s+(?:of|for|about|showing|where))?\s*",
        r"^(?:a\s+video\s+(?:of|for|about|showing|where))\s+",
    ]

    for pattern in instruction_patterns:
        text = re.sub(pattern, "", text, flags=re.IGNORECASE).strip()

    # Ensure it starts with a capital letter or descriptive word
    if text and text[0].islower():
        text = text[0].upper() + text[1:]

    return _sanitise_for_veo(text)


def _sanitise_for_veo(text: str) -> str:
    """Remove content that triggers Veo safety filters.

    - Brand names / company names
    - Real person titles that imply identity (CEO, President of X)
    - URLs
    """
    import re

    # Remove URLs
    text = re.sub(r"https?://\S+", "", text)
    text = re.sub(r"\b\w+\.\w+\.\w+(?:/\S*)?", "", text)  # domain.tld.cc/path

    # Remove common brand-name patterns
    # (Mobily, موبايلي, etc. — add more as needed)
    brand_terms = [
        r"\bmobily\b", r"\bموبايلي\b",
        r"\bstc\b", r"\bzain\b",
    ]
    for brand in brand_terms:
        text = re.sub(brand, "", text, flags=re.IGNORECASE)

    # Remove "CEO of X", "President of X" patterns (real-person implication)
    text = re.sub(
        r"\b(?:ceo|president|chairman|founder|director)\s+(?:of\s+\w+)",
        "professional leader",
        text,
        flags=re.IGNORECASE,
    )

    # Clean up — but keep standalone "ceo" as "professional executive"
    text = re.sub(r"\bceo\b", "professional executive", text, flags=re.IGNORECASE)

    # Deduplicate adjacent repeated words ("professional professional" → "professional")
    text = re.sub(r"\b(\w+)\s+\1\b", r"\1", text, flags=re.IGNORECASE)

    # Remove double spaces and trim
    text = re.sub(r"\s{2,}", " ", text).strip()

    # Remove dangling prepositions left after brand removal
    # English: "from", "by", "of", "at" at end
    # Arabic: "من" (from), "في" (in), "لـ" (for) at end
    text = re.sub(r"\s+(?:from|by|of|at|for|with)\s*$", "", text, flags=re.IGNORECASE)
    text = re.sub(r"\s+(?:من|في|لـ|عن)\s*$", "", text)

    # Remove trailing/leading punctuation artifacts
    text = re.sub(r"^[\s,.\-]+|[\s,.\-]+$", "", text)

    return text
