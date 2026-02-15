"""
Video Prompt Generator — builds structured prompts for Veo video generation.

Veo understands natural language descriptions of scenes. The prompt is
assembled following a cinematic hierarchy:

    1. Visual Style           (photorealistic, cinematic, anime, etc.)
    2. Camera Angle + Lens    (medium shot, shallow DOF, anamorphic, etc.)
    3. Camera Movement        (static, pan, dolly, tracking, etc.)
    4. Subject + Action       (who/what doing what)
    5. Scene Context          (where/when)
    6. Temporal / Pacing      (slow-motion, time-lapse, loop, etc.)
    7. Sound / Dialogue       (ambience, spoken text for lip-sync)
    8. Branding               (website URL as end-card / watermark cue)
    9. Constraints            (negative_prompt — handled separately in API)

The prompt is a SINGLE string passed to Veo. The negative_prompt is a
separate API parameter (not embedded in the prompt).
"""

from dataclasses import dataclass, field

from src.config import (
    VIDEO_DEFAULTS,
    DEFAULT_BRAND,
    CAMERA_ANGLE_MAP,
    CAMERA_MOVEMENT_MAP,
    LENS_EFFECT_MAP,
    VIDEO_STYLE_MAP,
    TEMPORAL_MAP,
    SOUND_AMBIENCE_MAP,
)


@dataclass
class VideoPromptInputs:
    """All user-supplied creative inputs for a video generation request."""

    # --- Core prompt (required — can include subject + action inline) ---
    prompt: str                                 # e.g. "A CEO giving a Ramadan greeting"

    # --- Subject & Scene (optional — merged into prompt if provided) ---
    subject: str = ""                           # e.g. "a CEO in a suit"
    action: str = ""                            # e.g. "saying congratulations for Ramadan"
    scene_context: str = ""                     # e.g. "in a modern office"

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
    dialogue: str = ""                          # spoken text for lip-sync

    # --- Branding ---
    company_website_url: str = ""               # e.g. "mobily.com.sa"
    use_default_brand: bool = False

    # --- Constraints (NOT in prompt — sent as separate Veo parameter) ---
    negative_prompt: str = ""


def build_video_prompt(inputs: VideoPromptInputs) -> str:
    """
    Assemble a Veo-optimised prompt from structured user inputs.

    Returns
    -------
    str
        A single prompt string for the Veo API's `prompt` parameter.
    """
    sections: list[str] = []

    # ── 1. Visual Style ──────────────────────────────────────────────────
    style_desc = VIDEO_STYLE_MAP.get(inputs.visual_style, "")
    if style_desc:
        sections.append(f"{style_desc} video.")

    # ── 2. Camera Angle + Lens ───────────────────────────────────────────
    angle_desc = CAMERA_ANGLE_MAP.get(inputs.camera_angle, "")
    if angle_desc:
        sections.append(angle_desc)

    lens_desc = LENS_EFFECT_MAP.get(inputs.lens_effect, "")
    if lens_desc:
        sections.append(lens_desc)

    # ── 3. Camera Movement ───────────────────────────────────────────────
    movement_desc = CAMERA_MOVEMENT_MAP.get(inputs.camera_movement, "")
    if movement_desc:
        sections.append(movement_desc)

    # ── 4. Subject + Action + Core Prompt ────────────────────────────────
    # Build the subject line from parts or use prompt directly
    subject_parts = []
    if inputs.subject:
        subject_parts.append(inputs.subject)
    if inputs.action:
        subject_parts.append(inputs.action)

    if subject_parts:
        subject_line = " ".join(subject_parts)
        # Merge with the core prompt
        if inputs.prompt:
            sections.append(f"{subject_line}. {inputs.prompt}")
        else:
            sections.append(f"{subject_line}.")
    else:
        sections.append(inputs.prompt)

    # ── 5. Scene Context ─────────────────────────────────────────────────
    if inputs.scene_context:
        sections.append(f"Setting: {inputs.scene_context}.")

    # ── 6. Temporal / Pacing ─────────────────────────────────────────────
    temporal_desc = TEMPORAL_MAP.get(inputs.temporal_elements, "")
    if temporal_desc:
        sections.append(temporal_desc)

    # ── 7. Sound & Dialogue (Veo 3 native audio cues) ───────────────────
    ambience_desc = SOUND_AMBIENCE_MAP.get(inputs.sound_ambience, "")
    if ambience_desc:
        sections.append(f"Audio: {ambience_desc}.")

    if inputs.dialogue:
        sections.append(
            f"The subject says: '{inputs.dialogue}'"
        )

    # ── 8. Branding ──────────────────────────────────────────────────────
    website = inputs.company_website_url
    if inputs.use_default_brand and not website:
        website = DEFAULT_BRAND["website_url"]

    if website:
        sections.append(
            f"A small '{website}' text watermark appears "
            "subtly in the lower-right corner."
        )

    return " ".join(sections)
