"""
Image Prompt Generator — builds structured prompts for Gemini image generation.

Follows Nano Banana Pro best practices. Prompt hierarchy (earlier = more influence):

    1. Visual Style + Subject + Action
    2. Setting / Environment
    3. Camera Angle
    4. Framing / Composition
    5. Lighting
    6. Tone / Mood
    7. Color Grading
    8. Brand Integration (logo, text, URL)
    9. Custom Variation Prompt
   10. Constraints (negative prompt)
"""

from dataclasses import dataclass

from src.config import (
    DEFAULTS,
    BRAND_CONFIG,
    VISUAL_STYLE_MAP,
    CAMERA_ANGLE_IMG_MAP,
    FRAMING_MAP,
    LIGHTING_MAP,
    TONE_MAP,
    COLOR_GRADING_MAP,
)

_BRAND_STYLE = BRAND_CONFIG.get("style_context", "").strip()


@dataclass
class PromptInputs:
    """All user-supplied creative inputs for image generation."""

    # 1. Subject + Action (required)
    subject: str                            # "A family enjoying an Iftar feast"
    action: str = ""                        # "sitting together around the table"

    # 2. Setting / Environment
    setting: str = ""                       # "a modern Saudi living room at sunset"
    items_in_scene: str = ""                # "lanterns, dates, Arabic coffee"

    # 3-7. Creative Controls (all optional — empty = skip)
    visual_style: str = ""
    camera_angle: str = ""
    framing: str = ""
    lighting: str = ""
    tone: str = ""
    color_grading: str = ""

    # 8. Brand Integration
    campaign_text: str | None = None        # "رمضان كريم"
    text_style: str | None = None           # "bold modern sans-serif"

    # 9. Custom Variation
    custom_prompt: str = ""                 # free-text fine-tuning

    # 10. Constraints
    negative_prompt: str | None = None


def build_prompt(inputs: PromptInputs) -> str:
    """
    Assemble a production-quality image prompt from structured inputs.

    Style + Subject → Setting → Camera → Framing → Lighting → Tone →
    Color → Campaign Text → Custom → Constraints

    Note: Logo is handled by Pillow overlay, not in the prompt.
    """
    sections: list[str] = []

    # ── 1. Style + Subject + Action ───────────────────────────────────────
    style_phrase = VISUAL_STYLE_MAP.get(inputs.visual_style, "")
    subject_line = inputs.subject
    if inputs.action:
        subject_line = f"{inputs.subject} {inputs.action}"

    if style_phrase:
        sections.append(f"{style_phrase} of {subject_line}.")
    else:
        sections.append(f"{subject_line}.")

    # ── 1b. Brand Style Context ───────────────────────────────────────────
    if _BRAND_STYLE:
        sections.append(_BRAND_STYLE)

    # ── 2. Setting / Environment ──────────────────────────────────────────
    if inputs.setting:
        sections.append(f"Setting: {inputs.setting}.")
    if inputs.items_in_scene:
        sections.append(f"Include these elements: {inputs.items_in_scene}.")

    # ── 3. Camera Angle ───────────────────────────────────────────────────
    angle = CAMERA_ANGLE_IMG_MAP.get(inputs.camera_angle, "")
    if angle:
        sections.append(angle)

    # ── 4. Framing / Composition ──────────────────────────────────────────
    framing = FRAMING_MAP.get(inputs.framing, "")
    if framing:
        sections.append(framing)

    # ── 5. Lighting ───────────────────────────────────────────────────────
    light = LIGHTING_MAP.get(inputs.lighting, "")
    if light:
        sections.append(light)

    # ── 6. Tone / Mood ────────────────────────────────────────────────────
    tone = TONE_MAP.get(inputs.tone, "")
    if tone:
        sections.append(tone)

    # ── 7. Color Grading ──────────────────────────────────────────────────
    color = COLOR_GRADING_MAP.get(inputs.color_grading, "")
    if color:
        sections.append(color)

    # ── 8. Campaign Text ─────────────────────────────────────────────────
    # Logo is overlaid by Pillow post-generation — not in prompt.
    if inputs.campaign_text:
        text_dir = inputs.text_style or DEFAULTS.get("text_style", "elegant typography")
        sections.append(
            f"Render the text '{inputs.campaign_text}' in {text_dir} "
            "that blends naturally with the overall design."
        )

    # ── 9. Custom Variation ───────────────────────────────────────────────
    if inputs.custom_prompt:
        sections.append(inputs.custom_prompt)

    # ── 10. Constraints ───────────────────────────────────────────────────
    if inputs.negative_prompt:
        sections.append(f"Constraints: {inputs.negative_prompt}.")

    return " ".join(sections)
