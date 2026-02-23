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
    8. Specs Comments (composition guidance)
    9. Campaign Text (rich text blocks or legacy single text)
   10. Logo Comments
   11. Custom Variation Prompt
   12. Constraints (negative prompt)
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from src.config import (
    DEFAULTS,
    BRAND_CONFIG,
    VISUAL_STYLE_MAP,
    CAMERA_ANGLE_IMG_MAP,
    FRAMING_MAP,
    LIGHTING_MAP,
    TONE_MAP,
    COLOR_GRADING_MAP,
    VISUAL_TEXT_FONT,
    MOBILY_PALETTE,
)

_BRAND_STYLE = BRAND_CONFIG.get("style_context", "").strip()

# ── Text block lookup tables ───────────────────────────────────────────────────

_POS_MAP: dict[str, str] = {
    "top_left":      "upper-left corner",
    "top_center":    "upper-center",
    "top_right":     "upper-right corner",
    "center_left":   "middle-left area",
    "center":        "center of the image",
    "center_right":  "middle-right area",
    "bottom_left":   "lower-left corner",
    "bottom_center": "lower-center",
    "bottom_right":  "lower-right corner",
}

_SIZE_MAP: dict[str, str] = {
    "S": "small",
    "M": "medium-sized",
    "L": "large, prominent",
}

# (display name, is_rtl)
_LANG_MAP: dict[str, tuple[str, bool]] = {
    "ar": ("Arabic",  True),
    "ur": ("Urdu",    True),
    "fa": ("Persian", True),
    "en": ("English", False),
    "fr": ("French",  False),
}


def _get_color_desc(color_key: str) -> str:
    """Map a Mobily palette key to a prompt-friendly color description."""
    if not color_key:
        return ""
    entry = MOBILY_PALETTE.get(color_key)
    if entry:
        name    = entry.get("name", color_key)
        hex_val = entry.get("hex", "")
        hex_end = entry.get("hex_end", "")
        if hex_end:
            return f"{name} gradient ({hex_val} to {hex_end})"
        return f"{name} ({hex_val})" if hex_val else name
    # Unknown key — clean up underscores as a safe fallback
    return color_key.replace("_", " ")


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

    # 8. Rich text blocks from the frontend Text tab.
    # Each dict: {text, size, color, weight, position, language, comments, font}
    # Takes priority over campaign_text when non-empty.
    text_blocks: list[dict[str, Any]] = field(default_factory=list)

    # 8b. Legacy single-text fallback (used when text_blocks is empty)
    campaign_text: str | None = None        # "رمضان كريم"
    text_style: str | None = None           # "bold modern sans-serif"

    # 8c. Composition & logo guidance from the Specs / Logo tabs
    specs_comments: str = ""
    logo_comments: str = ""
    logo_position: str = ""
    logo_size: str = ""

    # 9. Custom Variation
    custom_prompt: str = ""                 # free-text fine-tuning

    # 10. Constraints
    negative_prompt: str | None = None


def build_prompt(inputs: PromptInputs) -> str:
    """
    Assemble a production-quality image prompt from structured inputs.

    Style + Subject → Setting → Camera → Framing → Lighting → Tone →
    Color → Specs → Campaign Text → Logo → Custom → Constraints

    Note: Logo is overlaid by Pillow post-generation — not rendered in prompt.
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

    # ── 8. Specs Comments (composition / technical guidance) ──────────────
    if inputs.specs_comments:
        sections.append(f"Composition note: {inputs.specs_comments}.")

    # ── 9. Campaign Text ─────────────────────────────────────────────────
    # Logo is overlaid by Pillow post-generation — not in prompt.
    active_blocks = [b for b in (inputs.text_blocks or []) if b.get("text", "").strip()]

    if active_blocks:
        block_parts: list[str] = []
        for block in active_blocks:
            text       = block["text"].strip()
            size_label = _SIZE_MAP.get(block.get("size", "M"), "medium-sized")
            weight     = block.get("weight", "bold")
            color_desc = _get_color_desc(block.get("color", ""))
            pos_label  = _POS_MAP.get(block.get("position", "center"), "center of the image")
            lang_code  = block.get("language", "en")
            lang_name, is_rtl = _LANG_MAP.get(lang_code, ("", False))
            comments   = block.get("comments", "").strip()

            font_name  = block.get("font", VISUAL_TEXT_FONT) or VISUAL_TEXT_FONT
            parts = [f"Render the text '{text}' using {font_name} font"]
            parts.append(f"in {weight} weight, {size_label}")
            if color_desc:
                parts.append(f"in {color_desc} color")
            if lang_name:
                direction = "right-to-left" if is_rtl else "left-to-right"
                parts.append(f"in {lang_name} ({direction})")
            parts.append(f"positioned at the {pos_label}")
            if comments:
                parts.append(comments)
            block_parts.append(", ".join(parts) + ".")

        sections.append(" ".join(block_parts))
        

    elif inputs.campaign_text:
        # Fallback: legacy single-text field (backward compatibility)
        text_dir = inputs.text_style or DEFAULTS.get("text_style", "elegant typography")
        sections.append(
            f"Render the text '{inputs.campaign_text}' using {VISUAL_TEXT_FONT} font, "
            f"in {text_dir} style, that blends naturally with the overall design."
        )

    # ── 10. Logo Comments ─────────────────────────────────────────────────
    if inputs.logo_comments:
        pos_label = _POS_MAP.get(inputs.logo_position, "buttom_right")
        size_label = _SIZE_MAP.get(inputs.logo_size, "small-sized")
        sections.append(
            f"Place the provided logo image at the {pos_label}, "
            f"rendered as a {size_label} element. {inputs.logo_comments}."
        )

    # ── 11. Custom Variation ──────────────────────────────────────────────
    if inputs.custom_prompt:
        sections.append(inputs.custom_prompt)

    # ── 12. Constraints ───────────────────────────────────────────────────
    if inputs.negative_prompt:
        sections.append(f"Constraints: {inputs.negative_prompt}.")

    return " ".join(sections)
