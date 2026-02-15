"""
Prompt Generator — builds structured prompts following Nano Banana Pro best practices.

Google's official guidance + community best practices recommend a prompt
hierarchy that goes from most-important to least-important, because
**earlier details have more influence on the final result**:

    1. Subject + Action       (who/what is the focus, what are they doing)
    2. Setting / Environment  (where + when)
    3. Style / Medium         (photorealistic, illustration, 3D, etc.)
    4. Composition / Camera   (framing, angle, depth of field)
    5. Lighting / Mood        (emotional tone, light source & colour)
    6. Brand Integration      (logo placement, campaign text)
    7. Constraints            (what to avoid)

The user fills in the *content* for each slot via the API; this module
assembles them into a single, well-ordered prompt string.

Reference:
  - https://ai.google.dev/gemini-api/docs/image-generation#prompt
  - https://blog.google/products/gemini/prompting-tips-nano-banana-pro/
"""

from dataclasses import dataclass

from src.config import (
    DEFAULTS,
    VISUAL_STYLE_MAP,
    LOGO_SIZE_MAP,
    COMPOSITION_MAP,
    LIGHTING_MAP,
)


@dataclass
class PromptInputs:
    """All user-supplied creative inputs for a single generation request.

    Default values are loaded from config.yaml via the DEFAULTS dict,
    so editing the YAML changes the API defaults without touching code.
    """

    # --- 1. Subject + Action (required) ---
    subject: str                        # e.g. "A barista crafting latte art"

    # --- 2. Setting / Environment ---
    setting: str = ""                   # e.g. "a cosy Tokyo café at dawn"
    items_in_scene: str = ""            # e.g. "espresso machine, pastries, flowers"

    # --- 3. Style  (default from YAML) ---
    visual_style: str = DEFAULTS["visual_style"]

    # --- 4. Composition  (default from YAML) ---
    composition: str = DEFAULTS["composition"]

    # --- 5. Lighting  (default from YAML) ---
    lighting: str = DEFAULTS["lighting"]

    # --- 6. Brand  (default from YAML) ---
    logo_size: str = DEFAULTS["logo_size"]
    campaign_text: str | None = None    # e.g. "رمضان كريم" or "Summer Sale 50%"
    text_style: str | None = None       # e.g. "bold serif" or "elegant Arabic calligraphy"
    website_url: str | None = None      # e.g. "mobily.com.sa"

    # --- 7. Constraints ---
    negative_prompt: str | None = None  # e.g. "no watermarks, no extra people"


def build_prompt(inputs: PromptInputs, has_logo: bool = True) -> str:
    """
    Assemble a production-quality prompt from structured user inputs.

    The prompt follows the Nano Banana Pro ordering:
      Subject → Setting → Style → Composition → Lighting → Brand → Constraints

    Parameters
    ----------
    inputs : PromptInputs
        All creative fields supplied by the caller.
    has_logo : bool
        Whether a logo image was uploaded (controls brand-integration wording).

    Returns
    -------
    str
        A single prompt string ready to send to the model.
    """
    sections: list[str] = []

    # ── 1. Style opener + Subject ────────────────────────────────────────
    style_phrase = VISUAL_STYLE_MAP.get(
        inputs.visual_style, VISUAL_STYLE_MAP["photorealistic"]
    )
    sections.append(f"{style_phrase} of {inputs.subject}.")

    # ── 2. Setting / Environment ─────────────────────────────────────────
    if inputs.setting:
        sections.append(f"Setting: {inputs.setting}.")

    if inputs.items_in_scene:
        sections.append(f"Include these elements in the scene: {inputs.items_in_scene}.")

    # ── 3. Composition / Camera ──────────────────────────────────────────
    comp = COMPOSITION_MAP.get(inputs.composition, "")
    if comp:
        sections.append(comp)

    # ── 4. Lighting / Mood ───────────────────────────────────────────────
    light = LIGHTING_MAP.get(inputs.lighting, "")
    if light:
        sections.append(light)

    # ── 5. Brand Integration ─────────────────────────────────────────────
    if has_logo and inputs.logo_size != "none":
        logo_desc = LOGO_SIZE_MAP.get(inputs.logo_size, LOGO_SIZE_MAP["medium"])
        sections.append(
            f"Brand Integration: Integrate the provided logo as {logo_desc}. "
            "Choose the best placement so the logo does not obstruct the main subject."
        )

    if inputs.campaign_text:
        text_direction = inputs.text_style or DEFAULTS["text_style"]
        sections.append(
            f"Render the text '{inputs.campaign_text}' in {text_direction} "
            "that blends naturally with the overall design."
        )

    if inputs.website_url:
        sections.append(
            f"Display the website URL '{inputs.website_url}' in a small, clean, "
            "legible font near the bottom of the image as a subtle call-to-action."
        )

    # ── 6. Constraints ───────────────────────────────────────────────────
    if inputs.negative_prompt:
        sections.append(f"Constraints: {inputs.negative_prompt}.")

    return " ".join(sections)
