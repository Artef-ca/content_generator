"""
Mobily Creative Studio — FastAPI application.

A generic, topic-agnostic marketing banner generator powered by
Gemini 3 Pro Image (Nano Banana Pro). The user decides the campaign
theme, season, product, or occasion — the API structures their inputs
into an optimised prompt following Google's best practices.
"""

import logging
from typing import Optional

from fastapi import FastAPI, File, UploadFile, Form, HTTPException
from fastapi.responses import JSONResponse

from src.config import (
    DEFAULTS,
    ALLOWED_ASPECT_RATIOS,
    ALLOWED_IMAGE_SIZES,
    VISUAL_STYLE_MAP,
    LOGO_SIZE_MAP,
    COMPOSITION_MAP,
    LIGHTING_MAP,
    MODEL_ID,
    PROJECT_ID,
    PORT,
)
from src.core.prompt_generator import PromptInputs, build_prompt
from src.helpers.utils import (
    generate_image,
    generate_output_path,
    upload_to_gcs,
)

logger = logging.getLogger("creative_studio")

# ---------------------------------------------------------------------------
# FastAPI App
# ---------------------------------------------------------------------------
app = FastAPI(
    title="Mobily Creative Studio",
    description=(
        "Generate branded marketing banners for **any campaign or topic** "
        "using Gemini 3 Pro Image (Nano Banana Pro) on Vertex AI.\n\n"
        "The API follows [Google's Nano Banana prompt best practices]"
        "(https://ai.google.dev/gemini-api/docs/image-generation#prompt) "
        "to structure your creative inputs into high-quality prompts."
    ),
    version="2.0.0",
)


# ---------------------------------------------------------------------------
# Discovery endpoint — list available options
# ---------------------------------------------------------------------------
@app.get("/options", summary="List all available parameter choices")
async def list_options():
    """Return every valid value for style, composition, lighting, etc.

    Useful for front-end dropdowns and client documentation.
    """
    return {
        "visual_styles": list(VISUAL_STYLE_MAP.keys()),
        "compositions": list(COMPOSITION_MAP.keys()),
        "lightings": list(LIGHTING_MAP.keys()),
        "logo_sizes": list(LOGO_SIZE_MAP.keys()),
        "aspect_ratios": ALLOWED_ASPECT_RATIOS,
        "image_sizes": ALLOWED_IMAGE_SIZES,
        "defaults": DEFAULTS,
    }


# ---------------------------------------------------------------------------
# Health Check
# ---------------------------------------------------------------------------
@app.get("/health")
async def health():
    """Liveness / readiness probe."""
    return {"status": "ok", "model": MODEL_ID, "project": PROJECT_ID}


# ---------------------------------------------------------------------------
# Banner Generation
# ---------------------------------------------------------------------------
@app.post("/generate-banner", summary="Generate a branded marketing banner")
async def generate_banner(
    # ── 1. Subject + Action (the only truly required creative field) ──────
    subject: str = Form(
        ...,
        description=(
            "**Primary subject & action** — the most important part of your prompt. "
            "Example: 'A family enjoying an Iftar feast', "
            "'A smartphone floating above clouds', "
            "'A football player celebrating a goal'"
        ),
    ),
    # ── 2. Setting / Environment ─────────────────────────────────────────
    setting: str = Form(
        "",
        description=(
            "Where & when the scene takes place. "
            "Example: 'a modern Saudi living room at sunset', "
            "'a futuristic city skyline at night', "
            "'a sandy beach with turquoise water'"
        ),
    ),
    items_in_scene: str = Form(
        "",
        description=(
            "Additional objects to include. "
            "Example: 'lanterns, dates, Arabic coffee', "
            "'5G tower, confetti, fireworks', "
            "'palm trees, surfboard'"
        ),
    ),
    # ── 3. Style ─────────────────────────────────────────────────────────
    visual_style: str = Form(
        DEFAULTS["visual_style"],
        description=(
            "Art direction / medium. "
            "Options: " + ", ".join(VISUAL_STYLE_MAP.keys())
        ),
    ),
    # ── 4. Composition ───────────────────────────────────────────────────
    composition: str = Form(
        DEFAULTS["composition"],
        description=(
            "Camera / framing preset. "
            "Options: " + ", ".join(COMPOSITION_MAP.keys())
        ),
    ),
    # ── 5. Lighting ──────────────────────────────────────────────────────
    lighting: str = Form(
        DEFAULTS["lighting"],
        description=(
            "Lighting / mood preset. "
            "Options: " + ", ".join(LIGHTING_MAP.keys())
        ),
    ),
    # ── 6. Brand ─────────────────────────────────────────────────────────
    logo_file: Optional[UploadFile] = File(
        None,
        description="Brand logo image (PNG recommended). Optional.",
    ),
    logo_size: str = Form(
        DEFAULTS["logo_size"],
        description="Logo prominence: " + ", ".join(LOGO_SIZE_MAP.keys()),
    ),
    campaign_text: str = Form(
        None,
        description=(
            "Text to render on the banner. "
            "Example: 'رمضان كريم', 'Summer Sale 50% OFF', 'عيد مبارك'"
        ),
    ),
    text_style: str = Form(
        None,
        description=(
            "Typography direction for campaign_text. "
            "Example: 'bold modern sans-serif', 'elegant Arabic calligraphy', "
            "'retro neon glowing text'"
        ),
    ),
    website_url: str = Form(
        None,
        description=(
            "Brand website URL to display on the banner as a call-to-action. "
            "Rendered in a small, clean font near the bottom. "
            "Example: 'mobily.com.sa', 'shop.mobily.com.sa/offers'"
        ),
    ),
    # ── 7. Constraints ───────────────────────────────────────────────────
    negative_prompt: str = Form(
        None,
        description=(
            "What to avoid. "
            "Example: 'no watermarks, no extra people, no blurry edges'"
        ),
    ),
    # ── Output controls ──────────────────────────────────────────────────
    aspect_ratio: str = Form(
        DEFAULTS["aspect_ratio"],
        description="Output aspect ratio: " + ", ".join(ALLOWED_ASPECT_RATIOS),
    ),
    image_size: str = Form(
        DEFAULTS["image_size"],
        description="Output resolution: " + ", ".join(ALLOWED_IMAGE_SIZES),
    ),
):
    """
    Generate a branded marketing banner for **any** campaign or topic.

    The user provides creative inputs (subject, setting, style, lighting, etc.)
    and the API assembles them into an optimised prompt following
    [Nano Banana Pro best practices](https://ai.google.dev/gemini-api/docs/image-generation).

    **Prompt structure** (applied automatically):
    `Style + Subject → Setting → Composition → Lighting → Brand → Constraints`
    """

    # ── Validation ────────────────────────────────────────────────────────
    if aspect_ratio not in ALLOWED_ASPECT_RATIOS:
        raise HTTPException(
            status_code=400,
            detail=f"Invalid aspect_ratio '{aspect_ratio}'. Allowed: {ALLOWED_ASPECT_RATIOS}",
        )
    if image_size not in ALLOWED_IMAGE_SIZES:
        raise HTTPException(
            status_code=400,
            detail=f"Invalid image_size '{image_size}'. Allowed: {ALLOWED_IMAGE_SIZES}",
        )
    if visual_style not in VISUAL_STYLE_MAP:
        raise HTTPException(
            status_code=400,
            detail=f"Invalid visual_style '{visual_style}'. Allowed: {list(VISUAL_STYLE_MAP.keys())}",
        )
    if composition not in COMPOSITION_MAP:
        raise HTTPException(
            status_code=400,
            detail=f"Invalid composition '{composition}'. Allowed: {list(COMPOSITION_MAP.keys())}",
        )
    if lighting not in LIGHTING_MAP:
        raise HTTPException(
            status_code=400,
            detail=f"Invalid lighting '{lighting}'. Allowed: {list(LIGHTING_MAP.keys())}",
        )

    # ── Read logo (optional) ──────────────────────────────────────────────
    has_logo = False
    reference_images: list[tuple[bytes, str]] = []

    if logo_file is not None:
        logo_bytes = await logo_file.read()
        if len(logo_bytes) > 0:
            logo_mime = logo_file.content_type or "image/png"
            reference_images.append((logo_bytes, logo_mime))
            has_logo = True
            logger.info(
                "Logo received: %s (%s, %d bytes)",
                logo_file.filename,
                logo_mime,
                len(logo_bytes),
            )

    # ── Build prompt ──────────────────────────────────────────────────────
    prompt_inputs = PromptInputs(
        subject=subject,
        setting=setting,
        items_in_scene=items_in_scene,
        visual_style=visual_style,
        composition=composition,
        lighting=lighting,
        logo_size=logo_size if has_logo else "none",
        campaign_text=campaign_text,
        text_style=text_style,
        website_url=website_url,
        negative_prompt=negative_prompt,
    )
    final_prompt = build_prompt(prompt_inputs, has_logo=has_logo)
    logger.info("Final prompt: %s", final_prompt)

    # ── Generate image ────────────────────────────────────────────────────
    try:
        result = generate_image(
            prompt=final_prompt,
            aspect_ratio=aspect_ratio,
            image_size=image_size,
            reference_images=reference_images if reference_images else None,
        )
    except RuntimeError as e:
        raise HTTPException(status_code=500, detail=str(e))
    except Exception as e:
        logger.exception("Gemini API call failed")
        raise HTTPException(status_code=502, detail=f"Image generation failed: {e}")

    # ── Upload to GCS ─────────────────────────────────────────────────────
    output_path = generate_output_path(subject=subject, prefix="banners")
    try:
        gcs_result = upload_to_gcs(result.image_bytes, output_path)
    except Exception as e:
        logger.exception("GCS upload failed")
        raise HTTPException(status_code=500, detail=f"GCS upload failed: {e}")

    # ── Respond ───────────────────────────────────────────────────────────
    return JSONResponse(
        status_code=200,
        content={
            "status": "success",
            "gcs_uri": gcs_result.gcs_uri,
            "public_url": gcs_result.public_url,
            "model": MODEL_ID,
            "prompt_used": final_prompt,
            "aspect_ratio": aspect_ratio,
            "image_size": image_size,
            "model_commentary": result.model_text,
        },
    )


# ---------------------------------------------------------------------------
# Entrypoint
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    import uvicorn

    uvicorn.run(
        "app.main:app",
        host="0.0.0.0",
        port=PORT,
        reload=True,
    )
