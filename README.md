# Mobily Creative Studio API — v4.1.1

Generate branded marketing **images and videos** for any campaign using Google's **Gemini** (image) and **Veo** (video) models on Vertex AI.

---

## Table of Contents

1. [Project Structure](#project-structure)
2. [How It Works — End-to-End Flow](#how-it-works--end-to-end-flow)
3. [API Endpoints](#api-endpoints)
   - [Image Generation](#image-generation)
   - [Video Generation](#video-generation)
   - [Brand Assets](#brand-assets)
   - [System / Data](#system--data)
4. [Field Reference](#field-reference)
5. [Presets & Allowed Values](#presets--allowed-values)
6. [Audio Support](#audio-support)
7. [Logo Handling](#logo-handling)
8. [Text Blocks](#text-blocks)
9. [Setup & Run](#setup--run)
10. [Docker & Cloud Run](#docker--cloud-run)
11. [Prompt Best Practices](#prompt-best-practices)

---

## Project Structure

```
content_generation/
├── app.py                             # FastAPI app — all endpoints live here
├── requirements.txt
├── Dockerfile
├── src/
│   ├── config.py                      # Env vars, presets, SDK clients
│   ├── schemas.py                     # Pydantic enums & request/response schemas
│   ├── schema/
│   │   └── responses.py               # API response Pydantic models
│   ├── core/
│   │   ├── prompt_generator.py        # Image prompt builder (PromptInputs + build_prompt)
│   │   ├── video_prompt_generator.py  # Video prompt builder (VideoPromptInputs + build_video_prompt)
│   │   ├── auth.py                    # JWT auth (currently disabled — endpoints commented out)
│   │   └── database.py                # SQLAlchemy async DB (used by auth)
│   └── helpers/
│       ├── utils.py                   # generate_image, upload_to_gcs, overlay_logo, trending events
│       ├── image_helpers.py           # Image-specific helpers
│       ├── video_helpers.py           # Video endpoint shared logic (validation, prompt, audio)
│       └── video_utils.py             # generate_video, TTS, audio merge, GCS video ops
```

| File | Responsibility |
|---|---|
| `app.py` | Mounts all endpoints, validates inputs, orchestrates helpers |
| `src/config.py` | All allowed values, preset maps, SDK clients (`genai_client`, `gcs_client`) |
| `src/core/prompt_generator.py` | Assembles image prompts from structured `PromptInputs` |
| `src/core/video_prompt_generator.py` | Assembles video prompts from `VideoPromptInputs` |
| `src/helpers/utils.py` | Calls Gemini image API, uploads to GCS, overlays logo |
| `src/helpers/video_helpers.py` | Validates video params, builds Veo prompt, handles audio pipeline |
| `src/helpers/video_utils.py` | Calls Veo API, synthesizes TTS speech, merges audio onto video |

---

## How It Works — End-to-End Flow

### Image Generation Flow

```
POST /generate-image
        │
        ▼
  1. Validate inputs (style, framing, lighting, tone, colors, logo)
        │
        ▼
  2. Read logo
     ├─ From upload (logo_file)
     └─ From GCS library (logo_name → download_gcs_logo)
        │
        ▼
  3. Parse text blocks
     ├─ Frontend: JSON array in "text_blocks" form field
     └─ Swagger tester: individual text_* fields assembled into one block
        │
        ▼
  4. build_prompt(PromptInputs) → structured text prompt
        │
        ▼
  5. For each variation (1–4):
     ├─ generate_image(prompt, aspect_ratio, resolution, [logo_bytes as ref])
     │   └─ Calls Gemini image model on Vertex AI
     ├─ If logo present but no logo_comments → overlay_logo() composites logo in post-processing
     └─ upload_to_gcs() → returns GCS URI + public URL
        │
        ▼
  6. Return JSON:
     { status, images: [{gcs_uri, public_url, model_commentary}], prompt_used, logo, ... }
```

> **Logo placement logic:** If `logo_comments` is provided, the logo is passed as a reference image to Gemini and it places it. If `logo_comments` is empty, the logo is composited onto the output image in post-processing via Pillow (`overlay_logo`).

---

### Video Generation Flow (Text-to-Video, Image-to-Video, Reference-to-Video)

```
POST /generate-video/text   (or /image or /reference)
        │
        ▼
  1. Clean & validate inputs
     └─ validate_video_params() — checks aspect ratio, duration, resolution, counts, dropdowns
        │
        ▼
  2. Read logo (optional) — upload or GCS library
        │
        ▼
  3. build_veo_prompt(VideoPromptInputs) → structured Veo prompt string
        │
        ▼
  4. For image/reference modes: upload source image(s) to GCS → get GCS URI(s)
        │
        ▼
  5. generate_video(prompt, mode, aspect_ratio, duration, resolution, ...)
     └─ Calls Veo model on Vertex AI → returns GCS URI(s)
        │
        ▼
  6. Apply logo overlay on video (if logo provided)
     └─ download video → overlay_logo_on_video → re-upload to GCS
        │
        ▼
  7. handle_audio() — auto-detects mode:
     ├─ "upload"  → merge uploaded audio file onto video
     ├─ "tts"     → synthesize_speech(dialogue, language) → merge onto video
     └─ "none"    → skip
        │
        ▼
  8. Return VideoGenerationResponse:
     { status, gcs_uri, gcs_uris, public_url, model, generation_mode,
       prompt_used, aspect_ratio, resolution, duration_seconds,
       audio, end_frame, reference_images, operation }
```

---

### Refine Flow (Image & Video)

- **`POST /refine-image`** — Send an existing image + edit instructions. Gemini edits only what is explicitly asked and preserves the logo.
- **`POST /refine-video`** — Upload a video + edit instructions. Veo extends/edits via `extend_video` mode (always 7 seconds).

---

### Trending Events

```
GET /api/trending-events
        │
        ▼
  get_trending_events_cached()
  └─ Returns a list of upcoming events (name, date, category, description)
     for campaign planning suggestions
```

---

## API Endpoints

### Image Generation

| Method | Path | Description |
|---|---|---|
| `POST` | `/generate-image` | Generate 1–4 branded image variations |
| `POST` | `/refine-image` | Edit / refine an existing image |

### Video Generation

| Method | Path | Description |
|---|---|---|
| `POST` | `/generate-video/text` | Text-to-Video |
| `POST` | `/generate-video/image` | Image-to-Video (animate a still image) |
| `POST` | `/generate-video/reference` | Reference-to-Video (identity preservation, Veo 3.1) |
| `POST` | `/refine-video` | Extend / edit a generated video |

### Brand Assets

| Method | Path | Description |
|---|---|---|
| `GET` | `/logos` | List all pre-uploaded logos with preview URLs |
| `GET` | `/logos/{name}` | Serve a logo image directly from GCS |
| `GET` | `/colors` | Mobily brand palette (grouped by category, with hex codes) |
| `GET` | `/brand-assets` | Returns GCS bucket name and path prefixes |

### System / Data

| Method | Path | Description |
|---|---|---|
| `GET` | `/health` | Liveness probe — returns version + model info |
| `GET` | `/options` | All valid values for image endpoint dropdowns |
| `GET` | `/video-options` | All valid values for video endpoint dropdowns |
| `GET` | `/content?gcs_uri=...` | Proxy-serve any GCS image or video to the frontend |
| `GET` | `/api/trending-events` | Upcoming events for campaign inspiration |

Swagger UI: **http://localhost:8080/docs**

---

## Field Reference

### `POST /generate-image`

| Field | Type | Required | Default | Description |
|---|---|---|---|---|
| `subject` | string | **yes** | — | Main subject of the image |
| `action` | string | no | `""` | What the subject is doing |
| `items_in_scene` | string | no | `""` | Extra objects / elements in the scene |
| `setting` | string | no | `""` | Environment, location, time of day |
| `logo_file` | file (PNG) | no | — | Upload a logo directly |
| `logo_name` | string | no | `""` | Name of a pre-uploaded GCS logo (see `GET /logos`) |
| `logo_size` | string | no | `medium` | Logo size on the image |
| `logo_position` | string | no | `top_right` | Logo placement position |
| `logo_comments` | string | no | `""` | Instructions for LLM-placed logo (e.g. "inside white circle") |
| `text_content` | string | no | `""` | Text to render on the image |
| `text_font` | string | no | `Arial` | Font for the text block |
| `text_size` | string | no | `M` | S / M / L |
| `text_color` | string | no | `white` | Colour key from brand palette |
| `text_weight` | string | no | `bold` | Font weight |
| `text_position` | string | no | `bottom_center` | Where text sits on the image |
| `text_language` | string | no | `ar` | Language of text content |
| `text_notes` | string | no | `""` | Extra styling instructions |
| `text_blocks` | JSON string | no | — | Array of text blocks (frontend use — overrides individual text_* fields) |
| `visual_style` | string | no | `photorealistic` | Art direction style |
| `framing` | string | no | `default` | Composition / camera framing |
| `lighting` | string | no | `default` | Lighting style |
| `tone` | string | no | `default` | Mood / tone |
| `color_grading` | string | no | `default` | Comma-separated colour palette key(s); first = dominant |
| `image_size` | string | no | `16:9` | Output aspect ratio |
| `resolution` | string | no | `1K` | Output resolution |
| `variations` | int | no | `1` | Number of images to generate (1–4) |
| `negative_prompt` | string | no | `""` | What to exclude from the image |

---

### `POST /generate-video/text`

| Field | Type | Required | Default | Description |
|---|---|---|---|---|
| `subject` | string | **yes** | — | Main subject |
| `action` | string | no | `""` | What the subject is doing |
| `dialogue` | string | no | `""` | Voiceover script — fed to TTS audio only |
| `scene_context` | string | no | `""` | Environment / location / time |
| `camera_angle` | string | no | `eye_level` | Camera perspective |
| `camera_movement` | string | no | `static` | Camera movement type |
| `framing` | string | no | `standard_50mm` | Lens / framing |
| `video_motion` | string | no | `normal` | Temporal motion pacing |
| `visual_style` | string | no | `photorealistic` | Visual style |
| `tone` | string | no | `warm` | Mood |
| `audio_file` | file | no | — | Upload custom audio (overrides TTS) |
| `sound_ambience` | string | no | `none` | Background sound bed |
| `Language` | string | no | `ar-XA` | TTS language code |
| `video_size` | string | no | `16:9` | Output aspect ratio |
| `duration_seconds` | int | no | `8` | Duration: 4, 6, or 8 seconds |
| `number_of_videos` | int | no | `1` | Variants to generate (1–4) |
| `resolution` | string | no | `720p` | Output resolution |
| `negative_prompt` | string | no | `""` | What to avoid |
| `logo_file` | file (PNG) | no | — | Upload a logo |
| `logo_name` | string | no | `""` | GCS logo name |
| `logo_position` | string | no | `top_right` | Logo placement |
| `logo_size` | string | no | `medium` | Logo size |

---

### `POST /generate-video/image`

Same as Text-to-Video **plus**:

| Field | Type | Required | Description |
|---|---|---|---|
| `source_image` | file | **yes** | Source / reference image to animate |
| `end_frame_image` | file | no | Optional end frame; triggers Veo model upgrade to support end-frame mode |

---

### `POST /generate-video/reference`

Same creative controls as other video endpoints **plus**:

| Field | Type | Required | Description |
|---|---|---|---|
| `source_image` | file | **yes** | Primary reference image (identity to preserve) |
| `ref_image_2` | file | no | Second reference image |
| `ref_image_3` | file | no | Third reference image |
| `subject` | string | **yes** | Subject description matching the images |

> Duration is fixed at **8 seconds** for reference-to-video.

---

## Presets & Allowed Values

Call `GET /options` (image) or `GET /video-options` (video) to get all valid values programmatically.

### Image Presets

| Parameter | Options |
|---|---|
| **Visual Style** | `photorealistic`, `3d_render`, `illustration`, `flat_design`, `watercolor`, `cinematic`, `anime`, `minimalist`, `vintage_poster`, `isometric` |
| **Framing** | `default`, `close_up`, `wide`, `medium`, `birds_eye`, `low_angle`, `isometric`, `rule_of_thirds` |
| **Lighting** | `default`, `golden_hour`, `blue_hour`, `studio`, `neon`, `dramatic`, `soft_natural`, `backlit`, `candlelight` |
| **Tone** | `default`, `warm`, `cool`, `neutral`, `dramatic`, `vibrant`, `muted` |
| **Color Grading** | `default`, `mobily_green`, `mobily_dark`, `warm_gold`, `cool_blue`, `vibrant`, `monochrome`, `sunset`, `neon_pop` |
| **Logo Size** | `small`, `medium`, `large`, `none` |
| **Logo Position** | `top_left`, `top_center`, `top_right`, `bottom_left`, `bottom_center`, `bottom_right`, `center` |
| **Image Size (aspect ratio)** | `1:1`, `3:2`, `2:3`, `3:4`, `4:3`, `4:5`, `5:4`, `9:16`, `16:9`, `21:9` |
| **Resolution** | `1K`, `2K`, `4K` |

### Video Presets

| Parameter | Options |
|---|---|
| **Camera Angle** | `eye_level`, `low_angle`, `high_angle`, `birds_eye`, `dutch_angle`, `close_up`, `extreme_close_up`, `wide_shot`, `medium_shot`, `over_the_shoulder` |
| **Camera Movement** | `static`, `pan`, `tilt`, `dolly_in`, `dolly_out`, `tracking`, `crane`, `handheld`, `zoom_in`, `zoom_out`, `orbit` |
| **Framing (Lens)** | `wide_angle_24mm`, `standard_50mm`, `telephoto_85mm`, `macro`, `fish_eye`, `tilt_shift`, `anamorphic`, `shallow_dof`, `bokeh`, `lens_flare` |
| **Video Motion** | `slow_motion`, `fast_paced`, `time_lapse`, `normal`, `reverse`, `freeze_frame`, `loop` |
| **Visual Style** | `photorealistic`, `cinematic`, `anime`, `watercolor`, `retro`, `neon`, `3d_render`, `stop_motion`, `minimalist` |
| **Tone** | `warm`, `cool`, `neutral`, `dramatic`, `vibrant`, `muted`, `serene` |
| **Sound Ambience** | `none`, `office_hum`, `nature`, `city`, `crowd`, `festive`, `corporate`, `dramatic` |
| **Video Size** | `16:9`, `9:16` |
| **Duration** | `4`, `6`, `8` (seconds) |
| **Resolution** | `720p`, `1080p` |
| **TTS Language** | `ar-XA`, `en-US`, `en-GB`, `fr-FR`, `es-ES` (and others — see `/video-options`) |

---

## Audio Support

Audio is **auto-detected** from what you provide:

| What you send | Mode | Result |
|---|---|---|
| `audio_file` upload | `upload` | Your audio merged onto the video |
| `dialogue` text (no file) | `tts` | Text synthesized to speech via Google TTS, merged onto video |
| Neither | `none` | Video returned as-is (Veo's built-in audio if `generate_audio=True`) |

The audio response block always tells you which mode was used:

```json
"audio": {
  "mode": "tts",
  "merged": true,
  "tts_language": "ar-XA",
  "tts_chars": 42
}
```

---

## Logo Handling

Two ways to supply a logo on **every** endpoint:

1. **Direct upload** — `logo_file` (PNG binary). Takes priority.
2. **GCS Library** — `logo_name` (e.g. `mobily_logo.png`). The API fetches it from GCS. Browse available logos via `GET /logos`.

**Placement is decided by `logo_comments`:**

- `logo_comments` is **empty** → logo is composited in post-processing at the specified `logo_position` and `logo_size` (Pillow overlay, pixel-perfect).
- `logo_comments` is **filled** → logo is passed as a reference image to Gemini so the model places it according to your instructions (e.g. "inside a white circle", "watermark in corner").

Logo info is always included in the response:

```json
"logo": {
  "applied": true,
  "source": "gcs:mobily_logo.png",
  "position": "top_right",
  "size": "medium"
}
```

---

## Text Blocks

The image endpoint supports **multiple independent text blocks** on one image.

**Frontend (JSON):** Send a `text_blocks` form field containing a JSON array:

```json
[
  {
    "text": "رمضان كريم",
    "font": "Noto Naskh Arabic",
    "size": "L",
    "color": "mobily_green",
    "weight": "bold",
    "position": "bottom_center",
    "language": "ar",
    "comments": "gold foil effect"
  }
]
```

**Swagger / single block:** Use the individual `text_content`, `text_font`, `text_size`, `text_color`, `text_weight`, `text_position`, `text_language`, `text_notes` fields.

Available fonts: see `GET /options` → `fonts`.

---

## Setup & Run

### Prerequisites

- Python 3.11+
- Google Cloud project with Vertex AI, GCS, and TTS APIs enabled
- Application Default Credentials (`gcloud auth application-default login`)

### Install

```bash
python -m venv venv && source venv/bin/activate   # Windows: venv\Scripts\activate
pip install -r requirements.txt
```

### Environment Variables

```bash
export GOOGLE_CLOUD_PROJECT=mobily-genai
export GOOGLE_CLOUD_LOCATION=global        # Required for Gemini 3 image models
export GCS_BUCKET=content_creation_data
```

### Run

```bash
# Direct
python app.py

# Or with uvicorn
uvicorn app:app --host 0.0.0.0 --port 8080 --reload
```

Swagger UI: **http://localhost:8080/docs**

---

## Docker & Cloud Run

### Build & run locally

```bash
docker build -t mobily-creative-studio .
docker run -p 8080:8080 \
  -e GOOGLE_CLOUD_PROJECT=mobily-genai \
  -e GOOGLE_CLOUD_LOCATION=global \
  -e GCS_BUCKET=content_creation_data \
  mobily-creative-studio
```

### Deploy to Cloud Run

```bash
gcloud run deploy mobily-creative-studio \
  --source . \
  --region us-central1 \
  --set-env-vars "GOOGLE_CLOUD_PROJECT=mobily-genai,GOOGLE_CLOUD_LOCATION=global,GCS_BUCKET=content_creation_data" \
  --allow-unauthenticated
```

The Dockerfile installs `ffmpeg` (required for audio merge) and font packages for Pillow text rendering.

---

## Prompt Best Practices

These principles are baked into the prompt builders, but useful when writing `subject` / `scene_context` descriptions:

1. **Be specific** — "warm golden-hour light with long soft shadows" beats "nice lighting"
2. **Put the most important detail first** — subject and action before secondary props
3. **Use descriptive adjectives** — paint a clear picture
4. **Name the style explicitly** — "photorealistic", "cinematic", "3D render"
5. **Keep text short** — text rendered in images works best under 25 characters
6. **Use negative prompts** — "no watermarks, no extra people" reduces artifacts
7. **Iterate** — start broad, refine with the `/refine-image` or `/refine-video` endpoints

---

## Key Technical Notes

1. **Gemini image models require `GOOGLE_CLOUD_LOCATION=global`** — `us-central1` will not work
2. **`response_modalities` must include both `"TEXT"` and `"IMAGE"`** — image-only output is unsupported
3. Uses the **`google-genai` SDK (≥ 1.51.0)**, not the deprecated `vertexai.generative_models`
4. Logo is sent as inline bytes via `Part.from_bytes()` when LLM-placed — positioned before the text prompt
5. **Video operations are long-running** — `generate_video` polls until complete; `operation` in the response is the Vertex AI operation name for debugging
6. **Auth endpoints exist** (`/api/auth/register`, `/api/auth/login`, `/api/auth/me`) but are currently **disabled** (commented out in `app.py`)

---

## License

Internal — Mobily × Artefact
