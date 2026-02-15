# Mobily Creative Studio API

Generate branded marketing banners for **any campaign or topic** using **Gemini 3 Pro Image** (Nano Banana Pro 🍌) on Vertex AI.

The user decides the theme — Ramadan, Eid, National Day, product launch, seasonal sale, or anything else. The API structures their inputs into an optimised prompt following [Google's Nano Banana best practices](https://ai.google.dev/gemini-api/docs/image-generation).

---

## Project Structure

```
mobily_creative_studio/
├── app/
│   ├── __init__.py
│   ├── config.py              # Env vars, constants, presets, SDK clients
│   ├── prompt_generator.py    # Structured prompt builder (Nano Banana best practices)
│   ├── utils.py               # GCS upload, Gemini API wrapper, response parsing
│   └── main.py                # FastAPI app, routes, entrypoint
├── requirements.txt
└── README.md
```

| Module | Responsibility |
|---|---|
| **`config.py`** | Environment variables, allowed values, and preset maps for visual styles, compositions, lighting, and logo sizes. Singleton SDK clients. |
| **`prompt_generator.py`** | `build_prompt()` — assembles user inputs into a structured prompt following the recommended hierarchy. Zero hardcoded themes. |
| **`utils.py`** | `generate_image()` — calls Gemini, handles reference images, parses response. `upload_to_gcs()` — stores the output PNG. |
| **`main.py`** | FastAPI app with `/generate-banner`, `/options`, and `/health` endpoints. |

---

## How the Prompt Builder Works

The prompt generator follows the hierarchy recommended by Google and the Nano Banana Pro community. **Earlier details have more influence on the final result**, so we order them by importance:

```
┌─────────────────────────────────────────────────────────────┐
│  1. Style + Subject    "A photorealistic photograph of..."  │  ← most influence
│  2. Setting            "Setting: a modern Saudi majlis..."  │
│  3. Scene Items        "Include lanterns, dates, coffee..." │
│  4. Composition        "Wide-angle shot, expansive view..." │
│  5. Lighting           "Warm golden-hour sunlight..."       │
│  6. Brand Integration  "Integrate the provided logo as..."  │
│  7. Campaign Text      "Render 'رمضان كريم' in elegant..." │
│  8. Constraints        "No watermarks, no extra people"     │  ← least influence
└─────────────────────────────────────────────────────────────┘
```

The user fills in whichever fields they want. Empty fields are simply skipped — only `subject` is required.

---

## Available Presets

### Visual Styles
`photorealistic` · `3d_render` · `illustration` · `flat_design` · `watercolor` · `cinematic` · `anime` · `minimalist` · `vintage_poster` · `isometric`

### Composition / Camera
`default` · `close_up` · `wide` · `medium` · `birds_eye` · `low_angle` · `isometric` · `rule_of_thirds`

### Lighting / Mood
`default` · `golden_hour` · `blue_hour` · `studio` · `neon` · `dramatic` · `soft_natural` · `backlit` · `candlelight`

### Logo Size
`small` · `medium` · `large` · `none`

### Aspect Ratios
`1:1` · `3:2` · `2:3` · `3:4` · `4:3` · `4:5` · `5:4` · `9:16` · `16:9` · `21:9`

### Resolution
`1K` · `2K` · `4K`

Call `GET /options` to retrieve all of these programmatically.

---

## Setup

```bash
cd mobily_creative_studio

python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

export GOOGLE_CLOUD_PROJECT=mobily-genai
export GOOGLE_CLOUD_LOCATION=global          # Required for Gemini 3 models
export GCS_BUCKET=content_creation_data
```

## Run

```bash
# Option A
python -m app.main

# Option B
uvicorn app.main:app --host 0.0.0.0 --port 8080 --reload
```

Swagger docs: **http://localhost:8080/docs**

---

## API Reference

### `GET /options`
Returns all valid values for every dropdown/select field.

### `GET /health`
Liveness probe.

### `POST /generate-banner`
Generate a banner. All fields are `multipart/form-data`. Only **`subject`** is required.

| Field | Type | Required | Description |
|---|---|---|---|
| `subject` | string | **yes** | Who / what + action (e.g. "A family enjoying Iftar") |
| `setting` | string | no | Where & when (e.g. "a modern Saudi living room at sunset") |
| `items_in_scene` | string | no | Extra objects (e.g. "lanterns, dates, Arabic coffee") |
| `visual_style` | string | no | Art direction (default: `photorealistic`) |
| `composition` | string | no | Camera preset (default: `default`) |
| `lighting` | string | no | Lighting mood (default: `default`) |
| `logo_file` | file | no | Brand logo PNG |
| `logo_size` | string | no | Logo prominence (default: `medium`) |
| `campaign_text` | string | no | Text to render on the banner |
| `text_style` | string | no | Typography direction for the text |
| `negative_prompt` | string | no | What to avoid |
| `aspect_ratio` | string | no | Output ratio (default: `16:9`) |
| `image_size` | string | no | Resolution (default: `1K`) |

---

## Usage Examples

### Ramadan Campaign

```bash
curl -X POST http://localhost:8080/generate-banner \
  -F 'subject=A warm family gathering around a beautifully set Iftar table' \
  -F 'setting=a luxurious modern Saudi majlis with arched windows overlooking the city at sunset' \
  -F 'items_in_scene=traditional lanterns, dates, Arabic coffee, crescent moon decorations' \
  -F 'visual_style=photorealistic' \
  -F 'lighting=golden_hour' \
  -F 'composition=wide' \
  -F 'logo_file=@mobily_logo.png' \
  -F 'logo_size=medium' \
  -F 'campaign_text=رمضان كريم' \
  -F 'text_style=elegant Arabic calligraphy with gold foil effect' \
  -F 'aspect_ratio=16:9' \
  -F 'image_size=2K'
```

### Eid Al-Fitr

```bash
curl -X POST http://localhost:8080/generate-banner \
  -F 'subject=Children in traditional Saudi thobes and dresses exchanging gifts joyfully' \
  -F 'setting=a festive outdoor garden decorated with balloons and lights' \
  -F 'items_in_scene=gift boxes, sweets, Eid decorations, fireworks in the sky' \
  -F 'visual_style=cinematic' \
  -F 'lighting=soft_natural' \
  -F 'logo_file=@mobily_logo.png' \
  -F 'campaign_text=عيد مبارك' \
  -F 'text_style=bold modern Arabic typography' \
  -F 'aspect_ratio=9:16' \
  -F 'image_size=2K'
```

### Saudi National Day (Sep 23)

```bash
curl -X POST http://localhost:8080/generate-banner \
  -F 'subject=A proud celebration of Saudi heritage and vision' \
  -F 'setting=iconic Saudi landmarks with desert dunes in the background' \
  -F 'items_in_scene=Saudi flag, green and white theme, sword dance performers' \
  -F 'visual_style=cinematic' \
  -F 'lighting=dramatic' \
  -F 'composition=wide' \
  -F 'logo_file=@mobily_logo.png' \
  -F 'campaign_text=اليوم الوطني السعودي' \
  -F 'text_style=bold serif with green and white national colors' \
  -F 'aspect_ratio=16:9' \
  -F 'image_size=4K'
```

### 5G Product Launch

```bash
curl -X POST http://localhost:8080/generate-banner \
  -F 'subject=A sleek smartphone floating above a glowing 5G network visualization' \
  -F 'setting=a futuristic digital environment with data streams and light particles' \
  -F 'items_in_scene=5G signal waves, speed indicators, connected devices' \
  -F 'visual_style=3d_render' \
  -F 'lighting=neon' \
  -F 'composition=close_up' \
  -F 'logo_file=@mobily_logo.png' \
  -F 'logo_size=large' \
  -F 'campaign_text=Experience 5G Speed' \
  -F 'text_style=bold futuristic sans-serif with glowing edge' \
  -F 'negative_prompt=no blurry edges, no realistic human faces' \
  -F 'aspect_ratio=16:9' \
  -F 'image_size=2K'
```

### Summer Sale (no logo)

```bash
curl -X POST http://localhost:8080/generate-banner \
  -F 'subject=Tropical beach scene with bold sale typography' \
  -F 'setting=a crystal-clear beach with palm trees and turquoise water' \
  -F 'items_in_scene=sunglasses, surfboard, tropical drinks' \
  -F 'visual_style=flat_design' \
  -F 'lighting=soft_natural' \
  -F 'campaign_text=Summer Sale 50% OFF' \
  -F 'text_style=bold playful sans-serif in hot pink and yellow' \
  -F 'negative_prompt=no people, no watermarks' \
  -F 'aspect_ratio=1:1' \
  -F 'image_size=1K'
```

### Minimal prompt (only subject required)

```bash
curl -X POST http://localhost:8080/generate-banner \
  -F 'subject=A cute cat wearing a tiny astronaut helmet floating in space'
```

---

## Prompt Best Practices (from Google)

These are baked into the API's `prompt_generator.py`, but useful if you're crafting `subject` / `setting` descriptions:

1. **Be specific, not vague** — "warm golden-hour sunlight with long soft shadows" beats "nice lighting"
2. **Put important details first** — subject and action should come before secondary details
3. **Use descriptive language** — detailed adjectives and adverbs paint a clearer picture
4. **Name the style explicitly** — "photorealistic", "watercolor", "3D render" controls the visual mood
5. **Keep text under 25 characters** — longer text may render poorly
6. **Use constraints** — "no watermarks, no extra people" helps avoid common artifacts
7. **Iterate** — start broad, refine with follow-up adjustments

---

## Key Technical Notes

1. **`response_modalities` must include both `"TEXT"` and `"IMAGE"`** — image-only output is not supported
2. **Gemini 3 models require the `global` endpoint**, not `us-central1`
3. Uses the **new `google-genai` SDK** (≥ 1.51.0), not the deprecated `vertexai.generative_models`
4. Logo is sent as inline bytes via `Part.from_bytes()` — placed before the text prompt for best results
5. Logo is **optional** — the API works without a logo upload

---

## Deployment (Cloud Run)

```bash
gcloud run deploy mobily-creative-studio \
  --source . \
  --region us-central1 \
  --set-env-vars "GOOGLE_CLOUD_PROJECT=mobily-genai,GOOGLE_CLOUD_LOCATION=global,GCS_BUCKET=content_creation_data" \
  --allow-unauthenticated
```

---

## License

Internal — Mobily × Artefact
