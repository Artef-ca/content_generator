"""
Brand Scraper — extracts brand assets from a website URL.

Takes a URL → scrapes logos, colors, fonts, brand elements →
saves to GCS bucket → generates brand context for prompts.

Usage as script:
    python -m src.tools.brand_scraper https://www.mobily.com.sa

Usage as import:
    from src.tools.brand_scraper import scrape_brand
    result = scrape_brand("https://www.mobily.com.sa")

What it extracts:
    - Logos: favicon, apple-touch-icon, og:image, logo tags
    - Colors: meta theme-color, CSS variables, dominant colors from images
    - Fonts: Google Fonts, CSS font-family declarations
    - Brand text: meta description, og:title, og:site_name
"""

import io
import re
import json
import logging
import hashlib
from collections import Counter
from dataclasses import dataclass, field
from urllib.parse import urljoin, urlparse

import requests
from bs4 import BeautifulSoup
from PIL import Image

logger = logging.getLogger("brand_scraper")


@dataclass
class BrandAsset:
    """A downloaded brand asset."""
    url: str
    data: bytes
    mime_type: str
    category: str      # "logo", "icon", "og_image", "style_ref"
    filename: str


@dataclass
class BrandProfile:
    """Extracted brand profile."""
    url: str
    name: str = ""
    description: str = ""
    colors: list[str] = field(default_factory=list)       # hex colors
    color_names: dict = field(default_factory=dict)        # hex → role
    fonts: list[str] = field(default_factory=list)
    assets: list[BrandAsset] = field(default_factory=list)
    style_context: str = ""
    errors: list[str] = field(default_factory=list)


# ---------------------------------------------------------------------------
# Color extraction from images
# ---------------------------------------------------------------------------
def _extract_dominant_colors(img_bytes: bytes, n: int = 5) -> list[str]:
    """Extract top N dominant colors from an image as hex strings."""
    try:
        img = Image.open(io.BytesIO(img_bytes)).convert("RGB")
        # Resize for speed
        img = img.resize((150, 150), Image.LANCZOS)
        pixels = list(img.getdata())

        # Quantize colors (round to nearest 16)
        quantized = []
        for r, g, b in pixels:
            qr = (r // 16) * 16
            qg = (g // 16) * 16
            qb = (b // 16) * 16
            quantized.append((qr, qg, qb))

        # Count and get top colors, skip near-white and near-black
        counts = Counter(quantized)
        colors = []
        for (r, g, b), _ in counts.most_common(n * 3):
            brightness = (r + g + b) / 3
            if 30 < brightness < 240:  # skip pure black/white
                hex_color = f"#{r:02x}{g:02x}{b:02x}"
                colors.append(hex_color)
                if len(colors) >= n:
                    break
        return colors
    except Exception as e:
        logger.warning("Color extraction failed: %s", e)
        return []


def _extract_css_colors(css_text: str) -> list[str]:
    """Extract color values from CSS text."""
    colors = set()

    # Hex colors
    for match in re.findall(r'#([0-9a-fA-F]{3,8})\b', css_text):
        if len(match) in (3, 6):
            colors.add(f"#{match.lower()}")

    # rgb/rgba
    for match in re.findall(r'rgb\((\d+)\s*,\s*(\d+)\s*,\s*(\d+)', css_text):
        r, g, b = int(match[0]), int(match[1]), int(match[2])
        if 30 < (r + g + b) / 3 < 240:
            colors.add(f"#{r:02x}{g:02x}{b:02x}")

    # CSS custom properties with color values
    for match in re.findall(r'--[\w-]*color[\w-]*:\s*([^;]+)', css_text, re.IGNORECASE):
        hex_match = re.search(r'#([0-9a-fA-F]{3,6})', match)
        if hex_match:
            colors.add(f"#{hex_match.group(1).lower()}")

    return list(colors)[:20]


# ---------------------------------------------------------------------------
# HTML scraping
# ---------------------------------------------------------------------------
def _download(url: str, timeout: int = 15) -> bytes | None:
    """Download a URL, return bytes or None."""
    try:
        resp = requests.get(url, timeout=timeout, headers={
            "User-Agent": "MobilyBrandScraper/1.0"
        })
        if resp.status_code == 200 and len(resp.content) > 0:
            return resp.content
    except Exception as e:
        logger.warning("Download failed %s: %s", url, e)
    return None


def _find_logo_urls(soup: BeautifulSoup, base_url: str) -> list[tuple[str, str]]:
    """Find logo image URLs from HTML. Returns [(url, category), ...]"""
    found = []

    # 1. Favicon
    for link in soup.find_all("link", rel=lambda r: r and "icon" in " ".join(r).lower()):
        href = link.get("href")
        if href:
            found.append((urljoin(base_url, href), "icon"))

    # 2. Apple touch icon
    for link in soup.find_all("link", rel=lambda r: r and "apple-touch-icon" in " ".join(r).lower()):
        href = link.get("href")
        if href:
            found.append((urljoin(base_url, href), "logo"))

    # 3. og:image
    og = soup.find("meta", property="og:image")
    if og and og.get("content"):
        found.append((urljoin(base_url, og["content"]), "og_image"))

    # 4. Images with "logo" in src, alt, class, or id
    for img in soup.find_all("img"):
        src = img.get("src", "")
        alt = img.get("alt", "")
        cls = " ".join(img.get("class", []))
        img_id = img.get("id", "")
        searchable = f"{src} {alt} {cls} {img_id}".lower()
        if "logo" in searchable:
            found.append((urljoin(base_url, src), "logo"))

    # 5. SVG with "logo" in class/id
    for svg in soup.find_all("svg"):
        cls = " ".join(svg.get("class", []))
        svg_id = svg.get("id", "")
        if "logo" in f"{cls} {svg_id}".lower():
            # Can't easily download inline SVG as image, skip
            pass

    # Deduplicate by URL
    seen = set()
    unique = []
    for url, cat in found:
        if url not in seen:
            seen.add(url)
            unique.append((url, cat))

    return unique


def _find_colors_from_html(soup: BeautifulSoup, html_text: str) -> dict[str, str]:
    """Extract colors from HTML meta tags and inline CSS."""
    colors = {}

    # meta theme-color
    theme = soup.find("meta", attrs={"name": "theme-color"})
    if theme and theme.get("content"):
        colors[theme["content"].lower()] = "theme-color"

    # msapplication-TileColor
    tile = soup.find("meta", attrs={"name": "msapplication-TileColor"})
    if tile and tile.get("content"):
        colors[tile["content"].lower()] = "tile-color"

    # Inline style colors
    for style in soup.find_all("style"):
        if style.string:
            for c in _extract_css_colors(style.string):
                if c not in colors:
                    colors[c] = "css"

    return colors


def _find_fonts(soup: BeautifulSoup, html_text: str) -> list[str]:
    """Extract font names from HTML."""
    fonts = set()

    # Google Fonts links
    for link in soup.find_all("link", href=True):
        href = link["href"]
        if "fonts.googleapis.com" in href:
            match = re.search(r'family=([^&:]+)', href)
            if match:
                font_name = match.group(1).replace("+", " ")
                fonts.add(font_name)

    # CSS font-family
    for style in soup.find_all("style"):
        if style.string:
            for match in re.findall(r'font-family:\s*["\']?([^;"\',]+)', style.string):
                name = match.strip().strip("'\"")
                if name and name not in ("inherit", "initial", "sans-serif", "serif", "monospace"):
                    fonts.add(name)

    return list(fonts)[:10]


# ---------------------------------------------------------------------------
# Main scraper
# ---------------------------------------------------------------------------
def scrape_brand(url: str) -> BrandProfile:
    """
    Scrape a website for brand assets.

    Parameters
    ----------
    url : str
        Website URL (e.g. 'https://www.mobily.com.sa')

    Returns
    -------
    BrandProfile
        Extracted brand profile with colors, logos, fonts.
    """
    profile = BrandProfile(url=url)

    # Fetch page
    logger.info("Scraping brand from: %s", url)
    html_bytes = _download(url)
    if not html_bytes:
        profile.errors.append(f"Failed to fetch {url}")
        return profile

    html_text = html_bytes.decode("utf-8", errors="replace")
    soup = BeautifulSoup(html_text, "html.parser")

    # Brand name
    og_name = soup.find("meta", property="og:site_name")
    if og_name and og_name.get("content"):
        profile.name = og_name["content"]
    elif soup.title:
        profile.name = soup.title.get_text(strip=True).split("|")[0].strip()

    # Description
    og_desc = soup.find("meta", property="og:description")
    if og_desc and og_desc.get("content"):
        profile.description = og_desc["content"]
    else:
        meta_desc = soup.find("meta", attrs={"name": "description"})
        if meta_desc and meta_desc.get("content"):
            profile.description = meta_desc["content"]

    # Colors from HTML/CSS
    html_colors = _find_colors_from_html(soup, html_text)
    profile.color_names = html_colors

    # Fonts
    profile.fonts = _find_fonts(soup, html_text)

    # Find and download logos
    logo_urls = _find_logo_urls(soup, url)
    logger.info("Found %d logo candidates", len(logo_urls))

    all_image_colors = []
    for img_url, category in logo_urls[:10]:  # cap at 10
        data = _download(img_url)
        if not data:
            continue

        # Determine filename
        parsed = urlparse(img_url)
        ext = parsed.path.split(".")[-1].lower() if "." in parsed.path else "png"
        if ext not in ("png", "jpg", "jpeg", "svg", "ico", "webp"):
            ext = "png"
        short_hash = hashlib.md5(img_url.encode()).hexdigest()[:8]
        filename = f"{category}_{short_hash}.{ext}"

        # Detect mime
        mime_map = {"png": "image/png", "jpg": "image/jpeg", "jpeg": "image/jpeg",
                    "svg": "image/svg+xml", "ico": "image/x-icon", "webp": "image/webp"}
        mime = mime_map.get(ext, "image/png")

        profile.assets.append(BrandAsset(
            url=img_url, data=data, mime_type=mime,
            category=category, filename=filename,
        ))

        # Extract colors from this image
        if ext not in ("svg", "ico"):
            img_colors = _extract_dominant_colors(data, n=3)
            all_image_colors.extend(img_colors)

    # Merge all colors
    all_colors = list(html_colors.keys()) + all_image_colors
    # Deduplicate preserving order
    seen = set()
    unique_colors = []
    for c in all_colors:
        if c not in seen:
            seen.add(c)
            unique_colors.append(c)
    profile.colors = unique_colors[:15]

    # Generate style context
    color_str = ", ".join(profile.colors[:6]) if profile.colors else "brand colors"
    font_str = ", ".join(profile.fonts[:3]) if profile.fonts else "modern typography"
    profile.style_context = (
        f"Brand-inspired visual style with {color_str} as accent colors. "
        f"Typography: {font_str}. "
        f"High production value, polished and aspirational feel."
    )

    logger.info(
        "Brand scrape complete: name=%s, %d colors, %d fonts, %d assets",
        profile.name, len(profile.colors), len(profile.fonts), len(profile.assets),
    )
    return profile


# ---------------------------------------------------------------------------
# GCS upload
# ---------------------------------------------------------------------------
def upload_brand_to_gcs(
    profile: BrandProfile,
    bucket_name: str = "content_creation_data",
    prefix: str = "brand_assets",
) -> dict:
    """Upload scraped brand assets to GCS.

    Folder structure:
        {prefix}/{brand_name}/logos/
        {prefix}/{brand_name}/colors/
        {prefix}/{brand_name}/brand_profile.json

    Returns dict with GCS paths.
    """
    from src.config import gcs_client

    bucket = gcs_client.bucket(bucket_name)
    brand_slug = re.sub(r'[^a-z0-9]+', '_', profile.name.lower()).strip('_') or "brand"
    base = f"{prefix}/{brand_slug}"

    uploaded = {"logos": [], "profile": None}

    # Upload assets
    for asset in profile.assets:
        blob_path = f"{base}/logos/{asset.filename}"
        blob = bucket.blob(blob_path)
        blob.upload_from_string(asset.data, content_type=asset.mime_type)
        gcs_uri = f"gs://{bucket_name}/{blob_path}"
        public_url = f"https://storage.googleapis.com/{bucket_name}/{blob_path}"
        uploaded["logos"].append({
            "filename": asset.filename,
            "category": asset.category,
            "source_url": asset.url,
            "gcs_uri": gcs_uri,
            "public_url": public_url,
        })
        logger.info("Uploaded %s → %s", asset.filename, gcs_uri)

    # Save color palette as JSON image (optional — useful for reference)
    # Save brand profile JSON
    profile_data = {
        "name": profile.name,
        "source_url": profile.url,
        "description": profile.description,
        "colors": profile.colors,
        "color_roles": profile.color_names,
        "fonts": profile.fonts,
        "style_context": profile.style_context,
        "assets_count": len(profile.assets),
    }
    profile_blob = f"{base}/brand_profile.json"
    blob = bucket.blob(profile_blob)
    blob.upload_from_string(
        json.dumps(profile_data, indent=2, ensure_ascii=False),
        content_type="application/json",
    )
    uploaded["profile"] = f"gs://{bucket_name}/{profile_blob}"

    logger.info("Brand profile saved: %s", uploaded["profile"])
    return uploaded


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    import sys
    logging.basicConfig(level=logging.INFO)

    if len(sys.argv) < 2:
        print("Usage: python -m src.tools.brand_scraper <URL> [--upload]")
        sys.exit(1)

    target_url = sys.argv[1]
    do_upload = "--upload" in sys.argv

    result = scrape_brand(target_url)

    print(f"\n{'='*60}")
    print(f"Brand: {result.name}")
    print(f"URL: {result.url}")
    print(f"Description: {result.description[:100]}")
    print(f"\nColors ({len(result.colors)}):")
    for c in result.colors:
        role = result.color_names.get(c, "image")
        print(f"  {c} ({role})")
    print(f"\nFonts ({len(result.fonts)}): {', '.join(result.fonts)}")
    print(f"\nAssets ({len(result.assets)}):")
    for a in result.assets:
        print(f"  [{a.category}] {a.filename} ({len(a.data)} bytes) ← {a.url[:80]}")
    print(f"\nStyle Context:\n  {result.style_context}")
    print(f"{'='*60}")

    if do_upload:
        print("\nUploading to GCS...")
        uploaded = upload_brand_to_gcs(result)
        print(f"Profile: {uploaded['profile']}")
        for logo in uploaded['logos']:
            print(f"  {logo['filename']} → {logo['gcs_uri']}")
    elif result.errors:
        print(f"\nErrors: {result.errors}")
