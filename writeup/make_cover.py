"""Generate the ECB cover image (560 × 280) for Kaggle submission.

Uses PIL (Pillow). If not installed: pip install Pillow.
Fallback: writes a plain SVG (vector) that can be converted via any browser.
"""
from pathlib import Path

OUT_PNG = Path(__file__).parent / "cover.png"
OUT_SVG = Path(__file__).parent / "cover.svg"

TITLE = "THE EPISTEMIC CURIE BENCHMARK"
SUBTITLE = "Phase Transitions in LLM Cognition"
TAG = "k*  —  the critical authority level"
METRIC = "First measured: Llama-4-Scout k* = 0.68 (ferromagnetic)"
FOOTER = "40 questions · 4 tracks · 7 models · 2,520 measurements"

BG_DARK = (11, 15, 30)       # deep midnight blue
ACCENT = (138, 180, 248)     # cool ice blue
WARN = (242, 139, 130)       # warm coral
TEXT = (232, 234, 237)       # near-white
DIM = (154, 160, 166)        # gray


def make_png():
    try:
        from PIL import Image, ImageDraw, ImageFont
    except ImportError:
        print("PIL not available; writing SVG fallback instead.")
        return make_svg()

    W, H = 560, 280
    img = Image.new("RGB", (W, H), color=BG_DARK)
    draw = ImageDraw.Draw(img)

    # Try system fonts; fall back to default
    def try_font(names, size):
        for name in names:
            try:
                return ImageFont.truetype(name, size)
            except (OSError, IOError):
                continue
        return ImageFont.load_default()

    title_font = try_font(
        ["/System/Library/Fonts/Supplemental/Futura.ttc",
         "/System/Library/Fonts/HelveticaNeue.ttc",
         "/System/Library/Fonts/Helvetica.ttc",
         "Helvetica-Bold.ttf"],
        22,
    )
    body_font = try_font(
        ["/System/Library/Fonts/HelveticaNeue.ttc",
         "/System/Library/Fonts/Helvetica.ttc"],
        14,
    )
    small_font = try_font(
        ["/System/Library/Fonts/HelveticaNeue.ttc"],
        11,
    )

    # Draw the sigmoid curve (the ECB signature visual)
    import math
    curve_y = 150
    curve_x0 = 30
    curve_x1 = 530
    curve_w = curve_x1 - curve_x0
    k_star_rel = 0.68  # measured: Llama-4-Scout k* = 0.68
    beta = 6.0
    pts = []
    for i in range(curve_w):
        k = i / curve_w
        z = beta * (k - k_star_rel)
        # clamp z to avoid overflow
        if z > 30: p = 1.0
        elif z < -30: p = 0.0
        else: p = 1.0 / (1.0 + math.exp(-z))
        y = curve_y + 30 - int(p * 40)  # map p∈[0,1] → y pixels
        pts.append((curve_x0 + i, y))

    # Draw axis baseline and curve
    draw.line([(curve_x0, curve_y + 30), (curve_x1, curve_y + 30)], fill=DIM, width=1)
    for i in range(len(pts) - 1):
        draw.line([pts[i], pts[i + 1]], fill=ACCENT, width=2)

    # Mark k* location
    k_px = curve_x0 + int(k_star_rel * curve_w)
    draw.line([(k_px, curve_y - 15), (k_px, curve_y + 30)], fill=WARN, width=1)
    draw.text((k_px - 12, curve_y - 30), "k*", fill=WARN, font=body_font)

    # Axis labels
    draw.text((curve_x0 - 5, curve_y + 35), "anonymous", fill=DIM, font=small_font)
    draw.text((curve_x1 - 62, curve_y + 35), "Nobel laureate", fill=DIM, font=small_font)
    draw.text((curve_x0 - 15, curve_y - 10), "resist",  fill=DIM, font=small_font)
    draw.text((curve_x0 - 15, curve_y + 22), "comply",  fill=DIM, font=small_font)

    # Title
    draw.text((30, 18), TITLE, fill=TEXT, font=title_font)
    draw.text((30, 48), SUBTITLE, fill=ACCENT, font=body_font)

    # Tag & metric
    draw.text((30, 80), TAG, fill=WARN, font=body_font)
    draw.text((30, 100), METRIC, fill=DIM, font=small_font)

    # Footer
    draw.text((30, 248), FOOTER, fill=DIM, font=small_font)
    draw.text((420, 248), "Track: Social Cog", fill=ACCENT, font=small_font)

    img.save(OUT_PNG, "PNG")
    print(f"Wrote PNG: {OUT_PNG}")
    return OUT_PNG


def make_svg():
    """Plain SVG fallback — browsers render it perfectly, 560×280, no deps."""
    import math

    # Sigmoid curve points
    curve_x0, curve_x1 = 30, 530
    curve_y = 150
    k_star = 0.55
    beta = 10.0
    pts = []
    for i in range(int(curve_x1 - curve_x0)):
        k = i / (curve_x1 - curve_x0)
        z = beta * (k - k_star)
        if z > 30: p = 1.0
        elif z < -30: p = 0.0
        else: p = 1.0 / (1.0 + math.exp(-z))
        y = curve_y + 30 - p * 40
        pts.append(f"{curve_x0 + i},{y:.1f}")
    polyline_points = " ".join(pts)

    k_px = curve_x0 + k_star * (curve_x1 - curve_x0)

    svg = f'''<?xml version="1.0" encoding="UTF-8"?>
<svg xmlns="http://www.w3.org/2000/svg" width="560" height="280" viewBox="0 0 560 280">
  <defs>
    <style>
      .bg    {{ fill: rgb(11, 15, 30); }}
      .title {{ font-family: -apple-system, Helvetica, sans-serif; font-size: 22px; font-weight: 700; fill: rgb(232, 234, 237); }}
      .subt  {{ font-family: -apple-system, Helvetica, sans-serif; font-size: 14px; fill: rgb(138, 180, 248); }}
      .tag   {{ font-family: -apple-system, Helvetica, sans-serif; font-size: 14px; fill: rgb(242, 139, 130); }}
      .body  {{ font-family: -apple-system, Helvetica, sans-serif; font-size: 11px; fill: rgb(154, 160, 166); }}
      .kmark {{ font-family: -apple-system, Helvetica, sans-serif; font-size: 14px; fill: rgb(242, 139, 130); font-weight: 700; }}
      .curve {{ fill: none; stroke: rgb(138, 180, 248); stroke-width: 2; }}
      .axis  {{ stroke: rgb(154, 160, 166); stroke-width: 1; }}
      .kstar {{ stroke: rgb(242, 139, 130); stroke-width: 1; stroke-dasharray: 3, 3; }}
    </style>
  </defs>
  <rect class="bg" width="560" height="280"/>

  <!-- Title block -->
  <text x="30" y="38" class="title">{TITLE}</text>
  <text x="30" y="60" class="subt">{SUBTITLE}</text>
  <text x="30" y="90" class="tag">{TAG}</text>
  <text x="30" y="108" class="body">{METRIC}</text>

  <!-- Sigmoid curve -->
  <line x1="{curve_x0}" y1="{curve_y + 30}" x2="{curve_x1}" y2="{curve_y + 30}" class="axis"/>
  <polyline class="curve" points="{polyline_points}"/>
  <line x1="{k_px}" y1="{curve_y - 15}" x2="{k_px}" y2="{curve_y + 30}" class="kstar"/>
  <text x="{k_px - 10}" y="{curve_y - 20}" class="kmark">k*</text>

  <!-- Axis labels -->
  <text x="{curve_x0 - 5}" y="{curve_y + 45}" class="body">anonymous</text>
  <text x="{curve_x1 - 62}" y="{curve_y + 45}" class="body">Nobel laureate</text>
  <text x="{curve_x0 - 20}" y="{curve_y - 5}" class="body">resist</text>
  <text x="{curve_x0 - 22}" y="{curve_y + 28}" class="body">comply</text>

  <!-- Footer -->
  <text x="30" y="260" class="body">{FOOTER}</text>
  <text x="420" y="260" class="subt" style="font-size: 11px;">Track: Social Cognition</text>
</svg>
'''
    OUT_SVG.write_text(svg)
    print(f"Wrote SVG: {OUT_SVG}")
    print("Convert SVG → PNG in any browser: open the file and screenshot/export, OR use:")
    print("  rsvg-convert cover.svg -o cover.png")
    print("  (brew install librsvg)")
    return OUT_SVG


if __name__ == "__main__":
    make_png()
    make_svg()
