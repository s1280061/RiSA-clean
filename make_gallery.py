import os
import argparse
import shutil
import html
from pathlib import Path
import pandas as pd

# =========================
# Label definitions
# =========================

LABEL_COLS = [
    "env_rural", "env_city", "env_snowy", "env_sunny", "env_rainy",
    "ego_speed_kmh",
    "ts_final", "brake_final",
    "lane_change_detected_llava",
    "llava_maneuver",
    "detected_vehicle_count"
]

LABEL_NAME_MAP = {
    "env_rural": "Rural",
    "env_city": "City",
    "env_snowy": "Snowy",
    "env_sunny": "Sunny",
    "env_rainy": "Rainy",
    "ego_speed_kmh": "Ego Speed (km/h)",
    "ts_final": "Turn Signal",
    "brake_final": "Brake",
    "lane_change_detected_llava": "LLaVA Lane-Change Recognition",
    "llava_maneuver": "LLaVA Suggested Maneuver",
    "detected_vehicle_count": "Detected Vehicles"
}

MODEL_NAME_MAP = {
    "llava": "LLaVA-1.5-7B",
    "llava_1.5_7b": "LLaVA-1.5-7B",
    "llava-1.5-7b": "LLaVA-1.5-7B"
}

# =========================
# Utilities
# =========================

def safe_get(row, col, default=""):
    if col not in row or pd.isna(row[col]):
        return default
    return row[col]


def format_badge(col, value):
    label = LABEL_NAME_MAP.get(col, col)

    if isinstance(value, float):
        v = f"{value:.1f}"
    else:
        v = str(value)

    cls = "badge"
    low = v.lower()
    if low in ["yes", "true", "1"]:
        cls += " badge-yes"
    elif low in ["no", "false", "0"]:
        cls += " badge-no"

    return (
        f'<span class="{cls}">'
        f'<span class="b-label">{html.escape(label)}</span> '
        f'{html.escape(v)}</span>'
    )


def make_img_src(image_path, out_dir, copy_images):
    if not image_path:
        return ""

    p = Path(image_path)
    if not p.exists():
        return ""

    if copy_images:
        images_dir = Path(out_dir) / "images"
        images_dir.mkdir(parents=True, exist_ok=True)
        dst = images_dir / p.name
        shutil.copy2(p, dst)
        return f"images/{dst.name}"
    else:
        return p.resolve().as_uri()

# =========================
# HTML Builder
# =========================

def build_html(df, out_dir, title, copy_images):
    css = """
    body { background:#0b0f14; color:#e5e7eb; font-family: system-ui; margin:0; }
    .wrap { max-width:1000px; margin:auto; padding:24px; }
    .card { background:#111827; border-radius:16px; margin-bottom:20px; padding:16px; }
    .meta { font-size:12px; color:#94a3b8; margin-bottom:10px; }
    .imgbox img { width:100%; border-radius:12px; }
    .badges { display:flex; flex-wrap:wrap; gap:6px; margin-top:10px; }
    .badge { border:1px solid #1f2a44; border-radius:999px; padding:4px 10px; font-size:12px; }
    .badge-yes { border-color:#16a34a; }
    .badge-no { border-color:#ef4444; }
    .b-label { color:#60a5fa; font-weight:600; margin-right:4px; }
    .section { margin-top:12px; }
    h3 { font-size:13px; color:#94a3b8; margin-bottom:4px; }
    .text { font-size:13px; white-space:pre-wrap; }
    """

    html_parts = [f"""
<!doctype html>
<html>
<head>
<meta charset="utf-8">
<title>{html.escape(title)}</title>
<style>{css}</style>
</head>
<body>
<div class="wrap">
<h1>{html.escape(title)}</h1>
<p style="color:#94a3b8;font-size:13px;">Rows: {len(df)}</p>
"""]

    for idx, row in df.iterrows():
        img_src = make_img_src(safe_get(row, "eval_image_path"), out_dir, copy_images)

        raw_model = safe_get(row, "model")
        model = MODEL_NAME_MAP.get(raw_model, raw_model)

        badges = []
        for col in LABEL_COLS:
            if col in row:
                badges.append(format_badge(col, safe_get(row, col)))

        html_parts.append(f"""
<div class="card">
  <div class="meta">
    <b>idx</b>: {idx} &nbsp;
    <b>model</b>: {html.escape(model)}
  </div>

  <div class="imgbox">
    {"<img src='"+img_src+"'>" if img_src else "<div style='color:#94a3b8'>Image not found</div>"}
  </div>

  <div class="badges">
    {''.join(badges)}
  </div>

  <div class="section">
    <h3>LLaVA Output</h3>
    <div class="text">{html.escape(str(safe_get(row, "llava_output")))}</div>
  </div>

  <div class="section">
    <h3>GPT-based Judge (Nano)</h3>
    <div class="text">Overall: {safe_get(row, "overall")}</div>
    <div class="text">{html.escape(str(safe_get(row, "comment")))}</div>
  </div>

  <div class="section">
    <h3>GPT-based Judge (Mini)</h3>
    <div class="text">Overall: {safe_get(row, "overall_mini")}</div>
    <div class="text">{html.escape(str(safe_get(row, "comment_mini")))}</div>
  </div>
</div>
""")

    html_parts.append("</div></body></html>")
    return "".join(html_parts)

# =========================
# Main
# =========================

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--title", default="RiSA Evaluation Gallery")
    ap.add_argument("--copy-images", action="store_true")
    args = ap.parse_args()

    df = pd.read_csv(args.csv)
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    html_str = build_html(df, out_path.parent, args.title, args.copy_images)
    out_path.write_text(html_str, encoding="utf-8")

    print("✅ HTML written:", out_path)

if __name__ == "__main__":
    main()
