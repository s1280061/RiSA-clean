import os
import argparse
import shutil
import html
from pathlib import Path
from typing import Optional, List, Tuple
import pandas as pd

# =========================
# Label definitions
# =========================

LABEL_COLS = [
    # Environment
    "env_rural", "env_city", "env_snowy", "env_sunny", "env_rainy",
    "ego_speed_kmh",
    "detected_vehicle_count",
    # Ground Truth
    "gt_lane_change",
    # Recognition
    "lane_change_detected_llava",
    "lane_change_detected_gpt4o",
    # Advice
    "llava_maneuver",
    "gpt4o_suggested_maneuver",
    # Signals
    "ts_final", "brake_final",
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
    "lane_change_detected_llava": "LLaVA Lane-Change",
    "llava_maneuver": "LLaVA Maneuver",
    "detected_vehicle_count": "Detected Vehicles",
    "gt_lane_change": "GT Lane-Change",
    "lane_change_detected_gpt4o": "GPT-4o Lane-Change",
    "gpt4o_suggested_maneuver": "GPT-4o Maneuver",
}

MODEL_NAME_MAP = {
    "llava": "LLaVA-1.5-7B",
    "llava_1.5_7b": "LLaVA-1.5-7B",
    "llava-1.5-7b": "LLaVA-1.5-7B"
}


# =========================
# Utilities
# =========================

def safe_get(row: pd.Series, col: str, default: str = "") -> str:
    """Safely get value from DataFrame row"""
    try:
        if col not in row.index or pd.isna(row[col]):
            return default
        return row[col]
    except Exception:
        return default


def format_badge(col: str, value) -> str:
    """Format a label-value pair as an HTML badge"""
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


def make_img_src(image_path: str, out_dir: Path, copy_images: bool) -> str:
    """Generate image source path, optionally copying the image"""
    if not image_path:
        return ""

    try:
        p = Path(image_path)
        if not p.exists():
            return ""

        if copy_images:
            images_dir = out_dir / "images"
            images_dir.mkdir(parents=True, exist_ok=True)
            dst = images_dir / p.name
            shutil.copy2(p, dst)
            return f"images/{dst.name}"
        else:
            return p.resolve().as_uri()
    except Exception as e:
        print(f"⚠️  Error processing image {image_path}: {e}")
        return ""


def resolve_image_path(
        image_dir: Path,
        image_name: str,
        prefixes: Tuple[str, ...] = ("3_", "4_", "5_", "6_")
) -> str:
    """
    Try to resolve image path by testing multiple prefixes.
    """
    if not image_name:
        return ""

    try:
        if not image_dir.exists():
            return ""

        # First try with prefixes
        for prefix in prefixes:
            candidate = image_dir / f"{prefix}{image_name}"
            if candidate.exists():
                return str(candidate)

        # Try without prefix
        candidate = image_dir / image_name
        if candidate.exists():
            return str(candidate)

    except Exception as e:
        print(f"⚠️  Error resolving image path for {image_name}: {e}")

    return ""


def normalize_text(text: Optional[str]) -> str:
    """Normalize text by removing extra whitespace and empty lines"""
    if not text:
        return ""

    text = str(text)
    lines = [line.strip() for line in text.splitlines()]

    # Consolidate empty lines
    normalized = []
    prev_empty = False
    for line in lines:
        if line == "":
            if not prev_empty:
                normalized.append("")
            prev_empty = True
        else:
            normalized.append(line)
            prev_empty = False

    return "\n".join(normalized)


def get_score_class(score_str: str) -> str:
    """Get CSS class based on overall score"""
    try:
        score = float(score_str)
        if score >= 4.0:
            return "score-high"
        elif score <= 2.0:
            return "score-low"
        else:
            return "score-mid"
    except (ValueError, TypeError):
        return "score-mid"


# =========================
# HTML Builder
# =========================

def get_css() -> str:
    """Return CSS stylesheet"""
    return """
    body {
      background:#0b0f14;
      color:#e5e7eb;
      font-family: system-ui, -apple-system, BlinkMacSystemFont, sans-serif;
      margin:0;
    }

    .wrap {
      max-width: 920px;
      margin: auto;
      padding: 5px 4px 5px;
    }

    h1 {
      font-size: 18px;
      margin-bottom: 3px;
    }

    .subtitle {
      color:#94a3b8;
      font-size:12px;
      margin-bottom:8px;
    }

    .card {
      background:#111827;
      border-radius:3px;
      margin-bottom:4px;
      padding:7px;
    }

    .meta {
      font-size:12px;
      color:#94a3b8;
      margin-bottom:6px;
    }

    .imgbox {
      text-align:center;
      margin-bottom:8px;
    }

    .imgbox img {
      max-width: 480px;
      width: 100%;
      height: auto;
      border-radius:10px;
    }

    .badges {
      display:flex;
      flex-wrap:wrap;
      gap:3px;
      margin-top:3px;
      margin-bottom:8px;
    }

    .badge {
      border:1px solid #1f2a44;
      border-radius:999px;
      padding:2px 5px;
      font-size:12px;
      white-space:nowrap;
    }

    .badge-yes { border-color:#16a34a; }
    .badge-no  { border-color:#ef4444; }

    .b-label {
      color:#60a5fa;
      font-weight:600;
      margin-right:4px;
    }

    .section {
      margin-top: 6px;
      padding-top: 0px;
    }

    h3 {
      font-size:13px;
      color:#93c5fd;
      margin:0 0 3px;
    }

    .text {
      font-size:13px;
      line-height:1.5;
      white-space:pre-wrap;
      word-break:break-word;
      margin-bottom:0;
    }

    .overall {
      font-size: 12px;
      color: #e5e7eb;
      font-weight: 500;
      white-space: nowrap;
    }

    .overall-score {
      color: #e5e7eb;
      font-weight: 600;
    }

    .score-high { color: #22c55e; font-weight: 600; }  /* 緑: 4.0以上 */
    .score-low { color: #ef4444; font-weight: 600; }   /* 赤: 2.0以下 */
    .score-mid { color: #e5e7eb; font-weight: 600; }   /* 白: それ以外 */
    """


def build_card(idx: int, row: pd.Series, out_dir: Path, copy_images: bool, image_dir: Path) -> str:
    """Build a single card HTML"""
    img_path = resolve_image_path(
        image_dir=image_dir,
        image_name=safe_get(row, "image_name"),
    )

    img_src = make_img_src(img_path, out_dir, copy_images)
    model = "LLaVA-1.5-7B + GPT-4o"

    # Build badges
    badges = []
    for col in LABEL_COLS:
        if col in row.index and not pd.isna(row[col]):
            badges.append(format_badge(col, safe_get(row, col)))

    # Build image box
    img_html = (
        f"<img src='{img_src}'>" if img_src
        else "<div style='color:#94a3b8'>Image not found</div>"
    )

    return f"""
    <div class="card">
      <div class="meta">
        <b>idx</b>: {idx} &nbsp; | &nbsp;
        <b>model</b>: {html.escape(model)}
      </div>

      <div class="imgbox">
        {img_html}
      </div>

      <div class="badges">
        {''.join(badges)}
      </div>

      <div class="section">
        <h3>LLaVA Output</h3>
        <div class="text">{html.escape(normalize_text(safe_get(row, "llava_output")))}
        </div>
      </div>

      <div class="section">
        <h3>GPT-4o Output</h3>
        <div class="text">{html.escape(normalize_text(safe_get(row, "gpt4o_reason")))}
        </div>
      </div>

      <div class="section">
        <h3>Judge (Nano) → LLaVA &nbsp; 
        Overall: <span class="{get_score_class(safe_get(row, 'overall'))}">{html.escape(str(safe_get(row, "overall")))}</span>
        </h3>
        <div class="text">{html.escape(normalize_text(safe_get(row, "comment")))}
        </div>
      </div>

      <div class="section">
        <h3>Judge (Mini) → LLaVA &nbsp; Overall: <span class="{get_score_class(safe_get(row, 'overall_mini'))}">{html.escape(str(safe_get(row, "overall_mini")))}</span></h3>
        <div class="text">{html.escape(normalize_text(safe_get(row, "comment_mini")))}
        </div>
      </div>

      <div class="section">
        <h3>Judge (Mini) → GPT-4o &nbsp; Overall: <span class="{get_score_class(safe_get(row, 'overall_gpt5_mini_togpt4o'))}">{html.escape(str(safe_get(row, "overall_gpt5_mini_togpt4o")))}</span></h3>
        <div class="text">{html.escape(normalize_text(safe_get(row, "judge_mini_comment_togpt4o")))}
        </div>
      </div>

      <div class="section">
        <h3>Judge (Nano) → GPT-4o &nbsp; Overall: <span class="{get_score_class(safe_get(row, 'overall_gpt5_nano_gpt4o'))}">{html.escape(str(safe_get(row, "overall_gpt5_nano_gpt4o")))}</span></h3>
        <div class="text">{html.escape(normalize_text(safe_get(row, "judge_comment_nano_gpt4o")))}
        </div>
      </div>

    </div>
    """


def build_html(df: pd.DataFrame, out_dir: Path, title: str, copy_images: bool, image_dir: Path) -> str:
    """Build complete HTML gallery from DataFrame"""
    css = get_css()

    parts = [f"""<!doctype html>
<html lang="ja">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>{html.escape(title)}</title>
<style>{css}</style>
</head>
<body>
<div class="wrap">
<h1>{html.escape(title)}</h1>
<div class="subtitle">Rows: {len(df)}</div>
"""]

    # Generate cards
    for idx, row in df.iterrows():
        parts.append(build_card(idx, row, out_dir, copy_images, image_dir))

    parts.append("</div></body></html>")
    return "".join(parts)


# =========================
# Main
# =========================

def load_csv(csv_path: Path) -> Optional[pd.DataFrame]:
    """Load CSV with multiple encoding attempts"""
    encodings = ["utf-8", "cp932", "shift-jis", "utf-8-sig"]

    for encoding in encodings:
        try:
            df = pd.read_csv(csv_path, encoding=encoding)
            print(f"📊 Loaded {len(df)} rows from {csv_path} ({encoding})")
            return df
        except UnicodeDecodeError:
            continue
        except Exception as e:
            print(f"❌ Error reading CSV with {encoding}: {e}")
            continue

    print(f"❌ Could not decode CSV with any supported encoding")
    return None


def main():
    ap = argparse.ArgumentParser(
        description="Generate HTML gallery from RiSA evaluation CSV"
    )
    ap.add_argument("--csv", required=True, help="Input CSV file path")
    ap.add_argument("--out", required=True, help="Output HTML file path")
    ap.add_argument("--image-dir", default="/Users/kas/Desktop/eval_images",
                    help="Directory containing images")
    ap.add_argument("--title", default="RiSA Evaluation Gallery",
                    help="Page title")
    ap.add_argument("--copy-images", action="store_true",
                    help="Copy images to output directory")
    args = ap.parse_args()

    # Validate inputs
    csv_path = Path(args.csv)
    if not csv_path.exists():
        print(f"❌ Error: CSV file not found: {csv_path}")
        return 1

    # Load CSV
    df = load_csv(csv_path)
    if df is None:
        return 1

    # Prepare output
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    # Generate HTML
    try:
        html_str = build_html(
            df,
            out_path.parent,
            args.title,
            args.copy_images,
            Path(args.image_dir)
        )
        out_path.write_text(html_str, encoding="utf-8")
        print(f"✅ HTML written: {out_path}")
        return 0
    except Exception as e:
        print(f"❌ Error generating HTML: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    exit(main())