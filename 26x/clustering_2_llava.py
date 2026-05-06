# -*- coding: utf-8 -*-
"""
ContextVLM – 2問ゼロショット版 (v6, Plan-B, conf削除)
- Q1: SURFACE → {"label": "SLIPPERY|DRY"}  ※2択（WET/SNOWYを統合）
- Q2: GOOD_VISIBILITY → {"label": "YES|NO"}  ※YES=晴/曇、NO=霧/雨/雪/夜間
- 入力画像は full + road-ROI(下45%) の2枚を同時投入
- 厳密JSONパース、ラベル正規化
- 出力は label のみ（confなし）
"""

import os, io, json, re, base64, requests
from typing import Dict, Any, Tuple, Optional
from PIL import Image

# ======== 設定 ========
OLLAMA_URL   = "http://localhost:11434/api/generate"
LLAVA_MODEL  = "llava:latest"
TIMEOUT_SEC  = 120

IMG_SIZE       = (448, 448)
JPEG_QUALITY   = 92
ROI_BOTTOM_FR  = 0.45

# フォールバック
SURFACE_FALLBACK_LABEL = "DRY"
GOODVIS_FALLBACK_LABEL = "YES"

# ======== 正規化マップ ========
SURFACE_CANON = {"SLIPPERY", "DRY"}
SURFACE_ALIAS = {
    "WET": "SLIPPERY", "RAIN": "SLIPPERY", "RAINY": "SLIPPERY", "WATER": "SLIPPERY",
    "PUDDLE": "SLIPPERY", "PUDDLES": "SLIPPERY", "SPRAY": "SLIPPERY",
    "REFLECTIVE": "SLIPPERY", "GLOSSY": "SLIPPERY",
    "SNOW": "SLIPPERY", "SNOWY": "SLIPPERY", "ICE": "SLIPPERY", "ICY": "SLIPPERY",
    "SLUSH": "SLIPPERY", "PACKED SNOW": "SLIPPERY", "FROZEN": "SLIPPERY",
    "SALT": "SLIPPERY", "RESIDUE": "SLIPPERY", "PLOWED": "SLIPPERY",
    "DRY": "DRY", "CLEAN": "DRY", "NO WET": "DRY", "NO SNOW": "DRY",
    "ASPHALT": "DRY", "MATTE": "DRY",
}

GOODVIS_YESNO = {"YES", "NO"}

# ======== 定義 ========
PROMPT_SURFACE_DEF = (
    "Choose exactly one SURFACE label for the road surface (entire ROAD AREA):\n"
    "- SLIPPERY: roadway conditions that reduce tire grip, including liquid water (rain, puddles, spray) "
    "and frozen materials (snow, packed snow, ice, slush, salt residue). Any visible snow/ice OR mirror-like wetness counts as SLIPPERY.\n"
    "- DRY     : matte asphalt, mostly dark; no mirror-like sheen, no puddles, and no snow/ice/salt.\n"
    "If there is any doubt, prefer SLIPPERY for safety.\n"
)

PROMPT_GOODVIS_DEF = (
    "Decide GOOD_VISIBILITY (YES/NO) based on overall visibility, ignoring SHRP2's normal low image quality:\n"
    "YES = Weather is clear or cloudy with good visibility into the distance.\n"
    "NO  = Weather is foggy, raining, snowing, or nighttime, making it harder to see far.\n"
)

STRICT_JSON = (
    'Return exactly this single JSON and nothing else:\n'
    '{"surface":{"label":"SLIPPERY|DRY"},'
    '"good_visibility":{"label":"YES|NO"}}'
)

# ======== 画像→base64 ========
def _resize(img: Image.Image) -> Image.Image:
    return img.convert("RGB").resize(IMG_SIZE)

def _b64(img: Image.Image) -> str:
    buf = io.BytesIO()
    img.save(buf, format="JPEG", quality=JPEG_QUALITY)
    return base64.b64encode(buf.getvalue()).decode("utf-8")

def _two_views_b64(path: str) -> Tuple[Optional[str], Optional[str]]:
    try:
        img = Image.open(path).convert("RGB")
    except Exception as e:
        print(f"[WARN] open failed: {path} ({e})")
        return None, None
    img = _resize(img)
    w, h = img.size
    y0 = int(h*(1-ROI_BOTTOM_FR))
    roi = img.crop((0, y0, w, h))
    return _b64(img), _b64(roi)

# ======== API呼び出し ========
def _post_ollama(payload: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    try:
        r = requests.post(OLLAMA_URL, json=payload, timeout=TIMEOUT_SEC)
        r.raise_for_status()
        try:
            return r.json()
        except json.JSONDecodeError:
            m = re.search(r'"response"\s*:\s*"([^"]+)"', r.text)
            if m:
                return {"response": m.group(1)}
            return None
    except Exception as e:
        print(f"[ERR] request: {e}")
        return None

# ======== JSON抽出 ========
def _extract_first_json_obj(s: str) -> Optional[dict]:
    if not s:
        return None
    s = s.strip()
    if s.startswith("```"):
        s = s.strip("`")
        s = re.sub(r"^\s*json\s*", "", s, flags=re.IGNORECASE).strip()
    start = s.find("{")
    if start < 0:
        return None
    depth, in_str, esc, end = 0, False, False, None
    for i in range(start, len(s)):
        ch = s[i]
        if in_str:
            if esc: esc = False
            elif ch == "\\": esc = True
            elif ch == '"': in_str = False
            continue
        else:
            if ch == '"': in_str = True; continue
            if ch == "{": depth += 1
            elif ch == "}":
                depth -= 1
                if depth == 0:
                    end = i + 1
                    break
    if end is None:
        return None
    try:
        return json.loads(s[start:end])
    except Exception:
        return None

# ======== 正規化 ========
def _norm_surface(x: Any) -> str:
    if isinstance(x, str):
        t = re.sub(r"[^A-Z ]+", " ", x.upper()).strip()
        if t in SURFACE_CANON:
            return t
        for k, v in SURFACE_ALIAS.items():
            if k in t:
                return v
        for w in t.split():
            if w in SURFACE_ALIAS:
                return SURFACE_ALIAS[w]
    return SURFACE_FALLBACK_LABEL

def _norm_goodvis(x: Any) -> str:
    if isinstance(x, str):
        t = x.strip().upper()
        if t in GOODVIS_YESNO:
            return t
        if "YES" in t: return "YES"
        if "NO"  in t: return "NO"
    return GOODVIS_FALLBACK_LABEL

# ======== 2問バッチ（ゼロショット） ========
def ask_two_questions(image_path: str) -> Dict[str, Dict[str, Any]]:
    full_b64, roi_b64 = _two_views_b64(image_path)
    if not full_b64:
        return {
            "surface": {"label": SURFACE_FALLBACK_LABEL},
            "good_visibility": {"label": GOODVIS_FALLBACK_LABEL},
        }

    prompt = (
        "You are a driving-scene rater.\n\n"
        + PROMPT_SURFACE_DEF + "\n"
        + PROMPT_GOODVIS_DEF + "\n\n"
        + STRICT_JSON
    )

    payload = {
        "model": LLAVA_MODEL,
        "prompt": prompt,
        "images": [full_b64, roi_b64],
        "stream": False,
        "format": "json",
        "options": {"temperature": 0.0, "mirostat": 0, "repeat_penalty": 1.1}
    }

    outer = _post_ollama(payload)
    raw = ""
    if isinstance(outer, dict) and ("surface" in outer) and ("good_visibility" in outer):
        parsed = outer
    else:
        raw = outer.get("response", "") if isinstance(outer, dict) else str(outer or "")
        parsed = _extract_first_json_obj(raw)

    if not parsed:
        print("   [WARN] json extract failed; fallback to DRY/YES")
        return {
            "surface": {"label": SURFACE_FALLBACK_LABEL},
            "good_visibility": {"label": GOODVIS_FALLBACK_LABEL},
        }

    s_label = _norm_surface(parsed.get("surface", {}).get("label"))
    g_label = _norm_goodvis(parsed.get("good_visibility", {}).get("label"))

    return {
        "surface": {"label": s_label},
        "good_visibility": {"label": g_label},
    }

# ======== 入出力パス ========
base_dir = r"C:\Users\s1280\Desktop\SHRP2rawdata\central_frames_clustering_v6\clustered_images"
clusters = [f"cluster_{i:02d}" for i in range(10)]

save_dir = os.path.join(base_dir, "llava_results_surface2_visibility2_v6_zeroshot_noconf")
os.makedirs(save_dir, exist_ok=True)

# ======== 実行 ========
print("Q1: SURFACE = {SLIPPERY, DRY}")
print("Q2: GOOD_VISIBILITY = {YES, NO}\n")

for cluster in clusters:
    cdir = os.path.join(base_dir, cluster)
    if not os.path.exists(cdir):
        print(f"⚠ {cluster} not found, skip")
        continue

    images = sorted(
        os.path.join(cdir, f)
        for f in os.listdir(cdir)
        if f.lower().endswith((".jpg", ".jpeg", ".png"))
    )
    print(f"📂 {cluster} images: {len(images)}")

    out_rows = []
    for i, img_path in enumerate(images):
        print(f"[{i+1}/{len(images)}] {os.path.basename(img_path)} ...")
        ans = ask_two_questions(img_path)
        print(f"   SURFACE         : {ans['surface']['label']}")
        print(f"   GOOD_VISIBILITY : {ans['good_visibility']['label']}")
        out_rows.append({"cluster": cluster, "image": os.path.basename(img_path), "answers": ans})

    out_path = os.path.join(save_dir, f"surface_visibility_{cluster}.json")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(out_rows, f, ensure_ascii=False, indent=2)
    print(f"✅ saved: {out_path}")

print("\n🎉 done.")
