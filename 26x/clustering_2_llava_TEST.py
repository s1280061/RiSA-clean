# -*- coding: utf-8 -*-
"""
ContextVLM – 単一画像 簡易版 (v6, Plan-B)
- SURFACE(= SLIPPERY / DRY) と GOOD_VISIBILITY(= YES / NO) を1枚だけ判定
- 方針: 3クラス(WET/SNOWY/DRY) → 2クラス(SLIPPERY/DRY) に再定義、ヒューリスティック不使用
- GOOD_VISIBILITY: YES=晴れ/曇りで見通し良好、NO=霧/雨/雪/夜間で見通し不良（SHRP2の常時低画質は無視）
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

# ======== フォールバック ========
SURFACE_FALLBACK_LABEL = "DRY"   # Plan-BでもDRYでOK
SURFACE_FALLBACK_CONF  = 50

GOODVIS_FALLBACK_LABEL = "YES"   # YES=見通し良好
GOODVIS_FALLBACK_CONF  = 50

# ======== 正規化マップ（Plan-B: SLIPPERY / DRY） ========
SURFACE_CANON = {"SLIPPERY", "DRY"}
SURFACE_ALIAS = {
    # 水系 → SLIPPERY
    "WET": "SLIPPERY", "RAIN": "SLIPPERY", "RAINY": "SLIPPERY", "WATER": "SLIPPERY",
    "PUDDLE": "SLIPPERY", "PUDDLES": "SLIPPERY", "SPRAY": "SLIPPERY",
    "REFLECTIVE": "SLIPPERY", "GLOSSY": "SLIPPERY",
    # 雪氷系 → SLIPPERY
    "SNOW": "SLIPPERY", "SNOWY": "SLIPPERY", "ICE": "SLIPPERY", "ICY": "SLIPPERY",
    "SLUSH": "SLIPPERY", "PACKED SNOW": "SLIPPERY", "FROZEN": "SLIPPERY",
    "SALT": "SLIPPERY", "RESIDUE": "SLIPPERY", "PLOWED": "SLIPPERY",
    # 乾燥 → DRY
    "DRY": "DRY", "CLEAN": "DRY", "ASPHALT": "DRY", "MATTE": "DRY",
    "NO WET": "DRY", "NO SNOW": "DRY",
}

GOODVIS_YESNO = {"YES", "NO"}

# ======== プロンプト ========
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
    '{"surface":{"label":"SLIPPERY|DRY","conf":0-100},'
    '"good_visibility":{"label":"YES|NO","conf":0-100}}'
)

# ======== 画像処理 ========
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
        # モデルにより 'response' フィールドのみ返す場合があるため両対応
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

def _norm_conf(x: Any, fallback: int) -> int:
    try:
        v = int(round(float(x)))
        return max(0, min(100, v))
    except Exception:
        return fallback

# ======== メイン関数 ========
def ask_two_questions(image_path: str) -> Dict[str, Dict[str, Any]]:
    full_b64, roi_b64 = _two_views_b64(image_path)
    if not full_b64:
        return {
            "surface": {"label": SURFACE_FALLBACK_LABEL, "conf": SURFACE_FALLBACK_CONF},
            "good_visibility": {"label": GOODVIS_FALLBACK_LABEL, "conf": GOODVIS_FALLBACK_CONF},
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
        "images": [full_b64, roi_b64],  # full + road-ROI
        "stream": False,
        "format": "json",
        "options": {"temperature": 0.0, "mirostat": 0, "repeat_penalty": 1.1}
    }

    outer = _post_ollama(payload)

    # 直接JSONが来る場合と 'response' にテキストが入る場合の両対応
    if isinstance(outer, dict) and ("surface" in outer) and ("good_visibility" in outer):
        parsed = outer
    else:
        raw = outer.get("response", "") if isinstance(outer, dict) else str(outer or "")
        parsed = _extract_first_json_obj(raw)

    if not parsed:
        print("[WARN] json extract failed; fallback to DRY/YES (50)")
        return {
            "surface": {"label": SURFACE_FALLBACK_LABEL, "conf": SURFACE_FALLBACK_CONF},
            "good_visibility": {"label": GOODVIS_FALLBACK_LABEL, "conf": GOODVIS_FALLBACK_CONF},
        }

    s = parsed.get("surface", {})
    g = parsed.get("good_visibility", {})

    s_label = _norm_surface(s.get("label"))
    s_conf  = _norm_conf(s.get("conf"), SURFACE_FALLBACK_CONF)
    g_label = _norm_goodvis(g.get("label"))
    g_conf  = _norm_conf(g.get("conf"), GOODVIS_FALLBACK_CONF)

    return {
        "surface": {"label": s_label, "conf": s_conf},
        "good_visibility": {"label": g_label, "conf": g_conf},
    }

# ======== 実行部分 ========
if __name__ == "__main__":
    # 任意の画像パスに変更して使ってください
    img_path = r"C:\Users\s1280\Desktop\SHRP2rawdata\central_frames_clustering_v6\clustered_images\cluster_09\scene_337_center.jpg"
    ans = ask_two_questions(img_path)
    print(json.dumps(ans, indent=2, ensure_ascii=False))
