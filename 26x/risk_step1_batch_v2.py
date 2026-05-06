# -*- coding: utf-8 -*-
"""
Step1 – LLaVA Evaluation Batch Script (using confidence, always perform estimation, v2_conf_nothresh)
- Target:  --root/<base>/new_divided/scene_XXX_fire within *pre_tid*.jpg ("post" excluded)
- Reference:  --root/<base>/new_divided/scene_XXX_context.json (using final / final_conf_window)
- Output:  --root/<base>/new_divided/llava_result_v2/scene_XXX/<image>.json
- Base: 3,4,5,6 (default)
- LLaVA: http://localhost:11434/api/generate with llava:latest running

Specifications (this version):
- Even if confidence is low, don't return unknown, always return best-guess (explicitly tell LLM).
- Reflect confidence values directly in inference and explicitly state numbers in reason (supplementary addition in later formatting).
- Output key uses "lane_change_detected" aligned with Stage2.
"""

from __future__ import annotations
import os
import re
import io
import json
import base64
import argparse
from glob import glob
from typing import Any, Dict, Optional

import requests
from PIL import Image

# ======== LLaVA / Ollama Configuration ========
OLLAMA_ENDPOINT = "http://localhost:11434/api/generate"
DEFAULT_MODEL = "llava:latest"
TIMEOUT_SEC = 90
RETRIES = 1  # Retry on failure (total RETRIES+1 times)


# ======== Image -> base64 (lightweight) ========
def b64_of_image(image_path: str) -> str:
    img = Image.open(image_path).convert("RGB").resize((224, 224))
    buf = io.BytesIO()
    img.save(buf, format="JPEG", quality=40)
    return base64.b64encode(buf.getvalue()).decode("utf-8")


# ======== Extract JSON from LLaVA response ========
def extract_inner_json(inner: Any) -> Optional[Dict[str, Any]]:
    if isinstance(inner, dict):
        return inner
    s = str(inner or "").strip()
    if not s:
        return None
    # Remove ```json ... ``` format
    if s.startswith("```"):
        s = s.strip("`")
        s = re.sub(r"^\s*json\s*", "", s, flags=re.IGNORECASE).strip()
    if not s.startswith("{"):
        i = s.find("{")
        if i >= 0:
            s = s[i:]
    # Extract up to outermost bracket
    depth = 0
    end = -1
    for i, ch in enumerate(s):
        if ch == "{":
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0:
                end = i + 1
                break
    if end > 0:
        s = s[:end]
    try:
        return json.loads(s)
    except Exception:
        return None


# ======== Normalization / Forbidden terms ========
def norm_lane_detected(x: Any) -> str:
    if not isinstance(x, str):
        return "unknown"
    t = x.strip().lower()
    return t if t in ("keep", "left", "right", "unknown") else "unknown"


def norm_suggested(x: Any) -> str:
    if not isinstance(x, str):
        return "unknown"
    t = x.strip().lower().replace("-", " ")
    t = re.sub(r"\s+", " ", t)
    if t in ("keep speed", "keep"):
        return "keep_speed"
    if t == "change lane left":
        return "change_lane_left"
    if t == "change lane right":
        return "change_lane_right"
    if t in ("decelerate", "slow down", "reduce speed", "brake", "deaccelerate"):
        return "decelerate"
    return "unknown"


_FORBIDDEN_TERMS = ("risk", "probability", "hazard", "level", "score")
_FACT_PREFIX_RE = re.compile(
    r'^\s*(Speed\s*=\s*[\d\.]+\s*km/h;?\s*)?(Risk\s*Zone\s*=\s*(YES|NO)\.?;?\s*)',
    re.IGNORECASE
)

def sanitize_reason(s: Any) -> str:
    msg = str(s or "").strip()
    for t in _FORBIDDEN_TERMS:
        msg = re.sub(rf"\b{t}\b", "", msg, flags=re.IGNORECASE)
    msg = _FACT_PREFIX_RE.sub("", msg)
    msg = re.sub(r"\s{2,}", " ", msg).strip()
    return msg


# ======== Extract final label/confidence from context ========
def extract_perception(context: Dict[str, Any]) -> Dict[str, Optional[Any]]:
    def get(d, *ks, default=None):
        for k in ks:
            if not isinstance(d, dict) or k not in d:
                return default
            d = d[k]
        return d
    ts_label = get(context, "perception", "turn_signal", "final", default=None)
    ts_conf  = get(context, "perception", "turn_signal", "final_conf_window", default=None)
    br_label = get(context, "perception", "brake", "final", default=None)
    br_conf  = get(context, "perception", "brake", "final_conf_window", default=None)
    return {
        "turn_signal_label": ts_label,
        "turn_signal_conf": ts_conf,
        "brake_label": br_label,
        "brake_conf": br_conf,
    }


def make_conf_summary(ctx_perc: Dict[str, Any]) -> str:
    # Example: [turn_signal: right (conf 0.63), brake: off (conf 0.54)]
    def fmt(v):
        try:
            return f"{float(v):.2f}"
        except Exception:
            return "n/a"
    parts = []
    if ctx_perc.get("turn_signal_label") is not None:
        parts.append(f"turn_signal: {ctx_perc['turn_signal_label']} (conf {fmt(ctx_perc.get('turn_signal_conf'))})")
    if ctx_perc.get("brake_label") is not None:
        parts.append(f"brake: {ctx_perc['brake_label']} (conf {fmt(ctx_perc.get('brake_conf'))})")
    return "[" + ", ".join(parts) + "]" if parts else ""


# ======== LLaVA call (present confidence, always estimate) ========
def call_llava(image_b64: str, context: Dict[str, Any],
               endpoint: Optional[str] = None,
               model: Optional[str] = None,
               temperature: float = 0.0) -> Dict[str, Any]:
    """
    Return (JSON for one image):
    {
      "risk_probability": 0..100 | null,
      "lane_change_detected": "keep|left|right|unknown",   # Avoid unknown even with low conf, best-guess
      "suggested_maneuver": "keep_speed|change_lane_left|change_lane_right|decelerate|unknown",
      "reason": "<short explanation (explicitly state confidence numbers)>",
      "parsed": true/false
    }
    """
    if endpoint is None:
        endpoint = OLLAMA_ENDPOINT
    if model is None:
        model = DEFAULT_MODEL

    ctx_perc = extract_perception(context)

    guidance = {
        "note": "Always make a best-guess even if confidence is low. Do NOT output 'unknown' solely because confidence is low.",
        "use_confidence": "Use provided confidence values as weights, but still decide.",
        "turn_signal": {
            "final": ctx_perc["turn_signal_label"],
            "final_conf_window": ctx_perc["turn_signal_conf"],
        },
        "brake": {
            "final": ctx_perc["brake_label"],
            "final_conf_window": ctx_perc["brake_conf"],
        }
    }

    prompt = f"""
You are a driving safety expert.
Use the following JSON facts as true. Consider their confidence values as soft weights.
Even if confidence is low or missing, you must still make a best-guess classification.
Do NOT output "unknown" just because confidence is low.

# CONTEXT (JSON)
```json
{json.dumps(context, ensure_ascii=False)}
```

# PERCEPTION CONFIDENCE (JSON)
```json
{json.dumps(guidance, ensure_ascii=False)}
```

Return ONE JSON object only, with this schema:
{{
"risk_probability": "an integer 0-100 or null if not sure",
"lane_change_detected": "one of [keep,left,right,unknown]",
"suggested_maneuver": "one of [keep_speed,change_lane_left,change_lane_right,decelerate]",
"reason": "short explanation mentioning the confidence numbers explicitly"
}}
"""

    payload = {
        "model": model,
        "prompt": prompt,
        "images": [image_b64],
        "stream": False,
        "format": "json",
        "options": {"temperature": temperature, "mirostat": 0},
    }

    last_exc: Optional[Exception] = None
    outer: Optional[Dict[str, Any]] = None
    for _ in range(RETRIES + 1):
        try:
            resp = requests.post(endpoint, json=payload, timeout=TIMEOUT_SEC)
            resp.raise_for_status()
            outer = resp.json()
            break
        except Exception as e:
            last_exc = e
            outer = None

    if outer is None:
        return {
            "risk_probability": None,
            "lane_change_detected": "unknown",
            "suggested_maneuver": "unknown",
            "reason": f"LLaVA request error: {last_exc}",
            "parsed": False,
        }

    result = extract_inner_json(outer.get("response"))
    if result is None:
        return {
            "risk_probability": None,
            "lane_change_detected": "unknown",
            "suggested_maneuver": "unknown",
            "reason": "LLaVA response could not be parsed.",
            "parsed": False,
        }

    # ---- Normalization ----
    rp = result.get("risk_probability", None)
    try:
        if rp is not None:
            rp = int(rp)
            if not (0 <= rp <= 100):
                rp = None
    except Exception:
        rp = None

    lane = norm_lane_detected(result.get("lane_change_detected", "unknown"))
    sugg = norm_suggested(result.get("suggested_maneuver", "unknown"))
    reason = sanitize_reason(result.get("reason", ""))

    # Always explicitly state conf numbers in reason (insurance when model doesn't write them)
    conf_tail = make_conf_summary(ctx_perc)
    if conf_tail and conf_tail not in reason:
        reason = (reason + " " + conf_tail).strip()

    return {
        "risk_probability": rp,
        "lane_change_detected": lane,
        "suggested_maneuver": sugg,
        "reason": reason,
        "parsed": True,
    }


# ======== Process one scene ========
def process_scene(base_dir: str, scene_dirname: str, out_root: str,
                  endpoint: str, model: str, temperature: float) -> None:
    m = re.match(r"^scene_(\d{3})_fire$", scene_dirname, flags=re.IGNORECASE)
    if not m:
        return
    scene_num = m.group(1)
    scene_dir = os.path.join(base_dir, scene_dirname)

    # Only pre_tid images (exclude names containing post)
    imgs = sorted([
        p for p in glob(os.path.join(scene_dir, "*.jpg"))
        if ("pre_tid" in os.path.basename(p).lower()) and ("post" not in os.path.basename(p).lower())
    ])
    if not imgs:
        return

    # Context directly under new_divided
    context_json = os.path.join(base_dir, f"scene_{scene_num}_context.json")
    if not os.path.isfile(context_json):
        print(f"  - {scene_dirname}: missing context -> skip")
        return

    with open(context_json, "r", encoding="utf-8") as f:
        context = json.load(f)

    scene_out = os.path.join(out_root, scene_dirname)
    os.makedirs(scene_out, exist_ok=True)

    for img_path in imgs:
        fname = os.path.basename(img_path)
        img_stem, _ = os.path.splitext(fname)
        out_json = os.path.join(scene_out, f"{img_stem}.json")
        if os.path.exists(out_json):
            continue

        # Extract frame_id from filename (e.g.: frame_001686_pre_tid2.jpg -> 1686)
        frame_id = None
        m_id = re.search(r"frame_(\d+)", img_stem, flags=re.IGNORECASE)
        if m_id:
            try:
                frame_id = int(m_id.group(1))
            except Exception:
                frame_id = None

        try:
            img_b64 = b64_of_image(img_path)
            ra = call_llava(img_b64, context, endpoint=endpoint, model=model, temperature=temperature)
        except Exception as e:
            print(f"  ✖ {scene_dirname}/{img_stem}: {e}")
            continue

        # Output format (one object per file)
        out_obj = {
            "frame_id": frame_id,
            "image_path": img_path,
            "risk_assessment": {
                "risk_probability": ra.get("risk_probability", None),
                "lane_change_detected": ra.get("lane_change_detected", "unknown"),
                "suggested_maneuver": ra.get("suggested_maneuver", "unknown"),
                "reason": ra.get("reason", ""),
                "parsed": ra.get("parsed", False)
            }
        }

        with open(out_json, "w", encoding="utf-8") as fj:
            json.dump(out_obj, fj, ensure_ascii=False, indent=2)

        print(f"  ✓ {scene_dirname}/{fname} -> saved")


# ======== Process base ========
def process_base(root: str, base: str, max_scenes: int,
                 endpoint: str, model: str, temperature: float) -> None:
    base_dir = os.path.join(root, base, "new_divided")
    if not os.path.isdir(base_dir):
        print(f"⚠️ Skip base {base}: not found -> {base_dir}")
        return

    out_root = os.path.join(base_dir, "llava_result_v2")
    os.makedirs(out_root, exist_ok=True)

    scenes = sorted([d for d in os.listdir(base_dir)
                     if os.path.isdir(os.path.join(base_dir, d))
                     and re.match(r"^scene_(\d{3})_fire$", d, flags=re.IGNORECASE)])
    if max_scenes > 0:
        scenes = scenes[:max_scenes]

    print(f"\n===== Base {base} | Scenes: {len(scenes)} -> out: {out_root}")
    for sd in scenes:
        process_scene(base_dir, sd, out_root, endpoint, model, temperature)
    print(f"✅ Base {base}: done")


# ======== Main ========
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default=r"C:\Users\s1280\Desktop\SHRP2rawdata")
    ap.add_argument("--bases", default="3,4,5,6", help="comma-separated base dirs under root")
    ap.add_argument("--max_scenes", type=int, default=0, help="limit scenes per base (0=no limit)")
    ap.add_argument("--model", default=DEFAULT_MODEL)
    ap.add_argument("--endpoint", default=OLLAMA_ENDPOINT)
    ap.add_argument("--temperature", type=float, default=0.0)
    args = ap.parse_args()

    bases = [b.strip() for b in args.bases.split(",") if b.strip()]
    for b in bases:
        process_base(args.root, b, args.max_scenes, args.endpoint, args.model, args.temperature)


if __name__ == "__main__":
    main()