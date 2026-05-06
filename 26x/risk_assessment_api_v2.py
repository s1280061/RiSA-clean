# risk_assessment_api_v2.py
# Stage1: Environment recognition (single-choice ×3)  ※YES/NOフラグ廃止
# Stage2: Safety advice (uses Stage1 summary text; no re-guessing weather/road)
#
# Public API:
#   assess_environment_groups(image_path, *, url=LLaVA_URL, model=LLaVA_MODEL, temperature=LLaVA_TEMPERATURE)
#       -> dict {weather, lighting, road_type, facts_text}
#   should_run_stage1(frame_idx, total_frames, interval=150) -> bool
#   assess_risk_from_image_with_context(image_path, driving_facts, env_summary_text="", *, url, model, temperature)
#       -> dict {risk_probability, primary_hazard, lane_change_detected, suggested_maneuver, reason}
#   assess_risk_from_image(image_path, driving_facts, *, url, model, temperature) -> dict
#
# 依存: PIL, requests

from __future__ import annotations

import base64
import io
import json
import re
from typing import Dict, Any, Tuple

import requests
from PIL import Image

# --- safe cast helpers ---
def _to_int_safe(x, default=None):
    try:
        return int(x)
    except (TypeError, ValueError):
        return default

def _to_float_safe(x, default=None):
    try:
        return float(x)
    except (TypeError, ValueError):
        return default

# ====== Config ======
LLaVA_URL = "http://localhost:11434/api/generate"
LLaVA_MODEL = "llava:latest"
LLaVA_TIMEOUT_SEC = 30
LLaVA_TEMPERATURE = 0.0


# ========== 共通ユーティリティ ==========
def _encode_image_to_b64(image_path: str) -> str:
    """画像を軽量JPEGにしてbase64化（帯域と応答時間のため 224px / quality=40）。"""
    img = Image.open(image_path).convert("RGB").resize((224, 224))
    buf = io.BytesIO()
    img.save(buf, format="JPEG", quality=40)
    return base64.b64encode(buf.getvalue()).decode("utf-8")


def _post_llava(prompt: str, img_b64: str,
                *, url: str = LLaVA_URL,
                model: str = LLaVA_MODEL,
                temperature: float = LLaVA_TEMPERATURE) -> str:
    """LLaVA エンドポイントに POST（タイムアウト＆例外処理込み）。"""
    payload = {
        "model": model,
        "prompt": prompt,
        "images": [img_b64],
        "stream": False,
        "options": {"temperature": float(temperature)},
    }
    try:
        resp = requests.post(url, json=payload, timeout=LLaVA_TIMEOUT_SEC)
        resp.raise_for_status()
        return resp.text
    except requests.RequestException as e:
        return json.dumps({"response": "{}", "error": str(e)})


def _extract_json(raw_text: str) -> dict:
    """
    Ollama/LLaVA 応答から JSON を robust に抽出。
    1) 全体json → 2) response中の{} → 3) 最長一致
    """
    try:
        obj = json.loads(raw_text)
        if isinstance(obj, dict):
            inner = obj.get("response", "")
            if isinstance(inner, str):
                m2 = re.search(r"\{.*?\}", inner, re.DOTALL)
                return json.loads(m2.group(0)) if m2 else obj
        return obj
    except Exception:
        pass
    m = re.search(r"\{.*?\}", raw_text, re.DOTALL)
    if not m:
        return {}
    try:
        outer = json.loads(m.group(0))
        inner = outer.get("response", "")
        m2 = re.search(r"\{.*?\}", inner, re.DOTALL)
        return json.loads(m2.group(0)) if m2 else outer
    except Exception:
        return {}


# ========== Stage1: 単一選択×3（UNKNOWNなし） ==========
def _ask_single_choice(img_b64: str, title: str, choices: list[str],
                       *, url: str = LLaVA_URL, model: str = LLaVA_MODEL, temperature: float = LLaVA_TEMPERATURE) -> str:
    """
    choices のいずれか 1 語だけを JSON {"answer":"<CHOICE>"} で返させる。
    UNKNOWN は許可しない（不確実でも最も尤もらしい 1 ラベルを選ばせる）。
    """
    schema = {"answer": "|".join(choices)}
    prompt = f"""
You are a driving scene analyst.

Select EXACTLY ONE label for "{title}" from this list:
{", ".join(choices)}.

Rules:
- You MUST choose one of the labels above (do NOT output UNKNOWN).
- If you are unsure, choose the MOST LIKELY single label.
- Answer ONLY with a single JSON object. No markdown/code, no extra text.

Schema:
{json.dumps(schema, indent=2)}
""".strip()

    raw = _post_llava(prompt, img_b64, url=url, model=model, temperature=temperature)
    obj = _extract_json(raw)
    ans = (obj.get("answer") or "").strip().upper()

    # 正規化の軽い救済
    norm_map = {
        "NIGHTTIME": "NIGHT-TIME",
        "NIGHT TIME": "NIGHT-TIME",
        "RURAL": "RURAL AREA",
        "CITY STREET": "CITY",
        "URBAN": "CITY",
        "SUNNY DAY": "SUNNY",
        "DAY": "DAYTIME",
        "DAY LIGHT": "DAYTIME",
        "TWILITE": "TWILIGHT",
    }
    ans = norm_map.get(ans, ans)

    if ans in choices:
        return ans

    # 自由文が返った場合のフォールバック：含有判定 → 先頭採用
    body = ""
    try:
        parsed = json.loads(raw)
        body = (parsed.get("response") if isinstance(parsed, dict) else "") or raw
    except Exception:
        body = raw or ""
    low = body.lower()
    for c in choices:
        if c.lower().replace("-", "").replace(" ", "") in low.replace("-", "").replace(" ", ""):
            return c
    return choices[0]


# （risk_assessment_api_v2.py の Stage1 部分だけ差し替え）

def assess_environment_groups(image_path: str,
                              *, url: str = LLaVA_URL, model: str = LLaVA_MODEL, temperature: float = LLaVA_TEMPERATURE) -> dict:
    """
    画像から 3グループ×各1回の問い合わせで単一カテゴリを得る（UNKNOWNなし）。
    戻り値:
    {
      "weather":   "FOGGY|RAINY|SNOWY|SUNNY",
      "lighting":  "DAYTIME|TWILIGHT|NIGHT-TIME",
      "road_type": "HIGHWAY|CITY|RURAL AREA|PARKING LOT|OFF ROAD",
      "facts_text":"WEATHER: ...; LIGHTING: ...; ROAD: ..."
    }
    """
    img_b64 = _encode_image_to_b64(image_path)

    weather = _ask_single_choice(  # ← 1回
        img_b64, "Weather",
        ["FOGGY", "RAINY", "SNOWY", "SUNNY"],
        url=url, model=model, temperature=temperature
    )
    lighting = _ask_single_choice(  # ← 1回
        img_b64, "Lighting",
        ["DAYTIME", "TWILIGHT", "NIGHT-TIME"],
        url=url, model=model, temperature=temperature
    )
    road = _ask_single_choice(      # ← 1回（5カテゴリに拡張）
        img_b64, "RoadType",
        ["HIGHWAY", "CITY", "RURAL AREA", "PARKING LOT", "OFF ROAD"],
        url=url, model=model, temperature=temperature
    )

    facts_text = f"WEATHER: {weather}; LIGHTING: {lighting}; ROAD: {road}"
    return {
        "weather": weather,
        "lighting": lighting,
        "road_type": road,
        "facts_text": facts_text,
    }



def should_run_stage1(frame_idx: int, total_frames: int, interval: int = 150) -> bool:
    """
    Stage1 を走らせる固定スケジュール:
      - 先頭 (0)
      - interval の倍数 (例: 150, 300, ...)
      - 最終フレーム
    """
    if frame_idx == 0:
        return True
    if interval > 0 and frame_idx % interval == 0:
        return True
    if frame_idx == max(0, total_frames - 1):
        return True
    return False


# ========== Stage2: 助言（Stage1の要約テキストだけを文脈として使用） ==========
def assess_risk_from_image_with_context(image_path: str,
                                        driving_facts: dict,
                                        env_summary_text: str = "",
                                        *,
                                        url: str = LLaVA_URL,
                                        model: str = LLaVA_MODEL,
                                        temperature: float = LLaVA_TEMPERATURE) -> dict:
    """
    改善点:
    - BBox中心(cx,cy)、BBox面積(area)、predicted_ttc_sec はプロンプトに一切含めない
    - env_summary_text は Stage1 の1行要約（WEATHER/ LIGHTING/ ROAD）
    - 理由は長文OK（制限削除）
    """
    img_b64 = _encode_image_to_b64(image_path)

    v_now  = driving_facts.get("v_now")              # km/h
    intent = driving_facts.get("intent")
    trig   = driving_facts.get("trigger_type", "current")

    # ------ Stage2 プロンプト（必要情報のみ）------
    stage2_prompt = f"""
You are a driving safety expert.

Fixed environment summary from Stage1 (treat as GIVEN; do NOT re-guess):
{env_summary_text}

Ego/target context (concise):
- Ego Speed (km/h): {v_now}
- Target Intent: {intent}
- Trigger: {trig}

TASK: Based on the image and fixed facts above, output JSON with EXACT schema:

{{
  "risk_probability": 0-100 (int),
  "primary_hazard": one of ["vehicle","pedestrian","cyclist","infrastructure","weather","unknown"],
  "lane_change_detected": one of ["keep","left","right"],
  "suggested_maneuver": one of ["keep speed","change lane left","change lane right","brake"],
  "reason": "Free-form explanation (no length limit)"
}}

Only the JSON object. No markdown, no extra text.
""".strip()

    raw = _post_llava(stage2_prompt, img_b64, url=url, model=model, temperature=temperature)
    res = _extract_json(raw)

    # 旧互換のため最低限返す（weather/road_conditionはここでは付与しない）
    return {
        "risk_probability": _to_int_safe(res.get("risk_probability")),
        "lane_change_detected": res.get("lane_change_detected", "unknown"),
        "suggested_maneuver": res.get("suggested_maneuver", "unknown"),
        "reason": res.get("reason", ""),
        "primary_hazard": res.get("primary_hazard", "unknown"),
    }



def assess_risk_from_image(image_path: str,
                           driving_facts: Dict[str, Any],
                           *,
                           url: str = LLaVA_URL,
                           model: str = LLaVA_MODEL,
                           temperature: float = LLaVA_TEMPERATURE) -> dict:
    """簡易API：環境要約なしで Stage2 のみ。"""
    return assess_risk_from_image_with_context(
        image_path, driving_facts, env_summary_text="", url=url, model=model, temperature=temperature
    )


__all__ = [
    "LLaVA_URL",
    "LLaVA_MODEL",
    "LLaVA_TIMEOUT_SEC",
    "LLaVA_TEMPERATURE",
    "assess_environment_groups",
    "should_run_stage1",
    "assess_risk_from_image_with_context",
    "assess_risk_from_image",
]
