# -*- coding: utf-8 -*-
"""
risk_assessment_api_xx1.py

Stage2（第一段階・最小構成）:
- 画像1枚と「背後情報JSON（stage1_env / ego_vehicle / perception / detected_vehicles）」を
  LLaVA(Ollama) に渡し、risk_assessment を厳密JSONで返す。
- 返すキー（最小セット）:
    {
      "risk_probability": int|None,                # 0..100
      "lane_change_detected": "keep|left|right|unknown",
      "suggested_maneuver": "keep_speed|change_lane_left|change_lane_right|decelerate|unknown",
      "reason": str,                                # 簡潔な説明（禁止語を自動除去）
      "parsed": bool
    }

依存:
    pip install requests pillow
Ollama:
    ollama run llava:latest  (http://localhost:11434 で稼働)
"""

from __future__ import annotations

import base64
import io
import json
import re
from typing import Any, Dict, Optional

import requests
from PIL import Image


# ========= LLaVA / Ollama 設定 =========
OLLAMA_ENDPOINT = "http://localhost:11434/api/generate"
DEFAULT_MODEL = "llava:latest"
TIMEOUT_SEC = 60
RETRIES = 1  # 失敗時の再送回数（合計 RETRIES+1 回）


# ========= 正規化 / 禁止語 =========
SUGGESTED_MAP = {
    "keep speed": "keep_speed",
    "change lane left": "change_lane_left",
    "change lane right": "change_lane_right",
    "decelerate": "decelerate",
    "brake": "decelerate",
    "increase following distance": "decelerate",
    "slow down": "decelerate",
    "reduce speed": "decelerate",
    "deaccelerate": "decelerate",  # 誤綴りも吸収
}
LANE_MAP = {"keep": "keep", "left": "left", "right": "right"}
FORBIDDEN_TERMS = ("risk", "probability", "hazard", "level", "score")

# 「Speed=..., RiskZone=...」などの事実プレフィックスを外す
_FACT_PREFIX_RE = re.compile(
    r'^\s*(Speed\s*=\s*[\d\.]+\s*km/h;?\s*)?(Risk\s*Zone\s*=\s*(YES|NO)\.?;?\s*)',
    re.IGNORECASE
)


# ========= ユーティリティ =========
def _normalize_suggested(x: Any) -> str:
    if not isinstance(x, str):
        return "unknown"
    t = x.strip().lower().replace("-", " ")
    t = re.sub(r"\s+", " ", t)
    return SUGGESTED_MAP.get(t, t.replace(" ", "_"))


def _normalize_lane(x: Any) -> str:
    if not isinstance(x, str):
        return "unknown"
    return LANE_MAP.get(x.strip().lower(), "unknown")


def _sanitize_reason(text: Any) -> str:
    s = str(text or "").strip()
    for t in FORBIDDEN_TERMS:
        s = re.sub(rf"\b{t}\b", "", s, flags=re.IGNORECASE)
    s = re.sub(r"\s{2,}", " ", s).strip()
    return s


def _b64_of_image(image_path: str) -> str:
    """224x224 / JPEG Q=40 で軽量化しBase64へ"""
    img = Image.open(image_path).convert("RGB").resize((224, 224))
    buf = io.BytesIO()
    img.save(buf, format="JPEG", quality=40)
    return base64.b64encode(buf.getvalue()).decode("utf-8")


def _extract_inner_json(inner: Any) -> Optional[Dict[str, Any]]:
    """Ollama outer['response'] が dict ならそのまま、str なら掃除して json.loads。"""
    if isinstance(inner, dict):
        return inner

    inner_str = str(inner or "").strip()
    if not inner_str:
        return None

    # ```json ... ``` のガード除去
    if inner_str.startswith("```"):
        inner_str = inner_str.strip("`")
        inner_str = re.sub(r"^\s*json\s*", "", inner_str, flags=re.IGNORECASE).strip()

    # 先頭に { が無ければ最初の { から
    if not inner_str.startswith("{"):
        brace = inner_str.find("{")
        if brace >= 0:
            inner_str = inner_str[brace:]

    # 最外括弧でトリム
    depth = 0
    end = -1
    for i, ch in enumerate(inner_str):
        if ch == "{":
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0:
                end = i + 1
                break
    if end > 0:
        inner_str = inner_str[:end]

    try:
        return json.loads(inner_str)
    except Exception as e:
        print("⚠️ inner JSON decode failed:", e)
        print("── inner raw ──\n", inner_str[:600])
        return None


# ========= メインAPI（最小構成） =========
def assess_risk_from_image_with_context(
    image_path: str,
    context: Dict[str, Any],
    *,
    model: str = DEFAULT_MODEL,
    endpoint: str = OLLAMA_ENDPOINT,
    temperature: float = 0.0,
) -> Dict[str, Any]:
    """
    入力:
      - image_path: 軌跡付き画像のパス（1枚）
      - context: 背後情報JSON（stage1_env / ego_vehicle / perception / detected_vehicles）
                 ※ Stage1(YES/NO) は「真」として扱わせる
    """
    img_b64 = _b64_of_image(image_path)

    # f-string 内で JSON スキーマの { } をリテラルにするため {{ }} にする
    prompt = f"""You are a driving safety expert.
Use the following JSON facts as true. Do not contradict them.

CONTEXT (JSON):
```json
{json.dumps(context, ensure_ascii=False)}
```

Return ONE JSON object only, with this schema:
{{
"risk_probability": "an integer 0-100",
"lane_change_detected": "one of [\\"keep\\",\\"left\\",\\"right\\"]",
"suggested_maneuver": "one of [\\"keep speed\\",\\"change lane left\\",\\"change lane right\\",\\"decelerate\\"]",
"reason": "short explanation without probabilities or scores"
}}"""

    payload = {
        "model": model,
        "prompt": prompt,
        "images": [img_b64],
        "stream": False,
        "format": "json",
        "options": {"temperature": temperature, "mirostat": 0},
    }

    # 送信（簡易リトライ）
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

    result_dict = _extract_inner_json(outer.get("response"))
    if result_dict is None:
        return {
            "risk_probability": None,
            "lane_change_detected": "unknown",
            "suggested_maneuver": "unknown",
            "reason": "LLaVA response could not be parsed.",
            "parsed": False,
        }

    # ---- 正規化 ----
    rp = result_dict.get("risk_probability", None)
    try:
        if rp is not None:
            rp = int(rp)
            if not (0 <= rp <= 100):
                rp = None
    except Exception:
        rp = None

    lane = _normalize_lane(result_dict.get("lane_change_detected", "unknown"))
    sugg = _normalize_suggested(result_dict.get("suggested_maneuver", "unknown"))

    # reason：禁止語の削除 + 事実系プレフィックスの除去
    reason = _sanitize_reason(result_dict.get("reason", ""))
    reason = _FACT_PREFIX_RE.sub("", reason).strip()

    return {
        "risk_probability": rp,
        "lane_change_detected": lane,
        "suggested_maneuver": sugg,
        "reason": reason,
        "parsed": True,
    }


if __name__ == "__main__":
    # 簡易テスト（画像パスを差し替えてください）
    ctx = {
        "stage1_env": {"rural_area": "NO", "city": "YES", "snowy": "NO", "sunny": "YES", "rainy": "NO"},
        "ego_vehicle": {"speed_kmh": 52.3, "risk_zone": False, "risk_zone_predicted": True},
        "perception": {
            "turn_signal": {"final": "right", "final_conf_window": 0.74, "window_counts": {"right": 3, "off": 2}},
            "brake": {"final": "off", "final_conf_window": 0.86, "window_counts": {"off": 5}},
        },
        "detected_vehicles": {"count": 2},
    }
    test_img = "path/to/your_image.jpg"
    print(json.dumps(assess_risk_from_image_with_context(test_img, ctx), ensure_ascii=False, indent=2))