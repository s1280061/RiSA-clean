# -*- coding: utf-8 -*-
"""
risk_assessment_api_xx2.py (v2.5 pedestrian-state & action, stricter defaults)

Stage2（最小構成・歩行者重視版）:
- 画像1枚と「背後情報JSON（stage1_env / perception / detected_vehicles）」を
  LLaVA(Ollama) に渡し、歩行者の状態推定＋自車アクションを JSON で返す。
- 返すキー：
    {
      "pedestrian_state": "go|stop|crossing|unknown",
      "suggested_action": "go|stop|unknown",
      "reason": str,
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
ACTION_MAP = {
    "go": "go",
    "proceed": "go",
    "keep going": "go",
    "continue": "go",
    "stop": "stop",
    "brake": "stop",
    "decelerate": "stop",
}
PED_STATE_MAP = {
    "go": "go",
    "walking": "go",
    "standing": "stop",
    "stopped": "stop",
    "stop": "stop",
    "cross": "crossing",
    "crossing": "crossing",
    "jaywalking": "crossing",
}
FORBIDDEN_TERMS = ("risk", "probability", "hazard", "level", "score")


def _normalize_action(x: Any) -> str:
    if not isinstance(x, str):
        return "unknown"
    t = re.sub(r"\s+", " ", x.strip().lower())
    return ACTION_MAP.get(t, t if t in ("go", "stop") else "unknown")


def _normalize_ped_state(x: Any) -> str:
    if not isinstance(x, str):
        return "unknown"
    t = re.sub(r"\s+", " ", x.strip().lower())
    return PED_STATE_MAP.get(t, t if t in ("go", "stop", "crossing") else "unknown")


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

    if inner_str.startswith("```"):
        inner_str = inner_str.strip("`")
        inner_str = re.sub(r"^\s*json\s*", "", inner_str, flags=re.IGNORECASE).strip()

    if not inner_str.startswith("{"):
        brace = inner_str.find("{")
        if brace >= 0:
            inner_str = inner_str[brace:]

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


# ========= メインAPI =========
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
      - context: 背後情報JSON（stage1_env / perception / detected_vehicles）
        例:
          {
            "stage1_env": {"rural_area":"NO", "city":"YES", ...},
            "perception": {"pedestrian_present": true, "pedestrian_count": 1},
            "detected_vehicles": {"count": 2}
          }
    出力:
      - pedestrian_state: go|stop|crossing|unknown
      - suggested_action: go|stop|unknown
      - reason: 説明（禁止語除去）
      - parsed: True/False
    """
    img_b64 = _b64_of_image(image_path)

    prompt = f"""You are a driving safety expert.
Use the following JSON facts as true. Do not contradict them.
Focus ONLY on pedestrians and the ego vehicle's response.

CONTEXT (JSON):
```json
{json.dumps(context, ensure_ascii=False)}
```

Return ONE JSON object only, with this schema:
{{
  "pedestrian_state": "one of ['go','stop','crossing']",
  "suggested_action": "one of ['go','stop']",
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
            "pedestrian_state": "unknown",
            "suggested_action": "unknown",
            "reason": f"LLaVA request error: {last_exc}",
            "parsed": False,
        }

    result_dict = _extract_inner_json(outer.get("response"))
    if result_dict is None:
        return {
            "pedestrian_state": "unknown",
            "suggested_action": "unknown",
            "reason": "LLaVA response could not be parsed.",
            "parsed": False,
        }

    # ---- 正規化 ----
    ped_state = _normalize_ped_state(result_dict.get("pedestrian_state", "unknown"))
    action = _normalize_action(result_dict.get("suggested_action", "unknown"))
    reason = _sanitize_reason(result_dict.get("reason", ""))

    # ---- Fallbacks: 歩行者がいるのにunknownなら保守的に補完 ----
    try:
        ped_present = bool(context.get("perception", {}).get("pedestrian_present", False))
    except Exception:
        ped_present = False
    if ped_present:
        if ped_state == "unknown":
            ped_state = "crossing"
        if action == "unknown":
            action = "stop"

    return {
        "pedestrian_state": ped_state,
        "suggested_action": action,
        "reason": reason,
        "parsed": True,
    }


if __name__ == "__main__":
    # 簡易テスト（画像パスを差し替えてください）
    ctx = {
        "stage1_env": {
            "rural_area": "NO",
            "city": "YES",
            "snowy": "NO",
            "sunny": "YES",
            "rainy": "NO"
        },
        "perception": {
            "pedestrian_present": True,
            "pedestrian_count": 1
        },
        "detected_vehicles": {
            "count": 2
        },
    }
    test_img = "path/to/your_image.jpg"
    print(json.dumps(
        assess_risk_from_image_with_context(test_img, ctx),
        ensure_ascii=False,
        indent=2
    ))