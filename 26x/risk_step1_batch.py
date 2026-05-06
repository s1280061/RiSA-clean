# -*- coding: utf-8 -*-
"""
Step1 – LLaVA評価一括スクリプト（confidence考慮版）
- 対象:  --root/<base>/new_divided/scene_XXX_fire 内の *pre_tid*.jpg（postは除外）
- 参照:  --root/<base>/new_divided/scene_XXX_context.json
- 出力:  --root/<base>/new_divided/llava_result_v2/scene_XXX/<image>.json
- ベース: 3,4,5,6 (デフォルト)
- LLaVA:  http://localhost:11434/api/generate に llava:latest が起動していること
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

# ========= LLaVA / Ollama 設定 =========
OLLAMA_ENDPOINT = "http://localhost:11434/api/generate"
DEFAULT_MODEL = "llava:latest"
TIMEOUT_SEC = 90
RETRIES = 1  # 失敗時リトライ回数（合計 RETRIES+1 回）

# ========= 画像 -> base64（軽量化） =========
def b64_of_image(image_path: str) -> str:
    img = Image.open(image_path).convert("RGB").resize((224, 224))
    buf = io.BytesIO()
    img.save(buf, format="JPEG", quality=40)
    return base64.b64encode(buf.getvalue()).decode("utf-8")

# ========= LLaVA 応答の JSON を抽出 =========
def extract_inner_json(inner: Any) -> Optional[Dict[str, Any]]:
    if isinstance(inner, dict):
        return inner
    s = str(inner or "").strip()
    if not s:
        return None
    # ```json ... ``` の体裁を剥がす
    if s.startswith("```"):
        s = s.strip("`")
        s = re.sub(r"^\s*json\s*", "", s, flags=re.IGNORECASE).strip()
    if not s.startswith("{"):
        i = s.find("{")
        if i >= 0:
            s = s[i:]
    # 最外カッコ対応
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

# ========= LLaVA 呼び出し（プロンプトは conf 再考指示つき） =========
def call_llava(image_b64: str, context: Dict[str, Any],
               endpoint: str = OLLAMA_ENDPOINT,
               model: str = DEFAULT_MODEL,
               temperature: float = 0.0) -> Dict[str, Any]:
    """
    戻り値スキーマ（最小セット）:
      {
        "risk_probability": int|None,                # 0..100
        "lane_change_detected": "keep|left|right|unknown",
        "suggested_maneuver": "keep_speed|change_lane_left|change_lane_right|decelerate|unknown",
        "reason": str,
        "parsed": bool
      }
    """
    # ---- プロンプト（turn_signal.conf を考慮して再考させる） ----
    prompt = f"""
You are a driving safety expert.
Use the following JSON facts as *inputs*, but do NOT blindly trust fields with confidence values.
Re-evaluate turn signal using the image evidence:

CONTEXT (JSON):
```json
{json.dumps(context, ensure_ascii=False)}
