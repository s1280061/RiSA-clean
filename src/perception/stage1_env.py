# -*- coding: utf-8 -*-
"""
Stage1: 環境認識 (YES/NO) - 5回個別呼び出し版（最小API）
- LLaVA (Ollama) に画像を渡し、各質問を毎回1問ずつ YES/NO で取得
  * rural_area / city / snowy / sunny / rainy
- 公開関数:
    assess_environment_stage1(image_path: str) -> Dict[str, str]
    assess_environment_stage1_from_frame(frame_bgr: np.ndarray) -> Dict[str, str]
"""

from typing import Dict, Any
import base64
import io
import json

import cv2
import requests
from PIL import Image
import numpy as np

# ===== LLaVA (Ollama) 設定 =====
OLLAMA_ENDPOINT = "http://localhost:11434/api/generate"
LLAVA_MODEL = "llava:latest"
TIMEOUT = 60
RETRIES = 1  # 簡易リトライ

# ===== YES/NO 正規化 =====
def _to_yes_no(x: Any, default: str = "NO") -> str:
    s = str(x or "").strip().upper()
    if s.startswith("Y") or s in ("TRUE", "1"):
        return "YES"
    if s.startswith("N") or s in ("FALSE", "0"):
        return "NO"
    return default

# ===== 画像エンコード =====
def _b64_of_image_path(image_path: str) -> str:
    img = Image.open(image_path).convert("RGB").resize((224, 224))
    buf = io.BytesIO()
    img.save(buf, format="JPEG", quality=40)
    return base64.b64encode(buf.getvalue()).decode("utf-8")

def _b64_of_frame_bgr(frame_bgr: np.ndarray) -> str:
    img_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
    pil = Image.fromarray(img_rgb).resize((224, 224))
    buf = io.BytesIO()
    pil.save(buf, format="JPEG", quality=40)
    return base64.b64encode(buf.getvalue()).decode("utf-8")

# ===== 1問だけを聞く（毎回 LLaVA 起動） =====
def _ask_yesno(image_b64: str, question: str) -> str:
    prompt = (
        "You are an assistant that answers strictly YES or NO.\n"
        "Look at the image and answer the question below using only one word: YES or NO.\n\n"
        f"Q: {question}\nA: "
    )
    payload = {
        "model": LLAVA_MODEL,
        "prompt": prompt,
        "images": [image_b64],
        "stream": False,
        "options": {"temperature": 0.0, "mirostat": 0},
    }

    last_err = None
    for _ in range(RETRIES + 1):
        try:
            resp = requests.post(OLLAMA_ENDPOINT, json=payload, timeout=TIMEOUT)
            resp.raise_for_status()
            raw = str(resp.json().get("response", "")).strip()
            first = raw.split()[0] if raw else ""
            return _to_yes_no(first)
        except Exception as e:
            last_err = e
    print(f"⚠️ LLaVA yes/no request failed: {last_err}")
    return "NO"

# ===== 共通：5質問を実行 =====
_QUESTIONS = {
    "rural_area": "Is this rural area?",
    "city":       "Is this city?",
    "snowy":      "Is this snowy?",
    "sunny":      "Is this sunny?",
    "rainy":      "Is this rainy?",
}

def _run_all(image_b64: str) -> Dict[str, str]:
    return {k: _ask_yesno(image_b64, q) for k, q in _QUESTIONS.items()}

# ===== 公開API：画像パスから =====
def assess_environment_stage1(image_path: str) -> Dict[str, str]:
    """
    画像1枚に対して5項目を YES/NO で返す:
      {
        "rural_area": "YES|NO",
        "city":       "YES|NO",
        "snowy":      "YES|NO",
        "sunny":      "YES|NO",
        "rainy":      "YES|NO"
      }
    """
    img_b64 = _b64_of_image_path(image_path)
    return _run_all(img_b64)

# ===== 公開API：OpenCVフレームから =====
def assess_environment_stage1_from_frame(frame_bgr: np.ndarray) -> Dict[str, str]:
    img_b64 = _b64_of_frame_bgr(frame_bgr)
    return _run_all(img_b64)

# ===== サンプル実行 =====
if __name__ == "__main__":
    # 1) 画像パス版
    test_image = "path/to/your_image.jpg"
    print(json.dumps(assess_environment_stage1(test_image), ensure_ascii=False, indent=2))

    # 2) フレーム版（テスト用に同じ画像を読み込んで変換）
    # frame = cv2.imread(test_image)  # 実運用では cap.read() の frame を渡す
    # print(json.dumps(assess_environment_stage1_from_frame(frame), ensure_ascii=False, indent=2))