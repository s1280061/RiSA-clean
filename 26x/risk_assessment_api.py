import base64
import requests
from PIL import Image
import io
import re
import json


def assess_risk_from_image(image_path, driving_facts):
    """
    LLaVA に画像＋コンテキストを送ってリスク評価を行い、
    数値ではなく、意味のある文字列に変換して返す。

    Returns:
        dict:
            {
              "risk_probability": 0-100,
              "lane_change_detected": "keep/left/right",
              "suggested_maneuver": "keep speed / change lane left / change lane right / brake",
              "weather": "clear/cloudy/rainy...",
              "road_condition": "dry/wet/icy...",
              "reason": "AIが返した説明"
            }
    """

    # === 画像を base64 にエンコード ===
    img = Image.open(image_path).convert("RGB").resize((224, 224))
    buffer = io.BytesIO()
    img.save(buffer, format="JPEG", quality=40)
    img_b64 = base64.b64encode(buffer.getvalue()).decode("utf-8")

    # === driving_facts を展開 ===
    v_now = driving_facts["v_now"]
    intent = driving_facts["intent"]
    cx = driving_facts["cx"]
    cy = driving_facts["cy"]
    area = driving_facts["area"]

    # === プロンプト構築 ===
    prompt = f"""
You are a driving safety expert.

Given this image and vehicle context:
- Ego Speed: {v_now:.1f} km/h
- Target Vehicle Intent: {intent}
- Target Vehicle Coordinates: ({cx},{cy})
- Target Vehicle Area: {area}

Analyze the scene and respond ONLY in JSON with:
{{
  "risk_probability": int (0-100),
  "lane_change_detected": one of ["keep", "left", "right"],
  "suggested_maneuver": one of ["keep speed", "change lane left", "change lane right", "brake"],
  "weather": "clear / cloudy / rainy / foggy / night",
  "road_condition": "dry / wet / icy / snowy / unclear",
  "reason": "short explanation"
}}
DO NOT wrap in triple backticks.
    """

    # === API送信 ===
    payload = {
        "model": "llava:latest",
        "prompt": prompt,
        "images": [img_b64],
        "stream": False,
        "options": {"temperature": 0.0}
    }

    response = requests.post("http://localhost:11434/api/generate", json=payload)
    raw_text = response.text
    print("📨 ChatGPTからの生レスポンス:")
    print(raw_text)

    # === JSONパース ===
    match = re.search(r'\{.*\}', raw_text, re.DOTALL)
    if match:
        try:
            outer = json.loads(match.group(0))
            inner_raw = outer.get("response", "")
            inner_match = re.search(r'\{.*\}', inner_raw, re.DOTALL)

            if inner_match:
                result = json.loads(inner_match.group(0))
            else:
                result = outer


            return result

        except Exception as e:
            print("⚠️ JSONパース失敗:", e)

    # === フォールバック ===
    return {
        "risk_probability": None,
        "lane_change_detected": "unknown",
        "suggested_maneuver": "unknown",
        "weather": "?",
        "road_condition": "?",
        "reason": "LLaVA response could not be parsed."
    }
