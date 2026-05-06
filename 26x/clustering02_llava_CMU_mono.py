# -*- coding: utf-8 -*-
"""
Single-image LLaVA Yes/No tester
- 指定した1枚の画像に対して、24コンテキストの Yes/No を問い合わせ
- 端末表示 + JSON/CSV 保存
- 本体スクリプトと同じ正規化・timeout/retry/backoff を実装

使い方:
  python test_single_llava_yesno.py --image "C:\\Users\\s1280\\Desktop\\SHRP2rawdata\\central_frames_clustering_v7\\clustered_images\\cluster_02\\scene_318_center.jpg"
"""

import os, io, re, json, time, base64, random, argparse, requests
from typing import List, Dict
from PIL import Image
import pandas as pd
import numpy as np

# ====== 設定（本体と合わせる） ======
OLLAMA_URL   = "http://localhost:11434/api/generate"
LLAVA_MODEL  = "llava:latest"
TIMEOUT_SEC  = 30
MAX_RETRIES  = 3
BACKOFF_BASE = 1.6
SLEEP_BETWEEN_CALLS = 0.03  # 質問間のわずかな待機（任意）

IMG_SIZE     = 224
JPEG_QUALITY = 40

SEED = 42
random.seed(SEED)
np.random.seed(SEED)

# === コンテキスト（24） ===
CONTEXTS = [
    "daytime", "night-time", "twilight", "sunny", "rainy", "snowy", "foggy", "dust/sandstorm",
    "trees overhead", "paved road", "lane markers visible", "off road", "parking lot", "indoors",
    "outdoors", "tunnel", "urban canyon", "rural area", "city", "highway", "construction zone",
    "heavy traffic", "bridge", "underpass"
]

def _normalize_yesno(raw: str) -> str:
    if not raw:
        return "Unknown"
    s = str(raw).strip().lower()
    if re.fullmatch(r"(yes|yes\.)", s): return "Yes"
    if re.fullmatch(r"(no|no\.)",  s): return "No"
    if ("yes" in s) and ("no" not in s): return "Yes"
    if ("no"  in s) and ("yes" not in s): return "No"
    return "Unknown"

def _ask_llava_once(payload: Dict) -> (bool, str):
    try:
        resp = requests.post(OLLAMA_URL, json=payload, timeout=TIMEOUT_SEC)
        resp.raise_for_status()
        # JSON優先
        try:
            data = resp.json()
            return True, str(data.get("response", ""))
        except Exception:
            txt = resp.text
            m = re.search(r'"response"\s*:\s*"([^"]*)"', txt)
            return True, (m.group(1) if m else txt)
    except Exception as e:
        return False, str(e)

def ask_llava_yesno_b64(img_b64: str, question: str) -> str:
    payload = {
        "model": LLAVA_MODEL,
        "prompt": f"{question} Answer only Yes or No.",
        "images": [img_b64],
        "stream": False,
        "options": {"temperature": 0.0}
    }
    for t in range(MAX_RETRIES):
        ok, raw = _ask_llava_once(payload)
        if ok:
            ans = _normalize_yesno(raw)
            if ans != "Unknown":
                return ans
        time.sleep((BACKOFF_BASE ** t) * 0.3 + random.uniform(0, 0.1))
    return "Unknown"

def encode_image_b64(image_path: str) -> str:
    img = Image.open(image_path).convert("RGB").resize((IMG_SIZE, IMG_SIZE))
    buf = io.BytesIO()
    img.save(buf, format="JPEG", quality=JPEG_QUALITY)
    return base64.b64encode(buf.getvalue()).decode("utf-8")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--image", type=str, required=True,
                        help="テストする画像のフルパス")
    parser.add_argument("--save_dir", type=str, default=None,
                        help="保存先（省略時は画像と同じディレクトリに llava_single_results）")
    args = parser.parse_args()

    image_path = args.image
    if not os.path.exists(image_path):
        raise SystemExit(f"画像が見つかりません: {image_path}")

    # 保存先
    if args.save_dir is None:
        base_dir = os.path.dirname(image_path)
        save_dir = os.path.join(base_dir, "llava_single_results")
    else:
        save_dir = args.save_dir
    os.makedirs(save_dir, exist_ok=True)

    img_name = os.path.basename(image_path)
    base, _ = os.path.splitext(img_name)

    # 画像エンコードは1回のみ
    try:
        img_b64 = encode_image_b64(image_path)
    except Exception as e:
        raise SystemExit(f"画像の読み込み/エンコードに失敗: {e}")

    # 24問を順番に実行
    qa_list = []
    print(f"\n🖼 Image: {image_path}")
    for ctx in CONTEXTS:
        q = f"Is this {ctx}?"
        pred = ask_llava_yesno_b64(img_b64, q)
        qa_list.append({"question": q, "pred": pred})
        print(f"   {ctx:20s}: {pred}")
        time.sleep(SLEEP_BETWEEN_CALLS)

    # 端末に見やすい表（Yes/No/Unknown）
    df = pd.DataFrame({
        "question": [qa["question"] for qa in qa_list],
        "pred":     [qa["pred"] for qa in qa_list],
    })
    print("\n=== Results (table) ===")
    print(df.to_string(index=False))

    # JSON保存
    out_json = os.path.join(save_dir, f"{base}_llava_qa.json")
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump({
            "image": img_name,
            "path": image_path,
            "qa_results": qa_list
        }, f, ensure_ascii=False, indent=2)
    print(f"\n✅ JSON saved: {out_json}")

    # CSV保存（1行＝この画像、列＝各質問）
    row = {"image": img_name, "path": image_path}
    for qa in qa_list:
        row[qa["question"]] = qa["pred"]
    df_row = pd.DataFrame([row])
    out_csv = os.path.join(save_dir, f"{base}_llava_qa.csv")
    df_row.to_csv(out_csv, index=False, encoding="utf-8-sig")
    print(f"✅ CSV saved:  {out_csv}")

    print("\n🎉 Done.")

if __name__ == "__main__":
    main()
