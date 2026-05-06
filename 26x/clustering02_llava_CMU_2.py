# -*- coding: utf-8 -*-
"""
LLaVA Yes/No QA over clustered frames (robust + fast + resumable + summaries)

改良点:
- timeout / retry / backoff / Unknown正規化 で止まりにくい
- 画像エンコードは1回 → 24問で共用（高速）
- 途中再開: 画像ごとに個別JSONを保存し、既存はスキップ
- クラスタ自動検出 & 拡張子を網羅（jpg/png/bmp等）
- 集計CSVを自動生成（クラスタ別/全体のYes率）
- GPUメモリログ＋empty_cache（任意）

出力:
  <BASE_DIR>/llava_results/
    ├─ per_image/
    │   └─ cluster_xx/<image>.json   ← 画像ごとのQ&A結果（再開用）
    ├─ llava_cluster_xx_results.json  ← クラスタの全画像Q&Aまとめ（互換）
    ├─ per_image_cluster_xx.csv       ← 画像×質問テーブル（クラスタ別）
    ├─ summary_cluster_xx.csv         ← Yes率まとめ（クラスタ別）
    ├─ per_image.csv                  ← 全クラスタ統合（画像×質問）
    └─ global_summary.csv             ← Yes率まとめ（全体）
"""

import os, io, re, json, time, base64, random, requests
from typing import List, Dict
from PIL import Image
import pandas as pd
import numpy as np
import torch

# ====== 設定 ======
BASE_DIR   = r"C:\Users\s1280\Desktop\SHRP2rawdata\central_frames_clustering_v7\clustered_images"
SAVE_DIR   = os.path.join(BASE_DIR, "llava_results")
PER_IMAGE_ROOT = os.path.join(SAVE_DIR, "per_image")

OLLAMA_URL   = "http://localhost:11434/api/generate"
LLAVA_MODEL  = "llava:latest"
TIMEOUT_SEC  = 30
MAX_RETRIES  = 3
BACKOFF_BASE = 1.6
SLEEP_BETWEEN_CALLS = 0.03  # 軽いレート制御（秒）

IMG_SIZE      = 224
JPEG_QUALITY  = 40

SEED = 42
random.seed(SEED)
np.random.seed(SEED)
if torch.cuda.is_available():
    torch.manual_seed(SEED)
    torch.cuda.manual_seed_all(SEED)

# === コンテキスト（24） ===
CONTEXTS = [
    "daytime", "night-time", "twilight", "sunny", "rainy", "snowy", "foggy", "dust/sandstorm",
    "trees overhead", "paved road", "lane markers visible", "off road", "parking lot", "indoors",
    "outdoors", "tunnel", "urban canyon", "rural area", "city", "highway", "construction zone",
    "heavy traffic", "bridge", "underpass"
]

# ====== ユーティリティ ======
def print_gpu_memory(prefix: str = ""):
    if torch.cuda.is_available():
        mem = torch.cuda.memory_allocated() / (1024 ** 2)
        res = torch.cuda.memory_reserved() / (1024 ** 2)
        print(f"{prefix} GPU Memory: allocated={mem:.1f} MB, reserved={res:.1f} MB")

def discover_clusters(base_dir: str) -> List[str]:
    return sorted([d for d in os.listdir(base_dir)
                   if os.path.isdir(os.path.join(base_dir, d)) and d.startswith("cluster_")])

def list_images(cluster_path: str) -> List[str]:
    exts = {".jpg", ".jpeg", ".png", ".bmp"}
    files = []
    for f in os.listdir(cluster_path):
        ext = os.path.splitext(f)[1].lower()
        if ext in exts:
            files.append(os.path.join(cluster_path, f))
    files.sort(key=lambda p: os.path.basename(p).lower())
    return files

def encode_image_b64(image_path: str) -> str:
    img = Image.open(image_path).convert("RGB").resize((IMG_SIZE, IMG_SIZE))
    buf = io.BytesIO()
    img.save(buf, format="JPEG", quality=JPEG_QUALITY)
    return base64.b64encode(buf.getvalue()).decode("utf-8")

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
            # 失敗時はテキストから拾う
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
        # backoff
        time.sleep((BACKOFF_BASE ** t) * 0.3 + random.uniform(0, 0.1))
    return "Unknown"

def per_image_json_path(cluster_name: str, img_name: str) -> str:
    d = os.path.join(PER_IMAGE_ROOT, cluster_name)
    os.makedirs(d, exist_ok=True)
    base, _ = os.path.splitext(img_name)
    return os.path.join(d, f"{base}.json")

def load_per_image_jsons(cluster_name: str) -> List[Dict]:
    d = os.path.join(PER_IMAGE_ROOT, cluster_name)
    if not os.path.exists(d):
        return []
    items = []
    for f in sorted(os.listdir(d)):
        if f.endswith(".json"):
            with open(os.path.join(d, f), "r", encoding="utf-8") as fh:
                items.append(json.load(fh))
    return items

# ====== メイン処理 ======
os.makedirs(SAVE_DIR, exist_ok=True)
clusters = discover_clusters(BASE_DIR)
if not clusters:
    raise SystemExit(f"クラスタフォルダが見つかりません: {BASE_DIR}")

print(f"🔎 対象クラスタ: {clusters}")

per_cluster_summary_paths = []
per_image_csv_paths = []

for cluster_name in clusters:
    cluster_path = os.path.join(BASE_DIR, cluster_name)
    frames = list_images(cluster_path)
    print(f"\n📂 {cluster_name} 対象画像数: {len(frames)} 枚")

    # === 画像ごとに処理（再開対応: 個別JSONがあればスキップ） ===
    for idx, img_path in enumerate(frames):
        img_name = os.path.basename(img_path)
        out_img_json = per_image_json_path(cluster_name, img_name)

        print(f"\n[{idx+1}/{len(frames)}] 🖼 {img_name} → 推論中...")

        if os.path.exists(out_img_json):
            print(f"   ↩ 既存結果をスキップ: {img_name}")
            continue

        try:
            img_b64 = encode_image_b64(img_path)
        except Exception as e:
            print(f"   ⚠ 画像読み込み失敗: {img_name} ({e})")
            continue

        qa_list = []
        for ctx in CONTEXTS:
            q = f"Is this {ctx}?"
            pred = ask_llava_yesno_b64(img_b64, q)
            qa_list.append({"question": q, "pred": pred})
            print(f"   {ctx:20s}: {pred}")
            time.sleep(SLEEP_BETWEEN_CALLS)

        with open(out_img_json, "w", encoding="utf-8") as f:
            json.dump({"cluster": cluster_name, "image": img_name, "qa_results": qa_list},
                      f, ensure_ascii=False, indent=2)

        print_gpu_memory(prefix="   After this image:")
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    # === クラスタJSON（互換フォーマット）を作成 ===
    per_image_items = load_per_image_jsons(cluster_name)
    cluster_results = []
    for it in per_image_items:
        cluster_results.append({
            "cluster": it["cluster"],
            "image": it["image"],
            "qa_results": it["qa_results"]
        })

    cluster_json = os.path.join(SAVE_DIR, f"llava_{cluster_name}_results.json")
    with open(cluster_json, "w", encoding="utf-8") as f:
        json.dump(cluster_results, f, ensure_ascii=False, indent=2)
    print(f"✅ {cluster_name} のQ&A結果を保存 → {cluster_json}")

    # === クラスタ別 CSV（画像×質問） ===
    # 各画像を1行、質問列は "Is this X?" カラム、値は Yes/No/Unknown
    rows = []
    for it in per_image_items:
        row = {"cluster": it["cluster"], "image": it["image"]}
        for qa in it["qa_results"]:
            row[qa["question"]] = qa["pred"]
        rows.append(row)
    df_img = pd.DataFrame(rows) if rows else pd.DataFrame(columns=["cluster","image"]+[f"Is this {c}?" for c in CONTEXTS])
    per_image_csv = os.path.join(SAVE_DIR, f"per_image_{cluster_name}.csv")
    df_img.to_csv(per_image_csv, index=False, encoding="utf-8-sig")
    per_image_csv_paths.append(per_image_csv)

    # === クラスタ別 Yes率サマリ ===
    def yes_rate(series: pd.Series) -> float:
        s = series.fillna("Unknown")
        return float((s == "Yes").mean()) if len(s) > 0 else 0.0

    cols = [c for c in df_img.columns if c.startswith("Is this ")]
    sum_row = {"cluster": cluster_name}
    for c in cols:
        sum_row[c] = yes_rate(df_img[c]) if c in df_img.columns else 0.0

    df_sum = pd.DataFrame([sum_row])
    per_cluster_summary = os.path.join(SAVE_DIR, f"summary_{cluster_name}.csv")
    df_sum.to_csv(per_cluster_summary, index=False, encoding="utf-8-sig")
    per_cluster_summary_paths.append(per_cluster_summary)

    print(f"✅ 集計CSVを保存: {per_image_csv}, {per_cluster_summary}")
    print_gpu_memory(prefix="   After cluster:")

# ====== 全体集計 ======
# Per-image 全統合
if per_image_csv_paths:
    all_img_df = pd.concat([pd.read_csv(p) for p in per_image_csv_paths], ignore_index=True)
    all_img_csv = os.path.join(SAVE_DIR, "per_image.csv")
    all_img_df.to_csv(all_img_csv, index=False, encoding="utf-8-sig")
    print(f"\n✅ 全体 per_image.csv を保存 → {all_img_csv}")

# クラスタサマリ統合
if per_cluster_summary_paths:
    all_sum_df = pd.concat([pd.read_csv(p) for p in per_cluster_summary_paths], ignore_index=True)
    global_csv = os.path.join(SAVE_DIR, "global_summary.csv")
    all_sum_df.to_csv(global_csv, index=False, encoding="utf-8-sig")
    print(f"✅ 全体 global_summary.csv を保存 → {global_csv}")

print("\n🎉 全クラスタの処理が完了しました！")
print(f"保存先: {SAVE_DIR}")
