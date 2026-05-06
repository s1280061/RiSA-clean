import json
import glob
import numpy as np
import os
from collections import defaultdict

# ルートと対象フォルダ
ROOT = r"C:\Users\s1280\Desktop\RISA_raw_data"
FOLDERS = ["3", "4", "5", "6"]

# stageごとの latency をためる
latency_stats = defaultdict(list)

def mean_or_none(xs):
    return float(np.mean(xs)) if xs else None

# ==== JSONL から読み込み ====
for folder in FOLDERS:
    latency_dir = os.path.join(ROOT, folder, "latency")
    files = sorted(glob.glob(os.path.join(latency_dir, "scene_*_latency.jsonl")))

    print(f"[Folder {folder}] Found {len(files)} latency files.")

    for file in files:
        with open(file, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    obj = json.loads(line)
                except Exception:
                    continue

                stage = obj.get("stage")
                if not stage:
                    continue

                # 最初の50フレームなどウォームアップは除外（warmup=True をスキップ）
                if obj.get("warmup", False):
                    continue

                lat = obj.get("latency_ms")  # ← ここが重要
                if isinstance(lat, (int, float)):
                    latency_stats[stage].append(lat)

# ==== 平均を計算 ====
summary = {
    stage: {
        "avg_latency_ms": mean_or_none(vals),
        "count": len(vals),
    }
    for stage, vals in latency_stats.items()
}

# ==== 表示用フォーマット ====
def fmt(v):
    return f"{v:.3f}" if isinstance(v, (float, int)) else "–"

print("\n=== Average Latency Over All Folders (warmup除外) ===")
for stage, vals in sorted(summary.items()):
    print(
        f"{stage:18s} | latency={fmt(vals['avg_latency_ms'])} ms"
        f" | count={vals['count']}"
    )
