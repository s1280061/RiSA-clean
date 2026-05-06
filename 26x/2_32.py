import os
import shutil
import glob

# === 入力フォルダ（3,4,5,6） ===
ROOTS = [
    r"C:\Users\s1280\Desktop\SHRP2_outputs_v2\3",
    r"C:\Users\s1280\Desktop\SHRP2_outputs_v2\4",
    r"C:\Users\s1280\Desktop\SHRP2_outputs_v2\5",
    r"C:\Users\s1280\Desktop\SHRP2_outputs_v2\6",
]

# === 出力先 ===
OUT_ROOT = r"C:\Users\s1280\Desktop\SHRP2_outputs_sorted"

# === 分類ルール（優先度の高いものから順に並べる） ===
CATEGORY_RULES = [
    ("lane_bbox_hist_future", "_lane_bbox_hist_future"),
    ("lane_bbox_hist", "_lane_bbox_hist"),
    ("lane_bbox", "_lane_bbox"),
    ("lane", "_lane"),
    ("bbox_hist_future", "_bbox_hist_future"),
    ("bbox_hist", "_bbox_hist"),
    ("bbox", "_bbox"),
    ("pre", "_pre"),
    ("raw", "_raw"),
]

# === 出力フォルダ作成 ===
for cat, _ in CATEGORY_RULES:
    os.makedirs(os.path.join(OUT_ROOT, cat), exist_ok=True)

# === 分類処理 ===
for root in ROOTS:
    print(f"▶ Scanning: {root}")

    for jpg in glob.glob(os.path.join(root, "scene_*", "*.jpg")):
        filename = os.path.basename(jpg)

        # 優先度の高い順に判定
        for cat, key in CATEGORY_RULES:
            if key in filename:
                dst = os.path.join(OUT_ROOT, cat, filename)
                shutil.copy2(jpg, dst)
                print(f"  → {filename}  ==>  {cat}")
                break
