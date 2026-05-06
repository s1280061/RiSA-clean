import pandas as pd
import json
import glob
import os

# ===============================
# パス設定
# ===============================
CSV_PATH = r"D:\merged_action4_paper_ready.csv"
JSON_ROOT = r"C:\Users\s1280\Desktop\experiment04(final)\JSON"
OUT_CSV = r"D:\merged_action4_paper_ready_with_json.csv"

# ===============================
# CSV 読み込み
# ===============================
df = pd.read_csv(CSV_PATH)

# マージキー（image_name）
df["image_name"] = df["image_name"].astype(str)

print("[INFO] CSV rows:", len(df))

# ===============================
# context.json から抽出
# ===============================
context_records = []

context_files = glob.glob(
    os.path.join(JSON_ROOT, "*", "*_context.json")
)

print("[INFO] Found context files:", len(context_files))

for path in context_files:
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)

    # 空配列対策
    if not isinstance(data, list) or len(data) == 0:
        continue

    for item in data:
        image_path = item.get("image_path")
        if image_path is None:
            continue

        image_name = os.path.basename(image_path)

        detected_count = (
            item
            .get("detected_vehicles", {})
            .get("count")
        )

        context_records.append({
            "image_name": image_name,
            "detected_vehicle_count": detected_count
        })

df_context = pd.DataFrame(context_records)
print("[INFO] Context records:", len(df_context))

# ===============================
# llava.json から抽出
# ===============================
llava_records = []

llava_files = glob.glob(
    os.path.join(JSON_ROOT, "*", "*_llava.json")
)

print("[INFO] Found llava files:", len(llava_files))

for path in llava_files:
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)

    # 空配列対策
    if not isinstance(data, list) or len(data) == 0:
        continue

    for item in data:
        image_path = item.get("image_path")
        if image_path is None:
            continue

        image_name = os.path.basename(image_path)

        lane_change = (
            item
            .get("risk_assessment", {})
            .get("lane_change_detected")
        )

        llava_records.append({
            "image_name": image_name,
            "lane_change_detected_llava": lane_change
        })

df_llava = pd.DataFrame(llava_records)
print("[INFO] Llava records:", len(df_llava))

# ===============================
# マージ（left join）
# ===============================
df = df.merge(df_context, on="image_name", how="left")
df = df.merge(df_llava, on="image_name", how="left")

# ===============================
# 保存
# ===============================
df.to_csv(OUT_CSV, index=False)

print("=== DONE ===")
print("Saved CSV:", OUT_CSV)

# ===============================
# 簡易チェック（任意）
# ===============================
print("\n[CHECK SAMPLE]")
print(
    df[[
        "image_name",
        "detected_vehicle_count",
        "lane_change_detected_llava"
    ]].head(15)
)
