import pandas as pd

# ===============================
# 入出力
# ===============================
INPUT_CSV = r"D:\merged_action4_all.csv"
OUTPUT_CSV = r"D:\merged_action4_paper_ready.csv"

df = pd.read_csv(INPUT_CSV)

# ===============================
# 1. 人手アノテーションを横持ちに変換
# ===============================
human_cols = ["frame_id", "scene", "image", "annotator_id", "human_action4", "unsure"]

df_human = df[human_cols].drop_duplicates()

df_human_wide = df_human.pivot(
    index=["frame_id", "scene", "image"],
    columns="annotator_id",
    values=["human_action4", "unsure"]
)

# 列名をフラット化
df_human_wide.columns = [
    f"{annot}_{col}"
    for col, annot in df_human_wide.columns
]

df_human_wide = df_human_wide.reset_index()

# ===============================
# 2. 非人手情報（RiSA / LLaVA）を抽出
# ===============================
drop_for_auto = ["annotator_id", "human_action4", "unsure"]
df_auto = df.drop(columns=[c for c in drop_for_auto if c in df.columns])

df_auto = df_auto.drop_duplicates(subset=["frame_id", "scene", "image"])

# ===============================
# 3. マージ
# ===============================
df = pd.merge(
    df_auto,
    df_human_wide,
    on=["frame_id", "scene", "image"],
    how="left"
)

# ===============================
# 4. 共通フレーム情報を正規化
# ===============================
df["scene_id"] = df["scene"]
df["image_name"] = df["image"]

df["base_scene_image_id"] = (
    df["base"].astype(str) + "_" + df["scene"] + "_" + df["image"]
)

def canonical_image_path(p):
    if isinstance(p, str):
        return p
    return None

df["image_path_canonical"] = df["image_path"].apply(canonical_image_path)

# ===============================
# 5. 冗長列削除
# ===============================
DROP_COLS = [
    "scene", "image",
    "base_nano", "scene_nano", "image_nano",
    "base_mini", "scene_mini", "image_mini",
]

df = df.drop(columns=[c for c in DROP_COLS if c in df.columns])

# ===============================
# 6. 列順整理（論文向け）
# ===============================
FRONT_COLS = [
    "frame_id",
    "scene_id",
    "image_name",
    "base_scene_image_id",
    "image_path_canonical",
    "A01_human_action4",
    "A02_human_action4",
    "A01_unsure",
    "A02_unsure",
]

FRONT_COLS = [c for c in FRONT_COLS if c in df.columns]
OTHER_COLS = [c for c in df.columns if c not in FRONT_COLS]

df = df[FRONT_COLS + OTHER_COLS]

# ===============================
# 7. 保存
# ===============================
df.to_csv(OUTPUT_CSV, index=False)

print("Saved:", OUTPUT_CSV)
print("Rows:", len(df))
print("Columns:", len(df.columns))
