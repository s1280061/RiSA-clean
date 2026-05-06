import pandas as pd
from wordcloud import WordCloud
import matplotlib.pyplot as plt
from collections import Counter
import re
import os

# ===============================
# 入力 / 出力
# ===============================
CSV_PATH = r"D:\merged_action4_paper_ready.csv"
OUT_DIR = r"D:\llava_wordclouds_score_split"
os.makedirs(OUT_DIR, exist_ok=True)

# ===============================
# 設定
# ===============================
ACTIONS = [
    "keep_speed",
    "decelerate",
    "change_lane_left",
    "change_lane_right"
]

# ★ WordCloud対象はコメント列 ★
COMMENT_COLUMNS = {
    "comment": "comment",
    "comment_mini": "comment_mini"
}

STOPWORDS = set([
    "the", "a", "an", "and", "or", "to", "of", "is", "are", "was", "were",
    "this", "that", "it", "as", "with", "for", "on", "at", "by",
    "image", "shows", "show", "road", "vehicle", "car",
    "driving", "lane", "ego", "current"
])

# ===============================
# 前処理
# ===============================
def clean_text(text):
    if not isinstance(text, str):
        return []
    text = text.lower()
    text = re.sub(r"[^a-z\s]", "", text)
    words = text.split()
    return [w for w in words if w not in STOPWORDS and len(w) > 2]

# ===============================
# データ読み込み
# ===============================
df = pd.read_csv(CSV_PATH)

# スコアは数値化
df["overall_num"] = pd.to_numeric(df["overall"], errors="coerce")
df = df[df["overall_num"].notna()].copy()

# コメント列を文字列化
for col in COMMENT_COLUMNS.values():
    df[col] = df[col].astype(str)

# maneuver 正規化（保険）
df["llava_maneuver"] = df["llava_maneuver"].str.strip().str.lower()

# ===============================
# WordCloud 生成
# ===============================
for score_tag in ["high", "low"]:
    if score_tag == "high":
        score_mask = df["overall_num"] >= 4
    else:
        score_mask = df["overall_num"] <= 2

    for comment_tag, col_name in COMMENT_COLUMNS.items():
        for action in ACTIONS:

            subset = df[
                (df["llava_maneuver"] == action) &
                score_mask
            ]

            words = []
            for txt in subset[col_name]:
                words.extend(clean_text(txt))

            if not words:
                print(f"[WARN] No words: {score_tag}/{comment_tag}/{action}")
                continue

            wc = WordCloud(
                width=1600,
                height=1200,
                background_color="white",
                max_words=100,
                collocations=False
            ).generate_from_frequencies(Counter(words))

            out_path = os.path.join(
                OUT_DIR,
                f"wordcloud_{score_tag}_{comment_tag}_{action}.png"
            )
            wc.to_file(out_path)

            plt.figure(figsize=(6, 4))
            plt.imshow(wc, interpolation="bilinear")
            plt.axis("off")
            plt.title(f"{score_tag} | {comment_tag} | {action}")
            plt.show()

            print(f"Saved: {out_path}")

print("=== ALL DONE (CORRECT SEMANTICS) ===")
