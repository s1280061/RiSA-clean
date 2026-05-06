import pandas as pd
import numpy as np
from scipy.stats import chi2_contingency

# =========================
# 表示設定（省略を完全に無効化）
# =========================
pd.set_option("display.max_rows", None)
pd.set_option("display.max_columns", None)
pd.set_option("display.width", 200)
pd.set_option("display.max_colwidth", None)

# =========================
# 1. CSV 読み込み
# =========================
csv_path = r"D:\merged_action4_paper_ready_with_json.csv"
df = pd.read_csv(csv_path)

# =========================
# 2. 必要列のみ抽出
# =========================
cols = ["lane_change_detected_llava", "llava_maneuver"]
df = df[cols]

# =========================
# 3. 前処理
# =========================
for c in cols:
    df[c] = (
        df[c]
        .astype(str)
        .str.strip()
        .str.lower()
    )

# =========================
# 4. ラベル確認
# =========================
print("=== Unique values (after cleaning) ===")
print("\nlane_change_detected_llava:")
print(df["lane_change_detected_llava"].value_counts())

print("\nllava_maneuver:")
print(df["llava_maneuver"].value_counts())

# =========================
# 5. クラス限定
# =========================
valid_lane = {"keep", "left", "right"}
valid_maneuver = {
    "keep_speed",
    "decelerate",
    "change_lane_left",
    "change_lane_right"
}

df = df[
    df["lane_change_detected_llava"].isin(valid_lane) &
    df["llava_maneuver"].isin(valid_maneuver)
]

# =========================
# 6. クロス表（Counts）
# =========================
ct = pd.crosstab(
    df["lane_change_detected_llava"],
    df["llava_maneuver"]
)

print("\n=== Cross Table (Counts) ===")
print(ct)
print("\nRow sums:")
print(ct.sum(axis=1))
print("\nColumn sums:")
print(ct.sum(axis=0))

# =========================
# 7. 条件付き確率
# =========================
ct_prob = ct.div(ct.sum(axis=1), axis=0)

print("\n=== Conditional Probability P(maneuver | lane_change) ===")
print(ct_prob.round(3))

# =========================
# 8. カイ二乗検定
# =========================
chi2, p, dof, expected = chi2_contingency(ct)

expected_df = pd.DataFrame(
    expected,
    index=ct.index,
    columns=ct.columns
)

print("\n=== Expected Frequencies (for Chi-square) ===")
print(expected_df.round(2))

print("\n=== Chi-square Test ===")
print(f"Chi-square statistic = {chi2:.3f}")
print(f"Degrees of freedom  = {dof}")
print(f"p-value             = {p:.4e}")

# =========================
# 9. Cramér's V
# =========================
n = ct.values.sum()
r, k = ct.shape
cramers_v = np.sqrt(chi2 / (n * (min(r - 1, k - 1))))

print("\n=== Cramér's V ===")
print(f"Cramér's V = {cramers_v:.3f}")
# =========================
# 10. 条件付き確率ヒートマップ（論文用）
# =========================
import matplotlib.pyplot as plt
import seaborn as sns

# 並び順を明示（論文で意味が通る順）
row_order = ["keep", "left", "right"]
col_order = [
    "keep_speed",
    "decelerate",
    "change_lane_left",
    "change_lane_right"
]

ct_prob_plot = ct_prob.loc[row_order, col_order]

plt.figure(figsize=(8, 4))
sns.heatmap(
    ct_prob_plot,
    annot=True,
    fmt=".2f",
    cmap="Blues",
    vmin=0,
    vmax=1,
    linewidths=0.5,
    cbar_kws={"label": "Conditional Probability"}
)

plt.xlabel("LLaVA-generated Maneuver")
plt.ylabel("Lane-change Detected by LLaVA")
plt.title("Conditional Probability of Maneuver Given Lane-change Detection")

plt.tight_layout()

# 論文用に保存（PNG & PDF 両方）
plt.savefig("conditional_probability_heatmap.png", dpi=300)
plt.savefig("conditional_probability_heatmap.pdf")

plt.show()
