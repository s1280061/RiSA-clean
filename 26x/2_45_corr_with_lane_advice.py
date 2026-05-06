import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import chi2_contingency

# =========================
# LOAD
# =========================
df = pd.read_csv(
    r"D:\merged_action4_paper_ready_with_json.csv"
)

# =========================
# NORMALIZE lane_change_detected_llava
# keep / left / right のみ
# =========================
df["lane_change_detected_llava"] = (
    df["lane_change_detected_llava"]
    .astype(str)
    .str.lower()
)

valid_lane = ["keep", "left", "right"]
df = df[df["lane_change_detected_llava"].isin(valid_lane)]

# =========================
# NORMALIZE llava_maneuver (4-class)
# =========================
m = df["llava_maneuver"].astype(str).str.lower()

df["llava_m4"] = np.nan
df.loc[m.str.contains("keep"),  "llava_m4"] = "keep_speed"
df.loc[m.str.contains("decel"), "llava_m4"] = "decelerate"
df.loc[m.str.contains("left"),  "llava_m4"] = "change_lane_left"
df.loc[m.str.contains("right"), "llava_m4"] = "change_lane_right"

df = df.dropna(subset=["llava_m4"])

print("lane_change_detected_llava distribution:")
print(df["lane_change_detected_llava"].value_counts(), "\n")

print("llava_m4 distribution:")
print(df["llava_m4"].value_counts(), "\n")

# =========================
# FUNCTION: Crosstab + stats + heatmap
# =========================
def analyze_relation(sub_df, tag):
    print(f"\n=== {tag.upper()} ===")

    # --- raw counts ---
    ct_raw = pd.crosstab(
        sub_df["lane_change_detected_llava"],
        sub_df["llava_m4"]
    )

    print("Raw counts:")
    print(ct_raw, "\n")

    # --- chi-square ---
    chi2, p, dof, expected = chi2_contingency(ct_raw)
    n = ct_raw.values.sum()
    r, k = ct_raw.shape
    cramers_v = np.sqrt((chi2 / n) / min(r - 1, k - 1))

    print(f"Chi-square = {chi2:.3f}")
    print(f"p-value    = {p:.6f}")
    print(f"Cramer's V = {cramers_v:.3f}")
    print("(0.1=small, 0.3=medium, 0.5=large)\n")

    # --- row-normalized ---
    ct_norm = pd.crosstab(
        sub_df["lane_change_detected_llava"],
        sub_df["llava_m4"],
        normalize="index"
    )

    # --- heatmap ---
    plt.figure(figsize=(7, 4))
    sns.heatmap(
        ct_norm,
        annot=True,
        fmt=".2f",
        cmap="Blues",
        vmin=0,
        vmax=1
    )
    plt.title(
        "Lane-change perception vs LLaVA maneuver\n"
        f"({tag})"
    )
    plt.ylabel("lane_change_detected_llava")
    plt.xlabel("llava_maneuver")
    plt.tight_layout()
    plt.savefig(
        f"heatmap_lane_change_vs_maneuver_{tag}.png",
        dpi=300
    )
    plt.show()

# =========================
# ALL DATA
# =========================
analyze_relation(df, "all")

# =========================
# LOW / HIGH QUALITY SPLIT
# =========================
df["low_quality"]  = (df["overall"] <= 2)
df["high_quality"] = (df["overall"] >= 4)

df_low  = df[df["low_quality"]]
df_high = df[df["high_quality"]]

print("Low-quality samples :", len(df_low))
print("High-quality samples:", len(df_high))

analyze_relation(df_low,  "low_quality")
analyze_relation(df_high, "high_quality")

print("=== ALL ANALYSIS DONE ===")
