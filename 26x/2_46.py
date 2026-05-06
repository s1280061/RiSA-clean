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
# QUALITY SPLIT (GPT judge)
# =========================
df["low_quality"]  = (df["overall"] <= 2)
df["high_quality"] = (df["overall"] >= 4)

# =========================
# TURN SIGNAL (no confidence available)
# =========================
df["ts_final"] = (
    df["ts_final"]
    .astype(str)
    .str.lower()
)

# left / right / off のみ残す
valid_ts = ["left", "right", "off"]
df = df[df["ts_final"].isin(valid_ts)]

# signal_on（二値化）
df["signal_on"] = 0
df.loc[df["ts_final"].isin(["left", "right"]), "signal_on"] = 1

print("Signal ON distribution:")
print(df["signal_on"].value_counts(), "\n")

# =========================
# NORMALIZE LLaVA maneuver (4-class)
# =========================
m = df["llava_maneuver"].astype(str).str.lower()

df["llava_m4"] = np.nan
df.loc[m.str.contains("keep"),  "llava_m4"] = "keep_speed"
df.loc[m.str.contains("decel"), "llava_m4"] = "decelerate"
df.loc[m.str.contains("left"),  "llava_m4"] = "change_lane_left"
df.loc[m.str.contains("right"), "llava_m4"] = "change_lane_right"

df = df.dropna(subset=["llava_m4"])

# =========================
# ANALYSIS FUNCTION
# =========================
def analyze_turn_signal(sub_df, tag):
    print(f"\n=== TURN SIGNAL vs MANEUVER ({tag}) ===")

    # Crosstab (raw)
    ct_raw = pd.crosstab(
        sub_df["signal_on"],
        sub_df["llava_m4"]
    )
    print("Raw counts:")
    print(ct_raw, "\n")

    # Chi-square
    chi2, p, dof, _ = chi2_contingency(ct_raw)
    n = ct_raw.values.sum()
    r, k = ct_raw.shape
    cramers_v = np.sqrt((chi2 / n) / min(r - 1, k - 1))

    print(f"Chi-square = {chi2:.3f}")
    print(f"p-value    = {p:.6f}")
    print(f"Cramer's V = {cramers_v:.3f}")
    print("(0.1=small, 0.3=medium, 0.5=large)\n")

    # Row-normalized heatmap
    ct_norm = pd.crosstab(
        sub_df["signal_on"],
        sub_df["llava_m4"],
        normalize="index"
    )

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
        "Turn signal (on/off) vs LLaVA maneuver\n"
        f"({tag})"
    )
    plt.ylabel("signal_on (1=on, 0=off)")
    plt.xlabel("llava_maneuver")
    plt.tight_layout()
    plt.savefig(
        f"heatmap_turn_signal_vs_maneuver_{tag}.png",
        dpi=300
    )
    plt.show()

# =========================
# RUN ANALYSIS
# =========================
analyze_turn_signal(df, "all")

df_low  = df[df["low_quality"]]
df_high = df[df["high_quality"]]

print("Low-quality samples :", len(df_low))
print("High-quality samples:", len(df_high))

analyze_turn_signal(df_low,  "low_quality")
analyze_turn_signal(df_high, "high_quality")

print("=== TURN SIGNAL ANALYSIS DONE ===")
