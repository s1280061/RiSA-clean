import pandas as pd
import numpy as np
from scipy.stats import chi2_contingency

# =========================
# 1. CSV 読み込み
# =========================
csv_path = r"D:\merged_action4_paper_ready_with_json.csv"
df = pd.read_csv(csv_path)

# =========================
# 2. 必要列
# =========================
cols = [
    "ts_final",              # other vehicle turn signal
    "brake_final",           # other vehicle brake
    "lane_change_detected_llava",
    "llava_maneuver"
]
df = df[cols]

# =========================
# 3. 前処理（必須）
# =========================
for c in cols:
    df[c] = (
        df[c]
        .astype(str)
        .str.strip()
        .str.lower()
    )

# =========================
# 4. クラス定義
# =========================
valid_ts = {"off", "left", "right"}
valid_brake = {"go", "brake"}
valid_lane = {"keep", "left", "right"}
valid_maneuver = {
    "keep_speed",
    "decelerate",
    "change_lane_left",
    "change_lane_right"
}

df = df[
    df["ts_final"].isin(valid_ts) &
    df["brake_final"].isin(valid_brake) &
    df["lane_change_detected_llava"].isin(valid_lane) &
    df["llava_maneuver"].isin(valid_maneuver)
]

# ======================================================
# Stage A: ts_final / brake_final → lane_change_detected
# ======================================================
print("\n================ Stage A =================")
print("ts_final × lane_change_detected_llava")

ct_ts_lane = pd.crosstab(
    df["ts_final"],
    df["lane_change_detected_llava"]
)
print(ct_ts_lane)

ct_ts_lane_prob = ct_ts_lane.div(ct_ts_lane.sum(axis=1), axis=0)
print("\nConditional Probability")
print(ct_ts_lane_prob.round(3))

chi2, p, dof, _ = chi2_contingency(ct_ts_lane)
n = ct_ts_lane.values.sum()
r, k = ct_ts_lane.shape
v = np.sqrt(chi2 / (n * (min(r - 1, k - 1))))
print(f"\nChi2={chi2:.3f}, p={p:.2e}, Cramer's V={v:.3f}")

# ------------------------------------------------------
print("\nbrake_final × lane_change_detected_llava")

ct_brake_lane = pd.crosstab(
    df["brake_final"],
    df["lane_change_detected_llava"]
)
print(ct_brake_lane)

ct_brake_lane_prob = ct_brake_lane.div(ct_brake_lane.sum(axis=1), axis=0)
print("\nConditional Probability")
print(ct_brake_lane_prob.round(3))

chi2, p, dof, _ = chi2_contingency(ct_brake_lane)
n = ct_brake_lane.values.sum()
r, k = ct_brake_lane.shape
v = np.sqrt(chi2 / (n * (min(r - 1, k - 1))))
print(f"\nChi2={chi2:.3f}, p={p:.2e}, Cramer's V={v:.3f}")

# ======================================================
# Stage B: ts_final / brake_final → llava_maneuver
# ======================================================
print("\n================ Stage B =================")
print("ts_final × llava_maneuver")

ct_ts_man = pd.crosstab(
    df["ts_final"],
    df["llava_maneuver"]
)
print(ct_ts_man)

ct_ts_man_prob = ct_ts_man.div(ct_ts_man.sum(axis=1), axis=0)
print("\nConditional Probability")
print(ct_ts_man_prob.round(3))

chi2, p, dof, _ = chi2_contingency(ct_ts_man)
n = ct_ts_man.values.sum()
r, k = ct_ts_man.shape
v = np.sqrt(chi2 / (n * (min(r - 1, k - 1))))
print(f"\nChi2={chi2:.3f}, p={p:.2e}, Cramer's V={v:.3f}")

# ------------------------------------------------------
print("\nbrake_final × llava_maneuver")

ct_brake_man = pd.crosstab(
    df["brake_final"],
    df["llava_maneuver"]
)
print(ct_brake_man)

ct_brake_man_prob = ct_brake_man.div(ct_brake_man.sum(axis=1), axis=0)
print("\nConditional Probability")
print(ct_brake_man_prob.round(3))

chi2, p, dof, _ = chi2_contingency(ct_brake_man)
n = ct_brake_man.values.sum()
r, k = ct_brake_man.shape
v = np.sqrt(chi2 / (n * (min(r - 1, k - 1))))
print(f"\nChi2={chi2:.3f}, p={p:.2e}, Cramer's V={v:.3f}")

# ======================================================
# Heatmaps (Paper-ready)
# ======================================================
import matplotlib.pyplot as plt
import seaborn as sns

plt.rcParams["pdf.fonttype"] = 42
plt.rcParams["ps.fonttype"] = 42

ts_order = ["off", "left", "right"]
lane_order = ["keep", "left", "right"]
brake_order = ["go", "brake"]

maneuver_order = [
    "keep_speed",
    "decelerate",
    "change_lane_left",
    "change_lane_right"
]

# ★ 短縮ラベル（方法②）
maneuver_labels_short = [
    "Keep",
    "Decel.",
    "LC-Left",
    "LC-Right"
]

# =========================
# Stage A-1
# =========================
plt.figure(figsize=(6, 4))
sns.heatmap(
    ct_ts_lane_prob.loc[ts_order, lane_order],
    annot=True,
    fmt=".2f",
    cmap="Blues",
    vmin=0,
    vmax=1,
    linewidths=0.5,
    cbar_kws={"label": "P(lane change | turn signal)"}
)
plt.xlabel("Lane-change detected by LLaVA")
plt.ylabel("Other vehicle turn signal")
plt.title("Stage A: Turn Signal → Lane-change Detection")
plt.tight_layout()
plt.savefig("stageA_ts_lane_heatmap.pdf")
plt.savefig("stageA_ts_lane_heatmap.png", dpi=300)
plt.show()

# =========================
# Stage A-2
# =========================
plt.figure(figsize=(6, 3))
sns.heatmap(
    ct_brake_lane_prob.loc[brake_order, lane_order],
    annot=True,
    fmt=".2f",
    cmap="Blues",
    vmin=0,
    vmax=1,
    linewidths=0.5,
    cbar_kws={"label": "P(lane change | brake)"}
)
plt.xlabel("Lane-change detected by LLaVA")
plt.ylabel("Other vehicle brake")
plt.title("Stage A: Brake → Lane-change Detection")
plt.tight_layout()
plt.savefig("stageA_brake_lane_heatmap.pdf")
plt.savefig("stageA_brake_lane_heatmap.png", dpi=300)
plt.show()

# =========================
# Stage B-1（短縮ラベル適用）
# =========================
plt.figure(figsize=(7, 4))
ax = sns.heatmap(
    ct_ts_man_prob.loc[ts_order, maneuver_order],
    annot=True,
    fmt=".2f",
    cmap="Blues",
    vmin=0,
    vmax=1,
    linewidths=0.5,
    cbar_kws={"label": "P(maneuver | turn signal)"}
)
ax.set_xticklabels(maneuver_labels_short, rotation=0)
plt.xlabel("LLaVA-generated maneuver")
plt.ylabel("Other vehicle turn signal")
plt.title("Stage B: Turn Signal → Maneuver Generation")
plt.tight_layout()
plt.savefig("stageB_ts_maneuver_heatmap.pdf")
plt.savefig("stageB_ts_maneuver_heatmap.png", dpi=300)
plt.show()

# =========================
# Stage B-2（短縮ラベル適用）
# =========================
plt.figure(figsize=(7, 3))
ax = sns.heatmap(
    ct_brake_man_prob.loc[brake_order, maneuver_order],
    annot=True,
    fmt=".2f",
    cmap="Blues",
    vmin=0,
    vmax=1,
    linewidths=0.5,
    cbar_kws={"label": "P(maneuver | brake)"}
)
ax.set_xticklabels(maneuver_labels_short, rotation=0)
plt.xlabel("LLaVA-generated maneuver")
plt.ylabel("Other vehicle brake")
plt.title("Stage B: Brake → Maneuver Generation")
plt.tight_layout()
plt.savefig("stageB_brake_maneuver_heatmap.pdf")
plt.savefig("stageB_brake_maneuver_heatmap.png", dpi=300)
plt.show()
