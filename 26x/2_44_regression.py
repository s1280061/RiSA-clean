import pandas as pd
import statsmodels.api as sm
import numpy as np
import matplotlib.pyplot as plt

# =========================
# LOAD
# =========================
df = pd.read_csv(
    r"D:\merged_action4_paper_ready_with_json.csv"
)

# =========================
# TARGET
# =========================
df["low_quality"] = (df["overall"] <= 2).astype(int)

# =========================
# ENV (YES/NO → 1/0)
# =========================
for c in ["env_rural", "env_city", "env_rainy"]:
    df[c] = df[c].map({"YES": 1, "NO": 0})

# =========================
# NEW 1: 車両台数（context）
# =========================
df["detected_vehicle_count"] = pd.to_numeric(
    df["detected_vehicle_count"],
    errors="coerce"
)

# =========================
# NEW 2: LLaVA lane-change 主張（2値）
# =========================
m_lane = df["lane_change_detected_llava"].astype(str).str.lower()

df["llava_lane_change"] = 0
df.loc[m_lane.isin(["left", "right"]), "llava_lane_change"] = 1

print("LLaVA lane-change distribution:")
print(df["llava_lane_change"].value_counts())

# =========================
# LLaVA maneuver（4値）
# =========================
m = df["llava_maneuver"].astype(str).str.lower()

df["llava_m4"] = "keep_speed"
df.loc[m.str.contains("decel"), "llava_m4"] = "decelerate"
df.loc[m.str.contains("left"),  "llava_m4"] = "change_left"
df.loc[m.str.contains("right"), "llava_m4"] = "change_right"

print("\nLLaVA maneuver distribution:")
print(df["llava_m4"].value_counts())

# =========================
# SUBSET
# =========================
use_cols = [
    "low_quality",
    "ego_speed_kmh",
    "detected_vehicle_count",   # ←追加
    "llava_lane_change",        # ←追加
    "env_rural",
    "env_city",
    "env_rainy",
    "llava_m4",
]

df_sub = df[use_cols].dropna()

print("\nAnalysis rows:", df_sub.shape)
print("Low-quality rate:", df_sub["low_quality"].mean())

# =========================
# DUMMY化（基準：keep_speed）
# =========================
X = pd.get_dummies(
    df_sub.drop(columns=["low_quality"]),
    columns=["llava_m4"],
    drop_first=True
)

X = sm.add_constant(X)
X = X.apply(pd.to_numeric, errors="coerce").astype(float)
y = df_sub["low_quality"].astype(int)

# =========================
# MODEL（正則化付き Logit）
# =========================
model = sm.Logit(y, X)
result = model.fit_regularized(
    alpha=1.0,
    maxiter=300
)

print(result.summary())

# =========================
# COEFFICIENT PLOT
# =========================
coef_df = pd.DataFrame({
    "coef": result.params,
    "std_err": result.bse
})

coef_df["ci_low"]  = coef_df["coef"] - 1.96 * coef_df["std_err"]
coef_df["ci_high"] = coef_df["coef"] + 1.96 * coef_df["std_err"]

coef_df = coef_df.drop(index="const", errors="ignore")

coef_df["abs_coef"] = coef_df["coef"].abs()
coef_df = coef_df.sort_values("abs_coef", ascending=False)

print("\nCoefficient table:")
print(coef_df)

# =========================
# PLOT
# =========================
plt.figure(figsize=(7.5, 5))

y_pos = np.arange(len(coef_df))

plt.errorbar(
    coef_df["coef"],
    y_pos,
    xerr=[
        coef_df["coef"] - coef_df["ci_low"],
        coef_df["ci_high"] - coef_df["coef"]
    ],
    fmt="o",
    ecolor="gray",
    capsize=4
)

plt.axvline(0, color="red", linestyle="--", linewidth=1)
plt.yticks(y_pos, coef_df.index)

plt.xlabel("Logistic regression coefficient")
plt.title(
    "Factors associated with low-quality driving advice (overall ≤ 2)\n"
    "including vehicle count and LLaVA lane-change assertion"
)

plt.grid(axis="x", alpha=0.3)
plt.tight_layout()

plt.savefig(
    "figure_logit_coefficients_low_quality_with_vehicle_and_lanechange.png",
    dpi=300
)
plt.show()
# =========================
# TARGET (HIGH QUALITY)
# =========================
df["high_quality"] = (df["overall"] >= 4).astype(int)

print("\nHigh-quality rate:", df["high_quality"].mean())
use_cols_hq = [
    "high_quality",
    "ego_speed_kmh",
    "detected_vehicle_count",
    "llava_lane_change",
    "env_rural",
    "env_city",
    "env_rainy",
    "llava_m4",
]

df_hq = df[use_cols_hq].dropna()

print("High-quality analysis rows:", df_hq.shape)
print("High-quality rate (subset):", df_hq["high_quality"].mean())
X_hq = pd.get_dummies(
    df_hq.drop(columns=["high_quality"]),
    columns=["llava_m4"],
    drop_first=True
)

X_hq = sm.add_constant(X_hq)
X_hq = X_hq.apply(pd.to_numeric, errors="coerce").astype(float)
y_hq = df_hq["high_quality"].astype(int)
model_hq = sm.Logit(y_hq, X_hq)
result_hq = model_hq.fit_regularized(
    alpha=1.0,
    maxiter=300
)

print(result_hq.summary())
coef_hq = pd.DataFrame({
    "coef": result_hq.params,
    "std_err": result_hq.bse
})

coef_hq["ci_low"]  = coef_hq["coef"] - 1.96 * coef_hq["std_err"]
coef_hq["ci_high"] = coef_hq["coef"] + 1.96 * coef_hq["std_err"]

coef_hq = coef_hq.drop(index="const", errors="ignore")

coef_hq["abs_coef"] = coef_hq["coef"].abs()
coef_hq = coef_hq.sort_values("abs_coef", ascending=False)

print("\nHigh-quality coefficient table:")
print(coef_hq)
plt.figure(figsize=(7.5, 5))

y_pos = np.arange(len(coef_hq))

plt.errorbar(
    coef_hq["coef"],
    y_pos,
    xerr=[
        coef_hq["coef"] - coef_hq["ci_low"],
        coef_hq["ci_high"] - coef_hq["coef"]
    ],
    fmt="o",
    ecolor="gray",
    capsize=4
)

plt.axvline(0, color="red", linestyle="--", linewidth=1)
plt.yticks(y_pos, coef_hq.index)

plt.xlabel("Logistic regression coefficient")
plt.title(
    "Factors associated with high-quality driving advice (overall ≥ 4)\n"
    "including vehicle count and LLaVA lane-change assertion"
)

plt.grid(axis="x", alpha=0.3)
plt.tight_layout()

plt.savefig(
    "figure_logit_coefficients_high_quality_with_vehicle_and_lanechange.png",
    dpi=300
)
plt.show()
