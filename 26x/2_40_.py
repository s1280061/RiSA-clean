import pandas as pd
import statsmodels.api as sm

# =========================
# LOAD
# =========================
df = pd.read_csv(r"D:\merged_all_models.csv")

# =========================
# TARGET（高評価）
# =========================
df["high_quality"] = (df["overall"] >= 4).astype(int)

# =========================
# ENV (YES/NO → 1/0)
# =========================
for c in ["env_rural", "env_city", "env_rainy"]:
    df[c] = df[c].map({"YES": 1, "NO": 0})

# =========================
# LLaVA maneuver（left/right分離）
# =========================
m = df["llava_maneuver"].astype(str).str.lower()

df["llava_m4"] = "keep_speed"
df.loc[m.str.contains("decel"), "llava_m4"] = "decelerate"
df.loc[m.str.contains("left"),  "llava_m4"] = "change_left"
df.loc[m.str.contains("right"), "llava_m4"] = "change_right"

print("LLaVA maneuver distribution:")
print(df["llava_m4"].value_counts())

# =========================
# SUBSET
# =========================
use_cols = [
    "high_quality",
    "ego_speed_kmh",
    "env_rural",
    "env_city",
    "env_rainy",
    "llava_m4",
]

df_sub = df[use_cols].dropna()

print("Analysis rows:", df_sub.shape)
print("High-quality rate:", df_sub["high_quality"].mean())

# =========================
# DUMMY化（基準：keep_speed）
# =========================
X = pd.get_dummies(
    df_sub.drop(columns=["high_quality"]),
    columns=["llava_m4"],
    drop_first=True   # keep_speed が基準
)

X = sm.add_constant(X)
X = X.apply(pd.to_numeric, errors="coerce").astype(float)
y = df_sub["high_quality"].astype(int)

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
# COEFFICIENT PLOT（重要度順）
# =========================
import numpy as np
import matplotlib.pyplot as plt

# ---- 回帰係数を DataFrame にまとめる ----
coef_df = pd.DataFrame({
    "coef": result.params,
    "std_err": result.bse
})

# 95%信頼区間
coef_df["ci_low"]  = coef_df["coef"] - 1.96 * coef_df["std_err"]
coef_df["ci_high"] = coef_df["coef"] + 1.96 * coef_df["std_err"]

# 定数項は除外
coef_df = coef_df.drop(index="const", errors="ignore")

# =========================
# 重要度（|coef|）で並び替え
# =========================
coef_df["abs_coef"] = coef_df["coef"].abs()
coef_df = coef_df.sort_values("abs_coef", ascending=True)

print("\nCoefficient table (sorted by importance |coef|):")
print(coef_df)

# ---- プロット ----
plt.figure(figsize=(7, 5))

y_pos = np.arange(len(coef_df))

plt.errorbar(
    coef_df["coef"],
    y_pos,
    xerr=[
        coef_df["coef"] - coef_df["ci_low"],
        coef_df["ci_high"] - coef_df["coef"]
    ],
    fmt="o",
    color="black",
    ecolor="gray",
    capsize=4
)

# 0ライン（効果なし）
plt.axvline(0, color="red", linestyle="--", linewidth=1)

plt.yticks(y_pos, coef_df.index)
plt.xlabel("Logistic regression coefficient")
plt.title("Factors associated with high-quality driving advice (overall ≥ 4)\n(sorted by |coefficient|)")

plt.grid(axis="x", alpha=0.3)
plt.tight_layout()

# 保存（論文・修論用）
plt.savefig("figure_logit_coefficients_high_quality_sorted.png", dpi=300)
plt.show()