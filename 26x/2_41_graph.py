# =========================
# COEFFICIENT PLOT
# =========================
# =========================
# COEFFICIENT PLOT
# =========================
import pandas as pd   # ← これを追加
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

# 定数項は除外（見づらいので）
coef_df = coef_df.drop(index="const", errors="ignore")

print("\nCoefficient table:")
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
plt.title("Factors associated with high-quality driving advice (overall ≥ 4)")

plt.grid(axis="x", alpha=0.3)
plt.tight_layout()

# 保存（論文・修論用）
plt.savefig("figure_logit_coefficients_high_quality.png", dpi=300)
plt.show()
