import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import chi2_contingency

# =========================
# 0. 設定
# =========================
csv_path = r"D:\merged_action4_paper_ready_with_json.csv"
output_dir = r"D:\analysis_outputs_action4_scores_rq3"
os.makedirs(output_dir, exist_ok=True)

SAVE_FIGS = True
SHOW_FIGS = True

# =========================
# 1. CSV 読み込み
# =========================
df = pd.read_csv(csv_path)

# =========================
# 2. 定義
# =========================
factors = [
    "env_rural", "env_city", "env_sunny", "env_rainy",
    "lane_change_detected_llava",
    "brake_final",
    "ts_final"
]

metrics = ["situation", "advice", "safety", "overall"]

score_cols = {
    "nano": {
        "situation": "situation_accuracy",
        "advice": "advice_appropriateness",
        "safety": "safety_risk_calibration",
        "overall": "overall"
    },
    "mini": {
        "situation": "situation_accuracy_mini",
        "advice": "advice_appropriateness_mini",
        "safety": "safety_risk_calibration_mini",
        "overall": "overall_mini"
    }
}

use_cols = factors + sum([list(v.values()) for v in score_cols.values()], [])
df = df[use_cols].copy()

# =========================
# 3. 前処理
# =========================
for c in factors:
    df[c] = df[c].astype(str).str.strip().str.lower()

def make_binary_label(x):
    if pd.isna(x):
        return np.nan
    x = float(x)
    if x >= 4.0:
        return "high"
    elif x <= 2.0:
        return "low"
    else:
        return np.nan

for model in score_cols:
    for metric, col in score_cols[model].items():
        df[f"{model}_{metric}_label"] = df[col].apply(make_binary_label)

# =========================
# 4. Cramér’s V 計算
# =========================
def cramers_v(df_in, factor, label_col):
    tmp = df_in[[factor, label_col]].dropna()
    ct = pd.crosstab(tmp[factor], tmp[label_col])

    if ct.shape[0] < 2 or ct.shape[1] < 2:
        return np.nan

    chi2, _, _, _ = chi2_contingency(ct)
    n = ct.values.sum()
    r, k = ct.shape
    return np.sqrt(chi2 / (n * min(r - 1, k - 1)))

# =========================
# 5. 可視化（最小構成 heatmap）
# =========================
for model in ["nano", "mini"]:
    V = np.zeros((len(factors), len(metrics)))

    for i, f in enumerate(factors):
        for j, m in enumerate(metrics):
            label_col = f"{model}_{m}_label"
            V[i, j] = cramers_v(df, f, label_col)

    # ★ サイズを少しだけコンパクトに
    fig, ax = plt.subplots(figsize=(6.2, 4.2))

    im = ax.imshow(V, cmap="YlOrBr", vmin=0, vmax=np.nanmax(V))

    # 軸目盛（説明ラベルは削除）
    ax.set_xticks(range(len(metrics)))
    ax.set_xticklabels(metrics, fontsize=9)
    ax.set_yticks(range(len(factors)))
    ax.set_yticklabels(factors, fontsize=9)

    # ★ タイトルは最小限
    ax.set_title(model.upper(), fontsize=11, pad=6)

    ax.set_xlabel("")
    ax.set_ylabel("")

    # 数値表示（小さめ）
    for i in range(len(factors)):
        for j in range(len(metrics)):
            ax.text(j, i, f"{V[i, j]:.2f}",
                    ha="center", va="center", fontsize=8)

    # ★ カラーバーも省スペース化
    cbar = plt.colorbar(im, ax=ax, fraction=0.045, pad=0.04)
    cbar.set_label("V", fontsize=9)
    cbar.ax.tick_params(labelsize=8)

    plt.tight_layout(pad=0.5)

    if SAVE_FIGS:
        out_path = os.path.join(output_dir, f"rq3_cramersv_matrix_{model}.png")
        plt.savefig(out_path, dpi=300)
        print(f"[Saved] {out_path}")

    if SHOW_FIGS:
        plt.show()
    else:
        plt.close()

print("RQ3 heatmap visualization done.")
