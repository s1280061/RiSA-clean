import pandas as pd
import numpy as np
from scipy.stats import chi2_contingency

# ===============================
# 設定
# ===============================
INPUT_CSV = r"C:\Users\s1280\Desktop\merged_action4_paper_ready_with_json.csv"
OUT_TXT = r"C:\Users\s1280\Desktop\rq2_recognition_vs_quality_ALL_report_strict3class.txt"

LLAVA_COL = "lane_change_detected_llava"
GT_COL = "lane_change_detected_label"

SCORE_COLS = [
    "situation_accuracy",
    "advice_appropriateness",
    "safety_risk_calibration",
    "overall",
    "situation_accuracy_mini",
    "advice_appropriateness_mini",
    "safety_risk_calibration_mini",
    "overall_mini",
]

HIGH_TH = 4
LOW_TH = 2

# ===============================
# 正規化関数（keep/left/right に揃える）
# ===============================
def normalize_llava(x):
    if pd.isna(x):
        return None
    x = str(x).lower().strip()
    if x in ["keep", "keep_lane"]:
        return "keep"
    if x in ["left", "right"]:
        return x
    return None

def normalize_gt(x):
    if pd.isna(x):
        return None
    x = str(x).lower().strip()
    if x == "keep_lane":
        return "keep"
    if x == "lane_change_left":
        return "left"
    if x == "lane_change_right":
        return "right"
    return None

# ===============================
# 効果量（Cramér's V）
# ===============================
def cramers_v(ct: pd.DataFrame) -> float:
    chi2, _, _, _ = chi2_contingency(ct)
    n = ct.to_numpy().sum()
    r, k = ct.shape
    denom = n * (min(r - 1, k - 1))
    return np.sqrt(chi2 / denom) if denom > 0 else np.nan

# ===============================
# Load & normalize
# ===============================
df = pd.read_csv(INPUT_CSV)

df["llava_norm"] = df[LLAVA_COL].apply(normalize_llava)
df["gt_norm"] = df[GT_COL].apply(normalize_gt)

df_valid = df.dropna(subset=["llava_norm", "gt_norm"]).copy()

# ★ここが修正点：3クラス完全一致で正誤判定
# keep/left/right が一致したときのみ correct (=1)
df_valid["correct_recognition_strict"] = (
    df_valid["llava_norm"] == df_valid["gt_norm"]
).astype(int)

# ===============================
# Main loop（8指標）
# ===============================
lines = []
lines.append("=== RQ2: Recognition correctness (STRICT 3-class match) vs Advice Quality ===")
lines.append("Recognition correctness: exact match on {keep, left, right}\n")
lines.append("Nano & Mini | 3 sub-scores + Overall\n")

for SCORE_COL in SCORE_COLS:
    if SCORE_COL not in df_valid.columns:
        continue

    df_valid[SCORE_COL] = pd.to_numeric(df_valid[SCORE_COL], errors="coerce")
    df_s = df_valid.dropna(subset=[SCORE_COL]).copy()

    # HQ / LQ（3は除外したいならここで除外も可能）
    df_s["high_quality"] = (df_s[SCORE_COL] >= HIGH_TH).astype(int)
    df_s["low_quality"] = (df_s[SCORE_COL] <= LOW_TH).astype(int)

    # contingency tables: (correct vs HQ/LQ)
    ct_high = pd.crosstab(df_s["correct_recognition_strict"], df_s["high_quality"]).reindex(
        index=[0, 1], columns=[0, 1], fill_value=0
    )
    ct_low = pd.crosstab(df_s["correct_recognition_strict"], df_s["low_quality"]).reindex(
        index=[0, 1], columns=[0, 1], fill_value=0
    )

    chi2_h, p_h, _, _ = chi2_contingency(ct_high)
    chi2_l, p_l, _, _ = chi2_contingency(ct_low)

    v_h = cramers_v(ct_high)
    v_l = cramers_v(ct_low)

    mean_scores = df_s.groupby("correct_recognition_strict")[SCORE_COL].mean()
    N = len(df_s)

    lines.append(f"--- {SCORE_COL} ---")
    lines.append(f"N = {N}")
    lines.append("Mean score by strict recognition correctness:")
    lines.append(mean_scores.to_string())
    lines.append(f"HighQuality (>=4): chi2={chi2_h:.3f}, p={p_h:.4g}, V={v_h:.3f}")
    lines.append(f"LowQuality  (<=2): chi2={chi2_l:.3f}, p={p_l:.4g}, V={v_l:.3f}\n")

report = "\n".join(lines)
print(report)

with open(OUT_TXT, "w", encoding="utf-8") as f:
    f.write(report)

print("\nSaved full report to:")
print(OUT_TXT)
