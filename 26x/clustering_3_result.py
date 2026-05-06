# -*- coding: utf-8 -*-
"""
Key metrics with 95% CI (Bootstrap for Prec/Rec/F1, Wilson for Acc)
from: cluster_assignments_gt_with_pred.csv

Outputs (under OUT_DIR):
  - summary_with_CI.csv
  - summary_with_CI.tex
  - metrics_bar_surface.png
  - metrics_bar_goodvis.png
"""

import os
import numpy as np
import pandas as pd
from math import sqrt
import matplotlib.pyplot as plt
from sklearn.metrics import (
    accuracy_score, precision_recall_fscore_support
)

# ====== 入力CSV ======
CSV_PATH = r"C:\Users\s1280\Desktop\SHRP2rawdata\central_frames_clustering_v6\clustered_images\cluster_assignments_gt_with_pred.csv"
BASE_DIR = os.path.dirname(CSV_PATH)
OUT_DIR  = os.path.join(BASE_DIR, "paper_key_metrics")
os.makedirs(OUT_DIR, exist_ok=True)

# ====== ラベル正規化 ======
def norm_surface(s):
    if not isinstance(s, str):
        return "DRY"
    t = s.strip().upper()
    if "SLIPPERY" in t: return "SLIPPERY"
    if "DRY" in t: return "DRY"
    wetwords = ["WET","RAIN","RAINY","WATER","PUDDLE","PUDDLES","SPRAY",
                "REFLECTIVE","GLOSSY","SNOW","SNOWY","ICE","ICY",
                "SLUSH","PACKED SNOW","FROZEN","SALT","RESIDUE","PLOWED"]
    return "SLIPPERY" if any(w in t for w in wetwords) else "DRY"

def norm_vis(s):
    if not isinstance(s, str):
        return "YES"
    t = s.strip().upper()
    if "NO" in t: return "NO"
    if "YES" in t: return "YES"
    return "YES"

# ====== CIユーティリティ ======
def wilson_ci(p, n, z=1.96):
    """Accuracyの95%CI（Wilson）"""
    if n == 0:
        return (0.0, 0.0)
    denom = 1 + z**2/n
    center = (p + z*z/(2*n)) / denom
    half = (z/denom) * sqrt((p*(1-p) + z*z/(4*n))/n)
    return (max(0.0, center - half), min(1.0, center + half))

def bootstrap_ci(y_true, y_pred, score_fn, B=1000, seed=42):
    """Prec/Rec/F1の95%CI（ブートストラップ, macro average）"""
    rng = np.random.default_rng(seed)
    n = len(y_true)
    idx = np.arange(n)
    vals = np.empty(B, dtype=float)
    for b in range(B):
        samp = rng.choice(idx, size=n, replace=True)
        vals[b] = score_fn(y_true[samp], y_pred[samp])
    lo = float(np.percentile(vals, 2.5))
    hi = float(np.percentile(vals, 97.5))
    return float(vals.mean()), lo, hi

def macro_precision(y_true, y_pred, labels):
    p, _, _, _ = precision_recall_fscore_support(y_true, y_pred, labels=labels,
                                                 average='macro', zero_division=0)
    return p

def macro_recall(y_true, y_pred, labels):
    _, r, _, _ = precision_recall_fscore_support(y_true, y_pred, labels=labels,
                                                 average='macro', zero_division=0)
    return r

def macro_f1(y_true, y_pred, labels):
    _, _, f1, _ = precision_recall_fscore_support(y_true, y_pred, labels=labels,
                                                  average='macro', zero_division=0)
    return f1

# ====== 可視化（棒＋誤差バー） ======
def plot_metrics_bar(task_name, rows, save_path):
    """
    rows: list of (metric_name, value, lo, hi) with values in [0,1]
    """
    names = [r[0] for r in rows]
    vals  = [r[1] for r in rows]
    los   = [max(0.0, r[1] - (r[2] if r[2] is not None else r[1])) for r in rows]
    his   = [max(0.0, (r[3] if r[3] is not None else r[1]) - r[1]) for r in rows]
    yerr = np.vstack([los, his])

    fig, ax = plt.subplots(figsize=(5.2, 3.6))
    x = np.arange(len(names))
    ax.bar(x, vals, width=0.6, edgecolor='black', linewidth=0.6)
    ax.errorbar(x, vals, yerr=yerr, fmt='none', elinewidth=1.2, capsize=4)
    ax.set_xticks(x)
    ax.set_xticklabels(names)
    ax.set_ylim(0, 1.0)
    ax.set_ylabel("Score (Macro Average)")
    ax.set_title(f"{task_name} – Precision, Recall, F1, Accuracy (95% CI)")
    # 数値ラベル
    for xi, v in zip(x, vals):
        ax.text(xi, v + 0.02, f"{v:.2f}", ha="center", va="bottom", fontsize=9)
    fig.tight_layout()
    fig.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close(fig)

# ====== メイン ======
def main():
    # 読み込み
    try:
        df = pd.read_csv(CSV_PATH, sep=None, engine="python")
    except Exception:
        df = pd.read_csv(CSV_PATH)
    df.columns = [c.replace("\ufeff","").strip() for c in df.columns]

    # 正規化（Surface）
    y_s_true = (
        df["surface_gt_norm"].map(str).map(str.upper)
        if "surface_gt_norm" in df.columns
        else df["surface_gt"].map(norm_surface)
    )
    y_s_pred = (
        df["surface_pred_norm"].map(str).map(str.upper)
        if "surface_pred_norm" in df.columns
        else df["surface_pred_for_gt"].map(norm_surface)
    )

    # 正規化（Good-Visibility）
    y_v_true = (
        df["good_visibility_gt_norm"].map(str).map(str.upper)
        if "good_visibility_gt_norm" in df.columns
        else df["good_visibility_gt"].map(norm_vis)
    )
    y_v_pred = (
        df["good_visibility_pred_norm"].map(str).map(str.upper)
        if "good_visibility_pred_norm" in df.columns
        else df["good_visibility_pred_for_gt"].map(norm_vis)
    )

    # numpy配列化
    y_s_true = y_s_true.values
    y_s_pred = y_s_pred.values
    y_v_true = y_v_true.values
    y_v_pred = y_v_pred.values

    labels_surface = ["DRY", "SLIPPERY"]
    labels_vis     = ["YES", "NO"]

    # ===== Surface =====
    acc_s = accuracy_score(y_s_true, y_s_pred)
    acc_s_lo, acc_s_hi = wilson_ci(acc_s, len(y_s_true))
    prec_s_mean, prec_s_lo, prec_s_hi = bootstrap_ci(y_s_true, y_s_pred,
        lambda a,b: macro_precision(a,b,labels_surface))
    rec_s_mean,  rec_s_lo,  rec_s_hi  = bootstrap_ci(y_s_true, y_s_pred,
        lambda a,b: macro_recall(a,b,labels_surface))
    f1_s_mean,   f1_s_lo,   f1_s_hi   = bootstrap_ci(y_s_true, y_s_pred,
        lambda a,b: macro_f1(a,b,labels_surface))

    # ===== Good-Visibility =====
    acc_v = accuracy_score(y_v_true, y_v_pred)
    acc_v_lo, acc_v_hi = wilson_ci(acc_v, len(y_v_true))
    prec_v_mean, prec_v_lo, prec_v_hi = bootstrap_ci(y_v_true, y_v_pred,
        lambda a,b: macro_precision(a,b,labels_vis))
    rec_v_mean,  rec_v_lo,  rec_v_hi  = bootstrap_ci(y_v_true, y_v_pred,
        lambda a,b: macro_recall(a,b,labels_vis))
    f1_v_mean,   f1_v_lo,   f1_v_hi   = bootstrap_ci(y_v_true, y_v_pred,
        lambda a,b: macro_f1(a,b,labels_vis))

    # ===== サマリ表 =====
    rows = [
        ["Surface", "Precision", prec_s_mean, prec_s_lo, prec_s_hi],
        ["Surface", "Recall",    rec_s_mean,  rec_s_lo,  rec_s_hi],
        ["Surface", "F1",        f1_s_mean,   f1_s_lo,   f1_s_hi],
        ["Surface", "Accuracy",  acc_s,       acc_s_lo,  acc_s_hi],
        ["GoodVis", "Precision", prec_v_mean, prec_v_lo, prec_v_hi],
        ["GoodVis", "Recall",    rec_v_mean,  rec_v_lo,  rec_v_hi],
        ["GoodVis", "F1",        f1_v_mean,   f1_v_lo,   f1_v_hi],
        ["GoodVis", "Accuracy",  acc_v,       acc_v_lo,  acc_v_hi],
    ]
    df_sum = pd.DataFrame(rows, columns=["Task","Metric","Value","CI_low","CI_high"])
    csv_out = os.path.join(OUT_DIR, "summary_with_CI.csv")
    tex_out = os.path.join(OUT_DIR, "summary_with_CI.tex")
    df_sum.to_csv(csv_out, index=False, encoding="utf-8-sig")

    with open(tex_out, "w", encoding="utf-8") as f:
        f.write(df_sum.to_latex(index=False, escape=False, float_format=lambda x: f"{x:.3f}"))

    print("[OK] wrote:", csv_out)
    print("[OK] wrote:", tex_out)

    # ===== 図（棒＋誤差バー） =====
    plot_metrics_bar(
        "Surface",
        [
            ("Precision", prec_s_mean, prec_s_lo, prec_s_hi),
            ("Recall",    rec_s_mean,  rec_s_lo,  rec_s_hi),
            ("F1",        f1_s_mean,   f1_s_lo,   f1_s_hi),
            ("Accuracy",  acc_s,       acc_s_lo,  acc_s_hi),
        ],
        os.path.join(OUT_DIR, "metrics_bar_surface.png")
    )

    plot_metrics_bar(
        "Good-Visibility",
        [
            ("Precision", prec_v_mean, prec_v_lo, prec_v_hi),
            ("Recall",    rec_v_mean,  rec_v_lo,  rec_v_hi),
            ("F1",        f1_v_mean,   f1_v_lo,   f1_v_hi),
            ("Accuracy",  acc_v,       acc_v_lo,  acc_v_hi),
        ],
        os.path.join(OUT_DIR, "metrics_bar_goodvis.png")
    )

    print("[OK] Wrote figures to:", OUT_DIR)

if __name__ == "__main__":
    main()
