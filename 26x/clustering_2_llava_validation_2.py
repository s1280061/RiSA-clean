# -*- coding: utf-8 -*-
"""
Paper-ready figures & tables generator for ContextVLM (Plan-B)
- Reads: evaluation_results/joined_gt_pred_clean.csv (from previous evaluation step)
- Writes: paper_figures/ *.png, *.csv, *.tex
- Note: Heatmaps use a blue↔white palette (cmap='Blues') for better readability.
"""

import os
import numpy as np
import pandas as pd
from math import sqrt
import matplotlib.pyplot as plt
from sklearn.metrics import (
    classification_report, confusion_matrix, accuracy_score,
    balanced_accuracy_score, cohen_kappa_score, matthews_corrcoef
)

# ======== パス設定（必要なら変更） ========
BASE_DIR = r"C:\Users\s1280\Desktop\SHRP2rawdata\central_frames_clustering_v6\clustered_images"
EVAL_DIR = os.path.join(BASE_DIR, "evaluation_results")
JOINED   = os.path.join(EVAL_DIR, "joined_gt_pred_clean.csv")
OUT_DIR  = os.path.join(BASE_DIR, "paper_figures")
os.makedirs(OUT_DIR, exist_ok=True)

# ======== 補助関数 ========
def wilson_ci(p, n, z=1.96):
    if n == 0:
        return (0.0, 0.0)
    denom = 1 + z**2/n
    center = (p + z*z/(2*n)) / denom
    half = (z/denom) * sqrt((p*(1-p) + z*z/(4*n))/n)
    return (max(0.0, center - half), min(1.0, center + half))

def bootstrap_ci_metric(y_true, y_pred, func, B=1000, seed=42):
    rng = np.random.default_rng(seed)
    n = len(y_true)
    vals = []
    idx = np.arange(n)
    for _ in range(B):
        b = rng.choice(idx, size=n, replace=True)
        vals.append(func(y_true[b], y_pred[b]))
    vals = np.sort(vals)
    lo = np.percentile(vals, 2.5)
    hi = np.percentile(vals, 97.5)
    return float(np.mean(vals)), float(lo), float(hi)

def _auto_text_color(val, vmin, vmax):
    """背景の濃さに応じて注釈色を黒/白で切替"""
    mid = (vmin + vmax) / 2.0
    return "white" if val > mid else "black"

def draw_confusion(cm, labels, title, save_path, normalize=False):
    cm_plot = cm.astype(float)
    if normalize:
        row_sums = cm_plot.sum(axis=1, keepdims=True)
        row_sums[row_sums == 0] = 1.0
        cm_plot = cm_plot / row_sums
        vmin, vmax = 0.0, 1.0
    else:
        vmin, vmax = 0.0, cm_plot.max() if cm_plot.max() > 0 else 1.0

    fig, ax = plt.subplots(figsize=(4.8, 4.2))
    im = ax.imshow(cm_plot, interpolation="nearest", cmap="Blues", vmin=vmin, vmax=vmax)
    cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.ax.set_ylabel("Proportion" if normalize else "Count", rotation=90, va="center")

    ax.set_title(title)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Ground Truth")
    ax.set_xticks(np.arange(len(labels)))
    ax.set_yticks(np.arange(len(labels)))
    ax.set_xticklabels(labels)
    ax.set_yticklabels(labels)

    # 値を描画（背景に応じて文字色を自動切替）
    for i in range(cm_plot.shape[0]):
        for j in range(cm_plot.shape[1]):
            val = cm_plot[i, j]
            txt = f"{val:.2f}" if normalize else f"{int(val)}"
            ax.text(j, i, txt, ha="center", va="center",
                    color=_auto_text_color(val, vmin, vmax), fontsize=10)

    # 罫線で区切り
    ax.set_ylim(len(labels)-0.5, -0.5)
    ax.grid(False)
    for edge in ["top","right","left","bottom"]:
        ax.spines[edge].set_visible(False)

    fig.tight_layout()
    fig.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close(fig)

def bar_dict_compare(d_gt, d_pred, title, save_path, order=None, ylabel="Count"):
    labels = sorted(set(d_gt.keys()) | set(d_pred.keys())) if order is None else order
    x = np.arange(len(labels))
    gt_vals = [d_gt.get(k, 0) for k in labels]
    pr_vals = [d_pred.get(k, 0) for k in labels]

    fig, ax = plt.subplots(figsize=(5.8, 3.4))
    w = 0.38
    ax.bar(x - w/2, gt_vals, width=w, label="GT", edgecolor="black", linewidth=0.5)
    ax.bar(x + w/2, pr_vals, width=w, label="Pred", edgecolor="black", linewidth=0.5)
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.legend()
    fig.tight_layout()
    fig.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close(fig)

def bar_per_cluster(df, key, title, save_path):
    g = df.groupby(key)
    rows = []
    for c, d in g:
        if len(d) == 0:
            continue
        s_acc = (d["surface_gt_norm"] == d["surface_pred_norm"]).mean()
        v_acc = (d["good_visibility_gt_norm"] == d["good_visibility_pred_norm"]).mean()
        rows.append((c, len(d), s_acc, v_acc))
    if not rows:
        return
    rows = sorted(rows, key=lambda x: x[0])
    clusters, ns, sacc, vacc = zip(*rows)
    x = np.arange(len(clusters))

    fig, ax = plt.subplots(figsize=(7.8, 4.2))
    w = 0.38
    ax.bar(x - w/2, sacc, width=w, label="Surface Acc", edgecolor="black", linewidth=0.5)
    ax.bar(x + w/2, vacc, width=w, label="GoodVis Acc", edgecolor="black", linewidth=0.5)
    ax.set_xticks(x)
    ax.set_xticklabels(clusters, rotation=45, ha="right")
    ax.set_ylim(0, 1.0)
    ax.set_ylabel("Accuracy")
    ax.set_title(title)
    ax.legend()
    fig.tight_layout()
    fig.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close(fig)

def miss_flow_heatmap(df, gt_col, pred_col, labels, title, save_path):
    cm = confusion_matrix(df[gt_col], df[pred_col], labels=labels)
    # 対角を 0 にし、誤り流入だけを表示
    err = cm.astype(float)
    for i in range(len(labels)):
        err[i, i] = 0
    # 行正規化
    row_sums = err.sum(axis=1, keepdims=True)
    row_sums[row_sums == 0] = 1.0
    err = err / row_sums

    vmin, vmax = 0.0, 1.0
    fig, ax = plt.subplots(figsize=(4.8, 4.2))
    im = ax.imshow(err, interpolation="nearest", cmap="Blues", vmin=vmin, vmax=vmax)
    cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.ax.set_ylabel("Proportion of errors", rotation=90, va="center")

    ax.set_title(title)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Ground Truth")
    ax.set_xticks(np.arange(len(labels)))
    ax.set_yticks(np.arange(len(labels)))
    ax.set_xticklabels(labels)
    ax.set_yticklabels(labels)

    for i in range(err.shape[0]):
        for j in range(err.shape[1]):
            val = err[i, j]
            ax.text(j, i, f"{val:.2f}", ha="center", va="center",
                    color=_auto_text_color(val, vmin, vmax), fontsize=10)

    ax.set_ylim(len(labels)-0.5, -0.5)
    for edge in ["top","right","left","bottom"]:
        ax.spines[edge].set_visible(False)

    fig.tight_layout()
    fig.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close(fig)

# ======== 読み込み ========
df = pd.read_csv(JOINED)

# クラスタキー選択（GT側が無ければpred側を使う）
cluster_key = "cluster_from_id" if ("cluster_from_id" in df.columns and df["cluster_from_id"].notna().any()) else "cluster_pred"

# ======== Surface 指標 & 図 ========
labels_surface = ["DRY", "SLIPPERY"]
y_s_true = df["surface_gt_norm"].values
y_s_pred = df["surface_pred_norm"].values

cm_s = confusion_matrix(y_s_true, y_s_pred, labels=labels_surface)
draw_confusion(cm_s, labels_surface, "Surface – Confusion Matrix (Counts)",
               os.path.join(OUT_DIR, "surface_cm_raw.png"), normalize=False)
draw_confusion(cm_s, labels_surface, "Surface – Confusion Matrix (Row-normalized)",
               os.path.join(OUT_DIR, "surface_cm_norm.png"), normalize=True)

rep_s = classification_report(y_s_true, y_s_pred, labels=labels_surface, output_dict=True, zero_division=0)
pd.DataFrame(rep_s).T.to_csv(os.path.join(OUT_DIR, "surface_class_report.csv"), encoding="utf-8-sig")

# Surface 指標＋CI
acc_s = accuracy_score(y_s_true, y_s_pred)
bacc_s = balanced_accuracy_score(y_s_true, y_s_pred)
kappa_s = cohen_kappa_score(y_s_true, y_s_pred, labels=labels_surface)
mcc_s = matthews_corrcoef(y_s_true, y_s_pred)
n_s = len(y_s_true)
acc_s_ci = wilson_ci(acc_s, n_s)

# Macro-F1 bootstrap CI
def macro_f1_surface(y_true, y_pred):
    rep = classification_report(y_true, y_pred, labels=labels_surface, output_dict=True, zero_division=0)
    return rep["macro avg"]["f1-score"]
macro_f1_s_mean, macro_f1_s_lo, macro_f1_s_hi = bootstrap_ci_metric(y_s_true, y_s_pred, macro_f1_surface, B=1000)

# ======== Visibility 指標 & 図 ========
labels_vis = ["YES", "NO"]
y_v_true = df["good_visibility_gt_norm"].values
y_v_pred = df["good_visibility_pred_norm"].values

cm_v = confusion_matrix(y_v_true, y_v_pred, labels=labels_vis)
draw_confusion(cm_v, labels_vis, "Good-Visibility – Confusion Matrix (Counts)",
               os.path.join(OUT_DIR, "goodvis_cm_raw.png"), normalize=False)
draw_confusion(cm_v, labels_vis, "Good-Visibility – Confusion Matrix (Row-normalized)",
               os.path.join(OUT_DIR, "goodvis_cm_norm.png"), normalize=True)

rep_v = classification_report(y_v_true, y_v_pred, labels=labels_vis, output_dict=True, zero_division=0)
pd.DataFrame(rep_v).T.to_csv(os.path.join(OUT_DIR, "goodvis_class_report.csv"), encoding="utf-8-sig")

acc_v = accuracy_score(y_v_true, y_v_pred)
bacc_v = balanced_accuracy_score(y_v_true, y_v_pred)
kappa_v = cohen_kappa_score(y_v_true, y_v_pred, labels=labels_vis)
mcc_v = matthews_corrcoef(y_v_true, y_v_pred)
n_v = len(y_v_true)
acc_v_ci = wilson_ci(acc_v, n_v)

def macro_f1_vis(y_true, y_pred):
    rep = classification_report(y_true, y_pred, labels=labels_vis, output_dict=True, zero_division=0)
    return rep["macro avg"]["f1-score"]
macro_f1_v_mean, macro_f1_v_lo, macro_f1_v_hi = bootstrap_ci_metric(y_v_true, y_v_pred, macro_f1_vis, B=1000)

# ======== ラベル分布バー図 ========
dist_s_gt   = df["surface_gt_norm"].value_counts().to_dict()
dist_s_pred = df["surface_pred_norm"].value_counts().to_dict()
bar_dict_compare(dist_s_gt, dist_s_pred, "Surface – Label Distribution (GT vs Pred)",
                 os.path.join(OUT_DIR, "label_dist_surface.png"),
                 order=labels_surface)

dist_v_gt   = df["good_visibility_gt_norm"].value_counts().to_dict()
dist_v_pred = df["good_visibility_pred_norm"].value_counts().to_dict()
bar_dict_compare(dist_v_gt, dist_v_pred, "Good-Visibility – Label Distribution (GT vs Pred)",
                 os.path.join(OUT_DIR, "label_dist_goodvis.png"),
                 order=labels_vis)

# ======== クラスタ別精度（バー図） ========
bar_per_cluster(df, cluster_key, "Per-Cluster Accuracy (Surface & Good-Visibility)",
                os.path.join(OUT_DIR, "per_cluster_accuracy.png"))

# ======== 誤分類の方向（ヒートマップ） ========
miss_flow_heatmap(df, "surface_gt_norm", "surface_pred_norm", labels_surface,
                  "Surface – Misclassification Flow (Row-normalized, diag=0)",
                  os.path.join(OUT_DIR, "miss_flow_surface.png"))
miss_flow_heatmap(df, "good_visibility_gt_norm", "good_visibility_pred_norm", labels_vis,
                  "Good-Visibility – Misclassification Flow (Row-normalized, diag=0)",
                  os.path.join(OUT_DIR, "miss_flow_goodvis.png"))

# ======== サマリ表（CSV & LaTeX） ========
summary_rows = [
    ["Surface", "Accuracy",        acc_s,  acc_s_ci[0], acc_s_ci[1]],
    ["Surface", "Balanced Acc.",   bacc_s, None,        None],
    ["Surface", "Cohen's kappa",   kappa_s, None,       None],
    ["Surface", "MCC",             mcc_s,   None,       None],
    ["Surface", "Macro-F1 (boot)", macro_f1_s_mean, macro_f1_s_lo, macro_f1_s_hi],

    ["GoodVis", "Accuracy",        acc_v,  acc_v_ci[0], acc_v_ci[1]],
    ["GoodVis", "Balanced Acc.",   bacc_v, None,        None],
    ["GoodVis", "Cohen's kappa",   kappa_v, None,       None],
    ["GoodVis", "MCC",             mcc_v,   None,       None],
    ["GoodVis", "Macro-F1 (boot)", macro_f1_v_mean, macro_f1_v_lo, macro_f1_v_hi],
]
df_sum = pd.DataFrame(summary_rows, columns=["Task","Metric","Value","CI_low","CI_high"])
df_sum.to_csv(os.path.join(OUT_DIR, "summary_with_CI.csv"), index=False, encoding="utf-8-sig")

# クラス別PRFテーブル（LaTeX出力）
def prf_tex(report_dict, labels, caption, save_tex):
    rep = pd.DataFrame(report_dict).T.loc[labels, ["precision","recall","f1-score","support"]]
    rep.index.name = "Class"
    with open(save_tex, "w", encoding="utf-8") as f:
        f.write(rep.to_latex(escape=False, index=True, float_format=lambda x: f"{x:.3f}"))
    rep.to_csv(save_tex.replace(".tex",".csv"), encoding="utf-8-sig")

prf_tex(rep_s, labels_surface, "Surface PRF", os.path.join(OUT_DIR, "surface_prf.tex"))
prf_tex(rep_v, labels_vis,     "GoodVis PRF", os.path.join(OUT_DIR, "goodvis_prf.tex"))

# サマリ（LaTeX）
with open(os.path.join(OUT_DIR, "summary_with_CI.tex"), "w", encoding="utf-8") as f:
    f.write(df_sum.to_latex(escape=False, index=False, float_format=lambda x: f"{x:.3f}"))

print("[OK] Wrote paper-ready figures & tables to:", OUT_DIR)
