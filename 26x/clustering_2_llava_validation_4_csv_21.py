# -*- coding: utf-8 -*-
"""
Make per-cluster & overall confusion matrices (Surface / Visibility)
from: cluster_assignments_gt_with_pred.csv

Outputs (under OUT_DIR):
  - Per cluster:
      surface_cm_counts_cluster_XX.csv
      surface_cm_counts_cluster_XX.png
      surface_cm_norm_cluster_XX.csv
      surface_cm_norm_cluster_XX.png
      goodvis_cm_counts_cluster_XX.csv
      goodvis_cm_counts_cluster_XX.png
      goodvis_cm_norm_cluster_XX.csv
      goodvis_cm_norm_cluster_XX.png
  - Overall:
      surface_cm_counts_overall.{csv,png}
      surface_cm_norm_overall.{csv,png}
      goodvis_cm_counts_overall.{csv,png}
      goodvis_cm_norm_overall.{csv,png}
"""

import os
import re
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix

# ========= パス設定 =========
CSV_PATH = r"C:\Users\s1280\Desktop\SHRP2rawdata\central_frames_clustering_v6\clustered_images\cluster_assignments_gt_with_pred.csv"
OUT_DIR  = os.path.join(os.path.dirname(CSV_PATH), "paper_confusions_by_cluster")
os.makedirs(OUT_DIR, exist_ok=True)

# ========= 正規化 =========
def norm_surface(s: str) -> str:
    if not isinstance(s, str):
        return "DRY"
    t = s.strip().upper()
    # 予測は SLIPPERY/DRY、GTは dry/slippery の可能性がある想定
    if "SLIPPERY" in t: return "SLIPPERY"
    if "DRY" in t: return "DRY"
    # 別名を吸収
    if any(k in t for k in ["WET", "RAIN", "RAINY", "WATER", "PUDDLE", "PUDDLES", "SPRAY",
                             "REFLECTIVE", "GLOSSY", "SNOW", "SNOWY", "ICE", "ICY",
                             "SLUSH", "PACKED SNOW", "FROZEN", "SALT", "RESIDUE", "PLOWED"]):
        return "SLIPPERY"
    return "DRY"

def norm_vis(s: str) -> str:
    if not isinstance(s, str):
        return "YES"
    t = s.strip().upper()
    if "NO" in t: return "NO"
    if "YES" in t: return "YES"
    # 低リスク側へ
    return "YES"

# ========= クラスタ名整形 =========
def to_cluster_name(x) -> str:
    """
    0 / '0' / 'cluster_0' / 'cluster_00' / 'cluster-00' などを 'cluster_XX' に正規化。
    無ければ 'cluster_unknown'
    """
    if x is None or (isinstance(x, float) and np.isnan(x)):
        return "cluster_unknown"
    s = str(x)
    m = re.search(r"(\d{1,2})", s)
    if m:
        return f"cluster_{int(m.group(1)):02d}"
    m2 = re.search(r"cluster[_\-\s]?(\d{1,2})", s, flags=re.IGNORECASE)
    if m2:
        return f"cluster_{int(m2.group(1)):02d}"
    return str(s)

# ========= 描画（青↔白系） =========
def _auto_text_color(val, vmin, vmax):
    mid = (vmin + vmax) / 2.0
    return "white" if val > mid else "black"

def save_confusion(cm, labels, title, save_png, save_csv, normalize=False):
    cm_plot = cm.astype(float)
    if normalize:
        row_sums = cm_plot.sum(axis=1, keepdims=True)
        row_sums[row_sums == 0] = 1.0
        cm_plot = cm_plot / row_sums
        vmin, vmax = 0.0, 1.0
    else:
        vmin, vmax = 0.0, max(cm_plot.max(), 1.0)

    # CSV 保存（見出し付き）
    df_cm = pd.DataFrame(cm_plot, index=labels, columns=labels)
    df_cm.to_csv(save_csv, encoding="utf-8-sig")

    # 画像保存
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

    for i in range(cm_plot.shape[0]):
        for j in range(cm_plot.shape[1]):
            val = cm_plot[i, j]
            txt = f"{val:.2f}" if normalize else f"{int(val)}"
            ax.text(j, i, txt, ha="center", va="center",
                    color=_auto_text_color(val, vmin, vmax), fontsize=10)

    ax.set_ylim(len(labels)-0.5, -0.5)
    for edge in ["top","right","left","bottom"]:
        ax.spines[edge].set_visible(False)

    fig.tight_layout()
    fig.savefig(save_png, dpi=300, bbox_inches="tight")
    plt.close(fig)

# ========= メイン処理 =========
def main():
    # 読み込み（区切り自動推定）
    try:
        df = pd.read_csv(CSV_PATH, sep=None, engine="python")
    except Exception:
        df = pd.read_csv(CSV_PATH)

    # 列名クリーニング
    df.columns = [c.replace("\ufeff", "").strip() for c in df.columns]

    # クラスタ列の決定
    cluster_col = None
    for cand in ["cluster_from_id", "cluster_id", "cluster", "cluster_pred"]:
        if cand in df.columns:
            cluster_col = cand
            break
    if cluster_col is None:
        # 無ければ unknown として扱う
        cluster_col = "cluster_unknown"
        df[cluster_col] = "unknown"

    # 正規化した列を作る（GT / Pred）
    # surface
    gt_surface_col_candidates = ["surface_gt_norm", "surface_gt"]
    gt_vis_col_candidates     = ["good_visibility_gt_norm", "good_visibility_gt"]
    pred_surface_col = "surface_pred_norm" if "surface_pred_norm" in df.columns else "surface_pred_for_gt"
    pred_vis_col     = "good_visibility_pred_norm" if "good_visibility_pred_norm" in df.columns else "good_visibility_pred_for_gt"

    # GT 正規化
    gt_surface_col = None
    for c in gt_surface_col_candidates:
        if c in df.columns:
            gt_surface_col = c
            break
    if gt_surface_col is None:
        raise ValueError("GT surface column not found (expected 'surface_gt' or 'surface_gt_norm').")
    if gt_surface_col == "surface_gt":
        df["surface_gt_norm"] = df["surface_gt"].map(lambda x: norm_surface(str(x)))
    # 可視性
    gt_vis_col = None
    for c in gt_vis_col_candidates:
        if c in df.columns:
            gt_vis_col = c
            break
    if gt_vis_col is None:
        raise ValueError("GT good-visibility column not found (expected 'good_visibility_gt' or 'good_visibility_gt_norm').")
    if gt_vis_col == "good_visibility_gt":
        df["good_visibility_gt_norm"] = df["good_visibility_gt"].map(lambda x: norm_vis(str(x)))

    # 予測側の標準列名も補正
    if "surface_pred_norm" not in df.columns and "surface_pred_for_gt" in df.columns:
        df["surface_pred_norm"] = df["surface_pred_for_gt"].str.upper().map(lambda x: "SLIPPERY" if "SLIPPERY" in x else "DRY")
    if "good_visibility_pred_norm" not in df.columns and "good_visibility_pred_for_gt" in df.columns:
        df["good_visibility_pred_norm"] = df["good_visibility_pred_for_gt"].str.upper().map(lambda x: "NO" if "NO" in x else "YES")

    # クラスタ名を正規化
    df["cluster_name"] = df[cluster_col].apply(to_cluster_name)

    labels_surface = ["DRY", "SLIPPERY"]
    labels_vis     = ["YES", "NO"]

    # ======= クラスタごと =======
    for c, g in df.groupby("cluster_name"):
        # surface
        y_s_true = g["surface_gt_norm"].values
        y_s_pred = g["surface_pred_norm"].values
        cm_s = confusion_matrix(y_s_true, y_s_pred, labels=labels_surface)
        save_confusion(cm_s, labels_surface,
                       f"Surface – Confusion Matrix (Counts) [{c}]",
                       os.path.join(OUT_DIR, f"surface_cm_counts_{c}.png"),
                       os.path.join(OUT_DIR, f"surface_cm_counts_{c}.csv"),
                       normalize=False)
        save_confusion(cm_s, labels_surface,
                       f"Surface – Confusion Matrix (Row-normalized) [{c}]",
                       os.path.join(OUT_DIR, f"surface_cm_norm_{c}.png"),
                       os.path.join(OUT_DIR, f"surface_cm_norm_{c}.csv"),
                       normalize=True)

        # visibility
        y_v_true = g["good_visibility_gt_norm"].values
        y_v_pred = g["good_visibility_pred_norm"].values
        cm_v = confusion_matrix(y_v_true, y_v_pred, labels=labels_vis)
        save_confusion(cm_v, labels_vis,
                       f"Good-Visibility – Confusion Matrix (Counts) [{c}]",
                       os.path.join(OUT_DIR, f"goodvis_cm_counts_{c}.png"),
                       os.path.join(OUT_DIR, f"goodvis_cm_counts_{c}.csv"),
                       normalize=False)
        save_confusion(cm_v, labels_vis,
                       f"Good-Visibility – Confusion Matrix (Row-normalized) [{c}]",
                       os.path.join(OUT_DIR, f"goodvis_cm_norm_{c}.png"),
                       os.path.join(OUT_DIR, f"goodvis_cm_norm_{c}.csv"),
                       normalize=True)

    # ======= 全体 =======
    y_s_true_all = df["surface_gt_norm"].values
    y_s_pred_all = df["surface_pred_norm"].values
    cm_s_all = confusion_matrix(y_s_true_all, y_s_pred_all, labels=labels_surface)
    save_confusion(cm_s_all, labels_surface,
                   "Surface – Confusion Matrix (Counts) [Overall]",
                   os.path.join(OUT_DIR, "surface_cm_counts_overall.png"),
                   os.path.join(OUT_DIR, "surface_cm_counts_overall.csv"),
                   normalize=False)
    save_confusion(cm_s_all, labels_surface,
                   "Surface – Confusion Matrix (Row-normalized) [Overall]",
                   os.path.join(OUT_DIR, "surface_cm_norm_overall.png"),
                   os.path.join(OUT_DIR, "surface_cm_norm_overall.csv"),
                   normalize=True)

    y_v_true_all = df["good_visibility_gt_norm"].values
    y_v_pred_all = df["good_visibility_pred_norm"].values
    cm_v_all = confusion_matrix(y_v_true_all, y_v_pred_all, labels=labels_vis)
    save_confusion(cm_v_all, labels_vis,
                   "Good-Visibility – Confusion Matrix (Counts) [Overall]",
                   os.path.join(OUT_DIR, "goodvis_cm_counts_overall.png"),
                   os.path.join(OUT_DIR, "goodvis_cm_counts_overall.csv"),
                   normalize=False)
    save_confusion(cm_v_all, labels_vis,
                   "Good-Visibility – Confusion Matrix (Row-normalized) [Overall]",
                   os.path.join(OUT_DIR, "goodvis_cm_norm_overall.png"),
                   os.path.join(OUT_DIR, "goodvis_cm_norm_overall.csv"),
                   normalize=True)

    print("[OK] Wrote per-cluster & overall confusion matrices to:")
    print(" ", OUT_DIR)

if __name__ == "__main__":
    main()
