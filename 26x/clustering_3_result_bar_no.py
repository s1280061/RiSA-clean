# -*- coding: utf-8 -*-
"""
Binary metrics (Precision / Recall / F1 / Accuracy), no CI.
Generates two plots:
  - metrics_bar_surface_no_ci.png
  - metrics_bar_goodvis_no_ci.png
from: cluster_assignments_gt_with_pred.csv
"""

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score

CSV_PATH = r"C:\Users\s1280\Desktop\SHRP2rawdata\central_frames_clustering_v6\clustered_images\cluster_assignments_gt_with_pred.csv"
OUT_DIR = os.path.dirname(CSV_PATH)

# ---- normalization helpers ----
def norm_surface(x: str) -> str:
    if not isinstance(x, str): return "DRY"
    t = x.strip().upper()
    if "SLIPPERY" in t: return "SLIPPERY"
    if "DRY" in t: return "DRY"
    wetwords = ["WET","RAIN","RAINY","WATER","PUDDLE","PUDDLES","SPRAY",
                "REFLECTIVE","GLOSSY","SNOW","SNOWY","ICE","ICY",
                "SLUSH","PACKED SNOW","FROZEN","SALT","RESIDUE","PLOWED"]
    return "SLIPPERY" if any(w in t for w in wetwords) else "DRY"

def norm_vis(x: str) -> str:
    if not isinstance(x, str): return "YES"
    t = x.strip().upper()
    if "NO" in t: return "NO"
    if "YES" in t: return "YES"
    return "YES"

def compute_metrics(y_true, y_pred, average='macro'):
    return {
        "Precision": precision_score(y_true, y_pred, average=average, zero_division=0),
        "Recall":    recall_score(y_true, y_pred,    average=average, zero_division=0),
        "F1":        f1_score(y_true, y_pred,        average=average, zero_division=0),
        "Accuracy":  accuracy_score(y_true, y_pred),
    }

def plot_bar(metrics_dict, title, save_path):
    names = ["Precision","Recall","F1","Accuracy"]
    vals  = [metrics_dict[n] for n in names]

    plt.figure(figsize=(7.2, 5.0))
    bars = plt.bar(names, vals, color='steelblue', edgecolor='black', linewidth=0.6)
    for b, v in zip(bars, vals):
        plt.text(b.get_x()+b.get_width()/2, v+0.015, f"{v:.2f}", ha="center", va="bottom", fontsize=10)
    plt.ylim(0, 1.0)
    plt.ylabel("Score (Macro Average)")
    plt.title(title)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close()
    print("[OK] saved:", save_path)

def main():
    df = pd.read_csv(CSV_PATH)
    # ---- fixed column mapping ----
    # GT
    s_gt = df["surface_gt"].map(norm_surface)
    v_gt = df["good_visibility_gt"].map(norm_vis)
    # Predictions (use *_pred_for_gt to align filename space with GT)
    s_pred = df["surface_pred_for_gt"].map(norm_surface)
    v_pred = df["good_visibility_pred_for_gt"].map(norm_vis)

    # numpy array
    s_gt, s_pred = np.array(s_gt), np.array(s_pred)
    v_gt, v_pred = np.array(v_gt), np.array(v_pred)

    # ---- metrics ----
    m_surface = compute_metrics(s_gt, s_pred, average="macro")
    m_vis     = compute_metrics(v_gt, v_pred, average="macro")

    # ---- plots (no CI) ----
    plot_bar(m_surface, "Surface – Precision, Recall, F1, Accuracy", os.path.join(OUT_DIR, "metrics_bar_surface_no_ci.png"))
    plot_bar(m_vis,     "Good-Visibility – Precision, Recall, F1, Accuracy", os.path.join(OUT_DIR, "metrics_bar_goodvis_no_ci.png"))

    # ---- save table (optional) ----
    pd.DataFrame([
        ["Surface","Precision",m_surface["Precision"]],
        ["Surface","Recall",   m_surface["Recall"]],
        ["Surface","F1",       m_surface["F1"]],
        ["Surface","Accuracy", m_surface["Accuracy"]],
        ["GoodVis","Precision",m_vis["Precision"]],
        ["GoodVis","Recall",   m_vis["Recall"]],
        ["GoodVis","F1",       m_vis["F1"]],
        ["GoodVis","Accuracy", m_vis["Accuracy"]],
    ], columns=["Task","Metric","Value"]).to_csv(
        os.path.join(OUT_DIR, "summary_no_ci.csv"), index=False, encoding="utf-8-sig"
    )
    print("[OK] saved:", os.path.join(OUT_DIR, "summary_no_ci.csv"))

if __name__ == "__main__":
    main()
