# -*- coding: utf-8 -*-
"""
eval_llava_clusters.py
- 予測JSON群（フォルダ内 *.json）と GT CSV を突き合わせ、
  カテゴリ（STRUCTURE/TIME/SCENE/VISIBILITY）ごとの
  Accuracy / F1（macro・weighted・micro）/ Precision macro / Recall macro を算出。
- 混同行列（PNG）・集計表（CSV）・クラス別の詳細表（CSV）を出力。

使い方（例/Windows）:
python eval_llava_clusters.py ^
  --pred_dirs "C:\Users\s1280\Desktop\SHRP2rawdata\central_frames_clustering_v4\clustered_images\llava_results_multiclass_visibility" ^
  --gt_csv   "C:\Users\s1280\Desktop\SHRP2rawdata\central_frames_clustering_v4\clustered_images\cluster_assignments_gt.csv" ^
  --out_dir  "C:\Users\s1280\Desktop\SHRP2rawdata\central_frames_clustering_v4\clustered_images\llava_eval_outputs"

必要ライブラリ:
pip install pandas numpy matplotlib scikit-learn
"""
import os
import json
import glob
import argparse
from itertools import product

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import (
    confusion_matrix,
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    classification_report,
)

# =============== ラベル正規化 ===============
def normalize_label(x: object):
    if x is None or (isinstance(x, float) and np.isnan(x)):
        return None
    s = str(x).strip()
    u = s.upper()
    # 表記ゆれを吸収
    mapping = {
        # STRUCTURE
        "INDOORS": "INDOORS",
        "OUTDOORS": "OUTDOORS",
        # TIME
        "DAYTIME": "DAYTIME",
        "NIGHTTIME": "NIGHT-TIME",
        "NIGHT-TIME": "NIGHT-TIME",
        # SCENE
        "ROAD": "ROAD",
        "PARKING": "PARKING LOT",
        "PARKING_LOT": "PARKING LOT",
        "PARKINGLOT": "PARKING LOT",
        # VISIBILITY
        "CLEAR": "CLEAR",
        "UNCLEAR": "UNCLEAR",
    }
    return mapping.get(u, u)

# =============== 予測JSON読込 ===============
def load_all_predictions(json_dirs):
    """
    予測JSONを一括読込。
    期待形式（例）: List[ { "cluster":..., "image":..., "predictions": {
        "STRUCTURE": ..., "TIME": ..., "SCENE": ..., "VISIBILITY": ...
    }}]
    ※ keysは多少の揺れに対応
    """
    rows = []
    for d in json_dirs:
        for fp in glob.glob(os.path.join(d, "*.json")):
            try:
                with open(fp, "r", encoding="utf-8") as fr:
                    data = json.load(fr)
                # list or {"items": list}
                items = data.get("items") if isinstance(data, dict) else data
                if not isinstance(items, list):
                    continue
                for item in items:
                    cluster = item.get("cluster") or item.get("cluster_id")
                    image = item.get("image") or item.get("filename")
                    preds = item.get("predictions") or item.get("prediction") or {}
                    rows.append(
                        {
                            "cluster": cluster,
                            "image": image,
                            "P_STRUCTURE": preds.get("STRUCTURE"),
                            "P_TIME": preds.get("TIME"),
                            "P_SCENE": preds.get("SCENE"),
                            "P_VISIBILITY": preds.get("VISIBILITY"),
                            "pred_json": os.path.basename(fp),
                        }
                    )
            except Exception as e:
                print(f"[WARN] parse failed: {fp}: {e}")
    return pd.DataFrame(rows)

# =============== GT 整形 ===============
def prepare_ground_truth(df_gt: pd.DataFrame) -> pd.DataFrame:
    cols_lower = [c.lower() for c in df_gt.columns]

    # 画像名列を推定
    image_col = None
    for cand in ["image", "filename", "file", "img", "image_name"]:
        if cand in cols_lower:
            image_col = df_gt.columns[cols_lower.index(cand)]
            break

    # ラベル列をマッピング
    label_cols_map = {}
    for k in ["STRUCTURE", "TIME", "SCENE", "VISIBILITY"]:
        lk = k.lower()
        if lk in cols_lower:
            label_cols_map[k] = df_gt.columns[cols_lower.index(lk)]
        else:
            # 部分一致（例: category_visibility）
            found = [c for c in df_gt.columns if lk in c.lower()]
            label_cols_map[k] = found[0] if found else None

    out = df_gt.copy()
    if image_col is not None:
        out = out.rename(columns={image_col: "image"})

    for k, col in label_cols_map.items():
        if col is not None:
            out = out.rename(columns={col: f"T_{k}"})
            out[f"T_{k}"] = out[f"T_{k}"].map(normalize_label)
        else:
            out[f"T_{k}"] = np.nan

    # cluster 列があれば残す
    if "cluster" not in out.columns and "cluster" in cols_lower:
        out = out.rename(columns={df_gt.columns[cols_lower.index("cluster")]: "cluster"})
    return out

# =============== マージ ===============
def merge_gt_pred(gt_df: pd.DataFrame, pred_df: pd.DataFrame) -> pd.DataFrame:
    # image キーでのマージを優先
    if "image" in gt_df.columns and "image" in pred_df.columns:
        merged = pd.merge(gt_df, pred_df, on="image", how="inner", suffixes=("_GT", "_P"))
        # clusterが両方ある場合の整列
        if "cluster_x" in merged.columns and "cluster_y" in merged.columns:
            merged = merged.rename(columns={"cluster_x": "T_cluster", "cluster_y": "P_cluster"})
        elif "cluster" in merged.columns and "cluster" in pred_df.columns:
            merged = merged.rename(columns={"cluster": "T_cluster"})
        return merged

    # 代替: cluster+image でのマージ
    if all(k in gt_df.columns for k in ["cluster", "image"]) and all(
        k in pred_df.columns for k in ["cluster", "image"]
    ):
        return pd.merge(gt_df, pred_df, on=["cluster", "image"], how="inner", suffixes=("_GT", "_P"))

    print("[WARN] no merge key matched (image / cluster+image). Returning empty.")
    return pd.DataFrame()

# =============== 指標と図 ===============
def metrics_table(y_true, y_pred):
    return {
        "accuracy": accuracy_score(y_true, y_pred),
        "f1_macro": f1_score(y_true, y_pred, average="macro", zero_division=0),
        "f1_weighted": f1_score(y_true, y_pred, average="weighted", zero_division=0),
        "f1_micro": f1_score(y_true, y_pred, average="micro", zero_division=0),
        "precision_macro": precision_score(y_true, y_pred, average="macro", zero_division=0),
        "recall_macro": recall_score(y_true, y_pred, average="macro", zero_division=0),
        "support": int(len(y_true)),
        "n_classes": int(len(set(y_true) | set(y_pred))),
    }

def plot_confusion(y_true, y_pred, title, out_png, dpi=200):
    labels = sorted(list(set(list(y_true) + list(y_pred))))
    cm = confusion_matrix(y_true, y_pred, labels=labels)

    fig = plt.figure(figsize=(6, 5))
    ax = plt.gca()
    ax.imshow(cm, interpolation="nearest")  # 色は指定しない（環境規定）
    ax.set_title(title)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    ax.set_xticks(range(len(labels)))
    ax.set_yticks(range(len(labels)))
    ax.set_xticklabels(labels, rotation=45, ha="right")
    ax.set_yticklabels(labels)
    # セルに数値を描く
    for i, j in product(range(cm.shape[0]), range(cm.shape[1])):
        ax.text(j, i, cm[i, j], ha="center", va="center")
    fig.tight_layout()
    fig.savefig(out_png, dpi=dpi, bbox_inches="tight")
    plt.close(fig)

def save_class_report(y_true, y_pred, out_csv):
    """
    クラス別 precision/recall/f1/support をCSV保存（sklearn classification_reportのdict版）
    """
    report = classification_report(y_true, y_pred, output_dict=True, zero_division=0)
    # avg/weighted/macro等も含むが、そのまま保存しておく
    df = pd.DataFrame(report).T.reset_index().rename(columns={"index": "label"})
    df.to_csv(out_csv, index=False)

# =============== メイン ===============
def main(pred_dirs, gt_csv, out_dir, dpi):
    os.makedirs(out_dir, exist_ok=True)

    # 読み込み
    gt_df = pd.read_csv(gt_csv)
    pred_df = load_all_predictions(pred_dirs)

    # 整形と突合
    gt_ready = prepare_ground_truth(gt_df)
    merged = merge_gt_pred(gt_ready, pred_df)

    merged_out = os.path.join(out_dir, "merged_gt_pred.csv")
    merged.to_csv(merged_out, index=False)
    print(f"[INFO] merged -> {merged_out} (rows={len(merged)})")

    categories = ["STRUCTURE", "TIME", "SCENE", "VISIBILITY"]
    metric_rows = []

    if len(merged) == 0:
        warn = os.path.join(out_dir, "WARNING_no_overlap.txt")
        with open(warn, "w", encoding="utf-8") as f:
            f.write("No overlapping rows between GT and predictions. Check merge keys (image/cluster).\n")
        print(f"[WARN] {warn}")
    else:
        for cat in categories:
            t_col = f"T_{cat}"
            p_col = f"P_{cat}"
            if t_col not in merged.columns or p_col not in merged.columns:
                print(f"[WARN] missing columns for {cat}: {t_col} or {p_col}")
                continue

            sub = merged[[t_col, p_col]].dropna()
            if len(sub) == 0:
                print(f"[WARN] no valid rows for {cat}")
                continue

            y_true = sub[t_col].map(normalize_label).values
            y_pred = sub[p_col].map(normalize_label).values

            # 指標（表）
            m = metrics_table(y_true, y_pred)
            m["category"] = cat
            metric_rows.append(m)

            # 混同行列（図）
            cm_png = os.path.join(out_dir, f"cm_{cat}.png")
            plot_confusion(y_true, y_pred, f"Confusion Matrix – {cat}", cm_png, dpi=dpi)
            print(f"[INFO] saved: {cm_png}")

            # クラス別の詳細表
            rep_csv = os.path.join(out_dir, f"class_report_{cat}.csv")
            save_class_report(y_true, y_pred, rep_csv)
            print(f"[INFO] saved: {rep_csv}")

    # まとめ表
    if metric_rows:
        metrics_df = pd.DataFrame(metric_rows)[
            ["category", "accuracy", "f1_macro", "f1_weighted", "f1_micro", "precision_macro", "recall_macro", "support", "n_classes"]
        ]
    else:
        metrics_df = pd.DataFrame(
            columns=["category", "accuracy", "f1_macro", "f1_weighted", "f1_micro", "precision_macro", "recall_macro", "support", "n_classes"]
        )
    metrics_csv = os.path.join(out_dir, "metrics_summary.csv")
    metrics_df.to_csv(metrics_csv, index=False)
    print(f"[INFO] saved: {metrics_csv}")

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--pred_dirs", nargs="+", required=True, help="*.json が入ったディレクトリ（複数可）")
    ap.add_argument("--gt_csv", required=True, help="GTのCSVファイル")
    ap.add_argument("--out_dir", required=True, help="出力先ディレクトリ")
    ap.add_argument("--dpi", type=int, default=200, help="画像保存DPI（混同行列）")
    args = ap.parse_args()
    main(args.pred_dirs, args.gt_csv, args.out_dir, args.dpi)
