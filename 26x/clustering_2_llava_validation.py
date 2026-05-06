# -*- coding: utf-8 -*-
"""
Evaluation for ContextVLM (Plan-B) — robust GT loader
- GT CSV 列名の揺れ（BOM, 区切り, 別名）を吸収
- cluster_id が無い場合は saved_path/original_path から復元
- confは未使用。ラベルのみで評価。
"""

import os
import re
import json
import glob
import numpy as np
import pandas as pd
from typing import Dict, List, Tuple
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    classification_report,
    cohen_kappa_score,
    matthews_corrcoef,
    confusion_matrix,
)

# ======== パス設定 ========
BASE_DIR = r"C:\Users\s1280\Desktop\SHRP2rawdata\central_frames_clustering_v6\clustered_images"
GT_CSV   = os.path.join(BASE_DIR, "cluster_assignments_gt.csv")
PRED_DIR = os.path.join(BASE_DIR, "llava_results_surface2_visibility2_v6_zeroshot")
OUT_DIR  = os.path.join(BASE_DIR, "evaluation_results")
os.makedirs(OUT_DIR, exist_ok=True)

# ======== 正規化 ========
def norm_surface(x: str) -> str:
    if not isinstance(x, str):
        return "DRY"
    t = x.strip().upper()
    alias = {
        "WET": "SLIPPERY","RAIN":"SLIPPERY","RAINY":"SLIPPERY","WATER":"SLIPPERY",
        "PUDDLE":"SLIPPERY","PUDDLES":"SLIPPERY","SPRAY":"SLIPPERY","REFLECTIVE":"SLIPPERY","GLOSSY":"SLIPPERY",
        "SNOW":"SLIPPERY","SNOWY":"SLIPPERY","ICE":"SLIPPERY","ICY":"SLIPPERY","SLUSH":"SLIPPERY",
        "PACKED SNOW":"SLIPPERY","FROZEN":"SLIPPERY","SALT":"SLIPPERY","RESIDUE":"SLIPPERY","PLOWED":"SLIPPERY",
        "SLIPPERY":"SLIPPERY",
        "DRY":"DRY","CLEAN":"DRY","ASPHALT":"DRY","MATTE":"DRY","NO WET":"DRY","NO SNOW":"DRY",
    }
    for k, v in alias.items():
        if k in t:
            return v
    return t if t in {"SLIPPERY","DRY"} else "DRY"

def norm_vis(x: str) -> str:
    if not isinstance(x, str):
        return "YES"
    t = x.strip().upper()
    if "YES" in t: return "YES"
    if "NO"  in t: return "NO"
    # allow clear/cloudy heuristics only if needed; for now default YES
    return "YES"

def _clean_columns(cols: List[str]) -> List[str]:
    # 列名から BOM 等を除去 + 正規化
    cleaned = []
    for c in cols:
        c = c.replace("\ufeff", "")  # BOM
        c = c.strip()
        c = re.sub(r"\s+", "_", c.lower())  # 空白→アンダースコア、小文字化
        cleaned.append(c)
    return cleaned

def _try_read_csv_any_sep(path: str) -> pd.DataFrame:
    # まずは自動推定
    try:
        return pd.read_csv(path, sep=None, engine="python")
    except Exception:
        pass
    # タブ
    try:
        return pd.read_csv(path, sep="\t")
    except Exception:
        pass
    # カンマ
    return pd.read_csv(path)

def _extract_cluster_from_path(p: str) -> str:
    if not isinstance(p, str):
        return ""
    m = re.search(r"(cluster[_\-\s]?)(\d{1,2})", p, flags=re.IGNORECASE)
    if m:
        n = int(m.group(2))
        return f"cluster_{n:02d}"
    return ""

# ======== データ読込 ========
def load_gt_csv(path: str) -> pd.DataFrame:
    df = _try_read_csv_any_sep(path)
    # 列名クリーニング
    df.columns = _clean_columns(df.columns.tolist())

    # 想定列の別名取り込み
    # 必須：filename, surface_gt, good_visibility_gt
    # 任意：cluster_id / cluster / saved_path / original_path
    col_map = {}

    # filename
    for cand in ["filename","file_name","image","image_name","img","img_name"]:
        if cand in df.columns:
            col_map["filename"] = cand
            break
    if "filename" not in col_map:
        raise ValueError(f"GT CSV must contain a filename column. Found: {df.columns.tolist()}")

    # surface_gt
    for cand in ["surface_gt","surface","road_surface_gt","surface_label","surface_class"]:
        if cand in df.columns:
            col_map["surface_gt"] = cand
            break
    if "surface_gt" not in col_map:
        raise ValueError("GT CSV must contain 'surface_gt' column (or alias).")

    # good_visibility_gt
    for cand in ["good_visibility_gt","visibility_gt","goodvis_gt","good_visibility","visibility"]:
        if cand in df.columns:
            col_map["good_visibility_gt"] = cand
            break
    if "good_visibility_gt" not in col_map:
        raise ValueError("GT CSV must contain 'good_visibility_gt' column (or alias).")

    # cluster_id（あれば使う）
    cluster_col = None
    for cand in ["cluster_id","cluster","cluster_idx","clusterindex","cluster_no","clusterid"]:
        if cand in df.columns:
            cluster_col = cand
            break

    # パス列（cluster復元に使用可）
    saved_col = "saved_path" if "saved_path" in df.columns else None
    orig_col  = "original_path" if "original_path" in df.columns else None

    # 必要列抽出
    use_cols = [col_map["filename"], col_map["surface_gt"], col_map["good_visibility_gt"]]
    extra_cols = [c for c in [cluster_col, saved_col, orig_col] if c]
    df = df[use_cols + extra_cols].copy()

    # 標準名へリネーム
    rename_map = {
        col_map["filename"]: "filename",
        col_map["surface_gt"]: "surface_gt",
        col_map["good_visibility_gt"]: "good_visibility_gt",
    }
    if cluster_col: rename_map[cluster_col] = "cluster_id"
    if saved_col: rename_map[saved_col] = "saved_path"
    if orig_col: rename_map[orig_col] = "original_path"
    df = df.rename(columns=rename_map)

    # 正規化（GT）
    df["surface_gt_norm"] = df["surface_gt"].map(lambda s: norm_surface(str(s)))
    df["good_visibility_gt_norm"] = df["good_visibility_gt"].map(
        lambda s: "YES" if str(s).strip().lower()=="yes" else ("NO" if str(s).strip().lower()=="no" else norm_vis(str(s)))
    )
    df["filename"] = df["filename"].astype(str)

    # cluster_id が無ければ saved_path / original_path から復元
    if "cluster_id" not in df.columns:
        df["cluster_id"] = ""
    if "cluster_id" in df.columns:
        df["cluster_id"] = df["cluster_id"].astype(str).str.replace("\ufeff","").str.strip()

    need_fill = (df["cluster_id"] == "") | (df["cluster_id"].isna())
    if need_fill.any():
        from_saved = df["saved_path"].apply(_extract_cluster_from_path) if "saved_path" in df.columns else ""
        from_orig  = df["original_path"].apply(_extract_cluster_from_path) if "original_path" in df.columns else ""
        df.loc[need_fill, "cluster_id"] = df.loc[need_fill, "cluster_id"].fillna("")
        df.loc[need_fill, "cluster_id"] = np.where(
            (need_fill) & (from_saved != ""), from_saved,
            df.loc[need_fill, "cluster_id"]
        )
        df.loc[need_fill, "cluster_id"] = np.where(
            (need_fill) & (df.loc[need_fill, "cluster_id"] == "") & (from_orig != ""), from_orig,
            df.loc[need_fill, "cluster_id"]
        )
        # それでも空なら 'cluster_??' 不明 → 空のまま（評価は可能）

    # 数値IDに整形（オプション）
    def _to_cluster_name(x: str) -> str:
        # "0" or "cluster_0" or "cluster_00" などを "cluster_00" に
        if not isinstance(x, str) or x == "":
            return ""
        m = re.search(r"(\d{1,2})", x)
        if m:
            return f"cluster_{int(m.group(1)):02d}"
        # 既に cluster_XX ならそのまま
        m2 = re.search(r"cluster_(\d{2})", x, flags=re.IGNORECASE)
        if m2:
            return f"cluster_{int(m2.group(1)):02d}"
        return x

    df["cluster_from_id"] = df["cluster_id"].apply(_to_cluster_name)
    return df

def load_predictions(pred_dir: str) -> pd.DataFrame:
    rows = []
    files = sorted(glob.glob(os.path.join(pred_dir, "surface_visibility_cluster_*.json")))
    if not files:
        raise FileNotFoundError(f"No prediction JSONs found under: {pred_dir}")
    for jpath in files:
        with open(jpath, "r", encoding="utf-8") as f:
            data = json.load(f)
        for item in data:
            cluster = str(item.get("cluster", ""))
            image   = str(item.get("image", ""))
            ans     = item.get("answers", {}) or {}
            s       = (ans.get("surface", {}) or {}).get("label", "")
            v       = (ans.get("good_visibility", {}) or {}).get("label", "")
            rows.append({
                "cluster_pred": cluster,  # e.g., "cluster_00"
                "filename": image,
                "surface_pred_norm": norm_surface(str(s)),
                "good_visibility_pred_norm": norm_vis(str(v)),
            })
    return pd.DataFrame(rows)

# ======== 指標計算 ========
def compute_and_save_metrics(df_join: pd.DataFrame, out_dir: str, title_suffix: str="") -> None:
    dd = df_join.dropna(subset=["surface_pred_norm","good_visibility_pred_norm"]).copy()

    # Surface
    y_true_s = dd["surface_gt_norm"].values
    y_pred_s = dd["surface_pred_norm"].values
    labels_s = ["DRY", "SLIPPERY"]

    acc_s  = accuracy_score(y_true_s, y_pred_s)
    bacc_s = balanced_accuracy_score(y_true_s, y_pred_s)
    kappa_s= cohen_kappa_score(y_true_s, y_pred_s, labels=labels_s)
    mcc_s  = matthews_corrcoef(y_true_s, y_pred_s)
    report_s = classification_report(y_true_s, y_pred_s, labels=labels_s, output_dict=True, zero_division=0)
    cm_s = confusion_matrix(y_true_s, y_pred_s, labels=labels_s)

    # Visibility
    y_true_v = dd["good_visibility_gt_norm"].values
    y_pred_v = dd["good_visibility_pred_norm"].values
    labels_v = ["YES", "NO"]

    acc_v  = accuracy_score(y_true_v, y_pred_v)
    bacc_v = balanced_accuracy_score(y_true_v, y_pred_v)
    kappa_v= cohen_kappa_score(y_true_v, y_pred_v, labels=labels_v)
    mcc_v  = matthews_corrcoef(y_true_v, y_pred_v)
    report_v = classification_report(y_true_v, y_pred_v, labels=labels_v, output_dict=True, zero_division=0)
    cm_v = confusion_matrix(y_true_v, y_pred_v, labels=labels_v)

    # 分布
    dist_surface = pd.DataFrame({
        "GT": dd["surface_gt_norm"].value_counts(),
        "Pred": dd["surface_pred_norm"].value_counts()
    }).fillna(0).astype(int)
    dist_vis = pd.DataFrame({
        "GT": dd["good_visibility_gt_norm"].value_counts(),
        "Pred": dd["good_visibility_pred_norm"].value_counts()
    }).fillna(0).astype(int)

    # クラスタ別（cluster_from_id が無い場合は cluster_pred を使用）
    cl_key = "cluster_from_id" if "cluster_from_id" in dd.columns else "cluster_pred"
    if dd[cl_key].isna().all() or (dd[cl_key] == "").all():
        cl_key = "cluster_pred"
    per_cluster = []
    for c, g in dd.groupby(cl_key):
        per_cluster.append({
            "cluster": c,
            "n": len(g),
            "surface_acc": accuracy_score(g["surface_gt_norm"], g["surface_pred_norm"]),
            "goodvis_acc": accuracy_score(g["good_visibility_gt_norm"], g["good_visibility_pred_norm"]),
        })
    df_cluster = pd.DataFrame(per_cluster).sort_values("cluster")

    # 保存
    summary = pd.DataFrame({
        "metric": ["acc","balanced_acc","cohen_kappa","mcc"],
        "surface": [acc_s, bacc_s, kappa_s, mcc_s],
        "good_visibility": [acc_v, bacc_v, kappa_v, mcc_v],
    })
    summary.to_csv(os.path.join(out_dir, "summary_metrics.csv"), index=False, encoding="utf-8-sig")
    pd.DataFrame(report_s).T.to_csv(os.path.join(out_dir, "classification_report_surface.csv"), encoding="utf-8-sig")
    pd.DataFrame(report_v).T.to_csv(os.path.join(out_dir, "classification_report_good_visibility.csv"), encoding="utf-8-sig")
    pd.DataFrame(cm_s, index=["DRY","SLIPPERY"], columns=["DRY","SLIPPERY"]).to_csv(
        os.path.join(out_dir, "confusion_matrix_surface.csv"), encoding="utf-8-sig")
    pd.DataFrame(cm_v, index=["YES","NO"], columns=["YES","NO"]).to_csv(
        os.path.join(out_dir, "confusion_matrix_good_visibility.csv"), encoding="utf-8-sig")
    dist_surface.to_csv(os.path.join(out_dir, "label_distribution_surface.csv"), encoding="utf-8-sig")
    dist_vis.to_csv(os.path.join(out_dir, "label_distribution_good_visibility.csv"), encoding="utf-8-sig")
    df_cluster.to_csv(os.path.join(out_dir, "per_cluster_accuracy.csv"), index=False, encoding="utf-8-sig")

    # joinテーブルも保存（デバッグ/再現性）
    dd.to_csv(os.path.join(out_dir, "joined_gt_pred_clean.csv"), index=False, encoding="utf-8-sig")

    print("[OK] Saved metrics under:", out_dir)
    print(summary)

# ======== メイン ========
def main():
    print("[1/3] Loading GT:", GT_CSV)
    gt = load_gt_csv(GT_CSV)

    print("[2/3] Loading predictions from:", PRED_DIR)
    pred = load_predictions(PRED_DIR)

    # ファイル名で突き合わせ
    df = gt.merge(pred, how="left", on="filename", suffixes=("_gt", "_pred"))

    missing = df["surface_pred_norm"].isna().sum()
    if missing > 0:
        print(f"[WARN] Predictions not found for {missing} rows by filename.")

    # 保存：結合表（生）
    df.to_csv(os.path.join(OUT_DIR, "joined_gt_pred_raw.csv"), index=False, encoding="utf-8-sig")

    print("[3/3] Computing metrics...")
    compute_and_save_metrics(df, OUT_DIR, title_suffix="")

if __name__ == "__main__":
    main()
