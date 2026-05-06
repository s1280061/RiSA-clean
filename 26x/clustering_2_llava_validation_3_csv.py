# -*- coding: utf-8 -*-
"""
Add LLaVA predictions next to GT columns.
- Read GT CSV
- Read 10 JSON files (cluster_00..09)
- Merge by filename and append prediction columns
- Save as cluster_assignments_gt_with_pred.csv (no overwrite)
"""

import os, json, glob, re
import pandas as pd

# ====== パス設定 ======
BASE_DIR = r"C:\Users\s1280\Desktop\SHRP2rawdata\central_frames_clustering_v6\clustered_images"
GT_CSV   = os.path.join(BASE_DIR, "cluster_assignments_gt.csv")
PRED_DIR = os.path.join(BASE_DIR, "llava_results_surface2_visibility2_v6_zeroshot")
OUT_CSV  = os.path.join(BASE_DIR, "cluster_assignments_gt_with_pred.csv")

# ====== 正規化 ======
def norm_surface(x: str) -> str:
    if not isinstance(x, str):
        return "DRY"
    t = x.strip().upper()
    alias = {
        "WET":"SLIPPERY","RAIN":"SLIPPERY","RAINY":"SLIPPERY","WATER":"SLIPPERY",
        "PUDDLE":"SLIPPERY","PUDDLES":"SLIPPERY","SPRAY":"SLIPPERY","REFLECTIVE":"SLIPPERY","GLOSSY":"SLIPPERY",
        "SNOW":"SLIPPERY","SNOWY":"SLIPPERY","ICE":"SLIPPERY","ICY":"SLIPPERY","SLUSH":"SLIPPERY",
        "PACKED SNOW":"SLIPPERY","FROZEN":"SLIPPERY","SALT":"SLIPPERY","RESIDUE":"SLIPPERY","PLOWED":"SLIPPERY",
        "SLIPPERY":"SLIPPERY","DRY":"DRY","CLEAN":"DRY","ASPHALT":"DRY","MATTE":"DRY","NO WET":"DRY","NO SNOW":"DRY",
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
    return "YES"

# ====== 予測の読み込み ======
def load_predictions(pred_dir: str) -> pd.DataFrame:
    rows = []
    files = sorted(glob.glob(os.path.join(pred_dir, "surface_visibility_cluster_*.json")))
    if not files:
        raise FileNotFoundError(f"No prediction JSONs under: {pred_dir}")
    for jpath in files:
        with open(jpath, "r", encoding="utf-8") as f:
            data = json.load(f)
        for item in data:
            image = str(item.get("image", ""))
            ans = item.get("answers", {}) or {}
            s_raw = (ans.get("surface", {}) or {}).get("label", "")
            v_raw = (ans.get("good_visibility", {}) or {}).get("label", "")
            rows.append({
                "filename": image,
                "surface_pred_raw": s_raw,
                "surface_pred_norm": norm_surface(str(s_raw)),
                "good_visibility_pred_raw": v_raw,
                "good_visibility_pred_norm": norm_vis(str(v_raw)),
            })
    return pd.DataFrame(rows)

# ====== メイン ======
def main():
    # GT 読み込み（区切り自動推定＋BOM除去）
    try:
        gt = pd.read_csv(GT_CSV, sep=None, engine="python")
    except Exception:
        gt = pd.read_csv(GT_CSV)  # フォールバック

    # 列名クリーニング（BOM等）
    gt.columns = [c.replace("\ufeff","").strip() for c in gt.columns]
    if "filename" not in gt.columns:
        # よくある別名にも対応
        for alt in ["file_name","image","image_name","img","img_name"]:
            if alt in gt.columns:
                gt = gt.rename(columns={alt:"filename"})
                break
    if "filename" not in gt.columns:
        raise ValueError(f"'filename' column not found in GT CSV. Found: {gt.columns.tolist()}")

    preds = load_predictions(PRED_DIR)

    # マージ（filename）
    out = gt.merge(preds, how="left", on="filename")

    # GTと見た目を揃えた小文字列も追加（任意）
    out["surface_pred_for_gt"] = out["surface_pred_norm"].fillna("DRY").str.lower()           # dry | slippery
    out["good_visibility_pred_for_gt"] = out["good_visibility_pred_norm"].fillna("YES").str.lower()  # yes | no

    # 保存
    out.to_csv(OUT_CSV, index=False, encoding="utf-8-sig")
    print(f"[OK] wrote: {OUT_CSV}")
    # どの列が追加されたか表示
    added = ["surface_pred_raw","surface_pred_norm","surface_pred_for_gt",
             "good_visibility_pred_raw","good_visibility_pred_norm","good_visibility_pred_for_gt"]
    print("Added columns:", added)

if __name__ == "__main__":
    main()
