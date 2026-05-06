# -*- coding: utf-8 -*-
"""
Analyze joined_master.csv and produce compact paper-ready tables/figures.
Inputs:
  joined_master.csv (must contain: frame_core, label_A01, label_A02,
                                     pred_LLaVA, pred_GPT5nano, pred_GPT5mini)

Outputs (under --out_dir):
  - table_main_metrics.(tex/csv): GT x Model -> Accuracy, Macro-F1
  - table_class_dist.(tex/csv):   GT -> class distribution (#, %)
  - fig_acc_by_gt.png:            Accuracy bars
  - fig_f1_by_gt.png:             Macro-F1 bars
  - fig_recall_keyclasses.png:    Recall bars for DECELERATE and CHANGE_LANE
  - confusion_*.csv:              per GT x Model confusion matrices (CSV only)
"""
import argparse
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report, confusion_matrix

LABELS = ["KEEP_SPEED", "DECELERATE", "CHANGE_LEFT", "CHANGE_RIGHT"]
LABELS_A3 = ["KEEP_SPEED","DECELERATE","CHANGE_LANE"]

def to_a3(x: str):
    return "CHANGE_LANE" if x in ("CHANGE_LEFT","CHANGE_RIGHT") else x

def save_table_tex_csv(df: pd.DataFrame, out_tex: Path, out_csv: Path, floatfmt="%.3f"):
    df.to_csv(out_csv, index=True, encoding="utf-8-sig")
    with open(out_tex, "w", encoding="utf-8") as f:
        f.write(df.to_latex(escape=True, float_format=lambda v: (floatfmt % v) if isinstance(v,(float,np.floating)) else v))

def barplot(df_wide: pd.DataFrame, ylabel: str, title: str, out_png: Path):
    ax = df_wide.plot(kind="bar", figsize=(8,5))
    ax.set_title(title)
    ax.set_xlabel("GT (Ground Truth)")
    ax.set_ylabel(ylabel)
    ax.legend(title="Model")
    # annotate values
    for p in ax.patches:
        h = p.get_height()
        if np.isfinite(h):
            ax.annotate(f"{h:.2f}", (p.get_x()+p.get_width()/2, h),
                        ha="center", va="bottom", fontsize=8, xytext=(0,2), textcoords="offset points")
    plt.tight_layout()
    plt.savefig(out_png, dpi=180)
    plt.close()

def eval_one(df: pd.DataFrame, gt_col: str, model_cols: list, out_dir: Path, tag: str):
    out_dir.mkdir(parents=True, exist_ok=True)

    # クラス分布（4クラス）
    dist = df[gt_col].value_counts().reindex(LABELS, fill_value=0)
    dist_pct = (dist / dist.sum()).fillna(0)
    class_dist = pd.DataFrame({"count":dist, "ratio":dist_pct})
    save_table_tex_csv(class_dist, out_dir/"class_distribution.tex", out_dir/"class_distribution.csv")

    # メイン指標
    metrics = []
    percls_all = []
    for mcol in model_cols:
        sub = df.dropna(subset=[gt_col, mcol]).copy()
        if sub.empty: continue
        y_true = sub[gt_col].astype(str)
        y_pred = sub[mcol].astype(str)

        # Accuracy（手計算）
        acc = float((y_true == y_pred).mean())

        # classification_report（Macro-F1取得）
        rep = classification_report(
            y_true, y_pred, labels=LABELS, digits=3,
            output_dict=True, zero_division=0
        )
        macro_f1 = float(rep.get("macro avg",{}).get("f1-score",0.0))

        # 混同行列（CSV保存のみ）
        cm = confusion_matrix(y_true, y_pred, labels=LABELS)
        cm_df = pd.DataFrame(cm, index=[f"T_{l}" for l in LABELS], columns=[f"P_{l}" for l in LABELS])
        cm_df.to_csv(out_dir / f"confusion_{tag}_{mcol}.csv", encoding="utf-8-sig")

        # per-class（Precision/Recall/F1）蓄積
        for lab in LABELS:
            d = rep.get(lab, {"precision":0.0,"recall":0.0,"f1-score":0.0})
            percls_all.append({
                "GT": tag, "Model": mcol, "Class": lab,
                "Precision": float(d.get("precision",0.0)),
                "Recall":    float(d.get("recall",0.0)),
                "F1":        float(d.get("f1-score",0.0))
            })

        metrics.append({"GT":tag, "Model":mcol, "N_used":int(len(sub)),
                        "Accuracy":acc, "MacroF1":macro_f1})

    metrics_df = pd.DataFrame(metrics)
    return metrics_df, pd.DataFrame(percls_all)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in_csv", required=True)
    ap.add_argument("--out_dir", required=True)
    args = ap.parse_args()

    in_csv = Path(args.in_csv)
    out_dir = Path(args.out_dir); out_dir.mkdir(parents=True, exist_ok=True)

    # 読み込み
    df = pd.read_csv(in_csv, dtype=str)
    # 必須列チェック
    needed = {"label_A01","label_A02","pred_LLaVA","pred_GPT5nano","pred_GPT5mini"}
    missing = needed - set(df.columns)
    if missing:
        raise ValueError(f"joined_master.csv missing columns: {missing}")

    # 4クラスで評価
    model_cols = ["pred_LLaVA","pred_GPT5nano","pred_GPT5mini"]

    # 各GTのサブセット
    df_A01 = df.dropna(subset=["label_A01"]).copy()
    df_A02 = df.dropna(subset=["label_A02"]).copy()
    df_AG  = df.dropna(subset=["label_A01","label_A02"]).copy()
    df_AG  = df_AG[df_AG["label_A01"] == df_AG["label_A02"]].copy()
    df_AG  = df_AG.rename(columns={"label_A01":"label_AGREE"})

    # 評価
    m_A01, pc_A01 = eval_one(df_A01, "label_A01", model_cols, out_dir/"A01",  "A01")
    m_A02, pc_A02 = eval_one(df_A02, "label_A02", model_cols, out_dir/"A02",  "A02")
    m_AG,  pc_AG  = eval_one(df_AG,  "label_AGREE", model_cols, out_dir/"AGREE","AGREE")

    # まとめ表（本文用：Accuracy/Macro-F1）
    main_tbl = pd.concat([m_A01, m_A02, m_AG], ignore_index=True)
    main_tbl = main_tbl.sort_values(["GT","Model"])
    save_table_tex_csv(main_tbl.set_index(["GT","Model"])[["Accuracy","MacroF1"]],
                       out_dir/"table_main_metrics.tex", out_dir/"table_main_metrics.csv")

    # クラス分布（GT別）小表
    def class_dist(df_gt, gt_name):
        s = df_gt.iloc[:, df_gt.columns.str.startswith("label_")].iloc[:,0]
        v = s.value_counts().reindex(LABELS, fill_value=0)
        r = (v / v.sum()).fillna(0)
        return pd.DataFrame({(gt_name,"count"):v, (gt_name,"ratio"):r})
    cd_A01 = class_dist(df_A01.rename(columns={"label_A01":"lab"}), "A01")
    cd_A02 = class_dist(df_A02.rename(columns={"label_A02":"lab"}), "A02")
    cd_AG  = class_dist(df_AG .rename(columns={"label_AGREE":"lab"}), "AGREE")
    class_tbl = pd.concat([cd_A01, cd_A02, cd_AG], axis=1)
    save_table_tex_csv(class_tbl, out_dir/"table_class_dist.tex", out_dir/"table_class_dist.csv")

    # 図（Accuracy / Macro-F1）
    piv_acc = main_tbl.pivot_table(index="GT", columns="Model", values="Accuracy")
    piv_f1  = main_tbl.pivot_table(index="GT", columns="Model", values="MacroF1")
    barplot(piv_acc, "Accuracy", "Accuracy by Model under Different GT", out_dir/"fig_acc_by_gt.png")
    barplot(piv_f1,  "Macro-F1", "Macro-F1 by Model under Different GT", out_dir/"fig_f1_by_gt.png")

    # 重要クラスのRecall（DECELERATE と CHANGE_LANE=左右統合）を可視化
    percls_all = pd.concat([pc_A01, pc_A02, pc_AG], ignore_index=True)
    # 3クラス化（左右を統合）
    def to_keyclass(c):
        return "CHANGE_LANE" if c in ("CHANGE_LEFT","CHANGE_RIGHT") else c
    percls_all["KeyClass"] = percls_all["Class"].map(to_keyclass)
    # GTごと・モデルごと・KeyClassごとの Recall 平均
    keyrec = (percls_all.groupby(["GT","Model","KeyClass"])["Recall"]
              .mean().reset_index())
    # 抽出（KEEPは省き、DECELERATE/CHANGE_LANEのみ本文向けに）
    keyrec = keyrec[keyrec["KeyClass"].isin(["DECELERATE","CHANGE_LANE"])]

    # 表（付録/本文用小表）
    keyrec_tbl = keyrec.pivot_table(index=["GT","Model"], columns="KeyClass", values="Recall")
    save_table_tex_csv(keyrec_tbl, out_dir/"table_keyclass_recall.tex", out_dir/"table_keyclass_recall.csv")

    # 図（1枚にGT別バー：KeyClassごとに図を分けるより省スペースに1枚で）
    # 軸: index=GT, col=Model、系列=KeyClass → モデルの凡例が増えすぎるので、ここはGTをx軸、モデルを色、棒はKeyClassでfacet…は避けて1図にまとめる
    # ここでは GT を x、棒をモデル、棒上に (DECEL/CHANGE) を2段表示は複雑になるので、
    # 単純に GT 別に DECEL/CHANGE を連結した DataFrame を描画
    # → GT×Model×KeyClass を "GT|KeyClass" をx軸にして棒＝Model で比較
    keyrec["GT_Key"] = keyrec["GT"] + "|" + keyrec["KeyClass"]
    piv_key = keyrec.pivot_table(index="GT_Key", columns="Model", values="Recall")
    barplot(piv_key, "Recall", "Recall for DECELERATE & CHANGE_LANE", out_dir/"fig_recall_keyclasses.png")

    print("Done. Outputs in:", out_dir)

if __name__ == "__main__":
    main()
