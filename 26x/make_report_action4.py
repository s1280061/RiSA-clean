# -*- coding: utf-8 -*-
"""
Action-4 総合レポート生成スクリプト
- GT（基準）: A01, A02, A01∧A02(一致のみ)
- モデル: LLaVA（contextあり）, GPT-5-nano（画像のみ）, GPT-5-mini（画像のみ）
- 指標: Accuracy, Macro-F1, per-class, Confusion Matrix
- 出力: CSV, PNG(混同行列), LaTeX(表)

依存: pandas, numpy, scikit-learn, matplotlib
"""

import argparse, re, os, json
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report, confusion_matrix, cohen_kappa_score

LABELS = ["KEEP_SPEED", "DECELERATE", "CHANGE_LEFT", "CHANGE_RIGHT"]
LABELS_A3 = ["KEEP_SPEED", "DECELERATE", "CHANGE_LANE"]

def to_a3(x: str):
    return "CHANGE_LANE" if x in ("CHANGE_LEFT","CHANGE_RIGHT") else x

# ---- frame_core 抽出（image_path, filename など何からでも） ----
def extract_frame_core(s: str):
    if not isinstance(s, str): return None
    s = s.strip()
    pats = [
        r"frame[_\- ]+(\d{3,})",      # frame_000123 / frame-000123
        r"[\\/](\d{3,})[_\.]"         # .../000123_.jpg / ...\000123.
    ]
    for pat in pats:
        m = re.search(pat, s, flags=re.IGNORECASE)
        if m:
            return str(int(m.group(1)))  # 先頭ゼロ除去
    return None

# ---- 可視化（混同行列） ----
def save_confusion_png(cm: np.ndarray, labels, out_png: Path, title: str):
    fig, ax = plt.subplots(figsize=(6, 5))
    # cmap をブルー系に変更
    im = ax.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
    ax.set_title(title)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    ax.set_xticks(np.arange(len(labels)))
    ax.set_yticks(np.arange(len(labels)))
    ax.set_xticklabels(labels, rotation=30, ha="right")
    ax.set_yticklabels(labels)

    # セルに値を描画（色によって文字色を変えると見やすい）
    thresh = cm.max() / 2.
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, str(cm[i, j]),
                    ha="center", va="center",
                    color="white" if cm[i, j] > thresh else "black",
                    fontsize=9)
    fig.tight_layout()
    fig.savefig(out_png, dpi=160)
    plt.close(fig)

# ---- テーブルをCSV / LaTeX 両方で保存 ----
def save_table(df: pd.DataFrame, out_csv: Path, out_tex: Path, floatfmt="%.3f"):
    df.to_csv(out_csv, index=True, encoding="utf-8-sig")
    with open(out_tex, "w", encoding="utf-8") as f:
        f.write(df.to_latex(escape=True, float_format=lambda x: floatfmt % x if isinstance(x,(float,np.floating)) else x))

# ---- 入力読み込み・正規化 ----
def load_human(csv_path: Path, annotator_id: str):
    df = pd.read_csv(csv_path, dtype=str)
    for c in df.columns:
        df[c] = df[c].astype(str).str.strip()
    # unsure除外（列が無ければそのまま）
    if "unsure" in df.columns:
        df = df[df["unsure"].astype(str) != "1"]
    if "image_path" not in df.columns:
        raise ValueError(f"{annotator_id}: image_path 列が見つかりません")
    if "human_action4" not in df.columns:
        raise ValueError(f"{annotator_id}: human_action4 列が見つかりません")
    df["frame_core"] = df["image_path"].apply(extract_frame_core)
    df = df.dropna(subset=["frame_core", "human_action4"])
    df = df[["frame_core","human_action4"]].drop_duplicates(subset=["frame_core"], keep="last")
    df = df.rename(columns={"human_action4": f"label_{annotator_id}"})
    return df

def load_llava(csv_path: Path):
    df = pd.read_csv(csv_path, dtype=str)
    for c in df.columns:
        df[c] = df[c].astype(str).str.strip()
    # カラム名の想定: frame_core / suggested_maneuver_action4
    if "frame_core" not in df.columns:
        # 最低限 image_path から抽出
        df["frame_core"] = df.get("image_path","").apply(extract_frame_core)
    pred_col = None
    for cand in ["suggested_maneuver_action4","llava_action4","action_norm","pred_action"]:
        if cand in df.columns:
            pred_col = cand; break
    if pred_col is None:
        raise ValueError("LLaVA CSVに action 列（suggested_maneuver_action4 等）が見つかりません")
    out = df[["frame_core", pred_col]].dropna()
    out = out.rename(columns={pred_col: "pred_LLaVA"})
    out = out.drop_duplicates(subset=["frame_core"], keep="last")
    return out

def load_gpt(csv_path: Path, tag: str):
    df = pd.read_csv(csv_path, dtype=str)
    for c in df.columns:
        df[c] = df[c].astype(str).str.strip()
    # 画像名（image列）から frame_core
    name_col = "image" if "image" in df.columns else "image_path"
    df["frame_core"] = df[name_col].apply(extract_frame_core)
    # 列：action_norm（推奨）or action_raw
    pred_col = "action_norm" if "action_norm" in df.columns else "action_raw"
    out = df[["frame_core", pred_col]].dropna()
    out = out.rename(columns={pred_col: f"pred_{tag}"})
    out = out.drop_duplicates(subset=["frame_core"], keep="last")
    return out

# ---- 評価（1つのGTに対してすべてのモデルを評価） ----
def eval_against_gt(df: pd.DataFrame, gt_col: str, model_cols: list, out_dir: Path, tag: str):
    out_dir.mkdir(parents=True, exist_ok=True)

    # クラス分布（GT）
    dist = df[gt_col].value_counts().reindex(LABELS, fill_value=0)
    dist.to_csv(out_dir / f"class_distribution_{tag}.csv", encoding="utf-8-sig")

    metrics_rows = []
    for mcol in model_cols:
        sub = df.dropna(subset=[gt_col, mcol]).copy()
        if sub.empty:
            continue
        y_true = sub[gt_col].astype(str)
        y_pred = sub[mcol].astype(str)

        # ---- 精度は手計算（環境差でrep["accuracy"]が無い場合があるため）----
        acc = float((y_true == y_pred).mean())

        # ---- レポート（Macro-F1は安全に取得）----
        rep = classification_report(
            y_true, y_pred,
            labels=LABELS,
            digits=3,
            output_dict=True,
            zero_division=0
        )
        macro_f1 = float(rep.get("macro avg", {}).get("f1-score", 0.0))

        # ---- 混同行列 ----
        cm = confusion_matrix(y_true, y_pred, labels=LABELS)
        cm_df = pd.DataFrame(cm, index=[f"T_{l}" for l in LABELS], columns=[f"P_{l}" for l in LABELS])
        cm_csv = out_dir / f"confusion_{tag}_{mcol}.csv"
        cm_png = out_dir / f"confusion_{tag}_{mcol}.png"
        cm_tex = out_dir / f"confusion_{tag}_{mcol}.tex"
        cm_df.to_csv(cm_csv, encoding="utf-8-sig")
        save_confusion_png(cm, LABELS, cm_png, title=f"Confusion: GT={tag}, Model={mcol}")
        with open(cm_tex, "w", encoding="utf-8") as f:
            f.write(cm_df.to_latex(escape=True))

        # ---- per-class（precision/recall/f1）を堅牢に作る ----
        rows = []
        for lab in LABELS:
            d = rep.get(lab, {"precision":0.0,"recall":0.0,"f1-score":0.0})
            rows.append([lab, d.get("precision",0.0), d.get("recall",0.0), d.get("f1-score",0.0)])
        per_class = pd.DataFrame(rows, columns=["label","precision","recall","f1-score"]).set_index("label")
        save_table(per_class, out_dir / f"perclass_{tag}_{mcol}.csv", out_dir / f"perclass_{tag}_{mcol}.tex")

        metrics_rows.append({
            "GT": tag,
            "Model": mcol,
            "N_used": int(len(sub)),
            "Accuracy": acc,
            "MacroF1": macro_f1
        })

    # ---- モデル比較テーブル（精度） ----
    if metrics_rows:
        metrics = pd.DataFrame(metrics_rows).sort_values(["GT","Model"])
        save_table(metrics.set_index(["GT","Model"]),
                   out_dir / f"summary_metrics_{tag}.csv",
                   out_dir / f"summary_metrics_{tag}.tex")
    else:
        print(f"[WARN] {tag}: metrics_rows is empty (no overlapping rows).")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--a01_csv", required=True)
    ap.add_argument("--a02_csv", required=True)
    ap.add_argument("--llava_csv", required=True)
    ap.add_argument("--gpt_nano_csv", required=True)
    ap.add_argument("--gpt_mini_csv", required=True)
    ap.add_argument("--out_dir", required=True)
    args = ap.parse_args()

    out_dir = Path(args.out_dir); out_dir.mkdir(parents=True, exist_ok=True)

    # 1) 読み込み
    A01 = load_human(Path(args.a01_csv), "A01")
    A02 = load_human(Path(args.a02_csv), "A02")
    LLaVA = load_llava(Path(args.llava_csv))
    GPTn  = load_gpt(Path(args.gpt_nano_csv), "GPT5nano")
    GPTm  = load_gpt(Path(args.gpt_mini_csv), "GPT5mini")

    # 2) マスタ（frame_coreで結合）
    df = A01.merge(A02, on="frame_core", how="outer")
    for pred in [LLaVA, GPTn, GPTm]:
        df = df.merge(pred, on="frame_core", how="left")
    df.to_csv(out_dir / "joined_master.csv", index=False, encoding="utf-8-sig")

    # 3) 人間同士の一致度（κ, 4クラス）
    df_k = df.dropna(subset=["label_A01","label_A02"]).copy()
    if not df_k.empty:
        kappa4 = cohen_kappa_score(df_k["label_A01"], df_k["label_A02"], labels=LABELS)
        # 3クラス（左右統合）も参考で出しておく
        kappa3 = cohen_kappa_score(df_k["label_A01"].map(to_a3), df_k["label_A02"].map(to_a3), labels=LABELS_A3)
    else:
        kappa4 = np.nan; kappa3 = np.nan
    kdf = pd.DataFrame({"kappa_4class":[kappa4], "kappa_3class":[kappa3], "N_common":[len(df_k)]})
    save_table(kdf, out_dir / "human_interrater.csv", out_dir / "human_interrater.tex")

    # 4) 各GTで評価
    model_cols = ["pred_LLaVA", "pred_GPT5nano", "pred_GPT5mini"]

    # 4-1) GT = A01
    df_a01 = df.dropna(subset=["label_A01"]).copy()
    eval_against_gt(df_a01, "label_A01", model_cols, out_dir / "GT_A01", tag="A01")

    # 4-2) GT = A02
    df_a02 = df.dropna(subset=["label_A02"]).copy()
    eval_against_gt(df_a02, "label_A02", model_cols, out_dir / "GT_A02", tag="A02")

    # 4-3) GT = A01∧A02一致のみ
    agree = df.dropna(subset=["label_A01","label_A02"]).copy()
    agree = agree[agree["label_A01"] == agree["label_A02"]]
    agree = agree.rename(columns={"label_A01":"label_AGREE"})  # どちらでも同じ
    eval_against_gt(agree, "label_AGREE", model_cols, out_dir / "GT_AGREE", tag="AGREE")

    # 5) モデル間比較（精度の並べ表：各GTごとのまとめを縦結合）
    tables = []
    for name in ["GT_A01","GT_A02","GT_AGREE"]:
        p = out_dir / name / f"summary_metrics_{name.split('_')[-1]}.csv"
        if p.exists():
            t = pd.read_csv(p, index_col=[0,1])
            t = t.reset_index()
            tables.append(t)
    if tables:
        comp = pd.concat(tables, ignore_index=True)
        pivot = comp.pivot_table(index=["GT"], columns="Model", values="Accuracy")
        save_table(pivot, out_dir / "model_comparison_accuracy.csv", out_dir / "model_comparison_accuracy.tex")

    print("Done. Outputs under:", out_dir)

if __name__ == "__main__":
    main()
