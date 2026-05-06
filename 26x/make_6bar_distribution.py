# -*- coding: utf-8 -*-
"""
6本だけの横並び積み上げ棒グラフ（コンパクト版）:
A01(GT), A02(GT), AGREE(GT), LLaVA, GPT5-nano, GPT5-mini
各バーは 4クラス(KEEP_SPEED, DECELERATE, CHANGE_LEFT, CHANGE_RIGHT) の比率。

入力: joined_master.csv（列: label_A01, label_A02, pred_LLaVA, pred_GPT5nano, pred_GPT5mini）
出力: 図(PNG) + 表(TeX/CSV; クラス×6本の比率表)
"""

import argparse
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

LABELS = ["KEEP_SPEED", "DECELERATE", "CHANGE_LEFT", "CHANGE_RIGHT"]
BAR_ORDER = ["A01(GT)", "A02(GT)", "AGREE(GT)", "LLaVA", "GPT5-nano", "GPT5-mini"]

def series_ratio(s: pd.Series) -> pd.Series:
    s = s.dropna().astype(str)
    cnt = s.value_counts().reindex(LABELS, fill_value=0)
    tot = cnt.sum()
    return (cnt / tot) if tot > 0 else cnt.astype(float)

def build_table(df: pd.DataFrame) -> pd.DataFrame:
    tbl = {}
    # GT分布（A01/A02/AGREE）
    tbl["A01(GT)"]   = series_ratio(df["label_A01"])
    tbl["A02(GT)"]   = series_ratio(df["label_A02"])
    agree_mask = df["label_A01"].notna() & df["label_A02"].notna() & (df["label_A01"].astype(str) == df["label_A02"].astype(str))
    tbl["AGREE(GT)"] = series_ratio(df.loc[agree_mask, "label_A01"])  # A01とA02が一致した部分

    # モデル予測分布（全体）
    if "pred_LLaVA" in df.columns:
        tbl["LLaVA"]     = series_ratio(df["pred_LLaVA"])
    if "pred_GPT5nano" in df.columns:
        tbl["GPT5-nano"] = series_ratio(df["pred_GPT5nano"])
    if "pred_GPT5mini" in df.columns:
        tbl["GPT5-mini"] = series_ratio(df["pred_GPT5mini"])

    out = pd.DataFrame(tbl).reindex(LABELS)
    # 欠けがあっても順序をBAR_ORDERに合わせる（存在しない列は落ちる）
    cols = [c for c in BAR_ORDER if c in out.columns]
    return out[cols]

def plot_6bars_compact(ratio_df: pd.DataFrame, out_png: Path):
    n_bars = len(ratio_df.columns)
    x = np.arange(n_bars)

    # 図サイズをやや小さめに調整（横幅も詰め気味）
    fig, ax = plt.subplots(figsize=(7.5, 4.0))

    # バー幅を広げて隙間を減らす（0.8〜0.9推奨）
    width = 0.85
    bottom = np.zeros(n_bars)

    for lab in ratio_df.index:
        vals = ratio_df.loc[lab].values
        ax.bar(x, vals, width=width, bottom=bottom, label=lab)
        bottom += vals

    ax.set_xticks(x)
    ax.set_xticklabels(ratio_df.columns, rotation=0, fontsize=9)
    ax.set_ylabel("Ratio", fontsize=10)
    ax.set_ylim(0, 1.0)
    ax.set_title("Label/Prediction Distribution (6 bars)", fontsize=11)

    ax.legend(title="Class", fontsize=8, title_fontsize=9,
              bbox_to_anchor=(1.02, 1), loc="upper left", borderaxespad=0.8)

    plt.tight_layout(pad=0.5)  # 余白さらに詰める
    fig.savefig(out_png, dpi=200, bbox_inches="tight")
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in_csv",  required=True)
    ap.add_argument("--out_png", required=True)
    ap.add_argument("--out_tex", required=True)
    ap.add_argument("--out_csv", required=True)
    args = ap.parse_args()

    df = pd.read_csv(args.in_csv, dtype=str)

    # 比率表を作成
    ratio_df = build_table(df)

    # テーブルを保存（TeX/CSV） 小数3桁
    ratio_df.to_csv(args.out_csv, encoding="utf-8-sig")
    with open(args.out_tex, "w", encoding="utf-8") as f:
        f.write(ratio_df.to_latex(escape=True, float_format=lambda v: f"{v:.3f}"))

    # 図を保存（コンパクト）
    plot_6bars_compact(ratio_df, Path(args.out_png))

    print("Saved:")
    print("  Table CSV :", args.out_csv)
    print("  Table TeX :", args.out_tex)
    print("  Figure PNG:", args.out_png)

if __name__ == "__main__":
    main()
