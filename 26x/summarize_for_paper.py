# -*- coding: utf-8 -*-
"""
Summarize RiSA judge results for paper-ready tables & plots (scores only).
"""

import os, csv, argparse, math
from statistics import mean
import matplotlib.pyplot as plt
import pandas as pd

# 評価対象カラム（コスト類は除外）
NUM_COLS = [
    "situation_accuracy",
    "advice_appropriateness",
    "safety_risk_calibration",
    "overall",
]

PRETTY_NAMES = {
    "situation_accuracy": "Situation Acc.",
    "advice_appropriateness": "Advice",
    "safety_risk_calibration": "Risk Align.",
    "overall": "Overall",
}

def read_rows(path):
    if not os.path.exists(path):
        return []
    with open(path, "r", encoding="utf-8") as f:
        rd = csv.DictReader(f)
        return list(rd)

def to_float_safe(x):
    try:
        v = float(x)
        return v if math.isfinite(v) else None
    except:
        return None

def summarize_model(name, rows):
    """Compute counts & means for NUM_COLS."""
    out = {"Model": name, "Samples": len(rows)}
    for c in NUM_COLS:
        vals = [to_float_safe(r.get(c, "")) for r in rows]
        vals = [v for v in vals if v is not None]
        out[f"mean_{c}"] = round(mean(vals), 3) if vals else ""
    return out

def save_table_as_png(df, out_path):
    fig, ax = plt.subplots(figsize=(10, 2 + len(df) * 0.4))
    ax.axis("off")

    table = ax.table(
        cellText=df.values,
        colLabels=df.columns,
        cellLoc="center",
        loc="center"
    )
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1.2, 1.2)

    plt.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"🖼  Saved: {out_path}")

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", default=r"C:\Users\s1280\Desktop\SHRP2rawdata\for_GPT")
    ap.add_argument("--gpt5", default="results_gpt-5.csv")
    ap.add_argument("--mini", default="results_gpt-5-mini.csv")
    ap.add_argument("--nano50", default="results_gpt-5-nano.csv")
    ap.add_argument("--nano_all", default="nano_all_results.csv")
    args = ap.parse_args()

    root = args.dir
    rows_gpt5   = read_rows(os.path.join(root, args.gpt5))
    rows_mini   = read_rows(os.path.join(root, args.mini))
    rows_nano50 = read_rows(os.path.join(root, args.nano50))
    rows_nanoall= read_rows(os.path.join(root, args.nano_all))

    if not any([rows_gpt5, rows_mini, rows_nano50, rows_nanoall]):
        print("❌ No input rows. Check file paths.")
        return

    summaries = [
        summarize_model("gpt-5", rows_gpt5),
        summarize_model("gpt-5-mini", rows_mini),
        summarize_model("gpt-5-nano (50)", rows_nano50),
        summarize_model("gpt-5-nano (all)", rows_nanoall),
    ]

    # ---- CSV出力 ----
    out_csv = os.path.join(root, "paper_summary_by_model.csv")
    header = ["Model","Samples"] + [f"mean_{c}" for c in NUM_COLS]
    with open(out_csv, "w", newline="", encoding="utf-8") as f:
        wr = csv.DictWriter(f, fieldnames=header)
        wr.writeheader()
        wr.writerows(summaries)
    print(f"✅ Wrote summary: {out_csv}")

    # ---- 表をPNG化 ----
    df = pd.DataFrame(summaries)
    df = df.rename(columns={f"mean_{c}": PRETTY_NAMES[c] for c in NUM_COLS})
    table_png = os.path.join(root, "paper_summary_table.png")
    save_table_as_png(df, table_png)

    # ---- 棒グラフ ----
    labels = df["Model"]
    means_overall = df["Overall"]
    plt.figure()
    plt.bar(labels, means_overall, color="skyblue")
    plt.ylabel("Mean Overall Score")
    plt.title("Mean Overall by Model")
    plt.grid(axis="y", linestyle="--", alpha=0.6)
    plt.xticks(rotation=15, ha="right")
    plt.tight_layout()
    bar_path = os.path.join(root, "bar_overall_mean.png")
    plt.savefig(bar_path, dpi=300)
    plt.close()
    print(f"🖼  Saved: {bar_path}")

    # ---- 箱ひげ図 ----
    data_50 = [
        [to_float_safe(r.get("overall")) for r in rows_nano50 if to_float_safe(r.get("overall")) is not None],
        [to_float_safe(r.get("overall")) for r in rows_mini   if to_float_safe(r.get("overall")) is not None],
        [to_float_safe(r.get("overall")) for r in rows_gpt5   if to_float_safe(r.get("overall")) is not None],
    ]
    labels_50 = ["gpt-5-nano (50)","gpt-5-mini","gpt-5"]
    plt.figure()
    plt.boxplot(data_50, labels=labels_50, showmeans=True, meanprops={"marker":"o","markerfacecolor":"red"})
    plt.ylabel("Overall Score")
    plt.title("Score Distribution (50-case set)")
    plt.grid(axis="y", linestyle="--", alpha=0.6)
    plt.xticks(rotation=15, ha="right")
    plt.tight_layout()
    box_path = os.path.join(root, "box_overall_50.png")
    plt.savefig(box_path, dpi=300)
    plt.close()
    print(f"🖼  Saved: {box_path}")

    print("✅ Done. Tables & plots ready for paper (scores only).")

if __name__ == "__main__":
    main()
