# save as: make_table_and_figs.py
# usage (PowerShell):
#   python make_table_and_figs.py `
#     --in_csv "C:\Users\s1280\Desktop\SHRP2rawdata\project_root\model\model_comparison_accuracy.csv" `
#     --out_dir "C:\Users\s1280\Desktop\SHRP2rawdata\project_root\results_report"

import argparse
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

def to_long(df: pd.DataFrame) -> pd.DataFrame:
    cols = set(df.columns.str.strip())
    # Case 1: already long
    needed = {"GT","Model","Accuracy"}
    if needed.issubset(cols):
        if "MacroF1" not in cols:
            df["MacroF1"] = np.nan
        out = df[["GT","Model","Accuracy","MacroF1"]].copy()
        out["GT"] = out["GT"].astype(str)
        out["Model"] = out["Model"].astype(str)
        out["Accuracy"] = pd.to_numeric(out["Accuracy"], errors="coerce")
        out["MacroF1"] = pd.to_numeric(out["MacroF1"], errors="coerce")
        return out.dropna(subset=["GT","Model","Accuracy"])
    # Case 2: wide – try to detect by having GT column + model columns
    # Heuristics: first column is GT; others are models (either Accuracy or MacroF1 per model)
    # We'll support two flavors:
    #  (a) One metric wide (e.g., only Accuracy): columns like ["GT","pred_LLaVA","pred_GPT5nano","pred_GPT5mini"]
    #  (b) Two metrics wide with suffixes: e.g., pred_LLaVA_Acc, pred_LLaVA_MacroF1
    if "GT" in cols:
        long_rows = []
        metric_guess = None
        for c in df.columns:
            if c == "GT": continue
            col = c.strip()
            # try to split "model_metric" style
            if "_" in col:
                left, right = col.rsplit("_", 1)
                if right.lower() in ("acc","accuracy"):
                    metric = "Accuracy"
                    model = left
                elif right.lower() in ("f1","macrof1","macro-f1","macro"):
                    metric = "MacroF1"
                    model = left
                else:
                    # treat as Accuracy by default
                    metric = "Accuracy"
                    model = col
            else:
                metric = "Accuracy"
                model = col
            metric_guess = metric if metric_guess is None else metric_guess
            for _, r in df.iterrows():
                long_rows.append({
                    "GT": str(r["GT"]),
                    "Model": str(model),
                    metric: pd.to_numeric(r[c], errors="coerce")
                })
        tmp = pd.DataFrame(long_rows)
        # pivot back to single row per (GT,Model) with both metrics if available
        acc = tmp[tmp.columns.intersection(["GT","Model","Accuracy"])]
        f1  = tmp[tmp.columns.intersection(["GT","Model","MacroF1"])]
        if "Accuracy" in acc.columns and not acc.dropna(subset=["Accuracy"]).empty:
            acc = acc.groupby(["GT","Model"], as_index=False)["Accuracy"].mean()
        else:
            acc = None
        if "MacroF1" in f1.columns and not f1.dropna(subset=["MacroF1"]).empty:
            f1  = f1.groupby(["GT","Model"], as_index=False)["MacroF1"].mean()
        else:
            f1 = None
        if acc is not None and f1 is not None:
            out = acc.merge(f1, on=["GT","Model"], how="outer")
        elif acc is not None:
            out = acc.copy()
            out["MacroF1"] = np.nan
        elif f1 is not None:
            out = f1.copy()
            out["Accuracy"] = np.nan
        else:
            raise ValueError("Could not detect metrics in wide table.")
        return out.dropna(subset=["GT","Model"])
    raise ValueError("CSV format not recognized. Include columns (GT, Model, Accuracy[, MacroF1]) or a wide format with GT and model columns.")

def save_latex_table(df_long: pd.DataFrame, out_tex: Path):
    # Round for readability
    t = df_long.copy()
    t["Accuracy"] = t["Accuracy"].astype(float).round(3)
    if "MacroF1" in t.columns:
        t["MacroF1"] = t["MacroF1"].astype(float).round(3)
    # Sort by GT then Model
    t = t.sort_values(["GT","Model"])
    with open(out_tex, "w", encoding="utf-8") as f:
        f.write(t.to_latex(index=False, escape=True))

def plot_grouped_bars(pivot_df: pd.DataFrame, ylabel: str, title: str, out_png: Path):
    ax = pivot_df.plot(kind="bar", figsize=(8,5))
    ax.set_title(title)
    ax.set_xlabel("GT (Ground Truth Definition)")
    ax.set_ylabel(ylabel)
    ax.legend(title="Model")
    for p in ax.patches:
        height = p.get_height()
        if np.isfinite(height):
            ax.annotate(f"{height:.2f}", (p.get_x()+p.get_width()/2, height),
                        ha="center", va="bottom", fontsize=8, rotation=0, xytext=(0,2), textcoords="offset points")
    import matplotlib.pyplot as plt
    plt.tight_layout()
    plt.savefig(out_png, dpi=180)
    plt.close()

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in_csv", required=True)
    ap.add_argument("--out_dir", required=True)
    args = ap.parse_args()

    in_csv = Path(args.in_csv)
    out_dir = Path(args.out_dir); out_dir.mkdir(parents=True, exist_ok=True)

    # Load and normalize to long format
    raw = pd.read_csv(in_csv)
    long = to_long(raw)
    # Save normalized
    long_path = out_dir / "model_comparison_long.csv"
    long.to_csv(long_path, index=False, encoding="utf-8-sig")

    # Save LaTeX table
    tex_path = out_dir / "model_comparison_table.tex"
    save_latex_table(long, tex_path)

    # Build bar charts (Accuracy & Macro-F1 if available)
    # Accuracy
    if "Accuracy" in long.columns and not long["Accuracy"].isna().all():
        acc_pivot = long.pivot_table(index="GT", columns="Model", values="Accuracy")
        plot_grouped_bars(acc_pivot, "Accuracy", "Accuracy by Model under Different GT", out_dir / "model_accuracy_bar.png")

    # Macro-F1
    if "MacroF1" in long.columns and not long["MacroF1"].isna().all():
        f1_pivot = long.pivot_table(index="GT", columns="Model", values="MacroF1")
        plot_grouped_bars(f1_pivot, "Macro-F1", "Macro-F1 by Model under Different GT", out_dir / "model_macrof1_bar.png")

    print("Done.")
    print("Normalized CSV:", long_path)
    print("LaTeX table:", tex_path)
    print("Figures:", [p for p in [out_dir / 'model_accuracy_bar.png', out_dir / 'model_macrof1_bar.png'] if p.exists()])

if __name__ == "__main__":
    main()
