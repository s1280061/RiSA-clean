# make_model_trend_figs_combined.py
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from sklearn.metrics import confusion_matrix

LABELS = ["KEEP_SPEED", "DECELERATE", "CHANGE_LEFT", "CHANGE_RIGHT"]
MODELS = ["pred_LLaVA", "pred_GPT5nano", "pred_GPT5mini"]

def stacked_bar_multi_index(ratio_df: pd.DataFrame, out_png: Path):
    fig, ax = plt.subplots(figsize=(10,6))
    bottom = np.zeros(len(ratio_df))
    x = np.arange(len(ratio_df))
    for lab in LABELS:
        vals = ratio_df[lab].values
        ax.bar(x, vals, bottom=bottom, label=lab)
        bottom += vals
    ax.set_xticks(x)
    ax.set_xticklabels([f"{gt}\n{model}" for gt,model in ratio_df.index], rotation=45, ha='right')
    ax.set_ylabel("Prediction ratio")
    ax.set_title("Prediction distribution by GT and Model")
    ax.legend(title="Predicted class", bbox_to_anchor=(1.02,1), loc="upper left")
    plt.tight_layout()
    plt.savefig(out_png, dpi=180)
    plt.close(fig)

def make_combined_distribution(joined_csv: str, out_png_path: str):
    df = pd.read_csv(joined_csv, dtype=str)
    subsets = {
        "A01": df.dropna(subset=["label_A01"]),
        "A02": df.dropna(subset=["label_A02"]),
        "AGREE": df[df["label_A01"] == df["label_A02"]].dropna(subset=["label_A01","label_A02"])
    }
    all_ratio = []
    for gt_name, sub in subsets.items():
        for m in MODELS:
            if m not in sub.columns: continue
            s = sub[m].dropna().astype(str)
            cnt = s.value_counts().reindex(LABELS, fill_value=0)
            ratio = (cnt / cnt.sum()) if cnt.sum()>0 else cnt
            row = pd.Series(ratio, name=(gt_name,m))
            all_ratio.append(row)
    ratio_df = pd.DataFrame(all_ratio)
    ratio_df.index = pd.MultiIndex.from_tuples(ratio_df.index, names=["GT","Model"])
    stacked_bar_multi_index(ratio_df, Path(out_png_path))
    print("Saved:", out_png_path)
