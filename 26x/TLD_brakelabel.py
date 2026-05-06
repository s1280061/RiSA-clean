import os
import json
import pandas as pd
import matplotlib.pyplot as plt
from collections import defaultdict

# === ローカルパス設定 ===
base_dir = r"C:\Users\s1280\Desktop\SHRP2rawdata\central_frames_clustering_10\clustered_images"
gt_path = os.path.join(base_dir, "image_level_gt.json")
cluster_files = [os.path.join(base_dir, f"llava_cluster_{i:02d}_results.json") for i in range(10)]
cluster_files = [f for f in cluster_files if os.path.exists(f)]

# === Ground Truth 読み込み ===
with open(gt_path, "r", encoding="utf-8") as f:
    gt_data = json.load(f)
gt_dict = {os.path.basename(item["image"]): item["labels"] for item in gt_data}

# === 統計データ格納
stats = defaultdict(lambda: {"yes": {"correct": 0, "total": 0}, "no": {"correct": 0, "total": 0}})

# === 各クラスタ結果の処理
for file in cluster_files:
    with open(file, "r", encoding="utf-8") as f:
        pred_data = json.load(f)

    for pred in pred_data:
        img_name = os.path.basename(pred["image"])
        gt_labels = gt_dict.get(img_name, {})

        for qa in pred["qa_results"]:
            context = qa["question"].replace("Is this ", "").replace("?", "").strip().lower()
            pred_ans = qa["pred"].strip().lower()
            gt_ans = gt_labels.get(context, "").strip().lower()

            if gt_ans not in ["yes", "no"]:
                continue

            stats[context][pred_ans]["total"] += 1
            if pred_ans == gt_ans:
                stats[context][pred_ans]["correct"] += 1

# === DataFrame作成関数
def make_df(mode):
    df = pd.DataFrame([
        {
            "Context": ctx.upper(),
            f"{mode.upper()} Total": val[mode]["total"],
            f"{mode.upper()} Correct": val[mode]["correct"],
            f"{mode.upper()} Accuracy (%)": round(100 * val[mode]["correct"] / val[mode]["total"], 2)
            if val[mode]["total"] > 0 else 0
        }
        for ctx, val in stats.items()
    ])
    return df.sort_values(by=f"{mode.upper()} Accuracy (%)", ascending=False)

yes_df = make_df("yes")
no_df = make_df("no")

# === CSV保存
yes_df.to_csv(os.path.join(base_dir, "llava_context_accuracy_yes.csv"), index=False)
no_df.to_csv(os.path.join(base_dir, "llava_context_accuracy_no.csv"), index=False)

# === グラフ保存関数
def plot_accuracy_bar(df, mode):
    value_col = f"{mode.upper()} Accuracy (%)"
    total_col = f"{mode.upper()} Total"
    correct_col = f"{mode.upper()} Correct"

    plt.figure(figsize=(16, 8))
    bars = plt.bar(df["Context"], df[value_col], color="lightgreen", edgecolor="black")
    plt.ylim(0, 105)
    plt.xticks(rotation=90, fontsize=9)
    plt.ylabel("Accuracy (%)", fontsize=12, fontname="Times New Roman")
    plt.title(f"LLaVA Context-wise {mode.upper()} Accuracy", fontsize=14, fontweight="bold", fontname="Times New Roman")

    for bar, total, correct in zip(bars, df[total_col], df[correct_col]):
        label = f"{correct}/{total}"
        plt.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 1, label,
                 ha='center', va='bottom', fontsize=8)

    plt.tight_layout()
    plt.savefig(os.path.join(base_dir, f"llava_context_accuracy_{mode}.png"), dpi=200)
    plt.close()

# === PNG保存
plot_accuracy_bar(yes_df, "yes")
plot_accuracy_bar(no_df, "no")

print("✅ すべてのCSVとグラフ(PNG)を保存しました。")
