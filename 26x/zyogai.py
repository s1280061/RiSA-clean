import os
import json
import matplotlib.pyplot as plt
from collections import defaultdict
import pandas as pd

# === パスの設定 ===
gt_path = r"C:\Users\s1280\Desktop\SHRP2rawdata\central_frames_clustering_10\clustered_images\image_level_gt.json"
pred_dir = r"C:\Users\s1280\Desktop\SHRP2rawdata\central_frames_clustering_10\clustered_images"
cluster_ids = [f"{i:02}" for i in range(10)]  # cluster_00 ～ cluster_09

# === 正解ラベルの読み込み ===
with open(gt_path, "r", encoding="utf-8") as f:
    gt_data = json.load(f)

# === ground truth を辞書化: image_name -> label dict ===
gt_dict = {os.path.basename(item["image"]): item["labels"] for item in gt_data}

# === contextごとの正解数/全体数を記録 ===
context_stats = defaultdict(lambda: {"correct": 0, "total": 0})

# === 各クラスタの予測結果を処理 ===
for cid in cluster_ids:
    pred_path = os.path.join(pred_dir, f"llava_cluster_{cid}_results.json")
    if not os.path.exists(pred_path):
        print(f"⚠️ ファイルが見つかりません: {pred_path}")
        continue

    with open(pred_path, "r", encoding="utf-8") as f:
        pred_data = json.load(f)

    for pred in pred_data:
        img_name = pred["image"]
        gt_labels = gt_dict.get(img_name)
        if not gt_labels:
            continue

        for qa in pred["qa_results"]:
            context = qa["question"].replace("Is this ", "").replace("?", "").lower().strip()
            pred_ans = qa["pred"].lower()
            gt_ans = gt_labels.get(context)

            if not gt_ans or gt_ans == "ambiguous":
                continue

            context_stats[context]["total"] += 1
            if pred_ans == gt_ans:
                context_stats[context]["correct"] += 1

# === Accuracy計算 ===
accuracy_by_context = {
    ctx: 100 * val["correct"] / val["total"]
    for ctx, val in context_stats.items()
    if val["total"] > 0
}

# === DataFrameでソートして表示用に準備 ===
df_accuracy = pd.DataFrame([
    {"Context": k, "Accuracy (%)": round(v, 2)}
    for k, v in sorted(accuracy_by_context.items(), key=lambda x: x[1], reverse=True)
])

# === グラフの描画（最終改良版） ===
plt.figure(figsize=(16, 8))

bars = plt.bar(df_accuracy["Context"], df_accuracy["Accuracy (%)"],
               color="lightgreen", width=0.8, edgecolor='darkgreen', linewidth=0.5)

# xticks は消す（棒内に描画するため）
plt.xticks([])

plt.ylim(0, 105)
plt.xlabel("Contexts", fontsize=12)
plt.ylabel("Accuracy (%)", fontsize=12, labelpad=20, loc='center', fontname='Times New Roman')
plt.title("LLaVA: Context-wise Accuracy (All Clusters)", fontsize=16, fontweight='bold', fontname='Times New Roman')

# 各バーごとにラベルを描画
for i, (bar, row) in enumerate(zip(bars, df_accuracy.itertuples())):
    height = row._2  # _2 is "Accuracy (%)"
    x_pos = bar.get_x() + bar.get_width() / 2

    # Accuracy (%)（縦書き、バー中央）
    plt.text(x_pos, height / 2, f"{height:.1f}%",
             ha='center', va='center',
             fontsize=9, fontweight='bold',
             color='black', rotation=90,
             fontname='Times New Roman')

    # Context名（縦書き、バー下部、全大文字）
    label_text = str(row.Context).upper()
    plt.text(x_pos, 2, label_text,
             ha='center', va='bottom',
             fontsize=9, color='black',
             rotation=90, fontname='Times New Roman')

# グリッド・レイアウト
plt.grid(axis='y', linestyle='--', alpha=0.3)
plt.tight_layout()
plt.subplots_adjust(bottom=0.18)

plt.show()
