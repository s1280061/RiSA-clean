import json, os
import matplotlib.pyplot as plt
from collections import Counter

# === JSONがあるディレクトリ ===
base_dir = r"C:\Users\s1280\Desktop\SHRP2rawdata\central_frames_clustering_10\clustered_images"

# === 全24ラベル ===
all_labels = [
    "outdoors", "paved road", "highway", "rural area",
    "daytime", "twilight", "sunny", "foggy",
    "trees overhead", "rainy", "construction zone", "city",
    "bridge", "underpass", "night-time", "indoors",
    "tunnel", "urban canyon", "off road", "lane markers visible",
    "dust/sandstorm", "heavy traffic", "snowy", "parking lot"
]

label_counter = Counter()

# === cluster_00〜09 の JSON集計 ===
for idx in range(10):
    json_file = os.path.join(base_dir, f"llava_cluster_{idx:02d}_results.json")
    if not os.path.exists(json_file):
        print(f"⚠ Missing: {json_file}")
        continue

    with open(json_file, "r", encoding="utf-8") as f:
        data = json.load(f)

    for entry in data:
        for qa in entry["qa_results"]:
            question = qa["question"].replace("Is this ", "").replace("?", "").strip()
            pred = qa["pred"].strip().lower()
            if pred == "yes":
                label_counter[question] += 1

# === ゼロ件ラベルも補完 ===
for lbl in all_labels:
    if lbl not in label_counter:
        label_counter[lbl] = 0

# === 件数の多い順にソート ===
sorted_labels = sorted(label_counter.items(), key=lambda x: x[1], reverse=True)
labels = [k.upper() for k, v in sorted_labels]  # 大文字化
counts = [v for k, v in sorted_labels]

# === 黄金比で描画 ===
plt.figure(figsize=(12,7.4))  # 黄金比
bars = plt.bar(range(len(labels)), counts, color="#4682B4", edgecolor="black", width=0.6)

# ❌ 個別のX軸ラベルは表示しない
plt.xticks([])

# 1200だけを除外してY軸の目盛りを設定
max_count = max(counts)
y_ticks = [i for i in range(0, int(max_count * 1.2) + 200, 200) if i != 1200]
plt.yticks(y_ticks)

# X軸には1つだけ「Contexts」と書く
plt.xlabel("Contexts", fontsize=12)
plt.ylabel("Counts")
plt.title("Long-tail Distribution of Contexts", fontweight='bold')

# === Y軸の余白を大きめに確保 ===
plt.ylim(0, max(counts) * 1.2)  # 上に20%余白

# === 右上に全画像枚数を表示 ===
plt.text(0.98, 0.98, 'Total Images: 1033',
         transform=plt.gca().transAxes, fontsize=10,
         ha='right', va='top')

# === バー上に "LABEL: COUNT" を縦に表示 ===
for i, bar in enumerate(bars):
    height = bar.get_height()
    plt.text(
        bar.get_x() + bar.get_width()/2,
        height + max(counts)*0.02,  # 余白を少し多めに
        f"{labels[i]}: {counts[i]}",
        ha='center', va='bottom', fontsize=8, rotation=90
    )

plt.tight_layout()

# 保存
out_path = os.path.join(base_dir, "longtail_distribution_contexts_final.png")
plt.savefig(out_path, dpi=200)
plt.close()

print(f"✅ Final clean long-tail distribution saved to {out_path}")