import pandas as pd
import matplotlib.pyplot as plt

# === CSV読み込み ===
csv_path = r"D:\train_YT_100epochs\4\turn_test_results\summary_per_video.csv"
df = pd.read_csv(csv_path)

# === car と truck のみに絞る ===
df = df[df["vehicle_type"].isin(["car", "truck"])]

# === 全動画で車種ごとに合計 ===
agg = df.groupby("vehicle_type")[["go","brake","left","right"]].sum()

# --- car/truck それぞれ頻度多い順で並べる ---
car_sorted = agg.loc["car"].sort_values(ascending=False)
truck_sorted = agg.loc["truck"].sort_values(ascending=False)

# --- x軸ラベル（go, right, left, brake...） ---
labels = list(car_sorted.index) + list(truck_sorted.index)
values = list(car_sorted.values) + list(truck_sorted.values)
colors = ["blue"] * len(car_sorted) + ["red"] * len(truck_sorted)

x = range(len(labels))

# === プロット ===
plt.figure(figsize=(5,3))  # 小さめ

# バー表示（ほぼ詰める）
plt.bar(x, values, color=colors, width=0.95)

# x軸ラベル（車種名なし）
plt.xticks(x, labels)

plt.ylabel("Frequency")
plt.title("Signal Frequency (car/truck sorted)")
plt.grid(axis="y", linestyle="--", alpha=0.6)

# 凡例を右上に追加
legend_handles = [
    plt.Rectangle((0,0),1,1,color="blue",label="car"),
    plt.Rectangle((0,0),1,1,color="red",label="truck")
]
plt.legend(handles=legend_handles, loc="upper right", fontsize=8)

plt.tight_layout()

# === 小さめ画像を保存 ===
plt.savefig(r"D:\train_YT_100epochs\4\turn_test_results\signal_frequency_curve.png", dpi=200)
plt.show()

print("✅ グラフを小さめで保存しました！")
