import pandas as pd
import numpy as np
from tqdm import tqdm
import os

# === パラメータ設定 ===
csv_path = r"C:\Users\s1280\Desktop\SHRP2rawdata\3\LSTMdataset_new_1\3_final_3_filled_linear_by_id.csv"
output_dir = r"C:\Users\s1280\Desktop\SHRP2rawdata\3\LSTMdataset_new_1"
input_len = 10     # LSTMの入力ステップ数
output_len = 15    # 予測ステップ数
min_len = input_len + output_len

# === CSV読み込み ===
df = pd.read_csv(csv_path)

# === 有効なID（50フレーム以上）を抽出 ===
id_counts = df['id'].value_counts()
valid_ids = id_counts[id_counts >= 50].index
df_filtered = df[df['id'].isin(valid_ids)]

# === X, Y の整形処理 ===
X_list, Y_list = [], []

for track_id, group in tqdm(df_filtered.groupby("id"), desc="Processing track_ids"):
    group_sorted = group.sort_values("frame")
    coords = group_sorted[["x", "y"]].values

    for i in range(len(coords) - min_len + 1):
        x_seq = coords[i : i + input_len]
        y_seq = coords[i + input_len : i + input_len + output_len]
        X_list.append(x_seq)
        Y_list.append(y_seq)

# === numpy配列に変換して保存 ===
X = np.array(X_list)  # (N, 10, 2)
Y = np.array(Y_list)  # (N, 15, 2)

np.save(os.path.join(output_dir, "lstm_X.npy"), X)
np.save(os.path.join(output_dir, "lstm_Y.npy"), Y)

print("✅ 保存完了:")
print(" - 入力 X.shape:", X.shape)
print(" - 出力 Y.shape:", Y.shape)
print(" - 保存先:", output_dir)
