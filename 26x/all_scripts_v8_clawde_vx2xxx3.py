import pandas as pd
import os
import shutil
from sklearn.model_selection import train_test_split

# --- 設定 ---
input_csv = r"D:\TLD_data\2_merged_labels\merged_brake_go_labels.csv"
output_dir = r"D:\TLD_data\2_dataset_brake_clean_final"
split_ratio = 0.2  # val の割合

# --- CSV読み込み ---
df = pd.read_csv(input_csv)

# --- Stratified split（クラスの比率を保つ） ---
train_df, val_df = train_test_split(
    df,
    test_size=split_ratio,
    stratify=df['brake_go_label'],
    random_state=42
)

# --- 出力関数 ---
def copy_images(df, subset):
    for label in ['go', 'brake']:
        class_dir = os.path.join(output_dir, subset, label)
        os.makedirs(class_dir, exist_ok=True)

        for _, row in df[df['brake_go_label'] == label].iterrows():
            src = row['image_path']
            dst = os.path.join(class_dir, os.path.basename(src))
            if os.path.exists(src):
                shutil.copy2(src, dst)
            else:
                print(f"⚠️ 画像が存在しません: {src}")

# --- コピー実行 ---
copy_images(train_df, 'train')
copy_images(val_df, 'val')

print("✅ 2_dataset_brake_clean_final データセット構築完了")
print(f"Train: {len(train_df)} 枚 / Val: {len(val_df)} 枚")
