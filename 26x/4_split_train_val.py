import os
import random
import shutil

# === ディレクトリ設定 ===
base_dir = r"C:\Users\s1280\Desktop\SHRP2rawdata\yolo_augmented"
train_img_dir = os.path.join(base_dir, "train", "images")
train_lbl_dir = os.path.join(base_dir, "train", "labels")
val_img_dir = os.path.join(base_dir, "val", "images")
val_lbl_dir = os.path.join(base_dir, "val", "labels")

# === valフォルダを作成 ===
os.makedirs(val_img_dir, exist_ok=True)
os.makedirs(val_lbl_dir, exist_ok=True)

# === _aug を含まない元ファイルだけ抽出 ===
all_labels = [
    f for f in os.listdir(train_lbl_dir)
    if f.endswith(".txt") and "_aug" not in f
]

# === 20%をvalへ移動 ===
val_count = int(len(all_labels) * 0.2)
val_samples = random.sample(all_labels, val_count)

# === 移動処理 ===
for label_file in val_samples:
    image_file = label_file.replace(".txt", ".jpg")

    src_label = os.path.join(train_lbl_dir, label_file)
    src_image = os.path.join(train_img_dir, image_file)
    dst_label = os.path.join(val_lbl_dir, label_file)
    dst_image = os.path.join(val_img_dir, image_file)

    if os.path.exists(src_label):
        shutil.move(src_label, dst_label)
    if os.path.exists(src_image):
        shutil.move(src_image, dst_image)

print(f"✅ Train → Val に {val_count} 件移動しました（※_aug除外）")
