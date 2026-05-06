import os
import shutil
import re
import random
from collections import defaultdict

# === 正確な images / labels のパスを明示的に記述 ===
scene_paths = {
    "scene4": {
        "images": r"C:\Users\s1280\Desktop\SHRP2rawdata\4\YOLO\forward\images",
        "labels": r"C:\Users\s1280\Desktop\SHRP2rawdata\4\YOLO\forward\images\labels"
    },
    "scene5": {
        "images": r"C:\Users\s1280\Desktop\SHRP2rawdata\5\YOLO\forward\images",
        "labels": r"C:\Users\s1280\Desktop\SHRP2rawdata\5\YOLO\forward\images\labels"
    },
    "scene6": {
        "images": r"C:\Users\s1280\Desktop\SHRP2rawdata\6\YOLO\forward\images",
        "labels": r"C:\Users\s1280\Desktop\SHRP2rawdata\6\YOLO\forward\images\labels"
    },
}


# === 出力先（train/val） ===
output_base = r"C:\Users\s1280\Desktop\SHRP2rawdata\yolo_split_original"
train_img_dir = os.path.join(output_base, "train", "images")
train_lbl_dir = os.path.join(output_base, "train", "labels")
val_img_dir = os.path.join(output_base, "val", "images")
val_lbl_dir = os.path.join(output_base, "val", "labels")

for d in [train_img_dir, train_lbl_dir, val_img_dir, val_lbl_dir]:
    os.makedirs(d, exist_ok=True)

# === ID単位にグループ化 ===
id_to_files = defaultdict(list)

for scene, paths in scene_paths.items():
    label_dir = paths["labels"]
    image_dir = paths["images"]

    if not os.path.exists(label_dir):
        print(f"❌ ラベルパスが存在しません: {label_dir}")
        continue
    if not os.path.exists(image_dir):
        print(f"❌ 画像パスが存在しません: {image_dir}")
        continue

    for label_file in os.listdir(label_dir):
        if not label_file.endswith(".txt"):
            continue

        match = re.search(r"id_\d+", label_file)
        if not match:
            continue
        car_id = match.group()

        label_path = os.path.join(label_dir, label_file)
        img_file = os.path.splitext(label_file)[0] + ".jpg"
        img_path = os.path.join(image_dir, img_file)

        if os.path.exists(img_path):
            id_to_files[car_id].append((img_path, label_path))

print(f"🔍 車両ID数: {len(id_to_files)}")

# === train/val 分割（8:2）===
all_ids = list(id_to_files.keys())
random.shuffle(all_ids)
split_idx = int(len(all_ids) * 0.8)
train_ids = set(all_ids[:split_idx])
val_ids = set(all_ids[split_idx:])

# === コピー関数 ===
def copy_items(pairs, img_dst, lbl_dst):
    for img_path, lbl_path in pairs:
        shutil.copy(img_path, os.path.join(img_dst, os.path.basename(img_path)))
        shutil.copy(lbl_path, os.path.join(lbl_dst, os.path.basename(lbl_path)))

# === 振り分け実行 ===
for car_id, files in id_to_files.items():
    if car_id in train_ids:
        copy_items(files, train_img_dir, train_lbl_dir)
    else:
        copy_items(files, val_img_dir, val_lbl_dir)

print(f"\n✅ 分割完了: train={len(train_ids)} ID, val={len(val_ids)} ID")
print(f"📁 保存先: {output_base}")

train_count = sum(len(id_to_files[id]) for id in train_ids)
val_count = sum(len(id_to_files[id]) for id in val_ids)

print(f"📸 画像枚数 → train: {train_count}枚, val: {val_count}枚")

