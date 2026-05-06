import os
import random
import shutil
from collections import defaultdict

# パスを修正
label_dir = r"C:\Users\s1280\Desktop\SHRP2rawdata\yolo_augmented\train\labels"
image_dir = r"C:\Users\s1280\Desktop\SHRP2rawdata\yolo_augmented\train\images"

# 削除対象数（必要に応じて調整）
target_max_per_class = 1000

# 各クラスのファイル一覧を収集
class_to_files = defaultdict(list)

for label_file in os.listdir(label_dir):
    if not label_file.endswith(".txt"):
        continue
    path = os.path.join(label_dir, label_file)
    with open(path, "r", encoding="utf-8") as f:
        lines = f.readlines()
        classes = set(line.strip().split()[0] for line in lines)
        for cls in classes:
            class_to_files[cls].append(label_file)

# ダウンサンプリング処理
deleted_labels = 0
deleted_images = 0

for cls, files in class_to_files.items():
    if len(files) > target_max_per_class:
        to_delete = random.sample(files, len(files) - target_max_per_class)
        for f in to_delete:
            label_path = os.path.join(label_dir, f)
            image_path = os.path.join(image_dir, f.replace(".txt", ".jpg"))

            if os.path.exists(label_path):
                os.remove(label_path)
                deleted_labels += 1
            if os.path.exists(image_path):
                os.remove(image_path)
                deleted_images += 1

print(f"✅ ダウンサンプリング完了！")
print(f"🗑 削除されたラベル数: {deleted_labels}")
print(f"🗑 削除された画像数: {deleted_images}")

