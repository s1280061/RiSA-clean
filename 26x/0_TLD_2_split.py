import shutil
import random
from pathlib import Path

# === ベースディレクトリ ===
base_dir = Path(r"D:\TLD_data\yolo_dataset")
images_dir = base_dir / "images"
labels_dir = base_dir / "labels"

# === train/val の出力先を作る ===
(train_img_dir, val_img_dir) = (images_dir / "train", images_dir / "val")
(train_lbl_dir, val_lbl_dir) = (labels_dir / "train", labels_dir / "val")
for d in [train_img_dir, val_img_dir, train_lbl_dir, val_lbl_dir]:
    d.mkdir(parents=True, exist_ok=True)

# === 全画像ファイル一覧 ===
all_images = list(images_dir.glob("*.*"))
# jpg/pngのみ残す
all_images = [img for img in all_images if img.suffix.lower() in [".jpg", ".jpeg", ".png"]]

print(f"✅ 画像総数: {len(all_images)}")

# === シャッフルして8:2分割 ===
random.seed(42)  # 再現性確保
random.shuffle(all_images)
split_idx = int(len(all_images) * 0.8)
train_images = all_images[:split_idx]
val_images = all_images[split_idx:]

# === 画像＆ラベル移動 ===
def move_data(image_list, dest_img_dir, dest_lbl_dir):
    for img_path in image_list:
        # ラベルファイル名
        lbl_name = img_path.stem + ".txt"
        lbl_path = labels_dir / lbl_name

        # 移動先
        shutil.move(img_path, dest_img_dir / img_path.name)
        if lbl_path.exists():
            shutil.move(lbl_path, dest_lbl_dir / lbl_name)

move_data(train_images, train_img_dir, train_lbl_dir)
move_data(val_images, val_img_dir, val_lbl_dir)

print(f"✅ train: {len(train_images)} 枚, val: {len(val_images)} 枚に分割しました！")

# === dataset.yaml を修正 ===
yaml_path = base_dir / "dataset.yaml"
with open(yaml_path, "w", encoding="utf-8") as f:
    f.write(f"path: {base_dir}\n")
    f.write("train: images/train\n")
    f.write("val: images/val\n")
    f.write("nc: 6\n")
    f.write("names:\n")
    for name in ["cars","brake","go","left","right","vehicle"]:
        f.write(f"  - {name}\n")

print(f"✅ dataset.yaml を train/val 指定に修正しました！")
