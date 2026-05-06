import os
from glob import glob

# === ディレクトリパスを指定 ===
base_dir = r"C:\Users\s1280\Desktop\SHRP2rawdata\yolo_augmented"
train_image_dir = os.path.join(base_dir, "train", "images")
train_label_dir = os.path.join(base_dir, "train", "labels")
val_image_dir = os.path.join(base_dir, "val", "images")
val_label_dir = os.path.join(base_dir, "val", "labels")

def count_files(image_dir, label_dir):
    image_count = len(glob(os.path.join(image_dir, "*.jpg")))
    label_count = len(glob(os.path.join(label_dir, "*.txt")))
    return image_count, label_count

# カウント取得
train_img, train_lbl = count_files(train_image_dir, train_label_dir)
val_img, val_lbl = count_files(val_image_dir, val_label_dir)

# 合計から比率算出
total_img = train_img + val_img
total_lbl = train_lbl + val_lbl

print("📊 データ分割状況:")
print(f"  Train画像数: {train_img}")
print(f"  Val画像数  : {val_img}")
print(f"  Trainラベル数: {train_lbl}")
print(f"  Valラベル数  : {val_lbl}")
print()
print("📈 比率（画像数）:")
print(f"  Train: {train_img / total_img:.2%}")
print(f"  Val  : {val_img / total_img:.2%}")
print()
print("📈 比率（ラベル数）:")
print(f"  Train: {train_lbl / total_lbl:.2%}")
print(f"  Val  : {val_lbl / total_lbl:.2%}")

# 判定メッセージ
def check_ratio(ratio):
    return "✅ OK (約8:2)" if abs(ratio - 0.8) < 0.05 else "⚠️ 比率がズレています"

print("\n🔍 判定:")
print(f"  画像数 → {check_ratio(train_img / total_img)}")
print(f"  ラベル数 → {check_ratio(train_lbl / total_lbl)}")
