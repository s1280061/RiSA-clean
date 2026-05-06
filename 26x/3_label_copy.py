import os
import shutil

# 元の train ディレクトリ
src_image_dir = r"C:\Users\s1280\Desktop\SHRP2rawdata\yolo_split_original\train\images"
src_label_dir = r"C:\Users\s1280\Desktop\SHRP2rawdata\yolo_split_original\train\labels"

# コピー先の train_augmented ディレクトリ
dst_image_dir = r"C:\Users\s1280\Desktop\SHRP2rawdata\yolo_augmented\train\images"
dst_label_dir = r"C:\Users\s1280\Desktop\SHRP2rawdata\yolo_augmented\train\labels"

# ディレクトリ作成
os.makedirs(dst_image_dir, exist_ok=True)
os.makedirs(dst_label_dir, exist_ok=True)

# ファイルコピー関数
def copy_all_files(src_dir, dst_dir):
    count = 0
    for file_name in os.listdir(src_dir):
        src_path = os.path.join(src_dir, file_name)
        dst_path = os.path.join(dst_dir, file_name)
        shutil.copy2(src_path, dst_path)
        count += 1
    return count

# 実行
img_copied = copy_all_files(src_image_dir, dst_image_dir)
label_copied = copy_all_files(src_label_dir, dst_label_dir)

print(f"✅ 画像ファイルを {img_copied} 件コピー")
print(f"✅ ラベルファイルを {label_copied} 件コピー")
print("🎉 train データのコピーが完了しました！")
