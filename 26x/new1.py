import os
from glob import glob

# === ベースパス ===
base_dir = r"C:\Users\s1280\Desktop\SHRP2rawdata\central_frames_clustering_10\clustered_images"

total_images = 0

print("=== Cluster別の画像枚数 ===")

for idx in range(10):
    cluster_name = f"cluster_{idx:02d}"
    img_dir = os.path.join(base_dir, cluster_name, "images")

    # jpgファイルをカウント
    img_files = glob(os.path.join(img_dir, "*.jpg"))
    count = len(img_files)

    print(f"{cluster_name}: {count} 枚")
    total_images += count

print(f"\n✅ 合計画像枚数: {total_images} 枚")
