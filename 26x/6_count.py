import os
from collections import Counter, defaultdict

# === 各ステージのラベルパス ===
original_dir = r"C:\Users\s1280\Desktop\SHRP2rawdata\yolo_split_original\train\labels"
oversampled_dir = r"C:\Users\s1280\Desktop\SHRP2rawdata\yolo_augmented\train\labels"
downsampled_dir = r"C:\Users\s1280\Desktop\SHRP2rawdata\yolo_downsampled\train\labels"

# === クラス名辞書 ===
class_names = {0: "left_on", 1: "right_on", 2: "top_on"}

# === カウント関数 ===
def count_classes(label_dir):
    counter = Counter()
    scene_counter = defaultdict(Counter)
    scene_images = defaultdict(set)

    if not os.path.exists(label_dir):
        print(f"❌ パスが存在しません: {label_dir}")
        return counter, scene_counter, scene_images

    for fname in os.listdir(label_dir):
        if not fname.endswith(".txt"):
            continue
        with open(os.path.join(label_dir, fname), "r", encoding="utf-8") as f:
            for line in f:
                class_id = int(line.strip().split()[0])
                counter[class_id] += 1
                scene_id = "_".join(fname.split("_")[:3])
                scene_counter[scene_id][class_id] += 1
                scene_images[scene_id].add(fname.replace(".txt", ".jpg"))
    return counter, scene_counter, scene_images

# === 各段階のカウント ===
original_count, _, _ = count_classes(original_dir)
oversample_count, _, _ = count_classes(oversampled_dir)
downsample_count, scene_class_counter, scene_image_counter = count_classes(downsampled_dir)

# === 表示関数 ===
def print_stats(title, counts, base_counts=None):
    print(f"【{title}】")
    for i in range(3):
        now = counts.get(i, 0)
        diff = now - base_counts.get(i, 0) if base_counts else 0
        sign = "+" if diff >= 0 else ""
        print(f"  {i} ({class_names[i]}): {now} ({sign}{diff})")
    print("")

# === 出力 ===
print("=== クラス別アノテーション数 ===")
print_stats("初期状態（train）", original_count)
print_stats("オーバーサンプリング後", oversample_count, original_count)
print_stats("ダウンサンプリング後", downsample_count, original_count)

# === シーンごとの詳細 ===
print("\n=== シーンごとの画像数・クラス数（ダウンサンプリング後） ===")
total_images = 0
total_labels = 0
for scene_id, counter in scene_class_counter.items():
    img_count = len(scene_image_counter[scene_id])
    left = counter.get(0, 0)
    right = counter.get(1, 0)
    top = counter.get(2, 0)
    total_images += img_count
    total_labels += left + right + top
    print(f"{scene_id} ({img_count} images): left_on={left}, right_on={right}, top_on={top}")

print(f"\n✅ 合計画像数: {total_images}")
print(f"✅ 合計アノテーション数: {total_labels}")
