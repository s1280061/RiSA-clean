import os
from collections import Counter

def count_labels(label_dir):
    counter = Counter()
    for file in os.listdir(label_dir):
        if file.endswith(".txt"):
            with open(os.path.join(label_dir, file), "r", encoding="utf-8") as f:
                for line in f:
                    class_id = line.strip().split()[0]
                    counter[class_id] += 1
    return counter

base_dir = r"C:\Users\s1280\Desktop\SHRP2rawdata\yolo_augmented"

train_labels = os.path.join(base_dir, "train", "labels")
val_labels = os.path.join(base_dir, "val", "labels")

train_counts = count_labels(train_labels)
val_counts = count_labels(val_labels)

print("=== クラス別ラベル件数 ===")
print("[Train]")
for k, v in sorted(train_counts.items()):
    print(f"  Class {k}: {v} 件")
print("[Val]")
for k, v in sorted(val_counts.items()):
    print(f"  Class {k}: {v} 件")
