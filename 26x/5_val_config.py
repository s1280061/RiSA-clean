# valフォルダ内のlabelsのクラス分布をカウント
import os
from collections import Counter

val_label_dir = r"C:\Users\s1280\Desktop\SHRP2rawdata\yolo_split_original\val\labels"
class_counter = Counter()

for fname in os.listdir(val_label_dir):
    if fname.endswith(".txt"):
        with open(os.path.join(val_label_dir, fname), 'r') as f:
            for line in f:
                class_id = line.strip().split()[0]
                class_counter[class_id] += 1

print("📊 val クラス別件数:")
for class_id, count in class_counter.items():
    print(f"  {class_id}: {count} 件")
