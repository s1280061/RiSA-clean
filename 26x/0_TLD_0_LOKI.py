import json
import os
from collections import Counter

def scan_loki_coco_labels(root_dir):
    label_counter = Counter()
    category_map = {}

    # ディレクトリ走査
    for root, _, files in os.walk(root_dir):
        for file in files:
            if file.endswith(".json") and "coco" in file:
                json_path = os.path.join(root, file)
                with open(json_path, "r", encoding="utf-8") as f:
                    data = json.load(f)

                # categories マップ作成（id→name）
                for cat in data.get("categories", []):
                    category_map[cat["id"]] = cat["name"]

                # annotations を集計
                for ann in data.get("annotations", []):
                    cat_id = ann.get("category_id", -1)
                    label_name = category_map.get(cat_id, f"unknown_{cat_id}")
                    label_counter[label_name] += 1

    print("=== LOKIカテゴリの出現回数 ===")
    for lbl, cnt in label_counter.most_common():
        print(f"{lbl:20s} : {cnt}")

# 使い方
root_dir = r"D:\TLD_data\TLD-LOKI\TLD-LOKI\group_001"  # group_001フォルダ
scan_loki_coco_labels(root_dir)
