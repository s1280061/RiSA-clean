import cv2
import csv
import os

# === CSVとcrop画像フォルダ ===
csv_path = r"D:/TLD_data/loki_crops/labels.csv"
crop_dir = r"D:/TLD_data/loki_crops/crops"

# === CSV読み込み ===
rows = []
with open(csv_path, "r", encoding="utf-8") as f:
    reader = csv.DictReader(f)
    for r in reader:
        if r["turn_label"] == "left":  # leftだけ抽出
            rows.append(r)

print(f"✅ leftラベルは {len(rows)} 枚あります")

# === 画像ビューア ===
for i, r in enumerate(rows):
    img_path = os.path.join(crop_dir, r["filename"])
    if not os.path.exists(img_path):
        print(f"⚠ 画像が見つからない: {img_path}")
        continue

    img = cv2.imread(img_path)
    if img is None:
        continue

    # ラベル情報を画像に書く
    text1 = f"turn: {r['turn_label']}  brake: {r['brake_label']}"
    text2 = f"source: {r['source']}  orig: {r['original_image']}"
    cv2.putText(img, text1, (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,255,0), 2)
    cv2.putText(img, text2, (10, 55), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,255,255), 2)

    # ウィンドウ表示
    cv2.imshow("LEFT LABEL CHECK", img)
    key = cv2.waitKey(0)

    # 終了判定
    if key == ord('q'):
        break

cv2.destroyAllWindows()
