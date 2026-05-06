import os
import cv2
from ultralytics import YOLO

# =====================================================
# 設定
# =====================================================
IMAGE_PATH = r"C:\Users\s1280\Desktop\SHRP2rawdata\5\frames_bottom_left\frame_08758.jpg"
OUTPUT_DIR = r"C:\Users\s1280\Desktop\SHRP2rawdata\5\cropped"
MODEL = "yolov8n.pt"  # 小型で高速
# =====================================================

# 出力フォルダ作成
os.makedirs(OUTPUT_DIR, exist_ok=True)

# モデル読み込み
model = YOLO(MODEL)

# 画像読み込み
img = cv2.imread(IMAGE_PATH)
if img is None:
    raise FileNotFoundError(f"Image not found: {IMAGE_PATH}")

# 推論
results = model(IMAGE_PATH)[0]

crop_index = 0

for box in results.boxes:
    cls = int(box.cls[0])  # クラスID
    conf = float(box.conf[0])  # 確信度

    # COCOクラスID 2=car, 7=truck
    if cls not in [2, 7]:
        continue

    # バウンディングボックス座標取得
    x1, y1, x2, y2 = box.xyxy[0].cpu().numpy().astype(int)

    # Crop
    crop_img = img[y1:y2, x1:x2]

    # 保存パス
    save_path = os.path.join(OUTPUT_DIR, f"crop_{crop_index}_cls{cls}_conf{conf:.2f}.jpg")

    # 画像保存
    cv2.imwrite(save_path, crop_img)
    crop_index += 1

    print(f"Saved: {save_path}")

print("Completed.")
