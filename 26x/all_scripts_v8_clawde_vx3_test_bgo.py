import cv2
import torch
from ultralytics import YOLO
import numpy as np
import os
import csv

# === モデルパス ===
vehicle_detector_path = "yolov8n.pt"  # 車検出用 (COCO)
turn_model_path = r"C:\Users\s1280\PycharmProjects\yolo_classify_project\runs\classify\train2\weights\best.pt"

# === ディレクトリ設定 ===
input_dir = r"C:\Users\s1280\Desktop\SHRP2rawdata\3\new_divided"
output_dir = r"D:\train_YT_100epochs\3\turn_test_results"  # ★ここを変える
os.makedirs(output_dir, exist_ok=True)

csv_path = os.path.join(output_dir, "summary_per_video.csv")


# === モデルロード ===
vehicle_model = YOLO(vehicle_detector_path)
turn_model = YOLO(turn_model_path)

COCO_NAMES = vehicle_model.names  # 車種名取得用

# === 推論クラス名 ===
TURN_CLASSES = ["brake", "go", "left", "right"]
THRESH_OFF = 0.7  # この閾値未満ならOFF扱い

# === CSVのヘッダ ===
csv_header = ["video", "vehicle_type", "brake", "go", "left", "right", "off", "total"]
csv_summary = []  # 各動画×車種の集計結果を保存

def process_video(video_path, output_path):
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"⚠️ 動画を開けません: {video_path}")
        return []

    fps = cap.get(cv2.CAP_PROP_FPS)
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out = cv2.VideoWriter(output_path, fourcc, fps, (w, h))

    # === 車種別のカウンタ辞書 ===
    counts_per_type = {}

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # === 1. 車両検出 ===
        vehicle_results = vehicle_model(frame, verbose=False)[0]
        vehicle_boxes = []
        for box in vehicle_results.boxes:
            cls_id = int(box.cls)
            # COCO: car=2, motorcycle=3, bus=5, truck=7
            if cls_id in [2, 3, 5, 7]:
                x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
                cls_name = COCO_NAMES[cls_id]
                vehicle_boxes.append((x1, y1, x2, y2, cls_name))

        # === 2. 車ごとにクロップしてturnモデル判定 ===
        for (x1, y1, x2, y2, cls_name) in vehicle_boxes:
            crop = frame[y1:y2, x1:x2]
            if crop.size == 0:
                continue

            # 4クラスモデルで推論
            turn_res = turn_model.predict(crop, verbose=False)[0]

            if cls_name not in counts_per_type:
                # 初期化（車種ごとの集計辞書）
                counts_per_type[cls_name] = {"brake":0, "go":0, "left":0, "right":0, "off":0}

            if len(turn_res.boxes) > 0:
                best_idx = turn_res.boxes.conf.argmax()
                pred_cls = int(turn_res.boxes.cls[best_idx])
                conf = turn_res.boxes.conf[best_idx].item()

                if conf < THRESH_OFF:
                    label = f"OFF ({conf:.2f})"
                    counts_per_type[cls_name]["off"] += 1
                else:
                    label_name = TURN_CLASSES[pred_cls]
                    label = f"{label_name} ({conf:.2f})"
                    counts_per_type[cls_name][label_name] += 1

            else:
                label = "OFF (no-detect)"
                counts_per_type[cls_name]["off"] += 1

            # === 結果描画 ===
            cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 0, 0), 2)
            cv2.putText(frame, f"{cls_name}:{label}", (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)

        out.write(frame)

    cap.release()
    out.release()

    # === 1動画分の結果をまとめる ===
    summary_rows = []
    for vtype, counts in counts_per_type.items():
        total = sum(counts.values())
        summary_rows.append([
            os.path.basename(video_path),
            vtype,
            counts["brake"],
            counts["go"],
            counts["left"],
            counts["right"],
            counts["off"],
            total
        ])

    return summary_rows

# === すべての動画を走査 ===
for i in range(0, 188):
    filename = f"scene_{i:03d}.mp4"
    video_path = os.path.join(input_dir, filename)

    if not os.path.exists(video_path):
        print(f"⚠️ ファイルが存在しません: {video_path}")
        continue

    output_path = os.path.join(output_dir, f"scene_{i:03d}_turn_test.mp4")
    print(f"▶️ 処理開始: {video_path}")

    video_results = process_video(video_path, output_path)

    # 1動画内で複数車種の集計があるのでまとめて追加
    csv_summary.extend(video_results)

cv2.destroyAllWindows()
print("🎉 すべての動画処理が完了しました！")

# === CSVに保存 ===
with open(csv_path, mode="w", newline="", encoding="utf-8") as f:
    writer = csv.writer(f)
    writer.writerow(csv_header)
    writer.writerows(csv_summary)

print(f"✅ CSVに集計結果を保存しました: {csv_path}")

