import cv2
import os
import numpy as np
import pandas as pd
from ultralytics import YOLO
from yolox.tracker.byte_tracker import BYTETracker
import types

# === 入力・保存パス ===
video_path = r"C:\Users\s1280\PycharmProjects\yolo_classify_project\input_video.mp4"
save_base_dir = r"C:\Users\s1280\Desktop\SHRP2rawdata\merged_images"
os.makedirs(save_base_dir, exist_ok=True)

# === YOLO モデルロード ===
model = YOLO("yolov8x.pt").to("cuda")
target_classes = [2, 7]  # car, truck

# === ByteTrack 初期化 ===
args = types.SimpleNamespace(
    track_thresh=0.3,
    match_thresh=0.7,
    track_buffer=30,
    frame_rate=15,
    mot20=False
)
tracker = BYTETracker(args)

# === 動画読み込み ===
cap = cv2.VideoCapture(video_path)
if not cap.isOpened():
    print("❌ 動画が読み込めません。パスを確認してください。")
    exit()

frame_idx = 0
resize_size = (128, 128)
padding = 5  # 拡張ピクセル数

# === track_idごとのフレーム記録 ===
track_frame_info = {}

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    results = model(frame, verbose=False)[0]
    detections = []
    for box in results.boxes:
        cls = int(box.cls[0].item())
        if cls in target_classes:
            x1, y1, x2, y2 = box.xyxy[0].tolist()
            score = box.conf[0].item()
            detections.append([x1, y1, x2, y2, score])

    print(f"🟡 Frame {frame_idx:05d} — YOLO detections: {len(detections)}")

    if detections:
        detections_np = np.array(detections, dtype=np.float32)
        online_targets = tracker.update(detections_np, frame.shape[:2], frame.shape[:2])
        print(f"🟢 ByteTrack targets: {len(online_targets)}")

        for t in online_targets:
            track_id = t.track_id
            x, y, w, h = map(int, t.tlwh)

            # === +5px 拡張 & 範囲制限
            x = max(0, x - padding)
            y = max(0, y - padding)
            x2 = min(frame.shape[1], x + w + 2 * padding)
            y2 = min(frame.shape[0], y + h + 2 * padding)

            crop = frame[y:y2, x:x2]
            h_crop, w_crop = crop.shape[:2]
            if w_crop < 10 or h_crop < 20:
                print(f"    ⚠️ 小さすぎる切り出し（{w_crop}x{h_crop}）→ スキップ")
                continue

            crop_resized = cv2.resize(crop, resize_size, interpolation=cv2.INTER_LINEAR)

            # === フォルダを作らず、直接保存 ===
            save_path = os.path.join(
                save_base_dir,
                f"7_id_{track_id}_frame_{frame_idx:05d}.jpg"
            )
            cv2.imwrite(save_path, crop_resized)
            print(f"    ✅ Saved: {save_path} | 元サイズ: {w_crop}x{h_crop} → リサイズ: {resize_size[0]}x{resize_size[1]}")

            # === 開始・終了フレーム管理 ===
            if track_id not in track_frame_info:
                track_frame_info[track_id] = {"start": frame_idx, "end": frame_idx}
            else:
                track_frame_info[track_id]["end"] = frame_idx

    else:
        print(f"🔴 ByteTrackスキップ: detectionsなし")

    frame_idx += 1

cap.release()

# === CSV出力のみ（フォルダ名変更は不要） ===
csv_path = os.path.join(save_base_dir, "track_info.csv")
pd.DataFrame([
    {"track_id": track_id, "start_frame": info["start"], "end_frame": info["end"]}
    for track_id, info in track_frame_info.items()
]).to_csv(csv_path, index=False)

print("✅ 完了：画像保存（直置き）＆ track_info.csv 作成完了。")
