import cv2
import numpy as np
import pandas as pd
import os
import types
from ultralytics import YOLO
from yolox.tracker.byte_tracker import BYTETracker

# === パス設定 ===
video_path = r"C:\Users\s1280\Desktop\SHRP2rawdata\5\new\5_combined_frame_overlay_1.mp4"
output_video_path = video_path.replace(".mp4", "_combined_speed_yolo.mp4")
output_csv_path = video_path.replace(".mp4", "_trajectories_combined.csv")
csv_path = r"C:\Users\s1280\Desktop\SHRP2rawdata\5\original\5_acc_speed_filled_complete.csv"
base_path = r"C:\Users\s1280\PycharmProjects\yolo_classify_project\26x"

# === BEV変換用の座標読み込み ===
src = np.load(os.path.join(base_path, "src_points_3_forward.npy"))
dst = np.load(os.path.join(base_path, "dst_points_3_forward.npy"))
M = cv2.getPerspectiveTransform(src, dst)
Minv = cv2.getPerspectiveTransform(dst, src)
scale_px_per_m = 50.82

# === 速度・加速度データ読み込み ===
df = pd.read_csv(csv_path, dtype={"frame": int}, low_memory=False)

# === ByteTrack 初期化 ===
args = types.SimpleNamespace()
args.track_thresh = 0.3
args.match_thresh = 0.7
args.track_buffer = 30
args.frame_rate = 15
args.mot20 = False
tracker = BYTETracker(args)

# === YOLO モデルロード（GPU使用）===
model = YOLO("yolov8x.pt").to("cuda")
target_classes = [2, 7]  # car, truck
class_names = {2: "car", 7: "truck"}

# === 動画設定 ===
cap = cv2.VideoCapture(video_path)
fps = int(cap.get(cv2.CAP_PROP_FPS))
frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
out = cv2.VideoWriter(output_video_path, cv2.VideoWriter_fourcc(*"mp4v"), fps, (frame_width * 2, frame_height * 2))

# === 停止距離計算関数 ===
def compute_brake_distance(df, frame_idx, fixed_amax=6.0):
    speed_row = df.loc[df["frame"] == frame_idx, "vtti.speed_gps"]
    if speed_row.empty:
        print(f"\u26a0\ufe0f frame {frame_idx}：速度データなし → 停止距離 = 1.0 m【仮】")
        return 1.0
    v_now = speed_row.values[0] / 3.6
    return (v_now ** 2) / (2 * fixed_amax)

# === 初期化 ===
frame_idx = 0
trajectories = []
track_history = {}
print("\u25b6 統合処理を開始します...")

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    fixed_m = compute_brake_distance(df, frame_idx)
    bev_height = int(10 * scale_px_per_m)

    # === BEV & Canny ===
    bev = cv2.warpPerspective(frame, M, (360, bev_height))
    # === BEVマスク処理（左右直角三角形）===
    mask = np.ones((bev_height, 360), dtype=np.uint8) * 255
    x_ratio, y_ratio = 0.25, 0.7

    left_tri = np.array([
        [0, bev_height],
        [int(360 * x_ratio), bev_height],
        [0, int(bev_height * (1 - y_ratio))]
    ], np.int32)

    right_tri = np.array([
        [360, bev_height],
        [int(360 * (1 - x_ratio)), bev_height],
        [360, int(bev_height * (1 - y_ratio))]
    ], np.int32)

    cv2.fillPoly(mask, [left_tri], 0)
    cv2.fillPoly(mask, [right_tri], 0)

    bev_masked = cv2.bitwise_and(bev, bev, mask=mask)
    gray = cv2.cvtColor(bev_masked, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    edges = cv2.Canny(blur, 15, 30)
    binary = (edges > 0).astype(np.uint8)
    canny_vis = cv2.cvtColor(binary * 255, cv2.COLOR_GRAY2BGR)

    histogram = np.sum(binary[binary.shape[0] // 2:, :], axis=0)
    midpoint = histogram.shape[0] // 2
    leftx_base = np.argmax(histogram[:midpoint])
    rightx_base = np.argmax(histogram[midpoint:]) + midpoint

    sliding_vis = np.dstack([binary * 255] * 3)
    n_windows = 30
    margin = 40
    minpix = 30
    window_height = binary.shape[0] // n_windows

    nonzeroy, nonzerox = binary.nonzero()
    leftx_current, rightx_current = leftx_base, rightx_base
    left_trace, right_trace = [], []

    for window in range(n_windows):
        win_y_low = binary.shape[0] - (window + 1) * window_height
        win_y_high = binary.shape[0] - window * window_height
        win_xleft_low = leftx_current - margin
        win_xleft_high = leftx_current + margin
        win_xright_low = rightx_current - margin
        win_xright_high = rightx_current + margin

        cv2.rectangle(sliding_vis, (win_xleft_low, win_y_low), (win_xleft_high, win_y_high), (0, 255, 0), 2)
        cv2.rectangle(sliding_vis, (win_xright_low, win_y_low), (win_xright_high, win_y_high), (0, 255, 0), 2)

        good_left_inds = ((nonzeroy >= win_y_low) & (nonzeroy < win_y_high) &
                          (nonzerox >= win_xleft_low) & (nonzerox < win_xleft_high)).nonzero()[0]
        good_right_inds = ((nonzeroy >= win_y_low) & (nonzeroy < win_y_high) &
                           (nonzerox >= win_xright_low) & (nonzerox < win_xright_high)).nonzero()[0]

        if len(good_left_inds) > minpix:
            leftx_current = int(np.mean(nonzerox[good_left_inds]))
        if len(good_right_inds) > minpix:
            rightx_current = int(np.mean(nonzerox[good_right_inds]))

        left_center = (leftx_current, (win_y_low + win_y_high) // 2)
        right_center = (rightx_current, (win_y_low + win_y_high) // 2)
        left_trace.append(left_center)
        right_trace.append(right_center)

    def draw_polynomial_curve(vis_img, points, color=(255, 255, 0), thickness=2, degree=2):
        if len(points) < degree + 1:
            return
        points = np.array(points)
        x = points[:, 0]
        y = points[:, 1]
        coeffs = np.polyfit(y, x, degree)
        y_fit = np.linspace(min(y), max(y), num=100)
        x_fit = np.polyval(coeffs, y_fit)
        curve_pts = np.array([[int(x), int(y)] for x, y in zip(x_fit, y_fit)], dtype=np.int32)
        cv2.polylines(vis_img, [curve_pts], isClosed=False, color=color, thickness=thickness)

    draw_polynomial_curve(sliding_vis, left_trace)
    draw_polynomial_curve(sliding_vis, right_trace)

    # === 可変ポリゴン定義（BEVで作成 → 画像に逆射影）===
    fixed_length_px_bev = int(fixed_m * scale_px_per_m)  # 停止距離[m]→BEVピクセル
    fixed_length_px_bev = min(fixed_length_px_bev, bev_height)  # 上限 508px

    bottom_y_bev = bev_height - 1
    top_y_bev = bev_height - fixed_length_px_bev

    # 左右レーン中心の 2 次曲線係数を取得
    left_coef = np.polyfit([p[1] for p in left_trace], [p[0] for p in left_trace], 2)
    right_coef = np.polyfit([p[1] for p in right_trace], [p[0] for p in right_trace], 2)

    left_x_top = int(np.polyval(left_coef, top_y_bev))
    right_x_top = int(np.polyval(right_coef, top_y_bev))

    # BEV 上の台形ポリゴン
    poly_bev = np.array([
        [leftx_base, bottom_y_bev],
        [left_x_top, top_y_bev],
        [right_x_top, top_y_bev],
        [rightx_base, bottom_y_bev]
    ], dtype=np.float32)

    # BEV → 元画像へ逆射影
    poly_pts_image = cv2.perspectiveTransform(poly_bev[None, :, :], Minv)[0]

    results = model(frame, verbose=False)[0]
    detections, class_list = [], []
    for box in results.boxes:
        cls = int(box.cls[0].item())
        if cls in target_classes:
            x1, y1, x2, y2 = box.xyxy[0].tolist()
            score = box.conf[0].item()
            detections.append([x1, y1, x2, y2, score])
            class_list.append(cls)

    risky_found = False
    if detections:
        detections_np = np.array(detections, dtype=np.float32)
        online_targets = tracker.update(detections_np, [frame.shape[0], frame.shape[1]], [frame.shape[0], frame.shape[1]])
        for i, t in enumerate(online_targets):
            track_id = t.track_id
            x, y, w, h = map(int, t.tlwh)
            cx, cy = x + w // 2, y + h
            area = w * h
            cls = class_list[i] if i < len(class_list) else -1
            label = class_names.get(cls, "unknown")
            is_risky = cv2.pointPolygonTest(poly_pts_image.astype(np.int32), (cx, cy), False) >= 0
            if is_risky:
                risky_found = True
            cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)
            cv2.putText(frame, f"{track_id} {label}", (x, y - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
            if track_id not in track_history:
                track_history[track_id] = []
            track_history[track_id].append((cx, cy))
            for j in range(1, len(track_history[track_id])):
                cv2.line(frame, track_history[track_id][j - 1], track_history[track_id][j], (0, 255, 255), 2)
            speed_row = df.loc[df["frame"] == frame_idx, "vtti.speed_gps"]
            speed = 0.0 if speed_row.empty else speed_row.values[0]
            trajectories.append({"frame": frame_idx, "id": track_id, "x": cx, "y": cy, "area": area, "class": label, "speed": speed, "risky": int(is_risky)})

    poly_color = (0, 0, 255) if risky_found else (0, 255, 0)
    overlay_poly_bev = np.zeros_like(bev)
    poly_bev_int = poly_bev.astype(np.int32)
    cv2.fillPoly(overlay_poly_bev, [poly_bev_int], poly_color)
    overlay_poly = cv2.warpPerspective(overlay_poly_bev, Minv, (frame_width, frame_height))
    overlayed = cv2.addWeighted(frame, 0.7, overlay_poly, 1, 0)

    # === リサイズして表示用に合わせる（すべて 360x240） ===
    canny_vis_resized = cv2.resize(canny_vis, (360, 240))
    sliding_vis_resized = cv2.resize(sliding_vis, (360, 240))

    # === 出力用レイアウト合成（2x2） ===
    composed = np.vstack([
        np.hstack([frame, canny_vis_resized]),
        np.hstack([sliding_vis_resized, overlayed])
    ])
    out.write(composed)
    frame_idx += 1

cap.release()
out.release()
pd.DataFrame(trajectories).to_csv(output_csv_path, index=False)
print("\u2705 統合完了：動画とCSVを保存しました。")