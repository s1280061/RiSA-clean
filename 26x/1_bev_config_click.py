import cv2
import numpy as np
import os
import csv
import re

# === 入力情報設定 ===
img_path = r"C:\Users\s1280\Desktop\SHRP2rawdata\3\frames_bottom_left\frame_72000.jpg"
save_dir = r"C:\Users\s1280\PycharmProjects\yolo_classify_project\26x"
video_id = 3
view_name = "forward"

# === 画像読み込みと準備 ===
frame_filename = os.path.basename(img_path)
match = re.search(r"\d+", frame_filename)
frame_num = int(match.group()) if match else -1
original_img = cv2.imread(img_path)
if original_img is None:
    raise FileNotFoundError(f"❌ 画像が読み込めません: {img_path}")
w, h = original_img.shape[1], original_img.shape[0]
clone = original_img.copy()
click_points = []

# === 補助線表示 ===
cv2.line(clone, (0, int(h * 0.25)), (w, int(h * 0.25)), (200, 200, 200), 1)
cv2.line(clone, (0, int(h * 0.75)), (w, int(h * 0.75)), (200, 200, 200), 1)
print("[INFO] Step 3: 画像上で左下→右下→右上→左上の順で4点クリックしてください")

def click_event(event, x, y, flags, param):
    global click_points, clone
    if event == cv2.EVENT_LBUTTONDOWN and len(click_points) < 4:
        click_points.append((x, y))
        cv2.circle(clone, (x, y), 6, (0, 0, 255), -1)
        if len(click_points) > 1:
            cv2.line(clone, click_points[-2], click_points[-1], (0, 255, 0), 2)
        if len(click_points) == 4:
            # 水平補正
            bottom_y = int((click_points[0][1] + click_points[1][1]) / 2)
            top_y = int((click_points[2][1] + click_points[3][1]) / 2)
            click_points[0] = (click_points[0][0], bottom_y)
            click_points[1] = (click_points[1][0], bottom_y)
            click_points[2] = (click_points[2][0], top_y)
            click_points[3] = (click_points[3][0], top_y)

            # 再描画
            clone = original_img.copy()
            cv2.line(clone, (0, int(h * 0.25)), (w, int(h * 0.25)), (200, 200, 200), 1)
            cv2.line(clone, (0, int(h * 0.75)), (w, int(h * 0.75)), (200, 200, 200), 1)
            for pt in click_points:
                cv2.circle(clone, pt, 6, (0, 0, 255), -1)
            for i in range(4):
                cv2.line(clone, click_points[i], click_points[(i + 1) % 4], (0, 255, 0), 2)
            cv2.imwrite(os.path.join(save_dir, f"click_preview_{video_id}_{view_name}.jpg"), clone)
            print("🖼 クリック点を描画した画像を保存しました")

cv2.namedWindow("Click 4 src points")
cv2.setMouseCallback("Click 4 src points", click_event)
cv2.imshow("Click 4 src points", clone)
cv2.waitKey(0)
cv2.destroyAllWindows()

# === BEV変換 ===
src = np.array(click_points, dtype=np.float32)

scale_path = os.path.join(save_dir, f"scale_px_per_m_{view_name}.npy")
if os.path.exists(scale_path):
    scale_px_per_m = np.load(scale_path)[0]
else:
    scale_px_per_m = 50.82
    np.save(scale_path, np.array([scale_px_per_m]))
    print(f"⚠️ scale_px_per_m が見つからないため仮定値 {scale_px_per_m} を使用します")

desired_m = 10
bev_height = int(desired_m * scale_px_per_m)
dst = np.float32([
    [w * 0.25, bev_height],
    [w * 0.75, bev_height],
    [w * 0.75, 0],
    [w * 0.25, 0]
])
M = cv2.getPerspectiveTransform(src, dst)
Minv = cv2.getPerspectiveTransform(dst, src)
bev = cv2.warpPerspective(original_img, M, (w, bev_height))

# === ⬇ マスク処理（左右直角三角形）⬇ ===
mask = np.ones((bev_height, w), dtype=np.uint8) * 255
x_ratio = 0.25
y_ratio = 0.7
left_tri = np.array([
    [0, bev_height],
    [int(w * x_ratio), bev_height],
    [0, int(bev_height * (1 - y_ratio))]
], np.int32)
right_tri = np.array([
    [w, bev_height],
    [int(w * (1 - x_ratio)), bev_height],
    [w, int(bev_height * (1 - y_ratio))]
], np.int32)
cv2.fillPoly(mask, [left_tri], 0)
cv2.fillPoly(mask, [right_tri], 0)
bev_masked = cv2.bitwise_and(bev, bev, mask=mask)

# === 保存処理 ===
bev_save_path = os.path.join(save_dir, f"BEV_result_{video_id}_{view_name}.jpg")
mask_img_path = os.path.join(save_dir, f"BEV_mask_{video_id}_{view_name}.jpg")
mask_npy_path = os.path.join(save_dir, f"mask_binary_{video_id}_{view_name}.npy")

cv2.imwrite(bev_save_path, bev_masked)
cv2.imwrite(mask_img_path, mask)
np.save(mask_npy_path, mask)
np.save(os.path.join(save_dir, f"src_points_{video_id}_{view_name}.npy"), src)
np.save(os.path.join(save_dir, f"dst_points_{video_id}_{view_name}.npy"), dst)

# === dst点描画して保存 ===
bev_clone = bev_masked.copy()
dst_points_int = dst.astype(int)
for pt in dst_points_int:
    cv2.circle(bev_clone, tuple(pt), 6, (255, 0, 0), -1)
for i in range(4):
    cv2.line(bev_clone, tuple(dst_points_int[i]), tuple(dst_points_int[(i + 1) % 4]), (0, 255, 255), 2)
cv2.imwrite(os.path.join(save_dir, f"dst_points_preview_{video_id}_{view_name}.jpg"), bev_clone)

# === k_base 計算・保存 ===
pt_bottom = ((dst[0][0] + dst[1][0]) / 2, (dst[0][1] + dst[1][1]) / 2)
pt_top = ((dst[2][0] + dst[3][0]) / 2, (dst[2][1] + dst[3][1]) / 2)
dx, dy = pt_bottom[0] - pt_top[0], pt_bottom[1] - pt_top[1]
center_height_px = np.sqrt(dx ** 2 + dy ** 2)
k_base = center_height_px / scale_px_per_m
np.save(os.path.join(save_dir, f"k_base_{view_name}.npy"), np.array([k_base]))

# === summary CSV 更新 ===
summary_path = os.path.join(save_dir, "lane_detection_summary.csv")
row = [video_id, view_name, frame_num, f"frame_{frame_num}.jpg", round(center_height_px, 2), round(scale_px_per_m, 2), round(k_base, 3)]
file_exists = os.path.isfile(summary_path)
with open(summary_path, 'a', newline='') as csvfile:
    writer = csv.writer(csvfile)
    if not file_exists:
        writer.writerow(["video_id", "view", "frame_num", "frame_file", "center_height_px", "scale_px_per_m", "k_base_m"])
    writer.writerow(row)

# === 結果出力 ===
print(f"✅ BEV結果を保存しました → {bev_save_path}")
print(f"✅ マスク画像も保存しました → {mask_img_path}")
print(f"✅ マスクのnpy保存 → {mask_npy_path}")
print(f"📐 BEVの縦ピクセル長: {center_height_px:.2f} px")
print(f"📏 実距離 k_base: {k_base:.3f} m")
print("📝 summary CSV に追記しました")

# === 要求: src_points_3_forward.npy, dst_points_3_forward.npy を保存 ===
src_forward_path = os.path.join(save_dir, "src_points_3_forward.npy")
dst_forward_path = os.path.join(save_dir, "dst_points_3_forward.npy")
np.save(src_forward_path, src)
np.save(dst_forward_path, dst)
print(f"💾 src_points_3_forward.npy を保存しました → {src_forward_path}")
print(f"💾 dst_points_3_forward.npy を保存しました → {dst_forward_path}")
print(f"📐 BEV画像サイズ: 幅 = {w} px, 高さ = {bev_height} px")
print(f"🎯 BEV画像変換サイズは (幅={w}px, 高さ={bev_height}px) です")
# === BEV画像を表示して確認 ===
cv2.imshow("BEV Image", bev)
cv2.waitKey(0)
cv2.destroyAllWindows()  # ← これならすべてのウィンドウを確実に閉じる

