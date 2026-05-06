import cv2
import numpy as np
import os
import csv
import re

# === 入力情報設定 ===
img_path = r"C:\Users\s1280\Desktop\SHRP2rawdata\3\frames\frame_072000.jpg"
save_dir = r"C:\Users\s1280\PycharmProjects\yolo_classify_project\26x"
video_id = 3
view_name = "forward"

# === 画像読み込み ===
frame_filename = os.path.basename(img_path)
match = re.search(r"\d+", frame_filename)
frame_num = int(match.group()) if match else -1

original_img = cv2.imread(img_path)
if original_img is None:
    raise FileNotFoundError(f"❌ 画像が読み込めません: {img_path}")

h, w = original_img.shape[:2]
clone = original_img.copy()
click_points = []

# === 補助線表示 ===
cv2.line(clone, (0, int(h * 0.25)), (w, int(h * 0.25)), (200, 200, 200), 1)
cv2.line(clone, (0, int(h * 0.75)), (w, int(h * 0.75)), (200, 200, 200), 1)
print("[INFO] Step 3: 画像上で 左下→右下→右上→左上 の順で4点クリックしてください")

def redraw_click_preview():
    """クリック済みの点をすべて描画する"""
    global clone
    clone = original_img.copy()
    # 補助線を再描画
    cv2.line(clone, (0, int(h * 0.25)), (w, int(h * 0.25)), (200, 200, 200), 1)
    cv2.line(clone, (0, int(h * 0.75)), (w, int(h * 0.75)), (200, 200, 200), 1)

    # クリック点を描画
    for pt in click_points:
        cv2.circle(clone, pt, 6, (0, 0, 255), -1)  # 赤丸
    # 線でつなぐ（クリック数が2以上なら）
    if len(click_points) >= 2:
        for i in range(len(click_points)-1):
            cv2.line(clone, click_points[i], click_points[i+1], (0, 255, 0), 2)
    # 4点なら四角形閉じる
    if len(click_points) == 4:
        cv2.line(clone, click_points[3], click_points[0], (0, 255, 0), 2)

def click_event(event, x, y, flags, param):
    global click_points, clone
    if event == cv2.EVENT_LBUTTONDOWN and len(click_points) < 4:
        click_points.append((x, y))
        redraw_click_preview()
        # プレビュー更新
        cv2.imshow("Click 4 src points", clone)

        if len(click_points) == 4:
            # Y位置を補正（水平補正）
            bottom_y = int((click_points[0][1] + click_points[1][1]) / 2)
            top_y = int((click_points[2][1] + click_points[3][1]) / 2)
            click_points[0] = (click_points[0][0], bottom_y)
            click_points[1] = (click_points[1][0], bottom_y)
            click_points[2] = (click_points[2][0], top_y)
            click_points[3] = (click_points[3][0], top_y)

            # 補正後も再描画
            redraw_click_preview()

            # プレビュー保存（クリック点と線が描かれた画像）
            preview_path = os.path.join(save_dir, f"click_preview_{video_id}_{view_name}_3rd.jpg")
            cv2.imwrite(preview_path, clone)
            print(f"🖼 クリック点描画済みプレビューを保存しました → {preview_path}")

cv2.namedWindow("Click 4 src points")
cv2.setMouseCallback("Click 4 src points", click_event)
cv2.imshow("Click 4 src points", clone)
cv2.waitKey(0)
cv2.destroyAllWindows()

# === クリックした4点をnumpyへ ===
src = np.array(click_points, dtype=np.float32)

# ✅ Brandão論文寄せ: BEV出力は元画像と同じ解像度
dst = np.float32([
    [w * 0.25, h],     # 左下
    [w * 0.75, h],     # 右下
    [w * 0.75, 0],     # 右上
    [w * 0.25, 0]      # 左上
])

# ホモグラフィ行列
M = cv2.getPerspectiveTransform(src, dst)
bev = cv2.warpPerspective(original_img, M, (w, h))

# マスク処理（左右直角三角形カット）
mask = np.ones((h, w), dtype=np.uint8) * 255
x_ratio = 0.25
y_ratio = 0.7
left_tri = np.array([
    [0, h],
    [int(w * x_ratio), h],
    [0, int(h * (1 - y_ratio))]
], np.int32)
right_tri = np.array([
    [w, h],
    [int(w * (1 - x_ratio)), h],
    [w, int(h * (1 - y_ratio))]
], np.int32)
cv2.fillPoly(mask, [left_tri], 0)
cv2.fillPoly(mask, [right_tri], 0)
bev_masked = cv2.bitwise_and(bev, bev, mask=mask)

# 保存ファイル名（_2ndを付与）
bev_save_path = os.path.join(save_dir, f"BEV_result_{video_id}_{view_name}_3rd.jpg")
mask_img_path = os.path.join(save_dir, f"BEV_mask_{video_id}_{view_name}_3rd.jpg")
mask_npy_path = os.path.join(save_dir, f"mask_binary_{video_id}_{view_name}_3rd.npy")
src_save_path = os.path.join(save_dir, f"src_points_{video_id}_{view_name}_3rd.npy")
dst_save_path = os.path.join(save_dir, f"dst_points_{video_id}_{view_name}_3rd.npy")

# 保存処理
cv2.imwrite(bev_save_path, bev_masked)
cv2.imwrite(mask_img_path, mask)
np.save(mask_npy_path, mask)
np.save(src_save_path, src)
np.save(dst_save_path, dst)

# dstプレビューも作る
bev_clone = bev_masked.copy()
dst_points_int = dst.astype(int)
for pt in dst_points_int:
    cv2.circle(bev_clone, tuple(pt), 6, (255, 0, 0), -1)
for i in range(4):
    cv2.line(bev_clone, tuple(dst_points_int[i]), tuple(dst_points_int[(i + 1) % 4]), (0, 255, 255), 2)
cv2.imwrite(os.path.join(save_dir, f"dst_points_preview_{video_id}_{view_name}_3rd.jpg"), bev_clone)

# summary CSV更新
summary_path = os.path.join(save_dir, "lane_detection_summary.csv")
row = [f"{video_id}_3rd", view_name, frame_num, f"frame_{frame_num}.jpg", w, h]
file_exists = os.path.isfile(summary_path)
with open(summary_path, 'a', newline='') as csvfile:
    writer = csv.writer(csvfile)
    if not file_exists:
        writer.writerow(["video_id", "view", "frame_num", "frame_file", "width_px", "height_px"])
    writer.writerow(row)

# 結果出力
print(f"✅ BEV結果を保存しました → {bev_save_path}")
print(f"✅ マスク画像も保存しました → {mask_img_path}")
print(f"✅ マスクのnpy保存 → {mask_npy_path}")
print(f"💾 src/dstポイント(_2nd)を保存しました")
print(f"📐 BEV画像サイズ: 幅 = {w}px, 高さ = {h}px（論文寄せ）")

# BEV画像を表示
cv2.imshow("BEV Image", bev)
cv2.waitKey(0)
cv2.destroyAllWindows()

