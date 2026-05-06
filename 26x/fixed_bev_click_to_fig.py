# -*- coding: utf-8 -*-
"""
クリック4点 → BEV生成 → レーン抽出 → 停止距離ポリゴン → 前方画像へ逆写像 → 3パネル図
- src/dst/mask を npy/jpg で保存
- summary CSV を追記
"""

import os, re, csv
import cv2
import numpy as np
import matplotlib.pyplot as plt

# ========================= 入力と保存先 =========================
img_path = r"C:\Users\s1280\Desktop\SHRP2rawdata\3\frames\frame_072000.jpg"
save_dir = r"C:\Users\s1280\PycharmProjects\yolo_classify_project\26x"
video_id = 3
view_name = "forward"

# 速度とスケール（停止距離ポリゴン用）
ego_speed_kmh  = 60.0      # 任意
scale_px_per_m = 50.0      # 1mあたりpxの目安（図の見やすさ用）

os.makedirs(save_dir, exist_ok=True)

# ========================= 画像読み込み & 下準備 =========================
frame_filename = os.path.basename(img_path)
m = re.search(r"\d+", frame_filename)
frame_num = int(m.group()) if m else -1

original_img = cv2.imread(img_path)
if original_img is None:
    raise FileNotFoundError(f"❌ 画像が読み込めません: {img_path}")
h, w = original_img.shape[:2]

clone = original_img.copy()
click_points: list[tuple[int,int]] = []

# 補助線
cv2.line(clone, (0, int(h*0.25)), (w, int(h*0.25)), (200,200,200), 1)
cv2.line(clone, (0, int(h*0.75)), (w, int(h*0.75)), (200,200,200), 1)
print("[INFO] 左下→右下→右上→左上 の順で4点クリックしてください。Escで終了")

def redraw_click_preview():
    global clone
    clone = original_img.copy()
    cv2.line(clone, (0, int(h*0.25)), (w, int(h*0.25)), (200,200,200), 1)
    cv2.line(clone, (0, int(h*0.75)), (w, int(h*0.75)), (200,200,200), 1)
    for pt in click_points:
        cv2.circle(clone, pt, 6, (0,0,255), -1)
    if len(click_points) >= 2:
        for i in range(len(click_points)-1):
            cv2.line(clone, click_points[i], click_points[i+1], (0,255,0), 2)
    if len(click_points) == 4:
        cv2.line(clone, click_points[3], click_points[0], (0,255,0), 2)

def click_event(event, x, y, flags, param):
    if event == cv2.EVENT_LBUTTONDOWN and len(click_points) < 4:
        click_points.append((x, y))
        redraw_click_preview()
        cv2.imshow("Click 4 src points", clone)

        if len(click_points) == 4:
            # 下辺/上辺の水平補正（Yを平均化）
            bottom_y = int((click_points[0][1] + click_points[1][1]) / 2)
            top_y    = int((click_points[2][1] + click_points[3][1]) / 2)
            click_points[0] = (click_points[0][0], bottom_y)
            click_points[1] = (click_points[1][0], bottom_y)
            click_points[2] = (click_points[2][0], top_y)
            click_points[3] = (click_points[3][0], top_y)
            redraw_click_preview()
            prev_path = os.path.join(save_dir, f"click_preview_{video_id}_{view_name}_3rd.jpg")
            cv2.imwrite(prev_path, clone)
            print(f"🖼 クリック確認画像 → {prev_path}")

cv2.namedWindow("Click 4 src points")
cv2.setMouseCallback("Click 4 src points", click_event)
cv2.imshow("Click 4 src points", clone)
cv2.waitKey(0)
cv2.destroyAllWindows()

if len(click_points) != 4:
    raise RuntimeError("4点が取得できていません。もう一度実行してください。")

# ========================= 射影変換（src→dst） =========================
src = np.array(click_points, dtype=np.float32)
# Brandão風：BEV出力は元画像と同解像度の中央帯
dst = np.float32([
    [w*0.25, h],  # 左下
    [w*0.75, h],  # 右下
    [w*0.75, 0],  # 右上
    [w*0.25, 0],  # 左上
])

H    = cv2.getPerspectiveTransform(src, dst)
Hinv = cv2.getPerspectiveTransform(dst, src)
bev  = cv2.warpPerspective(original_img, H, (w, h))

# デバッグ：frontにsrc四角形を描画
front_dbg = original_img.copy()
cv2.polylines(front_dbg, [src.astype(np.int32)], True, (0,255,255), 3)
cv2.imwrite(os.path.join(save_dir, f"front_src_quad_check_{video_id}_{view_name}_3rd.jpg"), front_dbg)

# ========================= マスク（左右三角を落とす） =========================
mask = np.ones((h, w), np.uint8)*255
x_ratio, y_ratio = 0.25, 0.70
left_tri = np.array([[0,h],[int(w*x_ratio),h],[0,int(h*(1-y_ratio))]], np.int32)
right_tri= np.array([[w,h],[int(w*(1-x_ratio)),h],[w,int(h*(1-y_ratio))]], np.int32)
cv2.fillPoly(mask, [left_tri], 0)
cv2.fillPoly(mask, [right_tri], 0)
bev_masked = cv2.bitwise_and(bev, bev, mask=mask)

# ========================= BEVでレーン抽出 + 多項式フィット =========================
def sliding_fit_lanes(bev_img, n_win=20, margin=30, minpix=30):
    hh, ww = bev_img.shape[:2]
    gray  = cv2.cvtColor(bev_img, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(cv2.GaussianBlur(gray, (5,5), 0), 50, 150)
    biny  = (edges > 0).astype(np.uint8)

    hist = np.sum(biny[biny.shape[0]//2:, :], axis=0)
    mid  = hist.shape[0]//2
    lx   = int(np.argmax(hist[:mid])) if mid>0 else 0
    rx   = int(np.argmax(hist[mid:])+mid) if mid>0 else ww-1

    win_h = biny.shape[0] // n_win if n_win>0 else biny.shape[0]
    nz_y, nz_x = biny.nonzero()

    left_pts, right_pts = [], []
    for i in range(n_win):
        wy_low  = hh - (i+1)*win_h
        wy_high = hh - i*win_h
        li = ((nz_y>=wy_low)&(nz_y<wy_high)&(nz_x>=lx-margin)&(nz_x<lx+margin)).nonzero()[0]
        ri = ((nz_y>=wy_low)&(nz_y<wy_high)&(nz_x>=rx-margin)&(nz_x<rx+margin)).nonzero()[0]
        if len(li)>minpix: lx = int(np.mean(nz_x[li]))
        if len(ri)>minpix: rx = int(np.mean(nz_x[ri]))
        cy = (wy_low + wy_high)//2
        left_pts.append((lx, cy)); right_pts.append((rx, cy))

    def fit_cubic(points):
        if len(points) < 4: return None
        ys = np.array([p[1] for p in points], np.float32)
        xs = np.array([p[0] for p in points], np.float32)
        return np.polyfit(ys, xs, 3)  # x(y)=ay^3+by^2+cy+d

    return left_pts, right_pts, fit_cubic(left_pts), fit_cubic(right_pts), edges

left_pts, right_pts, lcoef, rcoef, edges = sliding_fit_lanes(bev_masked)
def poly_x(y, c): return int(np.polyval(c, y)) if c is not None else None

# ========================= 停止距離ポリゴン（BEV→Front逆写像も） =========================
stop_m  = min(ego_speed_kmh / 10.0, 10.0)
stop_px = int(stop_m * scale_px_per_m)
y_top, y_bot = max(0, h - stop_px), h - 1

poly_bev = None
if (lcoef is not None) and (rcoef is not None):
    poly_bev = np.array([
        [poly_x(y_bot, lcoef), y_bot],
        [poly_x(y_top, lcoef), y_top],
        [poly_x(y_top, rcoef), y_top],
        [poly_x(y_bot, rcoef), y_bot],
    ], dtype=np.int32)

# BEV可視化
bev_vis = bev_masked.copy()
edges_vis = cv2.cvtColor(edges, cv2.COLOR_GRAY2BGR)
bev_vis = cv2.addWeighted(bev_vis, 1.0, edges_vis, 0.25, 0.0)

ys = np.linspace(0, h-1, 120).astype(int)
if lcoef is not None:
    for i in range(1, len(ys)):
        cv2.line(bev_vis, (poly_x(ys[i-1], lcoef), ys[i-1]),
                          (poly_x(ys[i],   lcoef), ys[i]), (0,255,0), 2)
if rcoef is not None:
    for i in range(1, len(ys)):
        cv2.line(bev_vis, (poly_x(ys[i-1], rcoef), ys[i-1]),
                          (poly_x(ys[i],   rcoef), ys[i]), (0,0,255), 2)

if poly_bev is not None:
    ov = bev_vis.copy()
    cv2.fillPoly(ov, [poly_bev], (0,255,255))
    bev_vis = cv2.addWeighted(ov, 0.30, bev_vis, 0.70, 0)

# Frontへ逆写像して重畳
front_vis = original_img.copy()
if poly_bev is not None:
    poly_img = cv2.perspectiveTransform(poly_bev[None].astype(np.float32), Hinv)[0].astype(np.int32)
    ovf = front_vis.copy()
    cv2.fillPoly(ovf, [poly_img], (0,255,255))
    front_vis = cv2.addWeighted(ovf, 0.30, front_vis, 0.70, 0)

# ========================= 保存（画像・npy・CSV） =========================
bev_save_path  = os.path.join(save_dir, f"BEV_result_{video_id}_{view_name}_3rd.jpg")
mask_img_path  = os.path.join(save_dir, f"BEV_mask_{video_id}_{view_name}_3rd.jpg")
mask_npy_path  = os.path.join(save_dir, f"mask_binary_{video_id}_{view_name}_3rd.npy")
src_save_path  = os.path.join(save_dir, f"src_points_{video_id}_{view_name}_3rd.npy")
dst_save_path  = os.path.join(save_dir, f"dst_points_{video_id}_{view_name}_3rd.npy")

cv2.imwrite(bev_save_path, bev_masked)
cv2.imwrite(mask_img_path, mask)
np.save(mask_npy_path, mask)
np.save(src_save_path, src)
np.save(dst_save_path, dst)

cv2.imwrite(os.path.join(save_dir, f"bev_with_lanes_{video_id}_{view_name}_3rd.jpg"), bev_vis)
cv2.imwrite(os.path.join(save_dir, f"front_with_poly_{video_id}_{view_name}_3rd.jpg"), front_vis)

# dstプレビュー
dst_preview = bev_masked.copy()
dst_pts_i = dst.astype(int)
for pt in dst_pts_i:
    cv2.circle(dst_preview, tuple(pt), 6, (255,0,0), -1)
for i in range(4):
    cv2.line(dst_preview, tuple(dst_pts_i[i]), tuple(dst_pts_i[(i+1)%4]), (0,255,255), 2)
cv2.imwrite(os.path.join(save_dir, f"dst_points_preview_{video_id}_{view_name}_3rd.jpg"), dst_preview)

# summary CSV
summary_path = os.path.join(save_dir, "lane_detection_summary.csv")
row = [f"{video_id}_3rd", view_name, frame_num, frame_filename, w, h]
new_file = not os.path.isfile(summary_path)
with open(summary_path, "a", newline="") as f:
    writer = csv.writer(f)
    if new_file:
        writer.writerow(["video_id", "view", "frame_num", "frame_file", "width_px", "height_px"])
    writer.writerow(row)

# ========================= 論文用 3パネル図 =========================
fig, axes = plt.subplots(1, 3, figsize=(11.5, 3.4))
axes[0].imshow(cv2.cvtColor(front_vis, cv2.COLOR_BGR2RGB))
axes[0].set_title("Front (with polygon)"); axes[0].axis("off")
axes[1].text(0.5, 0.5, "Perspective Transform\n$\\mathbf{x}' \\sim \\mathbf{H}\\,\\mathbf{x}$",
             ha="center", va="center", fontsize=16)
axes[1].axis("off")
axes[2].imshow(cv2.cvtColor(bev_vis, cv2.COLOR_BGR2RGB))
axes[2].set_title("BEV (with lanes)"); axes[2].axis("off")
plt.tight_layout()
fig_path = os.path.join(save_dir, f"thesis_figure_bev_3panel_{video_id}_{view_name}_3rd.png")
plt.savefig(fig_path, dpi=300)
plt.close()

# ========================= 結果表示 =========================
print("✅ 保存しました：")
print("  ", bev_save_path)
print("  ", mask_img_path)
print("  ", os.path.join(save_dir, f"bev_with_lanes_{video_id}_{view_name}_3rd.jpg"))
print("  ", os.path.join(save_dir, f"front_with_poly_{video_id}_{view_name}_3rd.jpg"))
print("  ", fig_path)
print("💾 src/dst/mask npy も保存済み")
print(f"📐 出力サイズ: {w}x{h}")
