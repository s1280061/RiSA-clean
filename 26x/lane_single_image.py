# -*- coding: utf-8 -*-
"""
単体画像のレーン検出だけを可視化する最小スクリプト
- 入力: 1枚のフレーム画像
- 出力: レーン領域ポリゴンの塗りつぶし＆左右レーン線を元画像にオーバレイして表示/保存

必要ファイル:
- src_points_3_forward.npy, dst_points_3_forward.npy（射影用の対応点）
"""

import os
import cv2
import numpy as np

# ======== 入出力パス（ここだけ調整）========
img_path  = r"C:\Users\s1280\Desktop\SHRP2rawdata\5\frames_bottom_left\frame_08758.jpg"  # ←指定の画像
base_path = r"C:\Users\s1280\PycharmProjects\yolo_classify_project\26x"                 # ←src/dst の .npy を置いている場所
save_path = os.path.splitext(img_path)[0] + "_lanes.jpg"                                # 出力画像

# ======== パラメータ ========
scale_px_per_m = 50.82     # BEVスケール（px/m）
bev_height_m   = 10.0      # BEV画像の縦の物理長 [m]
bev_h          = int(bev_height_m * scale_px_per_m)  # BEV縦ピクセル
stop_m         = 10.0      # ポリゴン（停止距離相当）の長さ[m]。単体表示用に定数でOK

# スライディングウィンドウ設定（統合コード相当）
n_win  = 30
margin = 40
minpix = 30
gauss_ksize = (5, 5)
canny_lo, canny_hi = 15, 30

# ======== ユーティリティ ========
def compute_lane_length(pts, scale_px_per_m: float) -> float:
    if not pts or len(pts) < 2:
        return 0.0
    total_px = 0.0
    for i in range(1, len(pts)):
        dx = pts[i][0] - pts[i-1][0]
        dy = pts[i][1] - pts[i-1][1]
        total_px += (dx*dx + dy*dy) ** 0.5
    return total_px / scale_px_per_m

def compute_curvature_cubic(coeffs, y_eval, scale_px_per_m: float) -> float:
    # x = a y^3 + b y^2 + c y + d 形式の曲率（px基準→[1/m]へ換算）
    a, b, c, _ = coeffs
    dx_dy   = 3*a*y_eval*y_eval + 2*b*y_eval + c
    d2x_dy2 = 6*a*y_eval + 2*b
    curvature_px = abs(d2x_dy2) / ((1 + dx_dy*dx_dy) ** 1.5)
    return curvature_px * scale_px_per_m

# ======== メイン処理 ========
def main():
    # 1) 入力読み込み
    img = cv2.imread(img_path)
    if img is None:
        raise FileNotFoundError(f"画像が読み込めませんでした: {img_path}")
    h, w = img.shape[:2]

    # 2) 射影行列の準備
    src_path = os.path.join(base_path, "src_points_3_forward.npy")
    dst_path = os.path.join(base_path, "dst_points_3_forward.npy")
    if not (os.path.exists(src_path) and os.path.exists(dst_path)):
        raise FileNotFoundError("src/dst の .npy が見つかりません。base_path を確認してください。")
    src = np.load(src_path)
    dst = np.load(dst_path)
    M    = cv2.getPerspectiveTransform(src, dst)
    Minv = cv2.getPerspectiveTransform(dst, src)

    # 3) 画像→BEV へ射影
    bev = cv2.warpPerspective(img, M, (w, bev_h))

    # 4) マスク＆エッジ抽出（統合スクリプト相当の前処理）
    mask = np.ones((bev_h, w), np.uint8) * 255
    cv2.fillPoly(mask, [np.array([[0, bev_h], [90, bev_h], [0, int(bev_h*0.3)]], np.int32)], 0)
    cv2.fillPoly(mask, [np.array([[w, bev_h], [w-90, bev_h], [w, int(bev_h*0.3)]], np.int32)], 0)

    gray  = cv2.cvtColor(cv2.bitwise_and(bev, bev, mask=mask), cv2.COLOR_BGR2GRAY)
    blur  = cv2.GaussianBlur(gray, gauss_ksize, 0)
    edges = cv2.Canny(blur, canny_lo, canny_hi)
    binary = (edges > 0).astype(np.uint8)

    # 5) 下半分のヒストグラムから左右レーンの初期xを推定
    hist = np.sum(binary[binary.shape[0]//2:, :], axis=0)
    mid  = hist.shape[0] // 2 if hist.size > 0 else 0
    l_base = int(np.argmax(hist[:mid])) if hist.size > 0 and mid > 0 else 0
    r_base = int(np.argmax(hist[mid:]) + mid) if hist.size > 0 and mid > 0 else (w - 1)

    # 6) スライディングウィンドウで左右レーン点列を収集
    win_h = binary.shape[0] // n_win if n_win > 0 else binary.shape[0]
    nz_y, nz_x = binary.nonzero()
    lx_cur, rx_cur = l_base, r_base
    left_pts, right_pts = [], []
    for _ in range(n_win):
        wy_low  = binary.shape[0] - (len(left_pts) + 1) * win_h
        wy_high = binary.shape[0] - len(left_pts) * win_h
        win_l = ((nz_y >= wy_low) & (nz_y < wy_high) &
                 (nz_x >= lx_cur - margin) & (nz_x < lx_cur + margin)).nonzero()[0]
        win_r = ((nz_y >= wy_low) & (nz_y < wy_high) &
                 (nz_x >= rx_cur - margin) & (nz_x < rx_cur + margin)).nonzero()[0]
        if len(win_l) > minpix:
            lx_cur = int(np.mean(nz_x[win_l]))
        if len(win_r) > minpix:
            rx_cur = int(np.mean(nz_x[win_r]))
        cy = (wy_low + wy_high) // 2
        left_pts.append((lx_cur, cy))
        right_pts.append((rx_cur, cy))

    overlay = img.copy()
    vis    = img.copy()

    if left_pts and right_pts:
        # 7) 3次多項式フィット
        l_coef = np.polyfit([p[1] for p in left_pts],  [p[0] for p in left_pts],  3)
        r_coef = np.polyfit([p[1] for p in right_pts], [p[0] for p in right_pts], 3)

        # 8) “停止距離”相当の長さだけ上端を決め、BEV上の四角形ポリゴン→元画像へ逆射影
        stop_px = min(int(stop_m * scale_px_per_m), bev_h)
        y_bot, y_top = bev_h - 1, max(0, bev_h - stop_px)

        def poly_x(y, c): return int(np.polyval(c, y))

        poly_bev = np.array([
            [l_base,         y_bot],
            [poly_x(y_top, l_coef), y_top],
            [poly_x(y_top, r_coef), y_top],
            [r_base,         y_bot]
        ], np.float32)

        poly_img = cv2.perspectiveTransform(poly_bev[None], Minv)[0].astype(np.int32)

        # 9) レーン塗りつぶしを元画像へ描画
        cv2.fillPoly(overlay, [poly_img], (0, 255, 0))   # 緑で塗り
        vis = cv2.addWeighted(overlay, 0.3, vis, 0.7, 0)

        # 10) 左右レーン中心線を BEV→画像に逆射影して描画
        ys = np.linspace(y_top, y_bot, num=50).astype(np.float32)
        left_curve_bev  = np.stack([np.polyval(l_coef, ys), ys], axis=1).astype(np.float32)
        right_curve_bev = np.stack([np.polyval(r_coef, ys), ys], axis=1).astype(np.float32)

        left_curve_img  = cv2.perspectiveTransform(left_curve_bev[None],  Minv)[0].astype(np.int32)
        right_curve_img = cv2.perspectiveTransform(right_curve_bev[None], Minv)[0].astype(np.int32)

        cv2.polylines(vis,  [left_curve_img],  False, (255, 255, 255), 2)  # 左（白）
        cv2.polylines(vis,  [right_curve_img], False, (255, 255, 255), 2)  # 右（白）

        # 参考: 曲率・長さ（必要なら表示）
        y_eval = min(bev_h - 1, max(0, 120))
        left_curv_m  = compute_curvature_cubic(l_coef, y_eval, scale_px_per_m)
        right_curv_m = compute_curvature_cubic(r_coef, y_eval, scale_px_per_m)
        left_len_m   = compute_lane_length(left_pts, scale_px_per_m)
        right_len_m  = compute_lane_length(right_pts, scale_px_per_m)
        info = f"CurvL:{left_curv_m:.4f}  CurvR:{right_curv_m:.4f}  LenL:{left_len_m:.2f}m  LenR:{right_len_m:.2f}m"
        cv2.putText(vis, info, (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (20, 220, 20), 2, cv2.LINE_AA)
    else:
        cv2.putText(vis, "Lane not found", (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2, cv2.LINE_AA)

    # 11) 表示と保存
    cv2.imshow("Lane Detection (single image)", vis)
    cv2.imwrite(save_path, vis)
    print(f"保存しました: {save_path}")
    cv2.waitKey(0)
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
