# lane_bev_minimal_fixed.py
import os, cv2, numpy as np, matplotlib.pyplot as plt

# ========= 1) 入力とパラメータ =========
INPUT_IMAGE_PATH = r"C:\Users\s1280\Desktop\SHRP2rawdata\4\scene_keyframes\3\scene_045_middle.jpg"
SRC_NPY = r"C:\Users\s1280\PycharmProjects\yolo_classify_project\26x\src_points_3_forward.npy"
DST_NPY = r"C:\Users\s1280\PycharmProjects\yolo_classify_project\26x\dst_points_3_forward.npy"
# 参照解像度（.npy を作ったときの画像サイズを入れておく）
REF_W, REF_H = 1928, 1208

OUTPUT_DIR = "out_min"
os.makedirs(OUTPUT_DIR, exist_ok=True)

ego_speed_kmh   = 60.0
scale_px_per_m  = 50.0   # 図示用スケール

# ========= 2) 画像読み込み =========
img = cv2.imread(INPUT_IMAGE_PATH)
assert img is not None, f"cannot read: {INPUT_IMAGE_PATH}"
h, w = img.shape[:2]

# ========= ユーティリティ：点をTL,TR,BR,BLに並べる =========
def order_cw(pts):
    pts = np.asarray(pts, dtype=np.float32)
    # TL: min(x+y), BR: max(x+y)
    s = pts.sum(axis=1)
    tl, br = pts[np.argmin(s)], pts[np.argmax(s)]
    # 残り2点を左右に分ける
    remain = [p for i, p in enumerate(pts) if i not in [np.argmin(s), np.argmax(s)]]
    remain = np.array(remain, dtype=np.float32)
    # TR は x が大きい方（TL と同じ y 付近を仮定）
    tr, bl = (remain[np.argmax(remain[:,0])], remain[np.argmin(remain[:,0])])
    return np.array([tl, tr, br, bl], dtype=np.float32)

# ========= 3) 射影変換 H (src→dst) を作る =========
use_npy = os.path.exists(SRC_NPY) and os.path.exists(DST_NPY)
if use_npy:
    src_raw = np.load(SRC_NPY).astype(np.float32)
    dst_raw = np.load(DST_NPY).astype(np.float32)

    # 参照解像度 → 現フレーム解像度にスケール
    sx, sy = w/float(REF_W), h/float(REF_H)
    src = src_raw.copy(); src[:,0] *= sx; src[:,1] *= sy

    # dst を (0,0) 起点にリベース＆サイズ算出
    dst = dst_raw.copy()
    dst -= dst.min(axis=0, keepdims=True)
    bev_w = int(round(dst[:,0].max()))
    bev_h = int(round(dst[:,1].max()))
else:
    # フォールバック（手動想定）
    bev_w, bev_h = 360, int(10*scale_px_per_m)
    src = np.float32([
        [w*0.45, h*0.62],   # TL
        [w*0.55, h*0.62],   # TR
        [w*0.90, h*0.98],   # BR
        [w*0.10, h*0.98],   # BL
    ])
    dst = np.float32([
        [0.20*bev_w, 0.30*bev_h],
        [0.80*bev_w, 0.30*bev_h],
        [0.80*bev_w, 1.00*bev_h],
        [0.20*bev_w, 1.00*bev_h],
    ])

# 点の順序を強制（TL,TR,BR,BL）
src = order_cw(src)
dst = order_cw(dst)

# サイズ最終決定（npy由来なら上で決まっている）
if not use_npy:
    bev_w = int(np.max(dst[:,0]) - np.min(dst[:,0]))
    bev_h = int(np.max(dst[:,1]) - np.min(dst[:,1]))

H    = cv2.getPerspectiveTransform(src, dst)
Hinv = cv2.getPerspectiveTransform(dst, src)

# ========= 4) 画像→BEV 変換 =========
bev = cv2.warpPerspective(img, H, (bev_w, bev_h))

# ========= 5) スライディングウィンドウでレーン点抽出（BEV） =========
gray   = cv2.cvtColor(bev, cv2.COLOR_BGR2GRAY)
edges  = cv2.Canny(cv2.GaussianBlur(gray,(5,5),0), 50, 150)
binary = (edges>0).astype(np.uint8)

hist = np.sum(binary[binary.shape[0]//2:,:], axis=0)
mid  = hist.shape[0]//2
lx   = int(np.argmax(hist[:mid])) if mid>0 else 0
rx   = int(np.argmax(hist[mid:])+mid) if mid>0 else bev_w-1

n_win, margin, minpix = 20, 30, 30
win_h = max(1, binary.shape[0]//n_win)
nz_y, nz_x = binary.nonzero()

left_pts, right_pts = [], []
for i in range(n_win):
    wy_low, wy_high = bev_h-(i+1)*win_h, bev_h-i*win_h
    li = ((nz_y>=wy_low)&(nz_y<wy_high)&(nz_x>=lx-margin)&(nz_x<lx+margin)).nonzero()[0]
    ri = ((nz_y>=wy_low)&(nz_y<wy_high)&(nz_x>=rx-margin)&(nz_x<rx+margin)).nonzero()[0]
    if len(li)>minpix: lx = int(np.mean(nz_x[li]))
    if len(ri)>minpix: rx = int(np.mean(nz_x[ri]))
    cy = (wy_low+wy_high)//2
    left_pts.append((lx, cy)); right_pts.append((rx, cy))

def fit_cubic(points):
    if len(points) < 4: return None
    ys = np.array([p[1] for p in points], np.float32)
    xs = np.array([p[0] for p in points], np.float32)
    return np.polyfit(ys, xs, 3)  # a,b,c,d

def poly_x(y, c): return int(np.polyval(c, y)) if c is not None else None

l_coef = fit_cubic(left_pts)
r_coef = fit_cubic(right_pts)

ys = np.linspace(0, bev_h-1, 80).astype(int)
left_curve  = [(poly_x(y,l_coef), y) for y in ys] if l_coef is not None else []
right_curve = [(poly_x(y,r_coef), y) for y in ys] if r_coef is not None else []

# ========= 6) 速度ベースの停止距離ポリゴン =========
stop_m  = min(ego_speed_kmh/10.0, 10.0)
stop_px = int(stop_m*scale_px_per_m)
y_top, y_bot = max(0, bev_h-stop_px), bev_h-1

poly_bev = None
if l_coef is not None and r_coef is not None:
    poly_bev = np.array([
        [poly_x(y_bot,l_coef), y_bot],
        [poly_x(y_top,l_coef), y_top],
        [poly_x(y_top,r_coef), y_top],
        [poly_x(y_bot,r_coef), y_bot],
    ], np.int32)

# ========= 7) 可視化（BEV / Front） =========
bev_vis = bev.copy()
for i in range(1, len(left_curve)):  cv2.line(bev_vis, left_curve[i-1],  left_curve[i],  (0,255,0), 2)
for i in range(1, len(right_curve)): cv2.line(bev_vis, right_curve[i-1], right_curve[i], (0,0,255), 2)
if poly_bev is not None:
    ov = bev_vis.copy(); cv2.fillPoly(ov, [poly_bev], (0,255,255))
    bev_vis = cv2.addWeighted(ov, 0.3, bev_vis, 0.7, 0)

cv2.imwrite(os.path.join(OUTPUT_DIR, "bev_with_poly.png"), bev_vis)

front_vis = img.copy()
if poly_bev is not None:
    poly_img = cv2.perspectiveTransform(poly_bev[None].astype(np.float32), Hinv)[0].astype(np.int32)
    ov = front_vis.copy(); cv2.fillPoly(ov, [poly_img], (0,255,255))
    front_vis = cv2.addWeighted(ov, 0.3, front_vis, 0.7, 0)

cv2.imwrite(os.path.join(OUTPUT_DIR, "front_with_poly.png"), front_vis)

# ========= 8) 修論用 3パネル（Front → Concept → BEV） =========
fig, axes = plt.subplots(1, 3, figsize=(11, 3.2))
axes[0].imshow(cv2.cvtColor(front_vis, cv2.COLOR_BGR2RGB))
axes[0].set_title("Front (with polygon)"); axes[0].axis("off")
axes[1].imshow(np.ones((10,10,3))); axes[1].axis("off")
axes[1].text(0.5, 0.5, "Perspective Transform\n$\\mathbf{x}' \\sim \\mathbf{H}\\,\\mathbf{x}$",
             ha="center", va="center", fontsize=12)
axes[2].imshow(cv2.cvtColor(bev_vis, cv2.COLOR_BGR2RGB))
axes[2].set_title("BEV (with lanes)"); axes[2].axis("off")
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, "thesis_figure_bev_3panel.png"), dpi=300)
plt.close()

print("Saved:",
      os.path.join(OUTPUT_DIR, "front_with_poly.png"),
      os.path.join(OUTPUT_DIR, "bev_with_poly.png"),
      os.path.join(OUTPUT_DIR, "thesis_figure_bev_3panel.png"))
