# -*- coding: utf-8 -*-
"""
9枚の代表画像を余白なしで3x3に結合
出力: representatives_grid_3x3.png
"""

from PIL import Image
import os

IMG_PATHS = [
    r"C:\Users\s1280\Desktop\SHRP2rawdata\central_frames_clustering_v7\clustered_images\representative_frame\0\scene_125_center.jpg",
    r"C:\Users\s1280\Desktop\SHRP2rawdata\central_frames_clustering_v7\clustered_images\representative_frame\1\scene_099_center.jpg",
    r"C:\Users\s1280\Desktop\SHRP2rawdata\central_frames_clustering_v7\clustered_images\representative_frame\2\scene_192_center.jpg",
    r"C:\Users\s1280\Desktop\SHRP2rawdata\central_frames_clustering_v7\clustered_images\representative_frame\3\scene_133_center.jpg",
    r"C:\Users\s1280\Desktop\SHRP2rawdata\central_frames_clustering_v7\clustered_images\representative_frame\4\scene_257_center.jpg",
    r"C:\Users\s1280\Desktop\SHRP2rawdata\central_frames_clustering_v7\clustered_images\representative_frame\5\scene_323_center.jpg",
    r"C:\Users\s1280\Desktop\SHRP2rawdata\central_frames_clustering_v7\clustered_images\representative_frame\6\scene_053_center.jpg",
    r"C:\Users\s1280\Desktop\SHRP2rawdata\central_frames_clustering_v7\clustered_images\representative_frame\7\scene_165_center.jpg",
    r"C:\Users\s1280\Desktop\SHRP2rawdata\central_frames_clustering_v7\clustered_images\representative_frame\8\scene_203_center.jpg",
]

# サムネサイズ（全て同じに揃える）
THUMB_W, THUMB_H = 480, 270
COLS, ROWS = 3, 3

# 全画像をリサイズ
images = [Image.open(p).resize((THUMB_W, THUMB_H), Image.BILINEAR) for p in IMG_PATHS]

# 新しいキャンバス作成（余白なし）
canvas = Image.new("RGB", (COLS * THUMB_W, ROWS * THUMB_H))

for idx, img in enumerate(images):
    r = idx // COLS
    c = idx % COLS
    canvas.paste(img, (c * THUMB_W, r * THUMB_H))

out_path = r"C:\Users\s1280\Desktop\representatives_grid_3x3.png"
canvas.save(out_path, format="PNG", optimize=True)
print(f"✅ saved: {out_path}")
