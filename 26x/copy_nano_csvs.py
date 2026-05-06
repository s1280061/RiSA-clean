import cv2
import numpy as np

# 画像ファイルのリスト（小さい番号 → 大きい番号の順に並べる）
image_paths = [
    r"C:\Users\s1280\Desktop\SHRP2rawdata\5\new_divided\scene_019_fire\frame_008732_post_tid2.jpg",
    r"C:\Users\s1280\Desktop\SHRP2rawdata\5\new_divided\scene_019_fire\frame_008735_post_tid2.jpg",
    r"C:\Users\s1280\Desktop\SHRP2rawdata\5\new_divided\scene_019_fire\frame_008739_post_tid2.jpg",
    r"C:\Users\s1280\Desktop\SHRP2rawdata\5\new_divided\scene_019_fire\frame_008742_post_tid2.jpg",
    r"C:\Users\s1280\Desktop\SHRP2rawdata\5\new_divided\scene_019_fire\frame_008745_post_tid2.jpg",
    r"C:\Users\s1280\Desktop\SHRP2rawdata\5\new_divided\scene_019_fire\frame_008748_post_tid2.jpg",
    r"C:\Users\s1280\Desktop\SHRP2rawdata\5\new_divided\scene_019_fire\frame_008751_post_tid2.jpg",
    r"C:\Users\s1280\Desktop\SHRP2rawdata\5\new_divided\scene_019_fire\frame_008754_post_tid2.jpg",
    r"C:\Users\s1280\Desktop\SHRP2rawdata\5\new_divided\scene_019_fire\frame_008757_post_tid2.jpg"
]

# 画像を読み込み（すべて同じサイズ前提）
images = [cv2.imread(p) for p in image_paths]

# 1行ずつ結合 (3枚ずつ)
row1 = np.hstack(images[0:3])
row2 = np.hstack(images[3:6])
row3 = np.hstack(images[6:9])

# 行を縦方向に結合
grid = np.vstack([row1, row2, row3])

# 保存
out_path = r"C:\Users\s1280\Desktop\SHRP2rawdata\5\new_divided\scene_019_fire\scene019_grid.jpg"
cv2.imwrite(out_path, grid)

print("Saved:", out_path)
