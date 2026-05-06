import cv2
import numpy as np
import math

# === 初期設定 ===
img_path = r"C:\Users\s1280\Desktop\SHRP2rawdata\5\frames_bottom_left\frame_72000.jpg"  # 適宜変更
img = cv2.imread(img_path)
clone = img.copy()
points = []

# === 距離計算関数 ===
def euclidean_distance(p1, p2):
    return math.hypot(p2[0] - p1[0], p2[1] - p1[1])

# === マウスコールバック関数 ===
def click_event(event, x, y, flags, param):
    global points, clone
    if event == cv2.EVENT_LBUTTONDOWN:
        points.append((x, y))
        cv2.circle(clone, (x, y), 3, (0, 0, 255), -1)

        if len(points) == 2:
            d = euclidean_distance(points[0], points[1])
            print(f"🧭 距離: {d:.2f} px")
            cv2.line(clone, points[0], points[1], (255, 0, 0), 1)
            cv2.putText(clone, f"{d:.1f}px", (x + 10, y),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

        cv2.imshow("Distance Tool", clone)

# === 実行 ===
cv2.imshow("Distance Tool", clone)
cv2.setMouseCallback("Distance Tool", click_event)
cv2.waitKey(0)
cv2.destroyAllWindows()
