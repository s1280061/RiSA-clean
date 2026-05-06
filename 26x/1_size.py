import cv2

input_path = r"C:\Users\s1280\Desktop\SHRP2rawdata\3\new\3_combined_frame_overlay_1.mp4"
output_path = input_path.replace(".mp4", "_resized_1280x720.mp4")

# 動画読み込み
cap = cv2.VideoCapture(input_path)

# 元サイズの取得（capを定義した後に実行）
original_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
original_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
print(f"🎥 元のサイズ: {original_w}x{original_h}")

fourcc = cv2.VideoWriter_fourcc(*'mp4v')
fps = cap.get(cv2.CAP_PROP_FPS)

# 出力設定：1280x720
out = cv2.VideoWriter(output_path, fourcc, fps, (1280, 720))

while True:
    ret, frame = cap.read()
    if not ret:
        break
    resized = cv2.resize(frame, (1280, 720))
    out.write(resized)

cap.release()
out.release()
print("✅ リサイズ完了:", output_path)
