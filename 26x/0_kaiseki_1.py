import cv2
img = cv2.imread(r"C:\Users\s1280\Desktop\SHRP2rawdata\3\new\risk_frames_scene_0\frame_051977.jpg")
print(img is None)  # Trueなら破損
