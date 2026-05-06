import os
import cv2

# 入力パス
risk_path = r"C:\Users\s1280\Desktop\SHRP2rawdata\4\images_risk"
ref_path = r"C:\Users\s1280\Desktop\SHRP2rawdata\4\frames_bottom_right"
save_path = os.path.join(risk_path, "combined_output")
os.makedirs(save_path, exist_ok=True)

for fname in os.listdir(risk_path):
    if fname.startswith("4_risk_frame_") and fname.endswith(".jpg"):
        # === フレーム番号抽出し5桁ゼロ埋め ===
        try:
            frame_id_str = fname.split("_")[-1].replace(".jpg", "")
            frame_id = int(frame_id_str)
        except ValueError:
            print(f"❌ 無効なファイル名: {fname}")
            continue

        frame_id_padded = f"{frame_id:05d}"

        # === 対応するファイル名を生成 ===
        risk_img_path = os.path.join(risk_path, fname)
        bottom_img_path = os.path.join(ref_path, f"frame_{frame_id_padded}.jpg")
        if not os.path.exists(bottom_img_path):
            print(f"⚠️ Missing bottom frame: {bottom_img_path}")
            continue

        # === 読み込み ===
        img1 = cv2.imread(risk_img_path)
        img2 = cv2.imread(bottom_img_path)
        if img1 is None or img2 is None:
            print(f"❌ Error reading: {fname}")
            continue

        # === 高さ揃える（必要に応じてリサイズ）===
        if img1.shape[0] != img2.shape[0]:
            height = min(img1.shape[0], img2.shape[0])
            img1 = cv2.resize(img1, (int(img1.shape[1] * height / img1.shape[0]), height))
            img2 = cv2.resize(img2, (int(img2.shape[1] * height / img2.shape[0]), height))

        # === 横に連結 ===
        combined = cv2.hconcat([img1, img2])

        # === 保存 ===
        save_name = f"concat_{frame_id_padded}.jpg"
        save_full_path = os.path.join(save_path, save_name)
        cv2.imwrite(save_full_path, combined)

        print(f"✅ Saved: {save_name}")

