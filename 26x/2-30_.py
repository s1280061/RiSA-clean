import os
import glob

ROOT = r"C:\Users\s1280\Desktop\SHRP2rawdata"
OUTPUT_FILE = os.path.join(ROOT, "image_folders_list.txt")

# ======== フォルダ範囲の定義 ========
CONFIG = {
    3: (0, 268, "_fire"),
    4: (0, 237, "_fire"),
    5: (0, 337, "_fire"),   # ← 修正：ここも _fire を付ける
    6: (0, 337, "_fire"),
}

# ======== 画像拡張子 ========
IMAGE_EXT = ("*.jpg", "*.png", "*.jpeg")


def has_images(folder):
    """フォルダ内に画像ファイルが1枚以上あるか判定"""
    for ext in IMAGE_EXT:
        files = glob.glob(os.path.join(folder, ext))
        if len(files) > 0:
            return True
    return False


results = []

# ======== メイン処理 ========
for num, (start, end, suffix) in CONFIG.items():
    base_path = os.path.join(ROOT, str(num), "new_divided")

    for idx in range(start, end + 1):
        scene_name = f"scene_{idx:03d}{suffix}"
        scene_path = os.path.join(base_path, scene_name)

        if os.path.isdir(scene_path) and has_images(scene_path):
            results.append(scene_path)
            print(f"[画像あり] {scene_path}")
        else:
            print(f"[空 or なし] {scene_path}")

# ======== 結果を保存 ========
with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
    for path in results:
        f.write(path + "\n")

print("\n=== 完了！ ===")
print(f"画像入りフォルダ一覧を保存しました → {OUTPUT_FILE}")
