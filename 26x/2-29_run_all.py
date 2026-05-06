import glob
import os
import subprocess

# === 実行したい統合スクリプト ===
INTEGRATED_SCRIPT = r"C:\Users\s1280\PycharmProjects\yolo_classify_project\26x\2-28_all_scripts_vx2xxx.py"

# === .venv の Python を必ず使う ===
PYTHON_EXE = r"C:\Users\s1280\PycharmProjects\yolo_classify_project\.venv\Scripts\python.exe"

# === 走査対象フォルダ（3 / 4 / 5 / 6） ===
ROOTS = [
    r"C:\Users\s1280\Desktop\SHRP2rawdata\3\new_divided",
    r"C:\Users\s1280\Desktop\SHRP2rawdata\4\new_divided",
    r"C:\Users\s1280\Desktop\SHRP2rawdata\5\new_divided",
    r"C:\Users\s1280\Desktop\SHRP2rawdata\6\new_divided",
]

for root in ROOTS:
    # scene_000.mp4 ~ scene_999.mp4 を全取得
    videos = sorted(glob.glob(os.path.join(root, "scene_*.mp4")))

    print(f"▶ {root}: {len(videos)} 本の動画を検出")

    for video in videos:
        # scene番号を取得
        scene_no = os.path.splitext(os.path.basename(video))[0].split("_")[1]

        # CSV は 1つ上の階層の csv_divided にある
        parent = os.path.dirname(os.path.dirname(video))  # 例： ...\3\
        csv_path = os.path.join(parent, "csv_divided", f"scene_{scene_no}.csv")

        if not os.path.exists(csv_path):
            print(f"⚠ CSV がないためスキップ: {csv_path}")
            continue

        print("\n=========================================")
        print(f"▶ Processing {video}")
        print("=========================================\n")

        # 統合スクリプトを .venv の Python で確実に起動
        cmd = [
            PYTHON_EXE,
            INTEGRATED_SCRIPT,
            "--video", video,
            "--csv", csv_path,
        ]

        subprocess.run(cmd)
