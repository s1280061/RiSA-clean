import os
import subprocess

# === 統合スクリプトのパス ===
INTEGRATED_SCRIPT = r"C:\Users\s1280\PycharmProjects\yolo_classify_project\26x\2-28_all_scripts_vx2xxx.py"

# === リストファイル ===
LIST_FILE = r"C:\Users\s1280\Desktop\SHRP2rawdata\image_folders_list.txt"

# === ベースパス ===
BASE = r"C:\Users\s1280\Desktop\SHRP2rawdata"
PYTHON_EXE = r"C:\Users\s1280\PycharmProjects\yolo_classify_project\.venv\Scripts\python.exe"

# === 1行ずつ読み取る ===
with open(LIST_FILE, "r", encoding="utf-8") as f:
    lines = [line.strip() for line in f if line.strip()]

print(f"▶ {len(lines)} 個の発火シーンを検出")

for scene_path in lines:

    parts = scene_path.split("\\")
    folder_no = parts[-3]
    scene_name = parts[-1]

    # "scene_020_fire" → "020"
    scene_num = scene_name.split("_")[1]

    video_path = os.path.join(BASE, folder_no, "new_divided", f"scene_{scene_num}.mp4")
    csv_path   = os.path.join(BASE, folder_no, "csv_divided", f"scene_{scene_num}.csv")

    print("\n====================================================")
    print(f"▶ 処理開始: folder={folder_no} scene={scene_num}")
    print(f"動画: {video_path}")
    print(f"CSV : {csv_path}")

    if not os.path.exists(video_path):
        print("❌ 動画が見つかりません → スキップ")
        continue
    if not os.path.exists(csv_path):
        print("❌ CSV が見つかりません → スキップ")
        continue

    # === 実行コマンド (.venv の python を使う) ===
    cmd = [
        PYTHON_EXE,
        INTEGRATED_SCRIPT,
        "--video", video_path,
        "--csv", csv_path,
        "--folder", folder_no,
    ]

    # === ローカル ByteTrack を読ませない ===
    env = os.environ.copy()
    env["PYTHONPATH"] = ""   # ← ByteTrack/ を無効化

    subprocess.run(cmd, env=env)
