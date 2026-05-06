import os, sys, subprocess

ROOT = r"C:\Users\s1280\Desktop\SHRP2rawdata"
VIDEO_SUBDIR = "new_divided"
CSV_SUBDIR   = "csv_divided"
SCRIPT_PATH  = "2-25_all_scripts_vx2xxx.py"
PYTHON_PATH  = sys.executable  # 現在のPythonを使用

# === 対象フォルダ ===
folders = ["3", "4", "5", "6"]

for folder in folders:
    video_dir = os.path.join(ROOT, folder, VIDEO_SUBDIR)
    csv_dir   = os.path.join(ROOT, folder, CSV_SUBDIR)

    if not os.path.exists(video_dir):
        print(f"⚠️ フォルダが存在しません: {video_dir}")
        continue
    if not os.path.exists(csv_dir):
        print(f"⚠️ CSVフォルダが存在しません: {csv_dir}")
        continue

    video_files = sorted([f for f in os.listdir(video_dir) if f.lower().endswith(".mp4")])
    print(f"\n📁 Folder {folder}: {len(video_files)} 件のシーンを検出")

    for vf in video_files:
        scene_name = os.path.splitext(vf)[0]  # e.g., "scene_020"
        csv_path = os.path.join(csv_dir, f"{scene_name}.csv")
        video_path = os.path.join(video_dir, vf)

        if not os.path.exists(csv_path):
            print(f"⚠️ CSVが見つからないためスキップ: {scene_name}")
            continue

        print(f"\n▶ 実行中: folder={folder}, {scene_name}")
        cmd = [PYTHON_PATH, SCRIPT_PATH, "--video", video_path, "--csv", csv_path]
        ret = subprocess.call(cmd)
        if ret != 0:
            print(f"❌ 失敗: {folder}-{scene_name} (code={ret})")
        else:
            print(f"✅ 完了: {folder}-{scene_name}")
