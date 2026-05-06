import os, subprocess, sys

ROOT = r"C:\Users\s1280\Desktop\SHRP2rawdata"
VIDEO_SUBDIR = "new_divided"
CSV_SUBDIR   = "csv_divided"
SCRIPT_PATH  = "2-25_all_scripts_vx2xxx.py"
PYTHON_PATH  = sys.executable  # 現在のPython

# --- 対象フォルダ一覧 ---
folders = [f for f in os.listdir(ROOT) if os.path.isdir(os.path.join(ROOT, f))]

for folder in folders:
    video_dir = os.path.join(ROOT, folder, VIDEO_SUBDIR)
    csv_dir   = os.path.join(ROOT, folder, CSV_SUBDIR)
    if not (os.path.exists(video_dir) and os.path.exists(csv_dir)):
        continue

    # scene_xxx を走査
    for filename in os.listdir(video_dir):
        if not filename.lower().endswith(".mp4"):
            continue
        scene_name = os.path.splitext(filename)[0]
        csv_path = os.path.join(csv_dir, f"{scene_name}.csv")
        video_path = os.path.join(video_dir, filename)
        if not os.path.exists(csv_path):
            print(f"⚠️ CSVなしスキップ: {video_path}")
            continue

        print(f"\n▶ 実行中: folder={folder}, scene={scene_name}")
        cmd = [PYTHON_PATH, SCRIPT_PATH, "--video", video_path, "--csv", csv_path]
        ret = subprocess.call(cmd)
        if ret != 0:
            print(f"❌ 失敗: {folder}-{scene_name} (code={ret})")
        else:
            print(f"✅ 完了: {folder}-{scene_name}")
