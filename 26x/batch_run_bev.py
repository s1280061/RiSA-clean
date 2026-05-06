import os, glob, subprocess, re

ROOT = r"C:\Users\s1280\Desktop\SHRP2rawdata"
BASES = [3, 4, 5, 6]

# 元のスクリプト名（単発処理用）
SCRIPT = r"C:\Users\s1280\PycharmProjects\yolo_classify_project\26x\2-24_all_scripts_vx2xxx.py"

for b in BASES:
    vid_dir = os.path.join(ROOT, str(b), "new_divided")
    csv_dir = os.path.join(ROOT, str(b), "csv_divided")
    for vid in sorted(glob.glob(os.path.join(vid_dir, "scene_*.mp4"))):
        m = re.search(r"scene_(\d+)\.mp4$", os.path.basename(vid))
        if not m:
            continue
        scene_no = m.group(1)
        csv = os.path.join(csv_dir, f"scene_{scene_no}.csv")
        if not os.path.exists(csv):
            print(f"⚠ CSV not found → skip {vid}")
            continue

        print(f"\n▶ Processing {vid}")
        # サブプロセスで単発スクリプトを実行
        subprocess.run(
            ["python", SCRIPT, "--video", vid, "--csv", csv],
            check=False
        )
