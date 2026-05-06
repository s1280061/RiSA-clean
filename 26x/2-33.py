import os
import shutil
import glob

ROOT = r"D:\RISA_raw_data\6\traj"

# === BEV フォルダを作成（なければ作成） ===
folders = ["bev1", "bev2", "bev3"]
for f in folders:
    os.makedirs(os.path.join(ROOT, f), exist_ok=True)

# === bev1/bev2/bev3 に混在している全ファイルを拾う ===
all_files = []
for f in folders:
    all_files += glob.glob(os.path.join(ROOT, f, "*.csv"))

print(f"Found {len(all_files)} BEV files.\n")

# === 仕分け開始 ===
for f in all_files:
    fname = os.path.basename(f)

    # bev3
    if "with_bev_with_bev_with_bev_traj.csv" in fname:
        dst = os.path.join(ROOT, "bev3", fname)
        print(f"Move: {fname} → bev3")
        shutil.move(f, dst)
        continue

    # bev2
    if "with_bev_with_bev_traj.csv" in fname:
        dst = os.path.join(ROOT, "bev2", fname)
        print(f"Move: {fname} → bev2")
        shutil.move(f, dst)
        continue

    # bev1
    if "with_bev_traj.csv" in fname:
        dst = os.path.join(ROOT, "bev1", fname)
        print(f"Move: {fname} → bev1")
        shutil.move(f, dst)
        continue

print("\n✔ BEV ファイルの再仕分け完了！")
