# run_scenes.py
import sys, os, subprocess, argparse

def parse_pair(token: str):
    t = token.strip()
    if t.lower() == "s" or t == "":     # 先頭の 's' は無視
        return None
    if "-" not in t:
        raise ValueError(f"'{t}' は 'folder-scene' 形式ではありません（例: 5-19）")
    a, b = t.split("-", 1)
    if not (a.strip().isdigit() and b.strip().isdigit()):
        raise ValueError(f"'{t}' の数字が不正です")
    return int(a), int(b)

def build_paths(root, folder_num, scene_num, video_subdir, csv_subdir):
    tag = f"scene_{scene_num:03d}"
    v = os.path.join(root, f"{folder_num}", video_subdir, f"{tag}.mp4")
    c = os.path.join(root, f"{folder_num}", csv_subdir,  f"{tag}.csv")
    return v, c

def main():
    p = argparse.ArgumentParser(description="Batch runner for 2-25_all_scripts_vx2xxx.py")
    p.add_argument("pairs", nargs="+", help="folder-scene 羅列 (例: 5-19 5-20 3-7)")
    p.add_argument("--python", default=sys.executable, help="Python 実行パス")
    p.add_argument("--script", default="2-25_all_scripts_vx2xxx.py", help="元スクリプトのファイル名/パス")
    p.add_argument("--root", default=r"C:\Users\s1280\Desktop\SHRP2rawdata", help="SHRP2 ルート")
    p.add_argument("--video-subdir", default="new_divided", help="動画サブフォルダ")
    p.add_argument("--csv-subdir",   default="csv_divided", help="CSVサブフォルダ")
    p.add_argument("--dry-run", action="store_true", help="実行せずコマンドのみ表示")
    p.add_argument("--pass", dest="pass_through", nargs=argparse.REMAINDER,
                   help="以降は元スクリプトにそのまま渡す（例: --pass --gpu 0 --foo bar）")
    args = p.parse_args()

    targets = []
    for t in args.pairs:
        try:
            ps = parse_pair(t)
            if ps: targets.append(ps)
        except Exception as e:
            print(f"[スキップ] {t}: {e}")

    if not targets:
        print("有効なターゲットがありません。例: 5-19 5-20 3-7")
        sys.exit(1)

    for (folder, scene) in targets:
        video_path, csv_path = build_paths(args.root, folder, scene, args.video_subdir, args.csv_subdir)

        missing = [p for p in (video_path, csv_path) if not os.path.exists(p)]
        if missing:
            print("⚠️ 見つからないためスキップ:")
            for m in missing: print("   -", m)
            continue

        cmd = [args.python, args.script, "--video", video_path, "--csv", csv_path]
        if args.pass_through:
            cmd += args.pass_through

        print("\n▶ 実行:", " ".join(f'"{x}"' if " " in x else x for x in cmd))
        if args.dry_run:
            continue

        code = subprocess.call(cmd)
        if code != 0:
            print(f"❌ 失敗: folder={folder}, scene={scene} (returncode={code})")
        else:
            print(f"✅ 完了: folder={folder}, scene={scene})")

if __name__ == "__main__":
    main()
