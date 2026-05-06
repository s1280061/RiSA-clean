# run_all_scenes.py
# 使い方:
#   python run_all_scenes.py
#   python run_all_scenes.py --force              # 状態に関係なく再実行
#   python run_all_scenes.py --reset              # 状態ファイルを初期化
#   python run_all_scenes.py --dry-run            # 実行せず一覧だけ
#   python run_all_scenes.py --bases 3 4          # baseを絞る
#   python run_all_scenes.py --start 10 --end 20  # scene番号を範囲指定（全baseに適用）

import os
import sys
import json
import time
import argparse
import subprocess
from typing import Dict, List, Tuple

# ======== パス設定 ========
ROOT = r"C:\Users\s1280\Desktop\SHRP2rawdata"
MAIN = r"C:\Users\s1280\PycharmProjects\yolo_classify_project\26x\2-23_all_scripts_vx2xxx.py"
PY   = sys.executable  # 今のPythonで実行

# baseごとの最大scene番号（含む）
DEFAULT_BASES = {3: 268, 4: 237, 5: 187, 6: 337}

# 状態ファイル（処理済/失敗リスト）
STATE_PATH = os.path.join(os.path.dirname(MAIN), "run_all_state.json")


# ---------- ヘルパ ----------
def scene_paths(base: int, scene: int) -> Tuple[str, str, str]:
    """scene_id, video_path, csv_path を返す"""
    s = f"scene_{scene:03d}"
    video = os.path.join(ROOT, str(base), "new_divided", f"{s}.mp4")
    csv   = os.path.join(ROOT, str(base), "csv_divided", f"{s}.csv")
    return s, video, csv


def expected_outputs(video_path: str) -> Dict[str, str]:
    """
    MAINスクリプトの出力規約に合わせて、期待する出力ファイルのパスを推定。
      - 出力動画: {video}_combined_speed_yolo_vx1.mp4
      - 出力CSV : {video}_trajectories_combined.csv
      - context : 同ディレクトリ scene_XXX_context.json
      - llava   : 同ディレクトリ scene_XXX_llava.json
    """
    base_dir = os.path.dirname(video_path)
    name = os.path.splitext(os.path.basename(video_path))[0]  # scene_XXX
    scene_no = name.split("_")[1] if "_" in name else "unknown"

    return {
        "video_out": os.path.join(base_dir, f"{name}_combined_speed_yolo_vx1.mp4"),
        "traj_csv":  os.path.join(base_dir, f"{name}_trajectories_combined.csv"),
        "context":   os.path.join(base_dir, f"scene_{scene_no}_context.json"),
        "llava":     os.path.join(base_dir, f"scene_{scene_no}_llava.json"),
    }


def outputs_exist(video_path: str) -> bool:
    """
    「完了」と見なす条件。
    厳しめに: context.json と llava.json の両方があること（＋動画 or CSV があれば尚良）
    """
    outs = expected_outputs(video_path)
    must = [outs["context"], outs["llava"]]
    return all(os.path.exists(p) for p in must)


def load_state() -> Dict[str, List[str]]:
    if os.path.exists(STATE_PATH):
        try:
            with open(STATE_PATH, "r", encoding="utf-8") as f:
                data = json.load(f)
                return {"done": data.get("done", {}), "failed": data.get("failed", {})}
        except Exception:
            pass
    return {"done": {}, "failed": {}}


def save_state(done: Dict[str, List[str]], failed: Dict[str, List[str]]):
    tmp = STATE_PATH + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump({"done": done, "failed": failed}, f, ensure_ascii=False, indent=2)
    os.replace(tmp, STATE_PATH)


# ---------- メイン ----------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--force", action="store_true", help="状態や出力に関わらず強制再実行")
    ap.add_argument("--reset", action="store_true", help="状態ファイルを初期化してから実行")
    ap.add_argument("--dry-run", action="store_true", help="実行せず一覧のみ表示")
    ap.add_argument("--bases", nargs="*", type=int, help="処理する base を指定（例: --bases 3 5）")
    ap.add_argument("--start", type=int, default=None, help="scene開始番号（全base共通）")
    ap.add_argument("--end",   type=int, default=None, help="scene終了番号（全base共通・含む）")
    args = ap.parse_args()

    bases = DEFAULT_BASES.copy()
    if args.bases:
        bases = {b: DEFAULT_BASES[b] for b in args.bases if b in DEFAULT_BASES}

    if args.reset:
        save_state(done={}, failed={})
        print(f"[RESET] 状態ファイルを初期化しました: {STATE_PATH}")

    st = load_state()
    done   = {k: list(v) for k, v in st.get("done", {}).items()}
    failed = {k: list(v) for k, v in st.get("failed", {}).items()}

    print("=== Batch start ===")
    total = ok_cnt = skip_cnt = run_cnt = fail_cnt = 0

    for base, max_scene in bases.items():
        # scene範囲の確定
        s0 = args.start if args.start is not None else 0
        s1 = args.end   if args.end   is not None else max_scene
        s1 = min(s1, max_scene)

        print(f"\n[BASE {base}] {s0:03d}..{s1:03d}")

        done.setdefault(str(base), [])
        failed.setdefault(str(base), [])

        for sc in range(s0, s1 + 1):
            total += 1
            scene_id, video, csv = scene_paths(base, sc)

            # 入力存在チェック
            if not os.path.exists(video):
                print(f"  [SKIP] {scene_id} 動画なし -> {video}")
                skip_cnt += 1
                continue
            if not os.path.exists(csv):
                print(f"  [SKIP] {scene_id} CSVなし -> {csv}")
                skip_cnt += 1
                continue

            # 出力存在チェック
            have_outputs = outputs_exist(video)
            listed_done  = scene_id in done[str(base)]

            # スキップ条件
            if not args.force:
                if have_outputs and listed_done:
                    print(f"  [OK済]  {scene_id}")
                    skip_cnt += 1
                    continue
                if have_outputs and not listed_done:
                    # 出力はあるのに状態が古い → 状態を修復
                    done[str(base)].append(scene_id)
                    if scene_id in failed[str(base)]:
                        failed[str(base)].remove(scene_id)
                    save_state(done, failed)
                    print(f"  [FIX ] {scene_id} 出力あり → 状態を更新")
                    skip_cnt += 1
                    continue
                if (not have_outputs) and listed_done:
                    print(f"  [RETRY] {scene_id} 状態はOKだが出力が欠落 → 再実行")

            # 実行
            cmd = [PY, MAIN, "--video", video, "--csv", csv]
            print(f"  [RUN ] {scene_id} -> {os.path.basename(video)}")
            run_cnt += 1

            if args.dry_run:
                continue

            t0 = time.time()
            try:
                r = subprocess.run(cmd, shell=False)
                if r.returncode == 0 and outputs_exist(video):
                    # 正常完了
                    if scene_id not in done[str(base)]:
                        done[str(base)].append(scene_id)
                    if scene_id in failed[str(base)]:
                        failed[str(base)].remove(scene_id)
                    ok_cnt += 1
                    print(f"  [ OK ] {scene_id} ({time.time()-t0:.1f}s)")
                else:
                    # 失敗（リターンコード or 出力不足）
                    if scene_id not in failed[str(base)]:
                        failed[str(base)].append(scene_id)
                    fail_cnt += 1
                    print(f"  [FAIL] {scene_id} rc={r.returncode} outputs_ok={outputs_exist(video)}")
            except Exception as e:
                if scene_id not in failed[str(base)]:
                    failed[str(base)].append(scene_id)
                fail_cnt += 1
                print(f"  [ERR ] {scene_id} {e}")

            # 逐次保存（途中停止しても再開しやすい）
            save_state(done, failed)

    print("\n=== Batch done ===")
    print(f"Total:{total}  Run:{run_cnt}  OK:{ok_cnt}  Skip:{skip_cnt}  Fail:{fail_cnt}")
    save_state(done, failed)
    print(f"状態ファイル: {STATE_PATH}")


if __name__ == "__main__":
    main()
