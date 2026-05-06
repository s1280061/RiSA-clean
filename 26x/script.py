"""
統合版スクリプト v2 – 複数シーン対応版
- コマンドラインで複数シーンを指定可能
- 例: python script.py --scenes 5-19 5-20 3-7 6-105
"""

import os, re, time, types, json, math, sys
import cv2
import numpy as np
import pandas as pd
import torch
from collections import defaultdict, deque
from ultralytics import YOLO
from yolox.tracker.byte_tracker import BYTETracker
from PIL import ImageFont, ImageDraw, Image
from train_traj_seq2seq_px_rel import Seq2Seq
import argparse
from risk_assessment_api_xx1 import assess_risk_from_image_with_context
from stage1_env import assess_environment_stage1_from_frame
from latency_tracer import LatencyTracer

# ==== グローバル設定 ====
BASE_DATA_DIR = r"C:\Users\s1280\Desktop\SHRP2rawdata"
base_path = r"C:\Users\s1280\PycharmProjects\yolo_classify_project\26x"
font_path = r"C:\Users\s1280\PycharmProjects\yolo_classify_project\26x\fonts\RobotoMono-Regular.ttf"
turn_model_path = r"C:\Users\s1280\PycharmProjects\yolo_classify_project\runs\classify\turn_cls_with_noise_yolov8m3\weights\best.pt"
brake_model_path = r"C:\Users\s1280\PycharmProjects\yolo_classify_project\runs\classify\go_brake_with_noise_v3_f4\weights\best.pt"
traj_ckpt_path = r"C:\Users\s1280\PycharmProjects\yolo_classify_project\26x\checkpoints_traj_px_best15\best_ade_px.pt"

# ==== BEV 可視化のチューニング ====
EDGE_ALPHA = 0.2
EDGE_DILATE_ITER = 1
EDGE_KERNEL = (3, 3)
EDGE_UNSHARP = False
SHOW_SLIDING_WINDOWS = False
SLIDE_WIN_COLOR = (128, 128, 128)
SLIDE_WIN_THICK = 1

# === Stage1 環境認識（YES/NO） ===
ENV_REFRESH_EVERY_FRAMES = 100
DRAW_TTC_PANEL = False

# ==== 定数 ====
IMG_W, IMG_H = 360.0, 240.0
H_PAST, H_FUT = 30, 45
DEBUG_TRAJ = True
SPEED_MIN, SPEED_MAX = 0.0, 120.0
SPEED_COL_UNIT = "mph"
COORD_MODE = "pixel"
PRED_TYPE = "absolute_origin"
WIN = 5
LLAVA_COOLDOWN_FRAMES = 120
BEV_W = 240


# ========== ヘルパー関数（変更なし） ==========
def _yn_cap(s: str) -> str:
    return "Yes" if str(s).strip().upper() == "YES" else "No"


def speed_to_feature(v_kmh: float) -> float:
    v = (v_kmh - SPEED_MIN) / (SPEED_MAX - SPEED_MIN)
    return float(0.0 if v < 0 else 1.0 if v > 1 else v)


def to_kmh(x: float) -> float:
    return x * 1.60934 if SPEED_COL_UNIT == "mph" else (x * 3.6 if SPEED_COL_UNIT == "mps" else x)


def to_mps(x: float) -> float:
    return x * 0.44704 if SPEED_COL_UNIT == "mph" else (x if SPEED_COL_UNIT == "mps" else x / 3.6)


# [その他のヘルパー関数は省略 - 元のコードと同じ]
# simplify_for_llava, _iou_xyxy, draw_text_clean, etc...

def parse_scene_spec(spec: str):
    """
    シーン指定をパース
    例: "5-19" -> (5, 19)
        "3-007" -> (3, 7)
    """
    parts = spec.split('-')
    if len(parts) != 2:
        raise ValueError(f"無効なシーン指定: {spec}. フォーマット: subdir-scene_num (例: 5-19)")

    subdir = int(parts[0])
    scene_num = int(parts[1])
    return subdir, scene_num


def build_scene_paths(subdir: int, scene_num: int):
    """
    シーン番号からファイルパスを構築
    """
    scene_str = f"{scene_num:03d}"
    subdir_path = os.path.join(BASE_DATA_DIR, str(subdir))

    video_dir = os.path.join(subdir_path, "new_divided")
    csv_dir = os.path.join(subdir_path, "csv_divided")

    video_path = os.path.join(video_dir, f"scene_{scene_str}.mp4")
    csv_path = os.path.join(csv_dir, f"scene_{scene_str}.csv")

    # 出力パス
    output_video_path = os.path.join(video_dir, f"scene_{scene_str}_with_bev.mp4")
    output_csv_path = os.path.join(video_dir, f"scene_{scene_str}_with_bev_traj.csv")
    latency_csv_path = os.path.join(video_dir, f"scene_{scene_str}_latency.csv")
    latency_jsonl_path = os.path.join(video_dir, f"scene_{scene_str}_latency.jsonl")
    fire_dir = os.path.join(video_dir, f"scene_{scene_str}_fire")
    context_json_path = os.path.join(video_dir, f"scene_{scene_str}_context.json")
    llava_json_path = os.path.join(video_dir, f"scene_{scene_str}_llava.json")

    return {
        'subdir': subdir,
        'scene_num': scene_num,
        'scene_str': scene_str,
        'video_path': video_path,
        'csv_path': csv_path,
        'output_video_path': output_video_path,
        'output_csv_path': output_csv_path,
        'latency_csv_path': latency_csv_path,
        'latency_jsonl_path': latency_jsonl_path,
        'fire_dir': fire_dir,
        'context_json_path': context_json_path,
        'llava_json_path': llava_json_path
    }


def process_single_scene(paths: dict, models: dict):
    """
    1つのシーンを処理する関数
    """
    print("\n" + "=" * 70)
    print(f"🎬 処理開始: {paths['subdir']}/scene_{paths['scene_str']}")
    print("=" * 70)
    print(f"📹 動画: {paths['video_path']}")
    print(f"📊 CSV: {paths['csv_path']}")

    # ファイル存在チェック
    if not os.path.exists(paths['video_path']):
        print(f"❌ エラー: 動画ファイルが見つかりません")
        return False

    if not os.path.exists(paths['csv_path']):
        print(f"❌ エラー: CSVファイルが見つかりません")
        return False

    # 出力ディレクトリ作成
    os.makedirs(paths['fire_dir'], exist_ok=True)

    # レイテンシトレーサー初期化
    tr = LatencyTracer(paths['latency_csv_path'], paths['latency_jsonl_path'])
    print(f"📈 レイテンシログ: {paths['latency_csv_path']}")

    # BEV変換行列読み込み
    src = np.load(os.path.join(base_path, "src_points_3_forward.npy"))
    dst = np.load(os.path.join(base_path, "dst_points_3_forward.npy"))
    M = cv2.getPerspectiveTransform(src, dst)
    Minv = cv2.getPerspectiveTransform(dst, src)
    scale_px_per_m = 50.82

    # CSV読み込み
    df_speed = pd.read_csv(paths['csv_path'], dtype={"frame": int}, low_memory=False)

    # 速度列を探す
    def _find_speed_col(df):
        candidates = ["vtti.speed_gps", "vtti.gps_speed", "speed", "Speed", "ego_speed", "veh_speed"]
        for c in candidates:
            if c in df.columns:
                return c
        raise ValueError(f"速度列が見つかりません")

    SPEED_COL = _find_speed_col(df_speed)
    print(f"[speed] column='{SPEED_COL}'")

    # トラッカー初期化
    args_bt = types.SimpleNamespace(track_thresh=0.3, match_thresh=0.7,
                                    track_buffer=30, frame_rate=15, mot20=False)
    tracker = BYTETracker(args_bt)

    # 動画キャプチャ
    cap = cv2.VideoCapture(paths['video_path'])
    fps = cap.get(cv2.CAP_PROP_FPS)
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    # 動画出力
    out = cv2.VideoWriter(paths['output_video_path'],
                          cv2.VideoWriter_fourcc(*"mp4v"),
                          fps, (360 + BEV_W, 300))

    csv_first_frame = int(df_speed["frame"].iloc[0])
    offset = csv_first_frame
    print(f"開始オフセット: {offset} フレーム")

    # 変数初期化
    env_stage1_cache = {"result": None, "last_frame": -10 ** 9}
    traj_hist = defaultdict(list)
    turn_buffer = defaultdict(list)
    brake_buffer = defaultdict(list)
    llava_status_map = {}
    pred_inside = defaultdict(lambda: False)
    last_fire_frame = defaultdict(lambda: -10 ** 9)
    risk_zone_counter = defaultdict(int)
    curr_in_risk = defaultdict(lambda: False)
    sample_counter = defaultdict(int)
    detection_status = defaultdict(bool)
    track_bbox = defaultdict(lambda: None)
    prev_in_risk = defaultdict(lambda: False)
    last_origin = defaultdict(lambda: None)
    last_pred_px = {}

    turn_label_win = defaultdict(lambda: deque(maxlen=WIN))
    turn_conf_win = defaultdict(lambda: deque(maxlen=WIN))
    turn_probs_win = defaultdict(lambda: deque(maxlen=WIN))
    brake_label_win = defaultdict(lambda: deque(maxlen=WIN))
    brake_conf_win = defaultdict(lambda: deque(maxlen=WIN))
    brake_probs_win = defaultdict(lambda: deque(maxlen=WIN))

    turn_final_conf_map = defaultdict(lambda: 0.0)
    brake_final_conf_map = defaultdict(lambda: 0.0)

    llava_cooldown_until = -1
    FIRE_EVERY_N_FRAMES = max(1, int(round(fps / 5.0)))

    context_records = []
    llava_records = []
    final_snapshots = []
    trajectories = []
    track_history = {}

    frame_idx = 0
    start_time = time.time()

    print("▶ 統合処理を開始します...")

    try:
        while cap.isOpened():
            is_warmup = frame_idx < 50

            # フレーム読み取り
            with tr.span("frame_read", meta={"frame": int(frame_idx), "warmup": bool(is_warmup)}, gpu=False):
                ok, frame = cap.read()

            if not ok:
                break

            frame_raw = frame.copy()

            # [ここに元のメインループの処理を全て挿入]
            # 注意: tr, models, paths などを使用

            # Stage1環境認識
            if (frame_idx == 0) or (frame_idx % ENV_REFRESH_EVERY_FRAMES == 0) or (env_stage1_cache["result"] is None):
                try:
                    with tr.span("stage1_env", meta={"frame": int(frame_idx), "warmup": bool(is_warmup)}, gpu=True):
                        env_stage1_cache["result"] = assess_environment_stage1_from_frame(frame_raw)
                    env_stage1_cache["last_frame"] = frame_idx
                except Exception as e:
                    print(f"[Stage1] error: {e}")
                    env_stage1_cache["result"] = {"rural_area": "NO", "city": "NO", "snowy": "NO",
                                                  "sunny": "NO", "rainy": "NO"}

            # [残りの処理 - 元のコードと同じ]
            # ...（省略）

            frame_idx += 1

            if frame_idx % 100 == 0:
                elapsed = time.time() - start_time
                pct = (frame_idx) / max(1, total_frames) * 100
                print(f"  [{frame_idx:6d}/{total_frames}] {pct:5.1f}% elapsed:{elapsed:6.1f}s")

    except KeyboardInterrupt:
        print("⏹️ 中断されました")
        return False

    except Exception as e:
        print(f"❌ エラー: {e}")
        import traceback
        traceback.print_exc()
        return False

    finally:
        # クリーンアップ
        try:
            cap.release()
            out.release()
            tr.flush()
        except:
            pass

        # 結果保存
        try:
            if trajectories:
                pd.DataFrame(trajectories).to_csv(paths['output_csv_path'], index=False)

            with open(paths['context_json_path'], "w", encoding="utf-8") as f:
                json.dump(context_records, f, ensure_ascii=False, indent=2)

            with open(paths['llava_json_path'], "w", encoding="utf-8") as f:
                json.dump(llava_records, f, ensure_ascii=False, indent=2)

            total_elapsed = time.time() - start_time
            print(f"\n✅ 完了: {paths['subdir']}/scene_{paths['scene_str']}")
            print(f"   処理時間: {total_elapsed:.1f}秒")
            print(f"   フレーム数: {frame_idx:,}")

        except Exception as e:
            print(f"⚠️ 保存エラー: {e}")

    return True


def main():
    parser = argparse.ArgumentParser(
        description='統合版スクリプト - 複数シーン対応',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog='''
使用例:
  # 単一シーン
  python script.py --scenes 5-20

  # 複数シーン
  python script.py --scenes 5-19 5-20 3-7 6-105

  # 旧形式（後方互換）
  python script.py --video path/to/video.mp4 --csv path/to/csv.csv
        '''
    )

    parser.add_argument("--scenes", nargs='+', help="処理するシーン (例: 5-19 5-20 3-7)")
    parser.add_argument("--video", type=str, help="動画ファイルパス（旧形式）")
    parser.add_argument("--csv", type=str, help="CSVファイルパス（旧形式）")

    args = parser.parse_args()

    # モデル読み込み（1回だけ）
    print("=" * 70)
    print("🚀 モデル読み込み中...")
    print("=" * 70)

    det_device = "cuda" if torch.cuda.is_available() else "cpu"
    cls_device = "cuda" if torch.cuda.is_available() else "cpu"
    traj_device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    det_model = YOLO("yolov8n.pt").to(det_device)
    turn_model = YOLO(turn_model_path).to(cls_device)
    brake_model = YOLO(brake_model_path).to(cls_device)

    # 軌跡予測モデル
    ckpt = torch.load(traj_ckpt_path, map_location=traj_device)
    cfg = ckpt.get("cfg", {})
    hidden = int(cfg.get("hidden", 256))
    layers = int(cfg.get("layers", 3))
    dropout = float(cfg.get("dropout", 0.2))

    traj_model = Seq2Seq(input_size=5, hidden=hidden, n_layers=layers,
                         dropout=dropout, out_size=2).to(traj_device)
    traj_model.load_state_dict(ckpt["state_dict"])
    traj_model.eval()

    models = {
        'det_model': det_model,
        'turn_model': turn_model,
        'brake_model': brake_model,
        'traj_model': traj_model
    }

    print("✅ モデル読み込み完了")

    # シーン処理
    scenes_to_process = []

    if args.scenes:
        # 新形式: --scenes 5-19 5-20
        for spec in args.scenes:
            try:
                subdir, scene_num = parse_scene_spec(spec)
                paths = build_scene_paths(subdir, scene_num)
                scenes_to_process.append(paths)
            except Exception as e:
                print(f"⚠️ 無効なシーン指定をスキップ: {spec} ({e})")

    elif args.video and args.csv:
        # 旧形式: --video --csv
        print("⚠️ 旧形式の引数を使用しています。--scenes の使用を推奨します。")
        # 旧形式の処理（省略 - 必要に応じて実装）

    else:
        print("❌ エラー: --scenes または --video と --csv を指定してください")
        parser.print_help()
        return

    if not scenes_to_process:
        print("❌ 処理するシーンがありません")
        return

    # 処理予定の表示
    print(f"\n📊 処理予定のシーン: {len(scenes_to_process)}個")
    print("=" * 70)
    for p in scenes_to_process:
        print(f"  {p['subdir']}/scene_{p['scene_str']}")
    print("=" * 70)

    # 確認
    response = input("\n実行しますか？ (y/n): ")
    if response.lower() != 'y':
        print("❌ 処理を中断しました")
        return

    # 各シーンを処理
    success_count = 0
    fail_count = 0

    for i, paths in enumerate(scenes_to_process, 1):
        print(f"\n{'=' * 70}")
        print(f"🎬 [{i}/{len(scenes_to_process)}] {paths['subdir']}/scene_{paths['scene_str']}")
        print(f"{'=' * 70}")

        success = process_single_scene(paths, models)

        if success:
            success_count += 1
        else:
            fail_count += 1
            response = input("\n続行しますか？ (y/n): ")
            if response.lower() != 'y':
                print("❌ 処理を中断しました")
                break

    # 結果サマリー
    print("\n" + "=" * 70)
    print("📊 処理結果サマリー")
    print("=" * 70)
    print(f"✅ 成功: {success_count}件")
    print(f"❌ 失敗: {fail_count}件")
    print(f"📁 合計: {success_count + fail_count}件")
    print("\n🏁 すべての処理が完了しました")


if __name__ == "__main__":
    main()