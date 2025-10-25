"""
統合版スクリプト v2 – レイテンシ計測対応版
- モジュール別レイテンシをCSV+JSONLに記録
- GPU同期による厳密な計測
- ウォームアップフラグ付き（最初50フレーム）
"""

import os, re, time, types, json, math
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

# ========== レイテンシ計測の初期化 ==========
from latency_tracer import LatencyTracer

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
env_stage1_cache = {"result": None, "last_frame": -10**9}

DRAW_TTC_PANEL = False

context_records = []
llava_records   = []

def _yn_cap(s: str) -> str:
    return "Yes" if str(s).strip().upper() == "YES" else "No"

IMG_W, IMG_H = 360.0, 240.0
H_PAST, H_FUT = 30, 45
DEBUG_TRAJ = True

SPEED_MIN, SPEED_MAX = 0.0, 120.0
def speed_to_feature(v_kmh: float) -> float:
    v = (v_kmh - SPEED_MIN) / (SPEED_MAX - SPEED_MIN)
    return float(0.0 if v < 0 else 1.0 if v > 1 else v)

SPEED_COL_UNIT = "mph"

def to_kmh(x: float) -> float:
    return x*1.60934 if SPEED_COL_UNIT == "mph" else (x*3.6 if SPEED_COL_UNIT == "mps" else x)

def to_mps(x: float) -> float:
    return x*0.44704 if SPEED_COL_UNIT == "mph" else (x if SPEED_COL_UNIT == "mps" else x/3.6)

def simplify_for_llava(ctx: dict) -> dict:
    def _pct_int(x):
        try:
            return int(round(float(x) * 100.0))
        except:
            return 0

    def _round1(x):
        try:
            return round(float(x), 1)
        except:
            return 0.0

    out = {}
    s1 = ctx.get("stage1_env", {})
    ego = ctx.get("ego_vehicle", {})
    perc = ctx.get("perception", {})
    ts = perc.get("turn_signal", {})
    br = perc.get("brake", {})
    det = ctx.get("detected_vehicles", {})

    def _probs_pct_filtered(d):
        d = {k: _pct_int(v) for k, v in (d or {}).items()}
        d = {k: v for k, v in d.items() if v >= 1}
        if not d:
            return {}
        return d

    out["stage1_env"] = dict(s1)

    out["ego_vehicle"] = {
        "speed_kmh": _round1(ego.get("speed_kmh", 0.0)),
        "risk_zone": bool(ego.get("risk_zone", False)),
        "risk_zone_predicted": bool(ego.get("risk_zone_predicted", False)),
    }

    out["perception"] = {
        "turn_signal": {
            "final": ts.get("final", "off"),
            "final_conf_pct": _pct_int(ts.get("final_conf_window", 0.0)),
            "probs_pct": _probs_pct_filtered(ts.get("probs_window_avg")),
            "window_size": int(ts.get("window_size", 0)),
            "window_counts": {k: int(v) for k, v in (ts.get("window_counts") or {}).items()},
        },
        "brake": {
            "final": br.get("final", "off"),
            "final_conf_pct": _pct_int(br.get("final_conf_window", 0.0)),
            "probs_pct": _probs_pct_filtered(br.get("probs_window_avg")),
            "window_size": int(br.get("window_size", 0)),
            "window_counts": {k: int(v) for k, v in (br.get("window_counts") or {}).items()},
        },
    }

    out["detected_vehicles"] = {"count": int(det.get("count", 0))}
    return out

def _iou_xyxy(a, b):
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    inter_x1 = max(ax1, bx1); inter_y1 = max(ay1, by1)
    inter_x2 = min(ax2, bx2); inter_y2 = min(ay2, by2)
    iw = max(0.0, inter_x2 - inter_x1)
    ih = max(0.0, inter_y2 - inter_y1)
    inter = iw * ih
    if inter <= 0: return 0.0
    area_a = max(0.0, (ax2-ax1)) * max(0.0, (ay2-ay1))
    area_b = max(0.0, (bx2-bx1)) * max(0.0, (by2-by1))
    union = area_a + area_b - inter + 1e-6
    return inter / union

def _class_for_track(x0, y0, x1, y1, dets_with_cls, iou_thr=0.3):
    best, best_cls = 0.0, None
    for dx1, dy1, dx2, dy2, dconf, dcls in dets_with_cls:
        iou = _iou_xyxy((x0, y0, x1, y1), (dx1, dy1, dx2, dy2))
        if iou > best:
            best, best_cls = iou, dcls
    return best_cls if best >= iou_thr else None
# 既存importの下あたりに
from contextlib import contextmanager

def _global_from_meta(m: dict, offset: int) -> int:
    # CSVにフレーム列があればそれを最優先。無ければ offset + frame_idx
    return int(m["csv_frame"]) if m.get("csv_frame") is not None else int(offset + m["frame_idx"])

@contextmanager
def span_with_frame_edges(tr, name: str, base_meta: dict, gpu: bool, offset: int):
    """
    ts_start 時点のローカル/グローバルフレームを *_start として記録し、
    with を抜けた瞬間に、ts_end 時点のローカル/グローバルフレームを *_end として
    「空の end 印 span」で記録する。
    """
    meta_start = dict(base_meta)
    meta_start["frame_local_at_ts_start"]  = int(base_meta["frame_idx"])
    meta_start["frame_global_at_ts_start"] = _global_from_meta(base_meta, offset)

    # 実体の処理を本来の span に委譲（開始タイムスタンプはこの時に確定）
    with tr.span(name, meta=meta_start, gpu=gpu):
        yield

    # 終了メタは、直後のフレーム値をそのまま残す
    meta_end = dict(base_meta)
    meta_end["frame_local_at_ts_end"]  = int(base_meta["frame_idx"])
    meta_end["frame_global_at_ts_end"] = _global_from_meta(base_meta, offset)

    # 「ゼロ長の end 印 span」— 既存CSV/JSONLに確実に残せる
    with tr.span(f"{name}#end", meta=meta_end, gpu=gpu):
        pass

# ★ 追加: 黄→赤の"間"の連続データ
from collections import defaultdict

# tidごとに、黄→赤の間の連続サンプルを保存
# 例: between_series[tid] = [{"frame": 12345, "ms_from_yellow": 66.667, "ego_speed_kmh": 42.3, "stop_m": 4.2}, ...]
between_series = defaultdict(list)

# 連続記録中のtid（黄色検出後〜赤到達まで）
pending_series_tids = set()


# ---- Path settings for GitHub demo ----
BASE_DIR = os.path.dirname(__file__)
SAMPLE_DIR = os.path.join(BASE_DIR, "samples")

video_path = os.path.join(SAMPLE_DIR, "demo_scene.mp4")
csv_path   = os.path.join(SAMPLE_DIR, "demo_scene.csv")
font_path  = os.path.join(BASE_DIR, "src", "fonts", "RobotoMono-Regular.ttf")

turn_model_path  = os.path.join(BASE_DIR, "src", "weights", "turn_cls_best.pt")
brake_model_path = os.path.join(BASE_DIR, "src", "weights", "brake_go_best.pt")

# ---- Notes ----
# The sample video and CSV are lightweight demo examples.
# For reproducibility, replace these paths with your own dataset if needed.

parser = argparse.ArgumentParser()
parser.add_argument("--video", type=str, help="path to scene_###.mp4")
parser.add_argument("--csv",   type=str, help="path to scene_###.csv")
args, _ = parser.parse_known_args()
if args.video: video_path = args.video
if args.csv:   csv_path   = args.csv

output_video_path = os.path.splitext(video_path)[0] + "_with_bev.mp4"
output_csv_path   = os.path.splitext(video_path)[0] + "_with_bev_traj.csv"

BASE_SAVE_DIR = r"C:\Users\s1280\Desktop\SHRP2rawdata\5\new_divided"
scene_match = re.search(r"scene_(\d+)", os.path.basename(video_path))
scene_number = int(scene_match.group(1)) if scene_match else 0
scene_dir = os.path.join(BASE_SAVE_DIR, f"scene_{scene_number:03d}")
os.makedirs(scene_dir, exist_ok=True)

base_dir = os.path.dirname(video_path)
scene_no = scene_number

FIRE_DIR = os.path.join(scene_dir, "fire_snapshots")
os.makedirs(FIRE_DIR, exist_ok=True)

# 黄→赤遷移時間の保存リスト
risk_time_records = []
first_yellow_frame = {}
first_red_frame = {}

# ========== レイテンシ計測パスの設定 ==========
latency_csv_path = os.path.join(base_dir, f"scene_{scene_no}_latency.csv")
latency_jsonl_path = os.path.join(base_dir, f"scene_{scene_no}_latency.jsonl")
print("[INFO] Starting integrated processing...")
print(f"[INFO] Latency log (CSV): {latency_csv_path}")
print(f"[INFO] Latency log (JSONL): {latency_jsonl_path}")

# ========== 動画読み込み（offset推定） ==========
cap = cv2.VideoCapture(video_path)
if not cap.isOpened():
    print(f"[ERROR] Unable to open video: {video_path}")
    exit(1)

fps = cap.get(cv2.CAP_PROP_FPS)
total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
W_vid = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
H_vid = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

print(f"[INFO] Video file: {video_path}")
print(f"[INFO] Resolution: {W_vid}x{H_vid}, FPS: {fps:.2f}, Frames: {total_frames}")

offset = 0
try:
    if os.path.exists(csv_path):
        df_temp = pd.read_csv(csv_path, nrows=1)
        if "Frame" in df_temp.columns:
            offset = int(df_temp["Frame"].iloc[0])
            print(f"[INFO] CSV frame offset detected: {offset}")
except Exception as e:
    print(f"[WARN] Frame offset detection failed: {e}")

# ========== レイテンシ計測開始 ==========
tr = LatencyTracer(
    csv_path=latency_csv_path,
    jsonl_path=latency_jsonl_path,
    meta_keys=[
        "frame_idx",
        "csv_frame",
        "frame_local_at_ts_start",
        "frame_global_at_ts_start",
        "frame_local_at_ts_end",
        "frame_global_at_ts_end",
    ]
)

fourcc = cv2.VideoWriter_fourcc(*"mp4v")
out = cv2.VideoWriter(output_video_path, fourcc, fps, (720, 480))

print("[INFO] Loading models...")

# ========== YOLOv8検出モデル ==========
yolo_detect = YOLO(os.path.join(base_path, "best.pt"))
yolo_detect.to("cuda")

# ========== Seq2Seqモデル ==========
s2s_traj = Seq2Seq(
    input_size=2 + 1,
    hidden_size=256,
    num_layers=2,
    output_size=2,
    dropout=0.3
)
s2s_traj.to("cuda")
s2s_traj.load_state_dict(torch.load(
    os.path.join(base_path, "traj_seq2seq_best.pt"),
    map_location="cuda"
))
s2s_traj.eval()

# ========== 分類モデル ==========
try:
    turn_cls = YOLO(turn_model_path)
    turn_cls.to("cuda")
    print("[INFO] Turn signal model loaded.")
except Exception as e:
    print(f"[ERROR] Turn signal model loading failed: {e}")
    turn_cls = None

try:
    brake_cls = YOLO(brake_model_path)
    brake_cls.to("cuda")
    print("[INFO] Brake signal model loaded.")
except Exception as e:
    print(f"[ERROR] Brake signal model loading failed: {e}")
    brake_cls = None

print("[INFO] Models loaded successfully.")

# ========== BYTETracker初期化 ==========
tracker = BYTETracker(types.SimpleNamespace(
    track_thresh=0.5,
    track_buffer=30,
    match_thresh=0.8,
    mot20=False
))

# ========== CSV読み込み ==========
df_speed = None
if os.path.exists(csv_path):
    try:
        df_speed = pd.read_csv(csv_path)
        if "Frame" not in df_speed.columns:
            print("[WARN] 'Frame' column not found in CSV. Assuming sequential order from 0.")
            df_speed["Frame"] = df_speed.index
        if "Speed" not in df_speed.columns:
            print("[WARN] 'Speed' column not found in CSV. Using 0 for all speeds.")
            df_speed["Speed"] = 0.0
        df_speed = df_speed[["Frame", "Speed"]].set_index("Frame")
        print(f"[INFO] CSV loaded. Rows: {len(df_speed)}")
    except Exception as e:
        print(f"[WARN] CSV loading error: {e}")
        df_speed = None
else:
    print("[WARN] CSV file not found.")

def get_speed_kmh(global_fr: int) -> float:
    if df_speed is None:
        return 0.0
    if global_fr in df_speed.index:
        raw = df_speed.loc[global_fr, "Speed"]
        return to_kmh(raw)
    return 0.0

# ========== フォント設定 ==========
try:
    font_pil = ImageFont.truetype(font_path, 12)
except Exception:
    font_pil = ImageFont.load_default()

def put_text_pil(img_cv, text, x, y, font, color=(255, 255, 255)):
    img_rgb = cv2.cvtColor(img_cv, cv2.COLOR_BGR2RGB)
    pil_img = Image.fromarray(img_rgb)
    draw = ImageDraw.Draw(pil_img)
    draw.text((x, y), text, font=font, fill=color)
    return cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)

def draw_text_centered_x(img, text, y, font_size=12, color=(255, 255, 255)):
    try:
        font_ = ImageFont.truetype(font_path, font_size)
    except:
        font_ = ImageFont.load_default()
    dummy = Image.new("RGB", (1, 1))
    dummy_draw = ImageDraw.Draw(dummy)
    bbox = dummy_draw.textbbox((0, 0), text, font=font_)
    w_text = bbox[2] - bbox[0]
    w_img = img.shape[1]
    x = max(0, (w_img - w_text) // 2)
    return put_text_pil(img, text, x, y, font_, color=color)

# ========== BEV関連設定 ==========
BEV_W, BEV_H = 240, 480
Y_EGO = int(BEV_H * 0.83)

BEV_FRONT_M = 35
BEV_RIGHT_M = 3.5
BEV_LEFT_M = 3.5
BEV_MARGIN_BOTTOM = int(BEV_H * 0.07)
BEV_MARGIN_TOP = int(BEV_H * 0.05)
BEV_USABLE_H = BEV_H - BEV_MARGIN_TOP - BEV_MARGIN_BOTTOM

px_per_meter_y = BEV_USABLE_H / BEV_FRONT_M
px_per_meter_x = BEV_W / (BEV_LEFT_M + BEV_RIGHT_M)

def meter2bev_y(m: float) -> int:
    return int(Y_EGO - m * px_per_meter_y)

def meter2bev_x(m: float) -> int:
    return int(BEV_W // 2 + m * px_per_meter_x)

def meter2bev(mx: float, my: float):
    return (meter2bev_x(mx), meter2bev_y(my))

# ========== レーン設定 ==========
lane_def = [
    {"x_start": -3.0, "x_end": -2.8, "color": (255, 255, 255), "thick": 1},
    {"x_start": 0.0, "x_end": 0.0, "color": (255, 255, 0), "thick": 1},
    {"x_start": 2.8, "x_end": 3.0, "color": (255, 255, 255), "thick": 1},
]
lane_x_mid = [
    (item["x_start"] + item["x_end"]) / 2.0 for item in lane_def
]

def classify_lane_bev(cx_m: float):
    best_idx, best_dist = 0, abs(cx_m - lane_x_mid[0])
    for i, lm in enumerate(lane_x_mid):
        dist = abs(cx_m - lm)
        if dist < best_dist:
            best_dist, best_idx = dist, i
    if len(lane_x_mid) == 1:
        return 0
    if len(lane_x_mid) == 2:
        return 1 if best_idx > 0 else 0
    return 1 if best_idx == 1 else (2 if best_idx == 2 else 0)

# ========== 車両描画 ==========
def draw_vehicle_bev_with_arrow(
    bev_canvas, cx_m, cy_m, vx_m, vy_m,
    width_m=1.6, length_m=3.8,
    color=(0, 255, 0), thickness=1,
    arrow_len=3.5, track_len=15.0,
    track_hist=None, ego_line_thick=2
):
    cx_px = meter2bev_x(cx_m)
    cy_px = meter2bev_y(cy_m)

    wh = int(width_m * px_per_meter_x / 2)
    hh = int(length_m * px_per_meter_y / 2)
    cv2.rectangle(
        bev_canvas,
        (cx_px - wh, cy_px - hh),
        (cx_px + wh, cy_px + hh),
        color, thickness
    )

    speed_mps = math.sqrt(vx_m**2 + vy_m**2)
    if speed_mps > 0.5:
        vnorm_x = vx_m / speed_mps
        vnorm_y = vy_m / speed_mps
        tip_m_x = cx_m + vnorm_x * arrow_len
        tip_m_y = cy_m + vnorm_y * arrow_len
        tip_px = meter2bev(tip_m_x, tip_m_y)
        cv2.arrowedLine(
            bev_canvas,
            (cx_px, cy_px),
            tip_px,
            (255, 128, 0),
            1,
            tipLength=0.3
        )

    if track_hist and len(track_hist) > 1:
        pts_in_range = []
        dist_sum = 0.0
        for i in range(len(track_hist)-1, -1, -1):
            mx, my, _, _ = track_hist[i]
            pts_in_range.append((mx, my))
            if i > 0:
                mx_prev, my_prev, _, _ = track_hist[i-1]
                seg_dist = math.sqrt((mx - mx_prev)**2 + (my - my_prev)**2)
                dist_sum += seg_dist
                if dist_sum >= track_len:
                    break
        pts_bev = [meter2bev(mx, my) for (mx, my) in pts_in_range]
        if len(pts_bev) >= 2:
            pts_arr = np.array(pts_bev, dtype=np.int32)
            cv2.polylines(bev_canvas, [pts_arr], False, color, ego_line_thick)

# ========== 履歴管理 ==========
track_history = defaultdict(lambda: {
    "xy": deque(maxlen=45),
    "vxy": deque(maxlen=45),
    "meter": deque(maxlen=45),
    "vmeter": deque(maxlen=45),
})

# ========== ターンシグナル・ブレーキ ==========
turn_window_frames = 15
brake_window_frames = 10
turn_ring_buf = deque(maxlen=turn_window_frames)
brake_ring_buf = deque(maxlen=brake_window_frames)

# ========== Stage1 キャッシュ ==========
env_stage1_cache = {"result": None, "last_frame": -10**9}

# ========== メイン処理 ==========
frame_idx = 0
trajectories = []
start_time = time.time()

# 発火スナップショット保存用
final_snapshots = []

try:
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        global_frame_idx = offset + frame_idx
        csv_frame_val = global_frame_idx

        # ========== フレームごとのメタ情報 ==========
        base_meta = {
            "frame_idx": frame_idx,
            "csv_frame": csv_frame_val,
        }

        # ========== GPU同期でフレーム処理開始 ==========
        with span_with_frame_edges(tr, "frame", base_meta, gpu=True, offset=offset):
            orig = frame.copy()

            # ========== YOLOv8検出 ==========
            with span_with_frame_edges(tr, "yolo_detect", base_meta, gpu=True, offset=offset):
                results = yolo_detect(frame, verbose=False)
                result = results[0]
                dets_list = []
                dets_with_cls = []
                if result.boxes is not None and len(result.boxes) > 0:
                    for box in result.boxes:
                        x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                        conf = float(box.conf[0].cpu().numpy())
                        cls_id = int(box.cls[0].cpu().numpy())
                        dets_list.append([x1, y1, x2, y2, conf])
                        dets_with_cls.append([x1, y1, x2, y2, conf, cls_id])

            # ========== BYTETracker追跡 ==========
            with span_with_frame_edges(tr, "byte_track", base_meta, gpu=False, offset=offset):
                if len(dets_list) > 0:
                    dets_array = np.array(dets_list)
                    stracks = tracker.update(dets_array, [H_vid, W_vid], [H_vid, W_vid])
                else:
                    stracks = tracker.update(np.empty((0, 5)), [H_vid, W_vid], [H_vid, W_vid])

            # ========== トラック情報の記録 ==========
            with span_with_frame_edges(tr, "track_hist", base_meta, gpu=False, offset=offset):
                dt_sec = 1.0 / fps
                for st in stracks:
                    tid = st.track_id
                    cx = (st.tlbr[0] + st.tlbr[2]) / 2.0
                    cy = (st.tlbr[1] + st.tlbr[3]) / 2.0

                    if len(track_history[tid]["xy"]) > 0:
                        prev_cx, prev_cy = track_history[tid]["xy"][-1]
                        vx = (cx - prev_cx) / dt_sec
                        vy = (cy - prev_cy) / dt_sec
                    else:
                        vx, vy = 0.0, 0.0

                    track_history[tid]["xy"].append((cx, cy))
                    track_history[tid]["vxy"].append((vx, vy))

            # ========== 透視変換でメートル座標を計算 ==========
            with span_with_frame_edges(tr, "ipm", base_meta, gpu=False, offset=offset):
                src_pts = np.float32([
                    [170, 115],
                    [190, 115],
                    [320, 215],
                    [40, 215],
                ])
                h_ipm = 200
                w_ipm = 150
                dst_pts = np.float32([
                    [w_ipm * 0.50, 0],
                    [w_ipm * 0.50, 0],
                    [w_ipm, h_ipm],
                    [0, h_ipm],
                ])
                M_ipm = cv2.getPerspectiveTransform(src_pts, dst_pts)

                METER_PER_PX_Y = 35.0 / h_ipm
                METER_PER_PX_X = 3.0 / (w_ipm / 2.0)

                for st in stracks:
                    tid = st.track_id
                    if len(track_history[tid]["xy"]) == 0:
                        continue
                    cx_img, cy_img = track_history[tid]["xy"][-1]
                    pt = np.array([[cx_img, cy_img]], dtype=np.float32).reshape(-1, 1, 2)
                    tr_pt = cv2.perspectiveTransform(pt, M_ipm)
                    xd, yd = float(tr_pt[0, 0, 0]), float(tr_pt[0, 0, 1])

                    mx = (xd - w_ipm / 2.0) * METER_PER_PX_X
                    my = (h_ipm - yd) * METER_PER_PX_Y

                    if len(track_history[tid]["meter"]) > 0:
                        prev_mx, prev_my = track_history[tid]["meter"][-1]
                        vmx = (mx - prev_mx) / dt_sec
                        vmy = (my - prev_my) / dt_sec
                    else:
                        vmx, vmy = 0.0, 0.0

                    track_history[tid]["meter"].append((mx, my))
                    track_history[tid]["vmeter"].append((vmx, vmy))

            # ========== Seq2Seq予測 ==========
            with span_with_frame_edges(tr, "seq2seq_pred", base_meta, gpu=True, offset=offset):
                ego_speed_kmh = get_speed_kmh(global_frame_idx)
                speed_feat = speed_to_feature(ego_speed_kmh)

                for st in stracks:
                    tid = st.track_id
                    xys = list(track_history[tid]["xy"])
                    if len(xys) < H_PAST:
                        continue

                    recent_xys = xys[-H_PAST:]
                    arr = []
                    for px, py in recent_xys:
                        nx = px / IMG_W
                        ny = py / IMG_H
                        arr.append([nx, ny, speed_feat])

                    in_tensor = torch.tensor([arr], dtype=torch.float32).to("cuda")
                    with torch.no_grad():
                        pred_seq = s2s_traj(in_tensor, target_len=H_FUT)
                    pred_np = pred_seq[0].cpu().numpy()

                    pred_pixel = []
                    for i in range(pred_np.shape[0]):
                        px_pred = pred_np[i, 0] * IMG_W
                        py_pred = pred_np[i, 1] * IMG_H
                        pred_pixel.append((px_pred, py_pred))

                    st.pred_pixel = pred_pixel

                    px_last, py_last = pred_pixel[-1]
                    last_pt = np.array([[px_last, py_last]], dtype=np.float32).reshape(-1, 1, 2)
                    tr_last = cv2.perspectiveTransform(last_pt, M_ipm)
                    xd_pred, yd_pred = float(tr_last[0, 0, 0]), float(tr_last[0, 0, 1])
                    mx_pred = (xd_pred - w_ipm / 2.0) * METER_PER_PX_X
                    my_pred = (h_ipm - yd_pred) * METER_PER_PX_Y
                    st.pred_meter_last = (mx_pred, my_pred)

            # ========== ターンシグナル分類 ==========
            with span_with_frame_edges(tr, "turn_cls", base_meta, gpu=True, offset=offset):
                turn_probs = {"off": 0.0, "left": 0.0, "right": 0.0}
                if turn_cls is not None:
                    crop_h = int(0.2 * H_vid)
                    crop_bottom = H_vid
                    crop_top = crop_bottom - crop_h
                    img_crop = frame[crop_top:crop_bottom, :]
                    if img_crop.size > 0:
                        try:
                            t_results = turn_cls.predict(img_crop, verbose=False)
                            if t_results and t_results[0].probs is not None:
                                pr = t_results[0].probs.data.cpu().numpy()
                                nms = t_results[0].names
                                for i, nm in nms.items():
                                    key = nm.strip().lower()
                                    if key in turn_probs:
                                        turn_probs[key] = float(pr[i])
                        except Exception:
                            pass

                turn_ring_buf.append(turn_probs)

                window_probs = {"off": 0.0, "left": 0.0, "right": 0.0}
                window_counts = {"off": 0, "left": 0, "right": 0}
                for tp_dict in turn_ring_buf:
                    for k in tp_dict:
                        if k in window_probs:
                            window_probs[k] += tp_dict[k]
                    best_k = max(tp_dict, key=tp_dict.get)
                    if best_k in window_counts:
                        window_counts[best_k] += 1

                wsize = len(turn_ring_buf)
                if wsize > 0:
                    for k in window_probs:
                        window_probs[k] /= wsize

                final_turn = max(window_probs, key=window_probs.get)
                final_turn_conf = window_probs[final_turn]

            # ========== ブレーキ分類 ==========
            with span_with_frame_edges(tr, "brake_cls", base_meta, gpu=True, offset=offset):
                brake_probs = {"off": 0.0, "on": 0.0}
                if brake_cls is not None:
                    crop_h = int(0.15 * H_vid)
                    crop_bottom = H_vid
                    crop_top = crop_bottom - crop_h
                    img_crop = frame[crop_top:crop_bottom, :]
                    if img_crop.size > 0:
                        try:
                            b_results = brake_cls.predict(img_crop, verbose=False)
                            if b_results and b_results[0].probs is not None:
                                pr = b_results[0].probs.data.cpu().numpy()
                                nms = b_results[0].names
                                for i, nm in nms.items():
                                    key = nm.strip().lower()
                                    if key in brake_probs:
                                        brake_probs[key] = float(pr[i])
                        except Exception:
                            pass

                brake_ring_buf.append(brake_probs)

                window_probs_br = {"off": 0.0, "on": 0.0}
                window_counts_br = {"off": 0, "on": 0}
                for bp_dict in brake_ring_buf:
                    for k in bp_dict:
                        if k in window_probs_br:
                            window_probs_br[k] += bp_dict[k]
                    best_k = max(bp_dict, key=bp_dict.get)
                    if best_k in window_counts_br:
                        window_counts_br[best_k] += 1

                wsize_br = len(brake_ring_buf)
                if wsize_br > 0:
                    for k in window_probs_br:
                        window_probs_br[k] /= wsize_br

                final_brake = max(window_probs_br, key=window_probs_br.get)
                final_brake_conf = window_probs_br[final_brake]

            # ========== Stage1環境認識 ==========
            with span_with_frame_edges(tr, "stage1_env", base_meta, gpu=False, offset=offset):
                elapsed = frame_idx - env_stage1_cache["last_frame"]
                if elapsed >= ENV_REFRESH_EVERY_FRAMES:
                    env_stage1_cache["result"] = assess_environment_stage1_from_frame(frame)
                    env_stage1_cache["last_frame"] = frame_idx
                stage1_env = env_stage1_cache["result"] or {}

            # ========== 背景情報構築 ==========
            context_dict = {
                "frame": int(global_frame_idx),
                "scene_no": scene_no,
                "stage1_env": stage1_env,
                "ego_vehicle": {
                    "speed_kmh": round(ego_speed_kmh, 2),
                    "risk_zone": False,
                    "risk_zone_predicted": False,
                },
                "perception": {
                    "turn_signal": {
                        "final": final_turn,
                        "final_conf_window": round(final_turn_conf, 3),
                        "probs_window_avg": {k: round(v, 3) for k, v in window_probs.items()},
                        "window_size": wsize,
                        "window_counts": dict(window_counts),
                    },
                    "brake": {
                        "final": final_brake,
                        "final_conf_window": round(final_brake_conf, 3),
                        "probs_window_avg": {k: round(v, 3) for k, v in window_probs_br.items()},
                        "window_size": wsize_br,
                        "window_counts": dict(window_counts_br),
                    },
                },
                "detected_vehicles": {
                    "count": len(stracks),
                },
            }

            # ========== リスク評価（LLaVA） ==========
            with span_with_frame_edges(tr, "llava_assess", base_meta, gpu=True, offset=offset):
                fire_flag = False
                risk_label = "low"
                reasoning_dict = {}

                try:
                    result_llava = assess_risk_from_image_with_context(frame, context_dict)
                    if result_llava:
                        risk_label = result_llava.get("risk_label", "low").strip().lower()
                        reasoning_dict = result_llava.get("reasoning", {})
                        fire_flag = (risk_label in ["high", "very_high"])

                        llava_rec = {
                            "frame": int(global_frame_idx),
                            "scene_no": scene_no,
                            "risk_label": risk_label,
                            "reasoning": reasoning_dict,
                            "context_simple": simplify_for_llava(context_dict),
                        }
                        llava_records.append(llava_rec)
                except Exception:
                    pass

                context_dict["ego_vehicle"]["risk_zone"] = fire_flag
                context_records.append(context_dict)

            # ========== 停止距離計算 ==========
            speed_mps = to_mps(ego_speed_kmh)
            reaction_time = 1.0
            decel_ms2 = 8.0
            stop_m = speed_mps * reaction_time + (speed_mps ** 2) / (2.0 * decel_ms2)

            # ========== 予測リスク判定 ==========
            predicted_risk = False
            for st in stracks:
                tid = st.track_id
                if hasattr(st, "pred_meter_last"):
                    mx_pred, my_pred = st.pred_meter_last
                    if my_pred <= stop_m:
                        predicted_risk = True
                        break

            context_dict["ego_vehicle"]["risk_zone_predicted"] = predicted_risk

            # ========== 黄→赤遷移時間の記録 ==========
            for st in stracks:
                tid = st.track_id
                current_risk = fire_flag

                # 初回黄色検出
                if current_risk and tid not in first_yellow_frame:
                    first_yellow_frame[tid] = global_frame_idx
                    pending_series_tids.add(tid)

                # 初回赤色検出（高速度かつ接近）
                if predicted_risk and tid in first_yellow_frame and tid not in first_red_frame:
                    speed_threshold_kmh = 40.0
                    if ego_speed_kmh >= speed_threshold_kmh:
                        first_red_frame[tid] = global_frame_idx
                        yellow_fr = first_yellow_frame[tid]
                        delta_frames = global_frame_idx - yellow_fr
                        delta_ms = round(delta_frames / max(1.0, float(fps)) * 1000.0, 3)

                        risk_time_records.append({
                            "scene_no": scene_no,
                            "tid": int(tid),
                            "frame_yellow": int(yellow_fr),
                            "frame_red": int(global_frame_idx),
                            "fps": round(fps, 2),
                            "delta_frames": delta_frames,
                            "delta_ms": delta_ms,
                            "ego_speed_kmh": round(ego_speed_kmh, 2),
                            "stop_m": round(stop_m, 2),
                        })

                        pending_series_tids.discard(tid)

                # 黄→赤の間の連続データ記録
                if tid in pending_series_tids:
                    yellow_fr = first_yellow_frame.get(tid)
                    if yellow_fr is not None:
                        frames_since_yellow = global_frame_idx - yellow_fr
                        ms_from_yellow = round(frames_since_yellow / max(1.0, float(fps)) * 1000.0, 3)
                        between_series[tid].append({
                            "frame": int(global_frame_idx),
                            "ms_from_yellow": ms_from_yellow,
                            "ego_speed_kmh": round(ego_speed_kmh, 2),
                            "stop_m": round(stop_m, 2),
                        })

            # ========== 発火スナップショット ==========
            if fire_flag:
                for st in stracks:
                    tid = st.track_id
                    if tid in first_yellow_frame and tid not in first_red_frame:
                        final_snapshots.append((global_frame_idx, tid))

            # ========== トラジェクトリ保存 ==========
            for st in stracks:
                tid = st.track_id
                x1, y1, x2, y2 = st.tlbr
                cx = (x1 + x2) / 2.0
                cy = (y1 + y2) / 2.0

                vx, vy = 0.0, 0.0
                if len(track_history[tid]["vxy"]) > 0:
                    vx, vy = track_history[tid]["vxy"][-1]

                mx, my, vmx, vmy = 0.0, 0.0, 0.0, 0.0
                if len(track_history[tid]["meter"]) > 0:
                    mx, my = track_history[tid]["meter"][-1]
                if len(track_history[tid]["vmeter"]) > 0:
                    vmx, vmy = track_history[tid]["vmeter"][-1]

                lane_idx = classify_lane_bev(mx)

                cls_name = None
                if dets_with_cls:
                    cls_name = _class_for_track(x1, y1, x2, y2, dets_with_cls, iou_thr=0.3)

                traj_rec = {
                    "frame": int(global_frame_idx),
                    "track_id": int(tid),
                    "x1": float(x1), "y1": float(y1),
                    "x2": float(x2), "y2": float(y2),
                    "cx": float(cx), "cy": float(cy),
                    "vx_px": float(vx), "vy_px": float(vy),
                    "mx": float(mx), "my": float(my),
                    "vmx": float(vmx), "vmy": float(vmy),
                    "lane_idx": int(lane_idx),
                    "cls": cls_name,
                    "ego_speed_kmh": float(ego_speed_kmh),
                    "risk_zone": fire_flag,
                    "risk_zone_predicted": predicted_risk,
                }
                trajectories.append(traj_rec)

            # ========== 可視化描画 ==========
            with span_with_frame_edges(tr, "visualization", base_meta, gpu=False, offset=offset):
                combined_left = np.zeros((480, 480, 3), dtype=np.uint8)
                main_frame = orig.copy()

                # BBox & Track
                for st in stracks:
                    tid = st.track_id
                    x1, y1, x2, y2 = st.tlbr
                    color = (0, 255, 0)
                    cv2.rectangle(main_frame, (int(x1), int(y1)), (int(x2), int(y2)), color, 2)
                    label = f"ID{tid}"
                    cv2.putText(main_frame, label, (int(x1), int(y1) - 5),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)

                    xys = list(track_history[tid]["xy"])
                    if len(xys) > 1:
                        pts = np.array(xys, dtype=np.int32)
                        cv2.polylines(main_frame, [pts], False, color, 1)

                    if hasattr(st, "pred_pixel") and st.pred_pixel:
                        pts_pred = np.array(st.pred_pixel, dtype=np.int32)
                        cv2.polylines(main_frame, [pts_pred], False, (255, 0, 0), 1)

                main_frame = cv2.resize(main_frame, (360, 240))
                combined_left[0:240, 0:360] = main_frame

                info_panel = np.zeros((240, 120, 3), dtype=np.uint8)
                y_text = 15
                dy = 15

                txt = f"Frame: {global_frame_idx:6d}"
                info_panel = put_text_pil(info_panel, txt, 5, y_text, font_pil, (255, 255, 255))
                y_text += dy

                txt = f"Speed: {ego_speed_kmh:.1f} km/h"
                info_panel = put_text_pil(info_panel, txt, 5, y_text, font_pil, (255, 255, 255))
                y_text += dy

                txt = f"Stop: {stop_m:.1f} m"
                info_panel = put_text_pil(info_panel, txt, 5, y_text, font_pil, (255, 255, 255))
                y_text += dy

                txt = f"Tracks: {len(stracks)}"
                info_panel = put_text_pil(info_panel, txt, 5, y_text, font_pil, (255, 255, 255))
                y_text += dy

                txt = f"Turn: {final_turn} {final_turn_conf:.2f}"
                info_panel = put_text_pil(info_panel, txt, 5, y_text, font_pil, (255, 255, 0))
                y_text += dy

                txt = f"Brake: {final_brake} {final_brake_conf:.2f}"
                info_panel = put_text_pil(info_panel, txt, 5, y_text, font_pil, (0, 255, 255))
                y_text += dy

                txt = f"Risk: {risk_label}"
                info_panel = put_text_pil(info_panel, txt, 5, y_text, font_pil, (0, 255, 255))
                y_text += dy

                txt = f"RiskZone: {'Yes' if fire_flag else 'No'}"
                info_panel = put_text_pil(info_panel, txt, 5, y_text, font_pil, (0, 0, 255))
                y_text += dy

                txt = f"PredRisk: {'Yes' if predicted_risk else 'No'}"
                info_panel = put_text_pil(info_panel, txt, 5, y_text, font_pil, (0, 0, 255))
                y_text += dy

                combined_left[0:240, 360:480] = info_panel

                bottom_panel = np.zeros((240, 480, 3), dtype=np.uint8)
                y_bot = 10
                dy_bot = 12

                if stage1_env:
                    for k, v in stage1_env.items():
                        txt_stage1 = f"{k}: {_yn_cap(v)}"
                        bottom_panel = put_text_pil(bottom_panel, txt_stage1, 5, y_bot, font_pil, (255, 255, 255))
                        y_bot += dy_bot

                combined_left[240:480, 0:480] = bottom_panel

                # BEV描画
                bev_canvas = np.zeros((480, BEV_W, 3), dtype=np.uint8)
                bev_draw = np.zeros((BEV_H, BEV_W, 3), dtype=np.uint8)

                for ldef in lane_def:
                    x_s = ldef["x_start"]
                    x_e = ldef["x_end"]
                    color = ldef["color"]
                    thick = ldef["thick"]
                    px_s = meter2bev_x(x_s)
                    px_e = meter2bev_x(x_e)
                    cv2.line(bev_draw, (px_s, BEV_H), (px_s, 0), color, thick)
                    if x_s != x_e:
                        cv2.line(bev_draw, (px_e, BEV_H), (px_e, 0), color, thick)

                ego_cx_m = 0.0
                ego_cy_m = 0.0
                draw_vehicle_bev_with_arrow(
                    bev_draw, ego_cx_m, ego_cy_m, 0.0, 0.0,
                    width_m=1.6, length_m=3.8,
                    color=(255, 255, 255), thickness=2,
                    arrow_len=0, track_len=0, track_hist=None
                )

                for st in stracks:
                    tid = st.track_id
                    if len(track_history[tid]["meter"]) == 0:
                        continue
                    mx, my = track_history[tid]["meter"][-1]
                    vmx, vmy = 0.0, 0.0
                    if len(track_history[tid]["vmeter"]) > 0:
                        vmx, vmy = track_history[tid]["vmeter"][-1]

                    draw_vehicle_bev_with_arrow(
                        bev_draw, mx, my, vmx, vmy,
                        width_m=1.6, length_m=3.8,
                        color=(0, 255, 0), thickness=1,
                        arrow_len=3.5, track_len=15.0,
                        track_hist=list(track_history[tid]["meter"]),
                        ego_line_thick=2
                    )

                try:
                    if stop_m > 0.5:
                        poly_pts_m = [
                            (-BEV_LEFT_M, 0.0),
                            (BEV_RIGHT_M, 0.0),
                            (BEV_RIGHT_M, stop_m),
                            (-BEV_LEFT_M, stop_m),
                        ]
                        poly_bev = np.array([meter2bev(px, py) for px, py in poly_pts_m], dtype=np.float32)
                        tmp = bev_draw.copy()
                        cv2.fillPoly(tmp, [poly_bev.astype(np.int32)], (0, 255, 255))
                        bev_draw = cv2.addWeighted(tmp, 0.3, bev_draw, 0.7, 0.0)
                except Exception:
                    pass

                bev_resized = cv2.resize(bev_draw, (BEV_W, 240))
                bev_canvas[0:240, 0:BEV_W] = bev_resized

                try:
                    bev_canvas = draw_text_centered_x(bev_canvas, "BEV lane view", 245, font_size=11)
                    bev_canvas = draw_text_centered_x(bev_canvas, f"Stop {stop_m:.1f} m", 260, font_size=11)
                except Exception:
                    pass

                final_frame = np.hstack([combined_left, bev_canvas])

                if final_snapshots:
                    for (g_idx, t_id) in final_snapshots:
                        post_path = os.path.join(FIRE_DIR, f"frame_{g_idx:06d}_post_tid{t_id}.jpg")
                        cv2.imwrite(post_path, final_frame)
                    final_snapshots.clear()

                # ========== 動画書き出しの計測 ==========
                with span_with_frame_edges(tr, "video_write", base_meta, gpu=False, offset=offset):
                    out.write(final_frame)

                if frame_idx % 10 == 0:
                    elapsed = time.time() - start_time
                    pct = (frame_idx + 1) / max(1,total_frames) * 100
                    print(f"[{frame_idx:6d}/{total_frames}] {pct:5.1f}% det:{len(dets_list):2d} trk:{len(track_history):2d} elapsed:{elapsed:6.1f}s")

        frame_idx += 1

except KeyboardInterrupt:
    print("[WARN] Process interrupted. Saving current progress...")

except Exception as e:
    print(f"[ERROR] Exception occurred: {e}")

finally:
    print("[INFO] Cleanup in progress...")
    try:
        cap.release()
    except:
        pass

    try:
        out.release()
        print("[INFO] Video file saved.")
    except:
        print("[ERROR] Failed to save video file.")

    try:
        if trajectories:
            pd.DataFrame(trajectories).to_csv(output_csv_path, index=False)
            print("[INFO] CSV file saved.")
    except:
        print("[ERROR] Failed to save CSV file.")

    # 背後情報とLLaVA記録を保存
    context_json_path = os.path.join(base_dir, f"scene_{scene_no}_context.json")
    with open(context_json_path, "w", encoding="utf-8") as f:
        json.dump(context_records, f, ensure_ascii=False, indent=2)
    print(f"[INFO] Context JSON saved: {context_json_path}")

    llava_json_path = os.path.join(base_dir, f"scene_{scene_no}_llava.json")
    with open(llava_json_path, "w", encoding="utf-8") as f:
        json.dump(llava_records, f, ensure_ascii=False, indent=2)
    print(f"[INFO] LLaVA JSON saved: {llava_json_path}")

    # 基本情報出力
    total_elapsed = time.time() - start_time
    print(f"\n[INFO] Processing time: {total_elapsed:.1f}s")
    print(f"[INFO] Processed frames: {frame_idx:,}")
    print(f"[INFO] Output video: {output_video_path}")
    print(f"[INFO] Output CSV: {output_csv_path}")
    print(f"[INFO] Latency log (CSV): {latency_csv_path}")
    print(f"[INFO] Latency log (JSONL): {latency_jsonl_path}")
    print("[INFO] Process finished.")

    # ========== 黄→赤遷移時間の保存 ==========
    if risk_time_records:
        risk_csv_path = os.path.join(base_dir, f"scene_{scene_no}_risk_transition.csv")
        risk_json_path = os.path.join(base_dir, f"scene_{scene_no}_risk_transition.json")

        try:
            # CSV保存
            df_risk = pd.DataFrame(risk_time_records)
            df_risk.to_csv(risk_csv_path, index=False)
            print(f"[INFO] Risk transition CSV saved: {risk_csv_path}")

            # JSON保存（オプション）
            with open(risk_json_path, "w", encoding="utf-8") as f:
                json.dump(risk_time_records, f, ensure_ascii=False, indent=2)
            print(f"[INFO] Risk transition JSON saved: {risk_json_path}")

            # 統計サマリー出力
            times_ms = [r["delta_ms"] for r in risk_time_records]
            print(f"[INFO] Transition time stats: n={len(times_ms)} "
                  f"mean={np.mean(times_ms):.1f}ms median={np.median(times_ms):.1f}ms "
                  f"min={np.min(times_ms):.1f}ms max={np.max(times_ms):.1f}ms")

        except Exception as e:
            print(f"[ERROR] Failed to save transition time data: {e}")
    else:
        print("[INFO] No yellow-to-red transition events detected.")
    # ========== 黄→赤 "間" の連続データ保存 ==========
    try:
        if between_series:
            # 1️⃣ JSON形式（tidごとの配列）
            series_json_path = os.path.join(base_dir, f"scene_{scene_no}_risk_transition_series.json")
            series_payload = []
            for tid, series in between_series.items():
                f_yellow = first_yellow_frame.get(tid)
                f_red = first_red_frame.get(tid)
                if f_yellow is not None and f_red is not None:
                    delta_frames = f_red - f_yellow
                    delta_ms = round(delta_frames / max(1.0, float(fps)) * 1000.0, 3)
                else:
                    delta_frames = None
                    delta_ms = None

                # ms_from_yellow の統計（平均・最大など）
                ms_vals = [s["ms_from_yellow"] for s in series if "ms_from_yellow" in s]
                duration_mean_ms = float(np.mean(ms_vals)) if ms_vals else None
                duration_max_ms = float(np.max(ms_vals)) if ms_vals else None

                series_payload.append({
                    "scene_no": scene_no if scene_number is not None else "unknown",
                    "tid": int(tid),
                    "frame_yellow": int(f_yellow) if f_yellow is not None else None,
                    "frame_red": int(f_red) if f_red is not None else None,
                    "fps": round(fps, 2),
                    "delta_frames": delta_frames,
                    "delta_ms": delta_ms,
                    "duration_mean_ms": duration_mean_ms,
                    "duration_max_ms": duration_max_ms,
                    "series": series,
                })

            with open(series_json_path, "w", encoding="utf-8") as f:
                json.dump(series_payload, f, ensure_ascii=False, indent=2)
            print(f"[INFO] Continuous data JSON saved: {series_json_path}")

            # 2️⃣ CSV形式（縦長）
            series_csv_path = os.path.join(base_dir, f"scene_{scene_no}_risk_transition_series.csv")
            rows = []
            for tid, series in between_series.items():
                for s in series:
                    rows.append({
                        "scene_no": scene_no if scene_number is not None else "unknown",
                        "tid": int(tid),
                        "frame": int(s["frame"]),
                        "ms_from_yellow": float(s["ms_from_yellow"]),
                        "ego_speed_kmh": float(s["ego_speed_kmh"]),
                        "stop_m": float(s["stop_m"]),
                    })
            pd.DataFrame(rows).to_csv(series_csv_path, index=False)
            print(f"[INFO] Continuous data CSV saved: {series_csv_path}")
        else:
            print("[INFO] No continuous data between yellow-to-red transitions.")
    except Exception as e:
        print(f"[ERROR] Failed to save continuous data: {e}")

    # ========== 黄→赤イベントの発生シーン記録 ==========
    try:
        if risk_time_records:
            # フォルダ番号を自動推定（動画パスから取得）
            match_folder = re.search(r"\\(\d+)\\", video_path)
            folder_num = match_folder.group(1) if match_folder else "unknown"

            # サマリーファイルの保存先
            summary_dir = os.path.join(BASE_SAVE_DIR, "risk_transition_summary")
            os.makedirs(summary_dir, exist_ok=True)
            summary_path = os.path.join(summary_dir, "risk_transition_summary.json")

            # 今回のエントリ
            summary_entry = {
                "folder": folder_num,
                "scene_no": scene_no,
                "video_path": video_path,
                "csv_path": csv_path,
                "n_events": len(risk_time_records),
                "mean_ms": float(np.mean([r["delta_ms"] for r in risk_time_records])),
                "max_ms": float(np.max([r["delta_ms"] for r in risk_time_records])),
                "min_ms": float(np.min([r["delta_ms"] for r in risk_time_records])),
            }

            # 既存 summary に追記
            existing = []
            if os.path.exists(summary_path):
                try:
                    with open(summary_path, "r", encoding="utf-8") as f:
                        existing = json.load(f)
                except Exception:
                    existing = []

            existing.append(summary_entry)

            with open(summary_path, "w", encoding="utf-8") as f:
                json.dump(existing, f, ensure_ascii=False, indent=2)

            print(f"[INFO] Scene summary appended: {summary_path}")
        else:
            print("[INFO] No yellow-to-red events in this scene. Summary skipped.")
    except Exception as e:
        print(f"[ERROR] Failed to append scene summary: {e}")