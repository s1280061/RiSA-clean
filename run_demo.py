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

# ★ 追加: 黄→赤の“間”の連続データ
from collections import defaultdict

# tidごとに、黄→赤の間の連続サンプルを保存
# 例: between_series[tid] = [{"frame": 12345, "ms_from_yellow": 66.667, "ego_speed_kmh": 42.3, "stop_m": 4.2}, ...]
between_series = defaultdict(list)

# 連続記録中のtid（黄色検出後〜赤到達まで）
pending_series_tids = set()


# ---- パス設定（デフォルト）----
video_path        = r"C:\Users\s1280\Desktop\SHRP2rawdata\5\new_divided\scene_020.mp4"
csv_path          = r"C:\Users\s1280\Desktop\SHRP2rawdata\5\csv_divided\scene_020.csv"
base_path         = "/app/26x"
font_path         = "/app/26x/fonts/RobotoMono-Regular.ttf"
turn_model_path   = r"/app/models/classify/turn_best.pt"
brake_model_path  = r"/app/models/classify/brake_best.pt"

parser = argparse.ArgumentParser()
parser.add_argument("--video", type=str, help="path to scene_###.mp4")
parser.add_argument("--csv",   type=str, help="path to scene_###.csv")
parser.add_argument("--folder", type=str, required=True)     # ★ 追加

args, _ = parser.parse_known_args()
if args.video: video_path = args.video
if args.csv:   csv_path   = args.csv
folder_no = args.folder    # ★ 追加


output_video_path = os.path.splitext(video_path)[0] + "_with_bev.mp4"
output_csv_path   = os.path.splitext(video_path)[0] + "_with_bev_traj.csv"

BASE_SAVE_DIR = os.path.join(r"C:\Users\s1280\Desktop\SHRP2rawdata", folder_no, "new_divided")
scene_match = re.search(r"scene_(\d+)", os.path.basename(video_path))
scene_number = int(scene_match.group(1)) if scene_match else 0
scene_dir = os.path.join(BASE_SAVE_DIR, f"scene_{scene_number:03d}")
os.makedirs(scene_dir, exist_ok=True)

base_dir = os.path.dirname(video_path)
scene_match = re.search(r"(?i)scene_(\d+)", os.path.basename(video_path))

if scene_match:
    scene_no = scene_match.group(1).zfill(3)
    scene_number = int(scene_match.group(1))
else:
    scene_no = "unknown"
    scene_number = None

# ========== レイテンシトレーサーの初期化（シーン番号付き） ==========
latency_csv_path = os.path.join(base_dir, f"scene_{scene_no}_latency.csv")
latency_jsonl_path = os.path.join(base_dir, f"scene_{scene_no}_latency.jsonl")
tr = LatencyTracer(latency_csv_path, latency_jsonl_path)
print(f"📈 レイテンシログ: {latency_csv_path}")

OUTPUT_ROOT = r"C:\Users\s1280\Desktop\SHRP2_outputs_v2"
FIRE_DIR = os.path.join(OUTPUT_ROOT, folder_no, f"scene_{scene_no}")
os.makedirs(FIRE_DIR, exist_ok=True)


final_snapshots = []

# ========================== BEV変換 ==========================
src  = np.load(os.path.join(base_path, "src_points_3_forward.npy"))
dst  = np.load(os.path.join(base_path, "dst_points_3_forward.npy"))
M    = cv2.getPerspectiveTransform(src, dst)
Minv = cv2.getPerspectiveTransform(dst, src)
scale_px_per_m = 50.82

# ========================== データ/検出モデル ==========================
df_speed = pd.read_csv(csv_path, dtype={"frame": int}, low_memory=False)

def _find_speed_col(df):
    candidates = [
        "vtti.speed_gps", "vtti.gps_speed",
        "speed", "Speed", "ego_speed", "veh_speed"
    ]
    for c in candidates:
        if c in df.columns:
            return c
    raise ValueError(f"速度列が見つかりません。候補: {candidates} / columns={list(df.columns)[:10]}...")

def _guess_speed_unit(raw_series: pd.Series) -> str:
    s = pd.to_numeric(raw_series, errors="coerce").dropna().abs()
    if len(s) == 0:
        return "kmh"
    p99 = float(s.quantile(0.99))
    vmax = float(s.max())
    med  = float(s.median())
    if p99 <= 45 and vmax <= 60:
        return "mps"
    if med < 40 and vmax <= 120:
        return "mph"
    return "kmh"

SPEED_COL = _find_speed_col(df_speed)
SPEED_COL_UNIT = "kmh"
print(f"[speed] column='{SPEED_COL}'  unit='kmh' (forced)")

s = pd.to_numeric(df_speed[SPEED_COL], errors='coerce')
print(f"[speed] column='{SPEED_COL}'  unit='{SPEED_COL_UNIT}'  "
      f"p99={s.quantile(0.99):.2f}  max={s.max():.2f}  median={s.median():.2f}")

args_bt = types.SimpleNamespace(track_thresh=0.3, match_thresh=0.7,
                             track_buffer=30, frame_rate=15, mot20=False)
tracker = BYTETracker(args_bt)

det_device = "cuda" if torch.cuda.is_available() else "cpu"
det_model   = YOLO("yolov8n.pt").to(det_device)
cls_device = "cuda" if torch.cuda.is_available() else "cpu"
turn_model  = YOLO(turn_model_path).to(cls_device)
brake_model = YOLO(brake_model_path).to(cls_device)

# ========================== 予測モデル ==========================
traj_ckpt_path = "/app/26x/checkpoints_traj_px_best15/best_ade_px.pt"
traj_device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
ckpt = torch.load(traj_ckpt_path, map_location=traj_device)

cfg = ckpt.get("cfg", {})
H_PAST = int(cfg.get("h_past", 30))
H_FUT  = int(cfg.get("h_fut",  45))
IMG_W  = float(cfg.get("img_w", IMG_W))
IMG_H  = float(cfg.get("img_h", IMG_H))

sample_stride = int(cfg.get("sample_stride", 1))

hidden  = int(cfg.get("hidden", 256))
layers  = int(cfg.get("layers", 3))
dropout = float(cfg.get("dropout", 0.2))

traj_model = Seq2Seq(input_size=5, hidden=hidden, n_layers=layers, dropout=dropout, out_size=2).to(traj_device)
traj_model.load_state_dict(ckpt["state_dict"])
traj_model.eval()

last_pred_px = {}

# ========================== ユーティリティ ==========================
def get_detected_vehicle_list_simple(trajectories, frame_idx):
    return [{
        "id": int(tr["id"]),
        "type": tr.get("class", ""),
        "intent": tr.get("intent", ""),
        "in_risk_zone": bool(tr.get("risky", 0) == 1)
    } for tr in trajectories if int(tr.get("frame", -1)) == int(frame_idx)]

def draw_future_polyline(frame, fut_xy, color=(0,0,255), thickness=1):
    pts = np.asarray(fut_xy, dtype=np.int32)
    if pts.ndim != 2 or pts.shape[0] < 2 or pts.shape[1] != 2:
        return
    cv2.polylines(frame, [pts], False, color, thickness)

def pad_history_left(hist, target_length):
    if len(hist) >= target_length:
        return hist[-target_length:]
    if len(hist) == 0:
        return None
    first_val = hist[0]
    padded = [first_val] * (target_length - len(hist)) + hist
    return padded

COORD_MODE = "pixel"
PRED_TYPE  = "absolute_origin"

last_origin = defaultdict(lambda: None)

def px_to_model(cx, cy, w, h):
    return float(cx), float(cy)

def model_to_px(xm, ym, w, h):
    raise ValueError("model_to_px is unused in pixel mode")

def predict_trajectory(tid, traj_hist, current_speed_kmh, force_predict=False, *, base_meta=None, offset=0):
    global last_pred_px, w, h, H_PAST, H_FUT
    hist = traj_hist[tid]

    if len(hist) >= H_PAST:
        input_seq = hist[-H_PAST:]
    elif force_predict and len(hist) > 0:
        input_seq = pad_history_left(hist, H_PAST)
        if input_seq is None:
            return None
        if DEBUG_TRAJ:
            print(f"[early_pred] tid={tid} padded {len(hist)} -> {H_PAST}")
    else:
        return None

    try:
        with torch.no_grad():
            inp = np.asarray(input_seq, dtype=np.float32)
            inp[:, :2] -= inp[-1, :2]
            x_seq = torch.from_numpy(inp).unsqueeze(0).to(traj_device)

            # ★ 軌跡予測の計測（関数内で計測）
            meta_here = {**(base_meta or {}), "tid": int(tid), "hist_len": int(len(input_seq))}

            with span_with_frame_edges(tr, "traj_predict", meta_here, gpu=True, offset=offset):
                pred = traj_model(x_seq, tgt=None, tf_ratio=0.0, steps=H_FUT)

        fut_raw = pred.squeeze(0).detach().cpu().numpy()
        ox, oy = last_origin.get(tid, (None, None))
        if ox is None:
            return None

        if PRED_TYPE == "relative_delta":
            fut_abs = np.cumsum(fut_raw, axis=0)
        elif PRED_TYPE == "absolute_origin":
            fut_abs = fut_raw
        else:
            return None

        px_x = ox + fut_abs[:, 0]
        px_y = oy + fut_abs[:, 1]
        fut_px = np.stack([px_x, px_y], axis=1)
        fut_px_vis = np.vstack([[ox, oy], fut_px])

        good = np.isfinite(fut_px_vis).all(axis=1)
        fut_px_in = fut_px_vis[good]
        mask_in = (fut_px_in[:, 0] >= 0) & (fut_px_in[:, 0] < w) & (fut_px_in[:, 1] >= 0) & (fut_px_in[:, 1] < h)
        fut_px_in = fut_px_in[mask_in]

        if fut_px_in.shape[0] >= 2:
            fut_px_in = fut_px_in.astype(np.int32)
            last_pred_px[tid] = fut_px_in
            return fut_px_in
        else:
            return None

    except Exception as e:
        if DEBUG_TRAJ:
            print(f"[pred_error] tid={tid}: {e}")
        return None

def draw_text_clean(cv_img, text, position, font_size=16, text_color=(255,255,255)):
    try:
        cv_rgb = cv2.cvtColor(cv_img, cv2.COLOR_BGR2RGB)
    except Exception:
        cv_rgb = cv_img
    pil_img = Image.fromarray(cv_rgb)
    draw = ImageDraw.Draw(pil_img)
    try:
        font = ImageFont.truetype(font_path, font_size)
    except Exception:
        font = ImageFont.load_default()
    draw.text(position, text, font=font, fill=text_color)
    out = cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)
    return out

def draw_text_centered_x(cv_img, text, y, font_size=10, text_color=(255,255,255)):
    try:
        cv_rgb = cv2.cvtColor(cv_img, cv2.COLOR_BGR2RGB)
    except Exception:
        cv_rgb = cv_img
    pil_img = Image.fromarray(cv_rgb)
    draw = ImageDraw.Draw(pil_img)
    try:
        font = ImageFont.truetype(font_path, font_size)
    except Exception:
        font = ImageFont.load_default()

    bbox = draw.textbbox((0, 0), text, font=font)
    text_w = bbox[2] - bbox[0]
    x = max(0, (cv_img.shape[1] - text_w) // 2)

    draw.text((x, y), text, font=font, fill=text_color)
    out = cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)
    return out

def draw_multilines_with_bg(frame, lines, x, y, font_size=10):
    cv_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    pil_img = Image.fromarray(cv_rgb)
    draw = ImageDraw.Draw(pil_img)
    try:
        font = ImageFont.truetype(font_path, font_size)
    except Exception:
        font = ImageFont.load_default()
    text = "\n".join(lines)
    text_bbox = draw.multiline_textbbox((0, 0), text, font=font, spacing=2)
    text_w = text_bbox[2] - text_bbox[0]
    text_h = text_bbox[3] - text_bbox[1]
    draw.rectangle([x, y, x + text_w + 6, y + text_h + 6], fill=(0, 0, 0))
    draw.multiline_text((x + 3, y + 3), text, font=font, fill=(255, 255, 255), spacing=2)
    return cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)

def compute_lane_length(pts, scale_px_per_m):
    if not pts or len(pts) < 2: return 0.0
    total_len_px = 0.0
    for i in range(1, len(pts)):
        dx = pts[i][0] - pts[i-1][0]
        dy = pts[i][1] - pts[i-1][1]
        total_len_px += (dx**2 + dy**2) ** 0.5
    return total_len_px / scale_px_per_m

def compute_curvature_cubic(coeffs, y_eval, scale_px_per_m):
    a, b, c, _ = coeffs
    dx_dy = 3 * a * y_eval**2 + 2 * b * y_eval + c
    d2x_dy2 = 6 * a * y_eval + 2 * b
    curvature_px = abs(d2x_dy2) / ((1 + dx_dy**2) ** 1.5)
    return curvature_px * scale_px_per_m

def speed_based_lane_length(speed_kmh):
    if speed_kmh <= 0: return 0.0
    if speed_kmh < 100: return speed_kmh / 10.0
    return 10.0

def smoothed_speed_estimation(history, v_ego_ms, fps=15.0, scale_px_per_m=50.82):
    N = 5
    if len(history) < 2: return None
    recent = history[-N:]
    total_px = 0.0
    for i in range(1, len(recent)):
        x1, y1 = recent[i - 1]
        x2, y2 = recent[i]
        total_px += math.hypot(x2 - x1, y2 - y1)
    avg_px = total_px / max(1, (len(recent) - 1))
    avg_m = avg_px / scale_px_per_m
    v_rel = avg_m * fps
    delta_y = recent[-1][1] - recent[0][1]
    v_rel_signed = -v_rel if delta_y < 0 else v_rel
    v_other_ms = v_ego_ms + v_rel_signed
    v_other_kmh = v_other_ms * 3.6
    y_changes = [abs(recent[i][1] - recent[i - 1][1]) for i in range(1, len(recent))]
    avg_y_change = sum(y_changes) / len(y_changes) if y_changes else 0.0
    if delta_y > 1:   classification = "closer"
    elif delta_y < -1:classification = "farther"
    else:             classification = "same"
    return {
        'other_speed_ms': round(v_other_ms, 2),
        'other_speed_kmh': round(v_other_kmh, 1),
        'speed_classification': classification,
        'movement_smoothness': "smooth" if avg_y_change < 1.0 else "rough",
        'avg_y_change': round(avg_y_change, 2)
    }

def _first_pred_enter_step(seg_pts, poly_mask, thickness=1):
    if seg_pts is None or len(seg_pts) < 2: return None
    h_mask, w_mask = poly_mask.shape[:2]
    pts = np.asarray(seg_pts, dtype=np.int32)
    for i in range(1, len(pts)):
        x, y = int(pts[i,0]), int(pts[i,1])
        if 0 <= x < w_mask and 0 <= y < h_mask and poly_mask[y, x] != 0:
            return i
    for i in range(0, len(pts)-1):
        tmp = np.zeros_like(poly_mask, dtype=np.uint8)
        cv2.line(tmp, tuple(pts[i]), tuple(pts[i+1]), 255, thickness)
        inter = cv2.bitwise_and(tmp, poly_mask)
        if inter.any():
            return i+1
    return None

def compute_predicted_risk(future_segments, poly_pts_img, w, h, fps, sample_stride, thickness=1):
    pred_risk_map = {}
    any_pred = False
    if not (isinstance(poly_pts_img, np.ndarray) and poly_pts_img.ndim == 2 and poly_pts_img.shape[0] >= 3):
        return pred_risk_map, any_pred
    poly_mask = np.zeros((h, w), np.uint8)
    cv2.fillPoly(poly_mask, [poly_pts_img.astype(np.int32)], 255)
    dt = (max(1, sample_stride)) / max(1.0, float(fps))
    for tid, seg in future_segments.items():
        step = _first_pred_enter_step(seg, poly_mask, thickness=thickness)
        if step is None:
            pred_risk_map[tid] = None
        else:
            pred_risk_map[tid] = {"enter_step": int(step)}
            any_pred = True
    return pred_risk_map, any_pred

def build_caption(trigger, v_now, in_risk, reason):
    tag = "YES" if trigger == "current" else "PREDICTED"
    base = f"Ego {v_now:.1f} km/h. Risk zone: {tag if in_risk else 'NO'}."
    return f"{base} AI analysis: {reason}"

def other_forward_speed_bev_mps(tid, track_history, M, fps, scale_px_per_m):
    hist = track_history.get(tid, [])
    if len(hist) < 2: return None
    p_prev = np.float32([[hist[-2]]])
    p_cur  = np.float32([[hist[-1]]])
    y_prev = cv2.perspectiveTransform(p_prev, M)[0,0,1]
    y_cur  = cv2.perspectiveTransform(p_cur,  M)[0,0,1]
    v_mps = -(y_cur - y_prev) * fps / scale_px_per_m
    return float(v_mps)

def ttc_from_stopm(stop_m, v_ego_mps, v_other_mps, eps=1e-3):
    if v_ego_mps is None or v_other_mps is None: return None
    v_rel = v_ego_mps - v_other_mps
    if v_rel <= eps: return None
    return stop_m / v_rel

def pct_int(x) -> int:
    try:
        return max(0, min(100, int(round(float(x) * 100.0))))
    except Exception:
        return 0

FORBIDDEN_TERMS = ("risk", "probability", "hazard", "level", "score")

def build_reason(environment: dict, maneuver: dict, action_text: str) -> str:
    parts = []

    sig = maneuver.get("turn_signal", {})
    brk = maneuver.get("brake", {})
    spd = maneuver.get("speed_kmh", {})
    sig_state, sig_conf = sig.get("state"), sig.get("conf")
    brk_state, brk_conf = brk.get("state"), brk.get("conf")
    v = spd.get("value")

    m_bits = []
    if sig_state:
        if sig_conf is not None and sig_conf < 60:
            m_bits.append(f"{sig_state} turn signal may be on")
        else:
            m_bits.append(f"{sig_state} turn signal")
    if brk_state:
        if brk_conf is not None and brk_conf < 60:
            m_bits.append(f"brake may be {brk_state}")
        else:
            m_bits.append(f"brake {brk_state}")
    if v is not None:
        m_bits.append(f"ego speed {v:.0f} km/h")
    if m_bits:
        parts.append("Target maneuver: " + ", ".join(m_bits) + ".")

    w  = environment.get("weather", {})
    vis= environment.get("visibility", {})
    rd = environment.get("road_condition", {})
    env_bits = []
    if w.get("state"):   env_bits.append(w["state"])
    if vis.get("state"): env_bits.append(f"{vis['state']} visibility")
    if rd.get("state"):  env_bits.append(f"{rd['state']} road")
    if env_bits:
        parts.append("Conditions are " + ", ".join(env_bits) + ".")

    if action_text:
        parts.append(action_text if action_text.endswith('.') else action_text + ".")

    reason = " ".join(parts)

    low = reason.lower()
    if any(t in low for t in FORBIDDEN_TERMS):
        for t in FORBIDDEN_TERMS:
            reason = re.sub(rf"\b{t}\b", "", reason, flags=re.IGNORECASE)
        reason = re.sub(r"\s{2,}", " ", reason).strip()

    return reason

# ========================== YOLO-cls確率処理 ==========================
def yolo_probs_to_dict(res):
    names = res.names
    vec = res.probs.data.detach().float().cpu().numpy()
    prob_dict = {names[i]: float(vec[i]) for i in range(len(vec))}
    top1_idx = int(res.probs.top1)
    top1_name = names[top1_idx]
    top1_conf = float(res.probs.top1conf)
    return prob_dict, top1_name, top1_conf

def compute_window_majority_and_conf(label_deque, conf_deque):
    if not label_deque:
        return None, 0.0
    counts = defaultdict(int)
    for lb in label_deque:
        counts[lb] += 1
    final_label = max(counts.items(), key=lambda x: x[1])[0]
    vals = [c for l, c in zip(label_deque, conf_deque) if l == final_label]
    avg_conf = float(sum(vals)/len(vals)) if vals else 0.0
    return final_label, avg_conf

def avg_probs_over_window(prob_deque):
    if not prob_deque:
        return {}
    keys = set().union(*prob_deque)
    out = {}
    for k in keys:
        out[k] = float(sum(d.get(k, 0.0) for d in prob_deque) / len(prob_deque))
    return out

# ========================== Stage1推定ヘルパー ==========================
def _weather_from_stage1(s1: dict) -> str:
    if s1.get("snowy") == "YES": return "snowy"
    if s1.get("rainy") == "YES": return "rainy"
    if s1.get("sunny") == "YES": return "clear"
    return "cloudy"

def _road_from_stage1(s1: dict) -> str:
    if s1.get("snowy") == "YES": return "icy"
    if s1.get("rainy") == "YES": return "wet"
    return "dry"

# ========================== 変数 ==========================
traj_hist = defaultdict(list)

turn_buffer  = defaultdict(list)
brake_buffer = defaultdict(list)

llava_status_map = {}
pred_inside = defaultdict(lambda: False)
last_fire_frame = defaultdict(lambda: -10**9)
risk_zone_counter = defaultdict(int)
pred_risk_counter = defaultdict(int)
curr_in_risk = defaultdict(lambda: False)
sample_counter = defaultdict(int)
detection_status = defaultdict(bool)
track_bbox = defaultdict(lambda: None)
prev_in_risk = defaultdict(lambda: False)
pred_enter_seen = defaultdict(lambda: False)

WIN = 5
# ========== 黄→赤遷移時間計測用 ==========
first_yellow_frame = defaultdict(lambda: None)  # tidごとの初回黄色フレーム
first_red_frame    = defaultdict(lambda: None)  # tidごとの初回赤色フレーム
risk_time_records  = []                         # 計測結果バッファ
done_tid           = defaultdict(bool)          # 記録済みフラグ
# ★ 追加: 黄/赤検出時の概算メタ（速度・距離など）を保存
yellow_meta = defaultdict(dict)  # {tid: {"ego_speed_kmh": float, "stop_m": float}}
red_meta    = defaultdict(dict)  # 将来拡張用（必要なら）
turn_label_win  = defaultdict(lambda: deque(maxlen=WIN))
turn_conf_win   = defaultdict(lambda: deque(maxlen=WIN))
turn_probs_win  = defaultdict(lambda: deque(maxlen=WIN))
brake_label_win = defaultdict(lambda: deque(maxlen=WIN))
brake_conf_win  = defaultdict(lambda: deque(maxlen=WIN))
brake_probs_win = defaultdict(lambda: deque(maxlen=WIN))

turn_final_conf_map  = defaultdict(lambda: 0.0)
brake_final_conf_map = defaultdict(lambda: 0.0)

llava_cooldown_until = -1
LLAVA_COOLDOWN_FRAMES = 120

# ========================== 入出力 ==========================
cap = cv2.VideoCapture(video_path)
fps = cap.get(cv2.CAP_PROP_FPS)
FIRE_EVERY_N_FRAMES = max(1, int(round(fps / 5.0)))
w   = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
h   = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

BEV_W = 240
out = cv2.VideoWriter(output_video_path, cv2.VideoWriter_fourcc(*"mp4v"), fps, (360 + BEV_W, 300))

csv_first_frame = int(df_speed["frame"].iloc[0])
offset = csv_first_frame
print(f"このシーンの開始オフセット（CSV基準）: {offset} フレーム")

# COCOの車両クラスID
VEHICLE_CLASS_IDS = {2, 3, 5, 7}  # car, motorcycle, bus, truck
PEDESTRIAN_CLASS_ID = 0

# ========================== メインループ ==========================
frame_idx, trajectories, track_history = 0, [], {}
start_time = time.time()
print("▶ 統合処理を開始します...")

try:
    while cap.isOpened():
        # ========== ウォームアップフラグ ==========
        is_warmup = frame_idx < 50

        # ========== 共通メタデータの準備 ==========
        global_frame_idx = offset + frame_idx
        csv_frame_no = None
        if frame_idx < len(df_speed):
            v_row = df_speed.loc[df_speed["frame"] == global_frame_idx]
            if not v_row.empty:
                csv_frame_no = int(v_row["frame"].values[0])

        base_meta = {
            "scene_no": scene_no,
            "frame_idx": int(frame_idx),  # ← ローカル用フレーム
            "local_frame": int(frame_idx),  # ✅ 新規追加（0スタートの実行フレーム）
            "csv_frame": csv_frame_no,  # CSV上の絶対フレーム番号
            "warmup": bool(is_warmup)
        }

        # ========== フレーム読み取り ==========
        with span_with_frame_edges(tr, "frame_read", base_meta, gpu=False, offset=offset):
            ok, frame = cap.read()
        risky_found = False
        if not ok: break

        frame_raw = frame.copy()

        # === Stage1環境認識 ===
        if (frame_idx == 0) or (frame_idx % ENV_REFRESH_EVERY_FRAMES == 0) or (env_stage1_cache["result"] is None):
            try:
                # ========== Stage1環境認識の計測 ==========
                with span_with_frame_edges(tr, "stage1_env", base_meta, gpu=True, offset=offset):
                    env_stage1_cache["result"] = assess_environment_stage1_from_frame(frame_raw)
                env_stage1_cache["last_frame"] = frame_idx
            except Exception as e:
                print(f"[Stage1] error: {e}")
                env_stage1_cache["result"] = {"rural_area": "NO", "city": "NO", "snowy": "NO", "sunny": "NO",
                                              "rainy": "NO"}

        env_yesno = env_stage1_cache["result"] or {"rural_area": "NO", "city": "NO", "snowy": "NO", "sunny": "NO",
                                                   "rainy": "NO"}

        future_segments = {}
        current_frame_detected = set()

        # === Ego 値 ===
        v_row_all = df_speed.loc[df_speed["frame"] == global_frame_idx]

        if not v_row_all.empty:
            raw_speed = float(v_row_all[SPEED_COL].values[0])
            vtti_accel = float(v_row_all.get("vtti.accel_x", pd.Series([0.0])).values[0])
        else:
            raw_speed = 0.0
            vtti_accel = 0.0

        vtti_speed = float(raw_speed)
        v_ego_mps = float(raw_speed) / 3.6

        # --------- 速度ベースのポリゴン ----------
        current_speed_kmh = vtti_speed
        stop_m = speed_based_lane_length(current_speed_kmh)

        bev_h   = int(10*scale_px_per_m)

        # ========== BEV変換の計測 ==========
        with span_with_frame_edges(tr, "bev_warp", base_meta, gpu=False, offset=offset):
            bev_img = cv2.warpPerspective(frame, M, (360, bev_h))

        mask = np.ones((bev_h,360), np.uint8)*255
        cv2.fillPoly(mask,[np.array([[0,bev_h],[90,bev_h],[0,int(bev_h*0.3)]],np.int32)],0)
        cv2.fillPoly(mask,[np.array([[360,bev_h],[270,bev_h],[360,int(bev_h*0.3)]],np.int32)],0)
        gray   = cv2.cvtColor(cv2.bitwise_and(bev_img, bev_img, mask=mask), cv2.COLOR_BGR2GRAY)
        edges  = cv2.Canny(cv2.GaussianBlur(gray,(5,5),0), 15, 60)
        binary = (edges>0).astype(np.uint8)

        hist = np.sum(binary[binary.shape[0]//2:,:], axis=0)
        mid  = hist.shape[0]//2 if hist.size>0 else 0
        l_base = int(np.argmax(hist[:mid])) if hist.size>0 and mid>0 else 0
        r_base = int(np.argmax(hist[mid:])+mid) if hist.size>0 and mid>0 else 359

        n_win, margin, minpix = 30, 40, 30
        win_h = binary.shape[0]//n_win if n_win>0 else binary.shape[0]
        left_pts, right_pts = [], []
        left_x_curr, right_x_curr = l_base, r_base

        for win_idx in range(n_win):
            win_y_lo = bev_h - (win_idx+1)*win_h
            win_y_hi = bev_h - win_idx*win_h
            win_x_l_lo = max(0, left_x_curr-margin)
            win_x_l_hi = min(359, left_x_curr+margin)
            win_x_r_lo = max(0, right_x_curr-margin)
            win_x_r_hi = min(359, right_x_curr+margin)

            good_left = binary[win_y_lo:win_y_hi, win_x_l_lo:win_x_l_hi].nonzero()
            good_right= binary[win_y_lo:win_y_hi, win_x_r_lo:win_x_r_hi].nonzero()

            if len(good_left[0])>minpix:
                left_x_curr = int(np.mean(good_left[1])+win_x_l_lo)
            if len(good_right[0])>minpix:
                right_x_curr= int(np.mean(good_right[1])+win_x_r_lo)

            left_pts.append((left_x_curr, win_y_hi))
            right_pts.append((right_x_curr, win_y_hi))

        left_curv_m = None
        right_curv_m= None
        poly_pts_img = None

        if len(left_pts)>=3 and len(right_pts)>=3:
            l_arr = np.array(left_pts, np.float32)
            r_arr = np.array(right_pts, np.float32)
            l_y = l_arr[:,1]
            l_x = l_arr[:,0]
            r_y = r_arr[:,1]
            r_x = r_arr[:,0]

            coeffs_l = np.polyfit(l_y, l_x, 3)
            coeffs_r = np.polyfit(r_y, r_x, 3)
            y_eval = bev_h/2
            left_curv_m = compute_curvature_cubic(coeffs_l, y_eval, scale_px_per_m)
            right_curv_m= compute_curvature_cubic(coeffs_r, y_eval, scale_px_per_m)

            left_lane_len_m = compute_lane_length(left_pts, scale_px_per_m)
            right_lane_len_m= compute_lane_length(right_pts, scale_px_per_m)

            stop_px = int(stop_m * scale_px_per_m)
            stop_px = min(stop_px, bev_h)

            y_vals = np.linspace(bev_h-1, max(0,bev_h-stop_px), 50)
            left_x_vals = np.polyval(coeffs_l, y_vals)
            right_x_vals= np.polyval(coeffs_r, y_vals)

            poly_bev = []
            for x,y in zip(left_x_vals, y_vals):
                poly_bev.append([x,y])
            for x,y in zip(reversed(right_x_vals), reversed(y_vals)):
                poly_bev.append([x,y])
            poly_bev = np.array(poly_bev, np.float32)

            pts_img_list = cv2.perspectiveTransform(poly_bev.reshape(-1,1,2), Minv)
            poly_pts_img = pts_img_list.reshape(-1,2)

        # ========== YOLO検出の計測 ==========
        with span_with_frame_edges(tr, "detect_yolo", base_meta, gpu=True, offset=offset):
            det_res = det_model(frame, verbose=False, conf=0.4)[0]

        dets = []
        dets_with_cls = []

        for b in det_res.boxes:
            x1, y1, x2, y2 = b.xyxy[0].tolist()
            conf = float(b.conf[0])
            cls_id = int(b.cls[0])

            if cls_id not in (VEHICLE_CLASS_IDS | {PEDESTRIAN_CLASS_ID}):
                continue

            dets.append([x1, y1, x2, y2, conf])
            dets_with_cls.append((x1, y1, x2, y2, conf, cls_id))

        # ========== ByteTrackの計測 ==========
        with span_with_frame_edges(tr, "track_bytetrack", {**base_meta, "n_det": int(len(dets))}, gpu=False,
                                   offset=offset):
            tracks = tracker.update(np.array(dets, np.float32), [h, w], [h, w]) if dets else []

        min_w, min_h = 5, 10
        for t in tracks:
            tid, x, y, bw, bh = t.track_id, *map(int, t.tlwh)
            x0, y0 = x, y
            x1, y1 = min(x + bw, w - 1), min(y + bh, h - 1)
            bw_c, bh_c = x1 - x0, y1 - y0
            if bw_c < min_w or bh_c < min_h:
                continue
            cx = x0 + bw_c // 2
            cy = y1
            last_origin[tid] = (cx, cy)
            track_bbox[tid] = (x0, y0, x1, y1)

            current_frame_detected.add(tid)
            detection_status[tid] = True

            vehicle_crop = frame[y0:y1, x0:x1]
            if vehicle_crop is None or vehicle_crop.size == 0 or vehicle_crop.shape[0] < 5 or vehicle_crop.shape[1] < 5:
                if DEBUG_TRAJ:
                    print(f"[skip] bad crop tid={tid} f={frame_idx} shape={getattr(vehicle_crop,'shape',None)}")
                continue

            # ========== ウィンカー分類の計測 ==========
            with span_with_frame_edges(tr, "cls_turn", {**base_meta, "tid": int(tid)}, gpu=True, offset=offset):
                turn_result = turn_model.predict(vehicle_crop, imgsz=224, verbose=False)

            with span_with_frame_edges(tr, "cls_brake", {**base_meta, "tid": int(tid)}, gpu=True, offset=offset):
                brake_result = brake_model.predict(vehicle_crop, imgsz=224, verbose=False)

            turn_res  = turn_result[0]
            brake_res = brake_result[0]

            turn_probs,  turn_top1_name,  turn_top1_conf  = yolo_probs_to_dict(turn_res)
            brake_probs, brake_top1_name, brake_top1_conf = yolo_probs_to_dict(brake_res)

            turn_buffer[tid].append(turn_top1_name)
            brake_buffer[tid].append(brake_top1_name)
            if len(turn_buffer[tid])  > WIN: turn_buffer[tid].pop(0)
            if len(brake_buffer[tid]) > WIN: brake_buffer[tid].pop(0)

            turn_label_win[tid].append(turn_top1_name)
            turn_conf_win[tid].append(turn_top1_conf)
            turn_probs_win[tid].append(turn_probs)

            brake_label_win[tid].append(brake_top1_name)
            brake_conf_win[tid].append(brake_top1_conf)
            brake_probs_win[tid].append(brake_probs)

            turn_final,  turn_final_conf = compute_window_majority_and_conf(turn_label_win[tid],  turn_conf_win[tid])
            brake_final, brake_final_conf= compute_window_majority_and_conf(brake_label_win[tid], brake_conf_win[tid])

            turn_final_conf_map[tid]  = turn_final_conf
            brake_final_conf_map[tid] = brake_final_conf

            sample_counter[tid] += 1
            do_sample = (sample_counter[tid] % sample_stride == 0)

            if do_sample:
                x_n, y_n = px_to_model(cx, cy, w, h)
                s_n = speed_to_feature(current_speed_kmh)
                if traj_hist[tid]:
                    px_n, py_n, _, _, _ = traj_hist[tid][-1]
                    vx_n, vy_n = (x_n - px_n) * fps, (y_n - py_n) * fps
                else:
                    vx_n = vy_n = 0.0
                traj_hist[tid].append([x_n, y_n, vx_n, vy_n, s_n])
                if len(traj_hist[tid]) > H_PAST:
                    traj_hist[tid] = traj_hist[tid][-H_PAST:]

                fut_px = predict_trajectory(
                    tid, traj_hist, current_speed_kmh,
                    force_predict=True,
                    base_meta=base_meta,
                    offset=offset
                )
                if fut_px is not None:
                    last_pred_px[tid] = fut_px
                    future_segments[tid] = fut_px
                    if DEBUG_TRAJ and frame_idx % 10 == 0:
                        import numpy as _np
                        print(f"[pred(px)] tid={tid} start={_np.linalg.norm(fut_px[0]-_np.array([cx,cy])):.1f} "
                              f"end={_np.linalg.norm(fut_px[-1]-_np.array([cx,cy])):.1f}")

            if   brake_final == "go"    and turn_final == "left":  combined_label = "go_left"
            elif brake_final == "go"    and turn_final == "right": combined_label = "go_right"
            elif brake_final == "go"    and turn_final == "off":   combined_label = "go_off"
            elif brake_final == "brake" and turn_final == "left":  combined_label = "brake_left"
            elif brake_final == "brake" and turn_final == "right": combined_label = "brake_right"
            else:                                                  combined_label = "brake_off"

            action = "GO" if "go" in combined_label else "BRAKE"
            signal = "LEFT" if "left" in combined_label else "RIGHT" if "right" in combined_label else "OFF"

            other_speed_info = "Speed: - (-)"
            try:
                if len(track_history.get(tid, [])) >= 2:
                    speed_result = smoothed_speed_estimation(track_history[tid], v_ego_mps, fps, scale_px_per_m)
                    if speed_result:
                        other_speed_kmh = speed_result['other_speed_kmh']
                        other_speed_mps = speed_result['other_speed_ms']
                        sc = speed_result['speed_classification']
                        speed_class_short = {"closer": "C", "farther": "F", "same": "="}.get(sc, "-")
                        other_speed_info = f"Speed: {other_speed_kmh:.1f} km/h ({speed_class_short})"
            except Exception as e:
                print(f"速度推定エラー (ID:{tid}): {e}")
                other_speed_info = "Speed: Error"

            def _pct(x):
                try: return int(round(float(x)*100.0))
                except: return 0

            text_id     = f"ID: {tid}"
            text_action = f"Action: {action} ({_pct(brake_final_conf_map[tid])}%)"
            text_signal = f"Signal: {signal} ({_pct(turn_final_conf_map[tid])}%)"
            text_speed  = other_speed_info

            if DEBUG_TRAJ:
                hist_len = len(traj_hist[tid])
                status = "pred" if hist_len >= H_PAST else f"hist {hist_len}/{H_PAST}"
                print(f"[debug] f={frame_idx} tid={tid} {status}")

            try:
                text_y = max(0, y0 - 60)
                frame = draw_multilines_with_bg(frame, [text_id, text_action, text_signal, text_speed], x0, text_y, font_size=10)
            except Exception as e:
                print(f"4行描画エラー (ID:{tid}): {e}")
                text_y = max(0, y0 - 45)
                frame = draw_multilines_with_bg(frame, [text_id, text_action, text_signal], x0, text_y, font_size=10)

            cv2.rectangle(frame, (x0, y0), (x1, y1), (255, 0, 0), 1)
            # 追加：最大履歴秒数
            TRAIL_SECONDS = 2  # ← 過去2秒

            # 追加：最大保持フレーム数
            MAX_TRAIL_FRAMES = int(fps * TRAIL_SECONDS)

            # ---- 軌跡更新 ----
            track_history.setdefault(tid, []).append((cx, cy))

            # ★ 過去2秒分だけ残す（ここが重要）
            track_history[tid] = track_history[tid][-MAX_TRAIL_FRAMES:]

            # ---- 軌跡描画 ----
            if len(track_history[tid]) >= 2:
                for k in range(1, len(track_history[tid])):
                    cv2.line(frame, track_history[tid][k - 1], track_history[tid][k],
                             (0, 255, 255), 1)

            in_risk_zone = False
            if isinstance(poly_pts_img, np.ndarray) and poly_pts_img.ndim == 2 and poly_pts_img.shape[0] >= 3:
                in_risk_zone = cv2.pointPolygonTest(poly_pts_img.astype(np.int32), (cx, cy), False) >= 0
            risky_found |= in_risk_zone
            curr_in_risk[tid] = bool(in_risk_zone)
            prev_in_risk[tid] = in_risk_zone

            trajectories.append({
                "frame": frame_idx, "id": tid, "x": cx, "y": cy,
                "area": bw_c * bh_c, "class": combined_label, "speed": vtti_speed,
                "risky": int(in_risk_zone), "intent": combined_label
            })

            if in_risk_zone:
                risk_zone_counter[tid] += 1
            else:
                risk_zone_counter[tid] = 0

            if tid in llava_status_map:
                llava_status_map[tid]["active"] = False

        # --------- キャッシュのマージ ----------
        for tid in list(current_frame_detected):
            if sample_counter.get(tid, 0) % sample_stride != 0:
                cached_pred = last_pred_px.get(tid)
                if cached_pred is not None and len(cached_pred) >= 2:
                    future_segments[tid] = cached_pred
                    if DEBUG_TRAJ and frame_idx % 20 == 0:
                        print(f"[cached_pred] tid={tid} using cached trajectory ({len(cached_pred)} points)")

        for tid in list(last_pred_px.keys()):
            if tid not in current_frame_detected:
                del last_pred_px[tid]

        pred_risk_map, pred_risky_found = compute_predicted_risk(
            future_segments, poly_pts_img, w, h, fps, sample_stride, thickness=1
        )
        # ========== 黄→赤遷移時間の計測 ==========
        # ========== 黄→赤遷移時間の計測 ==========
        if isinstance(poly_pts_img, np.ndarray) and poly_pts_img.ndim == 2 and poly_pts_img.shape[0] >= 3:
            for tid in current_frame_detected:
                # 現在の状態を取得
                pred_info = pred_risk_map.get(tid)
                is_red = curr_in_risk.get(tid, False)  # 赤色（実侵入）
                is_yellow = (pred_info is not None) and (not is_red)  # 黄色（予測侵入）

                # 既に記録済みならスキップ
                if done_tid[tid]:
                    continue

                # (1) 黄色の初回検出
                if is_yellow and first_yellow_frame[tid] is None:
                    first_yellow_frame[tid] = global_frame_idx

                    # 概算メタ保存（既存）
                    yellow_meta[tid] = {
                        "ego_speed_kmh": float(vtti_speed),
                        "stop_m": float(stop_m),
                    }

                    # ★ 連続記録開始 + 初回サンプル（ms=0）
                    pending_series_tids.add(tid)
                    between_series[tid].append({
                        "frame": int(global_frame_idx),
                        "local_frame": int(frame_idx),  # ← 追加
                        "ms_from_yellow": 0.0,
                        "ego_speed_kmh": float(vtti_speed),
                        "stop_m": float(stop_m),
                    })

                    if DEBUG_TRAJ and frame_idx % 30 == 0:
                        print(f"[黄色検出] tid={tid} frame={global_frame_idx} "
                              f"speed≈{yellow_meta[tid]['ego_speed_kmh']:.1f}km/h "
                              f"stop≈{yellow_meta[tid]['stop_m']:.1f}m")

                # (2) 赤色の初回検出（黄色を経由している場合のみ）
                if is_red and first_yellow_frame[tid] is not None and first_red_frame[tid] is None:
                    first_red_frame[tid] = global_frame_idx
                    delta_frames = first_red_frame[tid] - first_yellow_frame[tid]

                    # 負の差分は除外（同フレームで黄→赤）
                    if delta_frames > 0:
                        delta_ms = delta_frames / fps * 1000.0

                        # 黄検出時メタ
                        ymeta = yellow_meta.get(tid, {})
                        approx_speed_kmh = float(ymeta.get("ego_speed_kmh")) if "ego_speed_kmh" in ymeta else None
                        approx_stop_m = float(ymeta.get("stop_m")) if "stop_m" in ymeta else None

                        # ★ 連続記録終了（このtidを取り除く）
                        if tid in pending_series_tids:
                            pending_series_tids.discard(tid)

                        # ★ 要約: series_len を入れる
                        series_len = int(len(between_series.get(tid, [])))

                        record = {
                            "scene_no": scene_no if scene_number is not None else "unknown",
                            "tid": int(tid),
                            "frame_yellow": int(first_yellow_frame[tid]),
                            "frame_red": int(first_red_frame[tid]),
                            "delta_frames": int(delta_frames),
                            "delta_ms": round(delta_ms, 1),
                            "fps": round(fps, 2),
                            "series_len": series_len,
                        }
                        if approx_speed_kmh is not None:
                            record["approx_ego_speed_kmh_at_yellow"] = round(approx_speed_kmh, 1)
                        if approx_stop_m is not None:
                            record["approx_stop_distance_m_at_yellow"] = round(approx_stop_m, 1)

                        risk_time_records.append(record)
                        done_tid[tid] = True  # 記録完了マーク

                        print(f"[黄→赤計測] tid={tid} {first_yellow_frame[tid]}→{first_red_frame[tid]} "
                              f"({delta_frames}fr = {delta_ms:.1f}ms) series_len={series_len} "
                              f"speed≈{record.get('approx_ego_speed_kmh_at_yellow', '-')}km/h "
                              f"stop≈{record.get('approx_stop_distance_m_at_yellow', '-')}m")
                        # === LatencyTracerにも黄→赤遷移を記録 ===
                        with tr.span(
                                "risk_transition",
                                meta={
                                    "scene_no": scene_no,
                                    "tid": int(tid),
                                    "local_frame": int(frame_idx),
                                    "csv_frame": int(global_frame_idx),
                                    "transition_ms": round(delta_ms, 3),
                                    "yellow_frame": int(first_yellow_frame[tid]),
                                    "red_frame": int(first_red_frame[tid]),
                                    "delta_frames": int(delta_frames),
                                    "series_len": int(series_len),
                                    "approx_ego_speed_kmh": float(
                                        approx_speed_kmh) if approx_speed_kmh is not None else None,
                                    "approx_stop_m": float(approx_stop_m) if approx_stop_m is not None else None,
                                },
                                gpu=False
                        ):
                            pass

            # ★ 黄→赤の“間”の連続データを毎フレーム追記（このforループの直後に）
            if pending_series_tids:
                for _tid in list(pending_series_tids):
                    y0 = first_yellow_frame.get(_tid)
                    if y0 is None:
                        continue
                    # 同一フレームの二重追加を防止
                    if between_series[_tid] and between_series[_tid][-1]["frame"] == int(global_frame_idx):
                        continue

                    ms_from_yellow = (global_frame_idx - int(y0)) / max(1.0, float(fps)) * 1000.0
                    between_series[_tid].append({
                        "frame": int(global_frame_idx),  # CSV上の絶対フレーム番号
                        "local_frame": int(frame_idx),  # ← 追加：0スタートの実行フレーム番号
                        "ms_from_yellow": float(round(ms_from_yellow, 3)),
                        "ego_speed_kmh": float(vtti_speed),
                        "stop_m": float(stop_m),
                    })

        # --------- 左上インフォ ----------
        info_lines = [f"Frame.No: {global_frame_idx}", f"FPS: {fps:.3f}",
                      f"Ego Speed: {vtti_speed:.1f} km/h", f"Accel X: {vtti_accel:.2f} m/s²"]
        for i, txt in enumerate(info_lines):
            frame = draw_text_clean(frame, txt, (5, 5 + i * 18), font_size=11, text_color=(255,255,255))

        # --------- 右上ターゲット ----------
        current_tracks = [tr for tr in trajectories if tr["frame"] == frame_idx]
        risky_tracks = [tr for tr in current_tracks if tr["risky"] == 1]
        if risky_tracks:
            target = risky_tracks[0]
        else:
            target = max(current_tracks, key=lambda t: t["area"]) if current_tracks else None
        if target:
            tid_t = target["id"]
            combined_action = target["intent"]
            vehicle_count = len(current_tracks)
            action = "GO" if "go" in combined_action else "BRAKE" if "brake" in combined_action else "-"
            if "left" in combined_action: signal="LEFT"
            elif "right" in combined_action: signal="RIGHT"
            else: signal = "OFF"
        else:
            tid_t, action, signal, vehicle_count = "-", "-", "OFF", 0

        show_tid = target["id"] if target else None

        if show_tid is None and isinstance(tid_t, int):
            show_tid = tid_t

        def _pct(x):
            try: return int(round(float(x) * 100.0))
            except: return 0

        a_pct = _pct(brake_final_conf_map.get(show_tid, 0.0)) if show_tid is not None else 0
        s_pct = _pct(turn_final_conf_map.get(show_tid, 0.0))  if show_tid is not None else 0

        target_info_lines = [
            f"Target ID: {tid_t}",
            f"Action: {action} ({a_pct}%)",
            f"Signal: {signal} ({s_pct}%)",
            f"#Vehicles: {vehicle_count}",
            f"Polygon Length: {stop_m:.1f}m",
            f"Curvature L: {left_curv_m:.4f}" if 'left_curv_m' in locals() and left_curv_m is not None else "Curvature L: -",
            f"Curvature R: {right_curv_m:.4f}" if 'right_curv_m' in locals() and right_curv_m is not None else "Curvature R: -",
        ]

        for i, txt in enumerate(target_info_lines):
            frame = draw_text_clean(frame, txt, (w - 157, 5 + i * 16), font_size=11, text_color=(255, 255, 255))

        # --------- ポリゴン描画 ----------
        overlay = frame.copy()
        if risky_found:
            poly_col = (0, 0, 255)
        elif pred_risky_found:
            poly_col = (0, 255, 255)
        else:
            poly_col = (0, 255, 0)

        if isinstance(poly_pts_img, np.ndarray) and poly_pts_img.ndim == 2 and poly_pts_img.shape[0] >= 3:
            cv2.fillPoly(overlay, [poly_pts_img.astype(np.int32)], poly_col)
        cv2.addWeighted(overlay, 0.3, frame, 0.7, 0, frame)

        # ===== 予測軌跡（最前面） =====
        for tid, seg in future_segments.items():
            if seg is None or len(seg) < 2: continue
            draw_future_polyline(frame, seg, color=(0, 0, 255), thickness=1)

        # ===== VLM 発火制御 =====
        if isinstance(poly_pts_img, np.ndarray) and poly_pts_img.ndim == 2 and poly_pts_img.shape[0] >= 3:
            for tid in list(current_frame_detected):
                pred_info = pred_risk_map.get(tid)
                now_inside_pred = (pred_info is not None)
                now_inside_cur = curr_in_risk.get(tid, False)

                if now_inside_cur:
                    pred_inside[tid] = False
                    if tid in llava_status_map:
                        llava_status_map[tid]["active"] = False
                    continue

                effective_inside = now_inside_pred and (not now_inside_cur)
                entered_now = (effective_inside and not pred_inside[tid])
                hit_rate_gate = effective_inside and ((frame_idx - last_fire_frame[tid]) >= FIRE_EVERY_N_FRAMES)
                should_fire = entered_now or hit_rate_gate
                pred_inside[tid] = effective_inside

                if not should_fire:
                    continue
                if frame_idx < llava_cooldown_until:
                    if DEBUG_TRAJ and (frame_idx % 10 == 0):
                        print(f"[LLaVA] cooldown... skip tid={tid} at frame={frame_idx}")
                    continue

                last_fire_frame[tid] = frame_idx

                vlm_img = frame_raw.copy()
                bb = track_bbox.get(tid)
                if bb:
                    x0, y0, x1, y1 = bb
                    cv2.rectangle(vlm_img, (x0, y0), (x1, y1), (255, 0, 0), 1)
                hist_pts = track_history.get(tid, [])
                if len(hist_pts) >= 2:
                    for k in range(1, len(hist_pts)):
                        cv2.line(vlm_img, hist_pts[k - 1], hist_pts[k], (0, 255, 255), 1)
                seg = future_segments.get(tid)
                if seg is not None and len(seg) >= 2:
                    cv2.polylines(vlm_img, [seg.astype(np.int32)], False, (0, 0, 255), 1)

                pre_path = os.path.join(FIRE_DIR, f"frame_{global_frame_idx:06d}_pre_tid{tid}.jpg")
                if os.path.exists(pre_path):
                    pre_path = os.path.join(FIRE_DIR, f"frame_{global_frame_idx:06d}_pre_tid{tid}_dup.jpg")
                cv2.imwrite(pre_path, vlm_img)
                print(f"[保存(pre)] {pre_path}")
                final_snapshots.append((global_frame_idx, tid))

                # ============================================================
                # ★ 5パターンの画像保存
                # ============================================================

                save_base = os.path.join(FIRE_DIR, f"frame_{global_frame_idx:06d}_tid{tid}")

                # ① 無加工
                cv2.imwrite(save_base + "_raw.jpg", frame_raw)

                # ② バウンディングボックス付き
                img_box = frame_raw.copy()
                if bb:
                    cv2.rectangle(img_box, (x0, y0), (x1, y1), (255, 0, 0), 1)
                cv2.imwrite(save_base + "_bbox.jpg", img_box)

                # ③ バウンディングボックス + 過去軌跡
                img_hist = img_box.copy()
                hist_pts = track_history.get(tid, [])
                if len(hist_pts) >= 2:
                    for k in range(1, len(hist_pts)):
                        cv2.line(img_hist, hist_pts[k - 1], hist_pts[k], (0, 255, 255), 1)
                cv2.imwrite(save_base + "_bbox_hist.jpg", img_hist)

                # ④ バウンディングボックス + 過去軌跡 + 未来予測軌跡
                img_future = img_hist.copy()
                seg = future_segments.get(tid)
                if seg is not None and len(seg) >= 2:
                    cv2.polylines(img_future, [seg.astype(np.int32)], False, (0, 0, 255), 1)
                cv2.imwrite(save_base + "_bbox_hist_future.jpg", img_future)

                print(f"[保存] 画像保存（BEV除く）4種類: {save_base}_*.jpg")
                # ⑥ レーン付き前方画像（BEV抜き）
                img_lane = frame_raw.copy()

                if isinstance(poly_pts_img, np.ndarray) and poly_pts_img.ndim == 2 and poly_pts_img.shape[0] >= 3:
                    # レーンポリゴンの色は現在と同じロジック
                    if risky_found:
                        poly_col = (0, 0, 255)  # 赤
                    elif pred_risky_found:
                        poly_col = (0, 255, 255)  # 黄
                    else:
                        poly_col = (0, 255, 0)  # 緑

                    overlay_lane = img_lane.copy()
                    cv2.fillPoly(overlay_lane, [poly_pts_img.astype(np.int32)], poly_col)
                    cv2.addWeighted(overlay_lane, 0.3, img_lane, 0.7, 0, img_lane)

                cv2.imwrite(save_base + "_lane.jpg", img_lane)

                # ② bbox + lane
                img_lane_bbox = img_lane.copy()
                if bb:
                    cv2.rectangle(img_lane_bbox, (x0, y0), (x1, y1), (255, 0, 0), 1)
                cv2.imwrite(save_base + "_lane_bbox.jpg", img_lane_bbox)

                # ③ bbox + hist + lane
                img_lane_bbox_hist = img_lane_bbox.copy()
                hist_pts = track_history.get(tid, [])
                if len(hist_pts) >= 2:
                    for k in range(1, len(hist_pts)):
                        cv2.line(img_lane_bbox_hist, hist_pts[k - 1], hist_pts[k], (0, 255, 255), 1)
                cv2.imwrite(save_base + "_lane_bbox_hist.jpg", img_lane_bbox_hist)

                # ④ bbox + hist + future + lane
                img_lane_bbox_hist_future = img_lane_bbox_hist.copy()
                seg = future_segments.get(tid)
                if seg is not None and len(seg) >= 2:
                    cv2.polylines(img_lane_bbox_hist_future, [seg.astype(np.int32)], False, (0, 0, 255), 1)
                cv2.imwrite(save_base + "_lane_bbox_hist_future.jpg", img_lane_bbox_hist_future)

                print(f"[保存] レーン付き画像4パターン: {save_base}_lane*.jpg")

                context_for_llava = simplify_for_llava({
                    "stage1_env": dict(env_yesno),
                    "ego_vehicle": {
                        "speed_kmh": float(vtti_speed),
                        "risk_zone": bool(curr_in_risk.get(tid, False)),
                        "risk_zone_predicted": bool(pred_info is not None)
                    },
                    "perception": {
                        "turn_signal": {
                            "final": max(
                                (lambda dq: {lb: list(dq).count(lb) for lb in dq})(turn_label_win[tid]).items(),
                                key=lambda x: x[1]
                            )[0] if turn_label_win[tid] else "off",
                            "final_conf_window": float(turn_final_conf_map.get(tid, 0.0)),
                            "probs_window_avg": avg_probs_over_window(turn_probs_win[tid]),
                            "window_size": len(turn_label_win[tid]),
                            "window_counts": (lambda dq: dict((lb, list(dq).count(lb)) for lb in dq))(
                                turn_label_win[tid]),
                        },
                        "brake": {
                            "final": max(
                                (lambda dq: {lb: list(dq).count(lb) for lb in dq})(brake_label_win[tid]).items(),
                                key=lambda x: x[1]
                            )[0] if brake_label_win[tid] else "off",
                            "final_conf_window": float(brake_final_conf_map.get(tid, 0.0)),
                            "probs_window_avg": avg_probs_over_window(brake_probs_win[tid]),
                            "window_size": len(brake_label_win[tid]),
                            "window_counts": (lambda dq: dict((lb, list(dq).count(lb)) for lb in dq))(
                                brake_label_win[tid]),
                        },
                    },
                    "detected_vehicles": {
                        "count": len([t for t in trajectories if t["frame"] == frame_idx])
                    }
                })

                try:
                    # ========== LLaVA呼び出しの計測 ==========
                    with span_with_frame_edges(tr, "llava_call", {**base_meta, "tid": int(tid)}, gpu=True,
                                               offset=offset):
                        result = assess_risk_from_image_with_context(pre_path, context_for_llava)
                except RuntimeError as e:
                    msg = str(e).lower()
                    if "cuda" in msg and "out of memory" in msg:
                        llava_cooldown_until = frame_idx + LLAVA_COOLDOWN_FRAMES
                        print(
                            f"[LLaVA] CUDA OOM → {LLAVA_COOLDOWN_FRAMES}フレーム冷却開始 (until {llava_cooldown_until})")
                        continue
                    else:
                        print(f"[LLaVA] 例外: {e}")
                        continue

                context_records.append({
                    "frame_id": int(global_frame_idx),
                    "image_path": pre_path,
                    **context_for_llava
                })

                llava_records.append({
                    "frame_id": int(global_frame_idx),
                    "image_path": pre_path,
                    "risk_assessment": {
                        "lane_change_detected": result.get("lane_change_detected", "unknown"),
                        "suggested_maneuver": result.get("suggested_maneuver", "keep_speed"),
                        "reason": result.get("reason", "")
                    }
                })
                image_ref = pre_path

                llava_status_map[tid] = {
                    "active": True,
                    "trigger_type": "predicted",
                    "action": result.get("suggested_maneuver", "keep_speed"),
                    "lane_change": result.get("lane_change_detected", "keep"),
                    "reason": result.get("reason", "")
                }

                in_risk_now = bool(curr_in_risk.get(tid, False))
                predicted_zone = bool(pred_info is not None)

                environment_eval = {
                    "weather": {"state": _weather_from_stage1(env_yesno)},
                    "visibility": {"state": "good"},
                    "road_condition": {"state": _road_from_stage1(env_yesno)}
                }

                turn_state = max(turn_label_win[tid], key=lambda x: list(turn_label_win[tid]).count(x)) if \
                turn_label_win[tid] else None
                brk_state = max(brake_label_win[tid], key=lambda x: list(brake_label_win[tid]).count(x)) if \
                brake_label_win[tid] else None
                maneuver_eval = {
                    "turn_signal": {"state": turn_state or "off", "conf": pct_int(turn_final_conf_map.get(tid, 0.0))},
                    "brake": {"state": brk_state or "off", "conf": pct_int(brake_final_conf_map.get(tid, 0.0))},
                    "speed_kmh": {"value": float(vtti_speed)}
                }

                advice_action = result.get("suggested_maneuver") or (
                    "Increase following distance" if "go" in combined_label else "Prepare to stop")
                advice_reason = build_reason(environment_eval, maneuver_eval, advice_action)

        # --------- LLaVA帯 ----------
        if DEBUG_TRAJ and frame_idx % 30 == 0:
            cached_count = sum(1 for c in last_pred_px.values() if c is not None)
            print(f"[trajectory_stats] frame={frame_idx} active_pred={len(future_segments)} cached={cached_count}")

        representative = next((tid for tid, info in llava_status_map.items() if info.get("active", False)), None)

        s1 = env_yesno
        llava_line1 = (
            f"Rural area: {_yn_cap(s1.get('rural_area'))}, "
            f"City: {_yn_cap(s1.get('city'))}, "
            f"Snowy: {_yn_cap(s1.get('snowy'))}, "
            f"Sunny: {_yn_cap(s1.get('sunny'))}, "
            f"Rainy: {_yn_cap(s1.get('rainy'))}"
        )

        llava_line2 = llava_line3 = llava_line4 = ""

        if representative is not None:
            info = llava_status_map[representative]
            act = str(info.get("action", "keep_speed"))
            reason_text = info.get("reason", "") or ""

            max_chars_per_line = 85

            def _wrap_reason_to_3_lines(s: str, width: int):
                s = (s or "").strip()
                if not s:
                    return "", "", ""
                words = s.split()
                lines = [""]
                for w in words:
                    new = (lines[-1] + " " + w).strip() if lines[-1] else w
                    if len(new) <= width:
                        lines[-1] = new
                    else:
                        if len(lines) == 3:
                            break
                        lines.append(w)
                while len(lines) < 3:
                    lines.append("")
                full = " ".join(words)
                used = " ".join([x for x in lines if x]).strip()
                if full != used:
                    if lines[2]:
                        if len(lines[2]) + 3 > width:
                            lines[2] = lines[2][:max(0, width - 3)]
                        lines[2] = lines[2].rstrip() + "..."
                    elif lines[1]:
                        if len(lines[1]) + 3 > width:
                            lines[1] = lines[1][:max(0, width - 3)]
                        lines[1] = lines[1].rstrip() + "..."
                return lines[0], lines[1], lines[2]

            l2, l3, l4 = _wrap_reason_to_3_lines(reason_text, max_chars_per_line)

            llava_line2 = f"{act}: {l2}" if l2 else act
            llava_line3 = l3
            llava_line4 = l4

        band_h = 60
        llava_vis = np.zeros((band_h, 360, 3), dtype=np.uint8)
        font_size = 7
        line_height = 14
        y_base = 6
        llava_vis = draw_text_clean(llava_vis, llava_line1, (5, y_base + 0 * line_height), font_size=font_size)
        if llava_line2:
            llava_vis = draw_text_clean(llava_vis, llava_line2, (5, y_base + 1 * line_height), font_size=font_size)
        if llava_line3:
            llava_vis = draw_text_clean(llava_vis, llava_line3, (5, y_base + 2 * line_height), font_size=font_size)
        if llava_line4:
            llava_vis = draw_text_clean(llava_vis, llava_line4, (5, y_base + 3 * line_height), font_size=font_size)

        resized_frame = cv2.resize(frame, (360, 240))
        combined_left = np.vstack([resized_frame, llava_vis])

        # ===== 右側BEVパネル =====
        bev_canvas = np.zeros((300, BEV_W, 3), dtype=np.uint8)

        bev_draw = np.zeros((bev_h, 360, 3), dtype=np.uint8)

        try:
            edges = cv2.Canny(cv2.cvtColor(bev_img, cv2.COLOR_BGR2GRAY), 15, 60)

            if EDGE_DILATE_ITER and EDGE_DILATE_ITER > 0:
                kernel = cv2.getStructuringElement(cv2.MORPH_RECT, EDGE_KERNEL)
                edges = cv2.dilate(edges, kernel, iterations=int(EDGE_DILATE_ITER))

            if EDGE_UNSHARP:
                blurred = cv2.GaussianBlur(edges, (0, 0), 1.0)
                edges = cv2.addWeighted(edges, 1.5, blurred, -0.5, 0)

            edges_vis = cv2.cvtColor(edges, cv2.COLOR_GRAY2BGR)
            bev_draw = cv2.addWeighted(bev_draw, 1.0, edges_vis, EDGE_ALPHA, 0.0)

            if left_pts and right_pts:
                for i in range(1, len(left_pts)):
                    cv2.line(bev_draw, left_pts[i - 1], left_pts[i], (0, 255, 0), 1)
                for i in range(1, len(right_pts)):
                    cv2.line(bev_draw, right_pts[i - 1], right_pts[i], (255, 0, 0), 1)

            if isinstance(poly_bev, np.ndarray) and poly_bev.shape[0] >= 3:
                tmp = bev_draw.copy()
                cv2.fillPoly(tmp, [poly_bev.astype(np.int32)], (0, 255, 255))
                bev_draw = cv2.addWeighted(tmp, 0.30, bev_draw, 0.70, 0.0)

            if SHOW_SLIDING_WINDOWS and 'n_win' in locals() and 'win_h' in locals():
                for i in range(n_win):
                    wy_low = bev_h - (i + 1) * win_h
                    wy_high = bev_h - i * win_h
                    cv2.rectangle(bev_draw, (0, max(0, wy_low)), (359, min(bev_h - 1, wy_high)), SLIDE_WIN_COLOR,
                                  SLIDE_WIN_THICK)

            if left_pts and right_pts:
                for i in range(1, len(left_pts)):
                    cv2.line(bev_draw, left_pts[i - 1], left_pts[i], (0, 255, 0), 1)
                for i in range(1, len(right_pts)):
                    cv2.line(bev_draw, right_pts[i - 1], right_pts[i], (255, 0, 0), 1)

            if isinstance(poly_bev, np.ndarray) and poly_bev.shape[0] >= 3:
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

        #if final_snapshots:
        #    for (g_idx, t_id) in final_snapshots:
        #        post_path = os.path.join(FIRE_DIR, f"frame_{g_idx:06d}_post_tid{t_id}.jpg")
        #        cv2.imwrite(post_path, final_frame)
        #    final_snapshots.clear()

        # ========== 動画書き出しの計測 ==========
        with span_with_frame_edges(tr, "video_write", base_meta, gpu=False, offset=offset):
            out.write(final_frame)

        if frame_idx % 10 == 0:
            elapsed = time.time() - start_time
            pct = (frame_idx + 1) / max(1,total_frames) * 100
            print(f"[{frame_idx:6d}/{total_frames}] {pct:5.1f}% det:{len(dets):2d} trk:{len(track_history):2d} elapsed:{elapsed:6.1f}s")

        frame_idx += 1

except KeyboardInterrupt:
    print("⏹️ 中断されました。現在の進捗まで保存します...")

except Exception as e:
    print(f"❌ エラーが発生しました: {e}")

finally:
    print("🔄 クリーンアップ中...")
    try:
        cap.release()
    except:
        pass

    try:
        out.release()
        print("✅ 動画ファイルを保存しました")
    except:
        print("⚠️ 動画保存でエラーが発生しました")

    try:
        if trajectories:
            pd.DataFrame(trajectories).to_csv(output_csv_path, index=False)
            print("✅ CSVファイルを保存しました")
    except:
        print("⚠️ CSV保存でエラーが発生しました")

    # 背後情報とLLaVA記録を保存
    context_json_path = os.path.join(base_dir, f"scene_{scene_no}_context.json")
    with open(context_json_path, "w", encoding="utf-8") as f:
        json.dump(context_records, f, ensure_ascii=False, indent=2)
    print(f"✅ 背後情報 JSON を保存しました → {context_json_path}")

    llava_json_path = os.path.join(base_dir, f"scene_{scene_no}_llava.json")
    with open(llava_json_path, "w", encoding="utf-8") as f:
        json.dump(llava_records, f, ensure_ascii=False, indent=2)
    print(f"✅ LLaVA最小 JSON を保存しました → {llava_json_path}")

    # 基本情報出力
    total_elapsed = time.time() - start_time
    print(f"\n📊 処理時間: {total_elapsed:.1f}秒")
    print(f"🎬 処理フレーム数: {frame_idx:,}")
    print(f"🎥 出力動画: {output_video_path}")
    print(f"📄 出力CSV: {output_csv_path}")
    print(f"📈 レイテンシログ: {latency_csv_path}")
    print(f"📈 レイテンシログ: {latency_jsonl_path}")
    print("🏁 完了")

    # ========== 黄→赤遷移時間の保存 ==========
    if risk_time_records:
        risk_csv_path = os.path.join(base_dir, f"scene_{scene_no}_risk_transition.csv")
        risk_json_path = os.path.join(base_dir, f"scene_{scene_no}_risk_transition.json")

        try:
            # CSV保存
            df_risk = pd.DataFrame(risk_time_records)
            df_risk.to_csv(risk_csv_path, index=False)
            print(f"✅ 黄→赤 遷移時間CSV保存: {risk_csv_path}")

            # JSON保存（オプション）
            with open(risk_json_path, "w", encoding="utf-8") as f:
                json.dump(risk_time_records, f, ensure_ascii=False, indent=2)
            print(f"✅ 黄→赤 遷移時間JSON保存: {risk_json_path}")

            # 統計サマリー出力
            times_ms = [r["delta_ms"] for r in risk_time_records]
            print(f"📊 遷移時間統計: n={len(times_ms)} "
                  f"平均={np.mean(times_ms):.1f}ms 中央値={np.median(times_ms):.1f}ms "
                  f"最小={np.min(times_ms):.1f}ms 最大={np.max(times_ms):.1f}ms")

        except Exception as e:
            print(f"⚠️ 遷移時間保存でエラー: {e}")
    else:
        print("ℹ️ 黄→赤遷移の記録なし（該当イベントが発生しませんでした）")
    # ========== 黄→赤 “間” の連続データ保存 ==========
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
            print(f"✅ 連続データ JSON 保存: {series_json_path}")

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
            print(f"✅ 連続データ CSV 保存: {series_csv_path}")
        else:
            print("ℹ️ 黄→赤の連続データなし")
    except Exception as e:
        print(f"⚠️ 連続データ保存でエラー: {e}")

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

            print(f"✅ シーンサマリー追記: {summary_path}")
        else:
            print("ℹ️ このシーンでは黄→赤イベントなし（サマリー追記スキップ）")
    except Exception as e:
        print(f"⚠️ シーンサマリー追記でエラー: {e}")
