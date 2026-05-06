# -*- coding: utf-8 -*-
"""
統合版スクリプト v2.3 – 最小表示・ウィンカー/ブレーキ分類/レーン認識なし・速度CSV不要版
- ByteTrack でトラッキング
- YOLOv8 (COCO) で車/トラック/歩行者検出
- 予測軌跡（Seq2Seq, 絶対座標）描画（学習と同期）※軌跡 ckpt が無い場合は自動で無効化
- LLaVA は「歩行者が存在」かつ「1秒ごと」に発火（tid ごとフォルダ保存）
- UI は最小限（左上: Frame.No / FPS、吹き出し: ID と Class）
"""

import os, re, time, types
import cv2
import numpy as np
import torch
from collections import defaultdict
from ultralytics import YOLO
from yolox.tracker.byte_tracker import BYTETracker
from PIL import ImageFont, ImageDraw, Image
from train_traj_seq2seq_px_rel import Seq2Seq
import argparse
from risk_assessment_api_xx2 import assess_risk_from_image_with_context
from stage1_env import assess_environment_stage1_from_frame
import textwrap

# ========================== 既定パラメタ ==========================
IMG_W, IMG_H = 360.0, 240.0
H_PAST, H_FUT = 30, 45
DEBUG_TRAJ = True
# 下部テキスト帯の高さ（環境行を消して詰めるので少し低く）
BOTTOM_BAND_H = 72
# シンプル描画（Windows の \x エスケープ事故を避けるため r"..." を使用）
font_path = r"C:\\Users\\s1280\\PycharmProjects\\yolo_classify_project\\26x\\fonts\\RobotoMono-Regular.ttf"

def draw_text(frame, text, xy, size=11, color=(255,255,255)):
    try:
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    except Exception:
        rgb = frame
    pil = Image.fromarray(rgb)
    d = ImageDraw.Draw(pil)
    try:
        f = ImageFont.truetype(font_path, size)
    except Exception:
        f = ImageFont.load_default()
    d.text(xy, text, font=f, fill=color)
    return cv2.cvtColor(np.array(pil), cv2.COLOR_RGB2BGR)

def draw_box_text(frame, lines, x, y, size=10):
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    pil = Image.fromarray(rgb)
    d = ImageDraw.Draw(pil)
    try:
        f = ImageFont.truetype(font_path, size)
    except Exception:
        f = ImageFont.load_default()
    txt = "\n".join(lines)  # ←ここは正しい
    bbox = d.multiline_textbbox((0,0), txt, font=f, spacing=2)
    w = bbox[2]-bbox[0]; h = bbox[3]-bbox[1]
    d.rectangle([x, y, x+w+6, y+h+6], fill=(0,0,0))
    d.multiline_text((x+3, y+3), txt, font=f, fill=(255,255,255), spacing=2)
    return cv2.cvtColor(np.array(pil), cv2.COLOR_RGB2BGR)


# ========================== 引数 / I/O パス ==========================
parser = argparse.ArgumentParser()
parser.add_argument("--video", type=str)
parser.add_argument("--traj", type=str, help="Seq2Seq ckpt path (optional)")
args, _ = parser.parse_known_args()

#video_path = args.video or r"C:\\Users\\s1280\\Desktop\\SHRP2rawdata\\2\\S06NDS_Sample_120406_1451_00186_Forward.mp4"
video_path = args.video or r"D:\JAAD_clips\JAAD_clips\video_0001.mp4"

output_video_path = os.path.splitext(video_path)[0] + "_minimal.mp4"
output_csv_path   = os.path.splitext(video_path)[0] + "_minimal_traj.csv"

base_dir = os.path.dirname(video_path)
scene_match = re.search(r"(?i)scene_(\d+)", os.path.basename(video_path))
scene_no = scene_match.group(1).zfill(3) if scene_match else "unknown"

# ===== 保存ディレクトリ =====
FIRE_DIR = os.path.join(base_dir, f"scene_{scene_no}_fire_min")
os.makedirs(FIRE_DIR, exist_ok=True)
FRAMES_DIR = os.path.join(base_dir, f"scene_{scene_no}_frames")
os.makedirs(FRAMES_DIR, exist_ok=True)
SAVE_EVERY_FRAME = True

# ========================== モデル ==========================
# cap を開いた“後”に fps を読んでから設定する
cap = cv2.VideoCapture(video_path)
fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
w   = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 1280)
h   = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 720)

args_bt = types.SimpleNamespace(track_thresh=0.15, match_thresh=0.8,
                                track_buffer=60, frame_rate=int(round(fps)), mot20=False)
tracker = BYTETracker(args_bt)


det_device = "cuda" if torch.cuda.is_available() else "cpu"
det_model = YOLO("yolov8x.pt").to(det_device)  # COCO: person=0, car=2, truck=7

# 予測軌跡モデル（ckpt が無ければ自動で無効化）
traj_device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
traj_ckpt_path = args.traj or r"C:\\Users\\s1280\\PycharmProjects\\yolo_classify_project\\26x\\checkpoints_traj_px_best15\\best_ade_px.pt"
USE_TRAJ = True
try:
    if not os.path.isfile(traj_ckpt_path):
        raise FileNotFoundError(traj_ckpt_path)
    ckpt = torch.load(traj_ckpt_path, map_location=traj_device)
    cfg = ckpt.get("cfg", {})
    H_PAST = int(cfg.get("h_past", H_PAST))
    H_FUT  = int(cfg.get("h_fut",  H_FUT))
    IMG_W  = float(cfg.get("img_w", IMG_W))
    IMG_H  = float(cfg.get("img_h", IMG_H))
    sample_stride = int(cfg.get("sample_stride", 1))
    hidden  = int(cfg.get("hidden", 256))
    layers  = int(cfg.get("layers", 3))
    dropout = float(cfg.get("dropout", 0.2))
    traj_model = Seq2Seq(input_size=5, hidden=hidden, n_layers=layers,
                         dropout=dropout, out_size=2).to(traj_device)
    traj_model.load_state_dict(ckpt["state_dict"])  # type: ignore
    traj_model.eval()
    print(f"[traj] loaded: {traj_ckpt_path}")
except Exception as e:
    USE_TRAJ = False
    sample_stride = 10**9  # 事実上オフ
    print(f"[traj] disabled ({e})")

# ========================== バッファ ==========================
traj_hist = defaultdict(list)
track_bbox = defaultdict(lambda: None)
track_history = {}
track_cls = defaultdict(lambda: None)  # 0:ped, 2:car, 7:truck

llava_status_map = {}
llava_cooldown_until = -1
LLAVA_COOLDOWN_FRAMES = 120

# Stage1 YES/NO（任意）
ENV_REFRESH_EVERY_FRAMES = 100
env_stage1_cache = {"result": None, "last_frame": -10**9}

# ========================== 補助関数 ==========================

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

def predict_trajectory(tid, traj_hist, force_predict=False):
    hist = traj_hist[tid]
    if len(hist) >= H_PAST:
        input_seq = hist[-H_PAST:]
    elif force_predict and len(hist) > 0:
        first = hist[0]
        input_seq = [first] * (H_PAST - len(hist)) + hist
        if DEBUG_TRAJ:
            print(f"[early_pred] tid={tid} padded {len(hist)} -> {H_PAST}")
    else:
        return None
    try:
        with torch.no_grad():
            inp = np.asarray(input_seq, dtype=np.float32)
            inp[:, :2] -= inp[-1, :2]
            x_seq = torch.from_numpy(inp).unsqueeze(0).to(traj_device)
            pred = traj_model(x_seq, tgt=None, tf_ratio=0.0, steps=H_FUT)
        fut_raw = pred.squeeze(0).detach().cpu().numpy()
        ox, oy = last_origin.get(tid, (None, None))
        if ox is None:
            return None
        fut_abs = fut_raw
        px_x = ox + fut_abs[:, 0]
        px_y = oy + fut_abs[:, 1]
        fut_px = np.stack([px_x, px_y], axis=1)
        fut_px_vis = np.vstack([[ox, oy], fut_px])
        if np.isfinite(fut_px_vis).all() and fut_px_vis.shape[0] >= 2:
            return fut_px_vis.astype(np.int32)
    except Exception as e:
        if DEBUG_TRAJ:
            print(f"[pred_error] tid={tid}: {e}")
    return None
# 置き換え：文字幅（px）で折り返すラッパ
def wrap_with_prefix(prefix: str, text: str, width_px: int, *, font_size: int = 9,
                     max_lines: int = 2) -> list[str]:
    """
    prefix を付けた1行目は prefix を含めて幅制約、2行目以降は本文のみで幅制約。
    width_px: ピクセル幅（帯の左右パディングを引いた残りの描画幅）
    max_lines: 最大行数（溢れたら "…" で省略）
    """
    try:
        f = ImageFont.truetype(font_path, font_size)
    except Exception:
        f = ImageFont.load_default()

    # 余計な空白除去
    text = " ".join((text or "").split())
    words = text.split()

    # 1行目（prefix つき）
    img = Image.new("RGB", (width_px, 20), (0, 0, 0))
    d = ImageDraw.Draw(img)

    def fit_line(head: str, words_iter):
        line = head
        consumed = 0
        # head がある場合は先に幅を確保
        cur = head
        for i, w in enumerate(words_iter):
            trial = (cur + (" " if cur and not cur.endswith(" ") else "") + w) if cur else w
            if d.textlength(trial, font=f) <= width_px:
                cur = trial
                consumed = i + 1
            else:
                break
        return cur, consumed

    # 1行目
    line1, used = fit_line(prefix, words)
    lines = [line1]

    # 2行目以降（prefix なし）
    remain = words[used:]
    while remain and len(lines) < max_lines:
        cur, used2 = fit_line("", remain)
        if not cur:  # 1語でも置けない場合は強制改行
            cur = remain[0]
            used2 = 1
        lines.append(cur)
        remain = remain[used2:]

    # まだ残りがあるなら末尾に "…" を付けて省略
    if remain:
        if lines:
            if not lines[-1].endswith("…"):
                # 末尾に "…" を無理なく付けられるよう微調整
                while d.textlength(lines[-1] + "…", font=f) > width_px and len(lines[-1]) > 0:
                    lines[-1] = lines[-1][:-1]
                lines[-1] = (lines[-1] + "…") if lines[-1] else "…"
        else:
            lines = ["…"]

    return lines




last_origin = defaultdict(lambda: None)
last_pred_px = {}
sample_counter = defaultdict(int)

# ========================== メイン処理 ==========================
cap = cv2.VideoCapture(video_path)
fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
w   = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 1280)
h   = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 720)

out = cv2.VideoWriter(output_video_path, cv2.VideoWriter_fourcc(*"mp4v"), fps, (360, int(240 + BOTTOM_BAND_H)))

frame_idx = 0
trajectories = []
start_time = time.time()

try:
    while cap.isOpened():
        ok, frame = cap.read()
        if not ok:
            break
        frame_raw = frame.copy()

        # Stage1 YES/NO（100フレームごと）
        if (frame_idx == 0) or (frame_idx % ENV_REFRESH_EVERY_FRAMES == 0) or (env_stage1_cache["result"] is None):
            try:
                env_stage1_cache["result"] = assess_environment_stage1_from_frame(frame_raw)
                env_stage1_cache["last_frame"] = frame_idx
            except Exception as e:
                print(f"[Stage1] error: {e}")
                env_stage1_cache["result"] = {"rural_area": "NO", "city": "NO", "snowy": "NO", "sunny": "NO", "rainy": "NO"}
        env_yesno = env_stage1_cache["result"]

        # 検出
        det_res = det_model(frame, imgsz=640, conf=0.4, iou=0.7, augment=True, verbose=False)[0]
        dets = []
        dets_with_cls = []
        for b in det_res.boxes:
            x1, y1, x2, y2 = b.xyxy[0].tolist()
            conf = float(b.conf[0])
            cls  = int(b.cls[0])
            if cls not in (0,2,7):
                continue
            if cls == 0 and conf < 0.15:
                continue
            if cls in (2,7) and conf < 0.40:
                continue
            dets.append([x1,y1,x2,y2,conf])
            dets_with_cls.append((x1,y1,x2,y2,conf,cls))

        tracks = tracker.update(np.array(dets, np.float32), [h,w], [h,w]) if dets else []

        current_frame_detected = set()
        future_segments = {}

        for t in tracks:
            tid, x, y, bw, bh = t.track_id, *map(int, t.tlwh)
            x0, y0 = x, y
            x1, y1 = min(x + bw, w - 1), min(y + bh, h - 1)
            if x1-x0 < 5 or y1-y0 < 10:
                continue

            cx = x0 + (x1-x0)//2
            cy = y1
            last_origin[tid] = (cx, cy)
            track_bbox[tid] = (x0,y0,x1,y1)
            current_frame_detected.add(tid)

            # 検出結果からクラスを復元
            cls_id_now = _class_for_track(x0, y0, x1, y1, dets_with_cls, iou_thr=0.3)
            track_cls[tid] = cls_id_now

            # 軌跡履歴（stride 同期）
            sample_counter[tid] += 1
            if USE_TRAJ and (sample_counter[tid] % sample_stride == 0):
                vx = vy = 0.0
                if traj_hist[tid]:
                    px, py, _, _, _ = traj_hist[tid][-1]
                    vx, vy = (cx - px)*fps, (cy - py)*fps
                # 入力 5 次元 [x, y, vx, vy, s] だが速度特徴は 0 とする
                traj_hist[tid].append([float(cx), float(cy), vx, vy, 0.0])
                if len(traj_hist[tid]) > H_PAST:
                    traj_hist[tid] = traj_hist[tid][-H_PAST:]
                fut = predict_trajectory(tid, traj_hist, force_predict=True)
                if fut is not None:
                    last_pred_px[tid] = fut
                    future_segments[tid] = fut

            if tid not in future_segments and tid in last_pred_px and last_pred_px[tid] is not None:
                future_segments[tid] = last_pred_px[tid]

            # 表示（IDとクラスのみ）
            cls_name = {0: "Pedestrian", 2: "Car", 7: "Truck"}.get(track_cls[tid], str(track_cls[tid]))
            frame = cv2.rectangle(frame, (x0,y0), (x1,y1), (255,0,0), 1)
            frame = draw_box_text(frame, [f"ID: {tid}", f"Class: {cls_name}"], x0, max(0,y0-40), size=10)

            # 履歴
            track_history.setdefault(tid, []).append((cx, cy))
            if len(track_history[tid]) >= 2:
                for k in range(1, len(track_history[tid])):
                    cv2.line(frame, track_history[tid][k-1], track_history[tid][k], (0,255,255), 1)

            # 予測線
            seg = future_segments.get(tid)
            if seg is not None and len(seg) >= 2:
                cv2.polylines(frame, [seg.astype(np.int32)], False, (0,0,255), 1)

            # 軽量CSV用ログ
            trajectories.append({
                "frame": frame_idx, "id": tid, "x": cx, "y": cy,
                "area": (x1-x0)*(y1-y0), "class": cls_name
            })

        # 左上インフォ（最小）
        info = [f"Frame.No: {frame_idx}", f"FPS: {fps:.3f}"]
        for i, tline in enumerate(info):
            frame = draw_text(frame, tline, (5, 5 + i*16), size=11)

        # LLaVA 帯
        # ---------------- LLaVA 帯（環境行は描かない） ----------------
        band = np.zeros((BOTTOM_BAND_H, 360, 3), dtype=np.uint8)

        # 環境YES/NOはLLaVAコンテキスト用に保持（帯には表示しない）
        s1 = env_yesno if isinstance(env_yesno, dict) else {}

        # === LLaVA 発火（歩行者が存在し、1秒ごと） ===
        FIRE_EVERY_N_FRAMES = max(1, int(round(fps * 1.0)))
        has_ped_now = any(track_cls.get(tid) == 0 for tid in current_frame_detected)
        if has_ped_now and (frame_idx % FIRE_EVERY_N_FRAMES == 0) and (frame_idx >= llava_cooldown_until):
            now_tracks = [tr for tr in trajectories if tr["frame"] == frame_idx and str(tr.get("class")).lower() == "pedestrian"]
            rep = max(now_tracks, key=lambda t: t["area"]) if now_tracks else None
            if rep is not None:
                tid = rep["id"]
                bb = track_bbox.get(tid)
                vlm_img = frame_raw.copy()
                if bb:
                    x0,y0,x1,y1 = bb
                    cv2.rectangle(vlm_img, (x0,y0), (x1,y1), (0,200,255), 1)  # pedestrian: シアン
                seg = last_pred_px.get(tid)
                if seg is not None and len(seg) >= 2:
                    cv2.polylines(vlm_img, [seg.astype(np.int32)], False, (0,0,255), 1)
                _tid_dir = os.path.join(FIRE_DIR, f"tid{tid:04d}")
                os.makedirs(_tid_dir, exist_ok=True)
                pre_path = os.path.join(_tid_dir, f"frame_{frame_idx:06d}_pre.jpg")
                cv2.imwrite(pre_path, vlm_img)

                context = {
                    "stage1_env": dict(s1),
                    "perception": {
                        "pedestrian_present": True,
                        "pedestrian_count": sum(1 for t in current_frame_detected if track_cls.get(t) == 0)
                    },
                    "detected_vehicles": {
                        "count": sum(1 for t in current_frame_detected if track_cls.get(t) in (2,7))
                    }
                }
                try:
                    result = assess_risk_from_image_with_context(pre_path, context)
                    llava_status_map[tid] = {
                        "active": True,
                        "ped_state": result.get("pedestrian_state", "unknown"),
                        "veh_action": result.get("suggested_action", "unknown"),
                        "reason": result.get("reason", "")
                    }
                except RuntimeError as e:
                    msg = str(e).lower()
                    if "cuda" in msg and "out of memory" in msg:
                        llava_cooldown_until = frame_idx + LLAVA_COOLDOWN_FRAMES
                        print(f"[LLaVA] CUDA OOM → cooldown {LLAVA_COOLDOWN_FRAMES} frames")
                except Exception as e:
                    print(f"[LLaVA] error: {e}")

        # LLaVA テキスト（帯の上端に詰めて表示）
        rep_tid = next((tid for tid, s in llava_status_map.items() if s.get("active", False)), None)
        if rep_tid is not None:
            s = llava_status_map[rep_tid]
            ped = s.get("ped_state", "unknown")
            act = s.get("veh_action", "unknown")
            prefix = f"Ped: {ped}  |  Ego: {act}  –  "
            reason = (s.get("reason", "") or "").strip()

            # ← ここを width=80（文字数）→ width_px=350（ピクセル幅）に
            lines = wrap_with_prefix(prefix, reason, width_px=350, font_size=9, max_lines=5)
            band = draw_box_text(band, lines, 5, 6, size=9)

        # 出力フレーム作成
        resized = cv2.resize(frame, (360, 240))
        final_frame = np.vstack([resized, band])
        out.write(final_frame)

        # 全フレーム画像の保存
        if SAVE_EVERY_FRAME:
            _frame_path = os.path.join(FRAMES_DIR, f"frame_{frame_idx:06d}.jpg")
            cv2.imwrite(_frame_path, final_frame)

        if frame_idx % 10 == 0:
            elapsed = time.time() - start_time
            print(f"[{frame_idx:6d}] elapsed:{elapsed:6.1f}s det:{len(dets):2d} trk:{len(track_history):2d}")
        frame_idx += 1

except KeyboardInterrupt:
    print("⏹️ 中断。保存します…")
except Exception as e:
    print(f"❌ エラー: {e}")
finally:
    try: cap.release()
    except: pass
    try: out.release(); print("✅ 動画を保存")
    except: print("⚠️ 動画保存エラー")
    # 軽量CSV出力（必要なければ削除可）
    try:
        import pandas as pd
        if trajectories:
            pd.DataFrame(trajectories).to_csv(output_csv_path, index=False)
            print("✅ CSVを保存")
    except Exception:
        try:
            with open(output_csv_path, "w", encoding="utf-8") as f:
                cols = ["frame","id","x","y","area","class"]
                f.write(",".join(cols)+"\n")
                for r in trajectories:
                    f.write(",".join(str(r.get(c, "")) for c in cols)+"\n")
            print("✅ 簡易CSVを保存")
        except Exception as e:
            print("⚠️ CSV保存エラー", e)
    print("🏁 完了")