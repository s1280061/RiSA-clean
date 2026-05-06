"""
統合版スクリプト v2  –  print 進捗ログ版 - RobotoMono フォント対応
────────────────────────────────────────
● ByteTrack で車両トラッキング
● YOLOv8x (COCO) で車・トラック検出
● best.pt で各 bbox 内のウインカー / ブレーキライト検出
● ルールベースで車両意図（左折・右折・ブレーキ・夜間走行・直進）を推定
● 緑のレーン検出ポリゴンを継続表示
● bbox 内に意図アイコン & 速度などをコンパクト描画
● フレーム10枚ごとに進捗を print
● 完成動画と CSV を保存
● RobotoMono フォントによる高品質テキスト描画
────────────────────────────────────────
"""
import cv2
import numpy as np
import pandas as pd
import os, types, time
from ultralytics import YOLO
from yolox.tracker.byte_tracker import BYTETracker
from collections import defaultdict
import re
from PIL import ImageFont, ImageDraw, Image
from risk_assessment_api import assess_risk_from_image
import textwrap
import json


# ---------- パス設定 ----------------------------------------------------------
video_path        = r"C:\Users\s1280\Desktop\SHRP2rawdata\5\new_divided\scene_020.mp4"
csv_path          = r"C:\Users\s1280\Desktop\SHRP2rawdata\5\csv_divided\scene_020.csv"
base_path         = r"C:\Users\s1280\PycharmProjects\yolo_classify_project\26x"
best_pt_path      = r"C:\Users\s1280\PycharmProjects\yolo_classify_project\runs\detect\train26\weights\best.pt"
font_path         = r"C:\Users\s1280\PycharmProjects\yolo_classify_project\26x\fonts\RobotoMono-Regular.ttf"

output_video_path = video_path.replace(".mp4", "_combined_speed_yolo_vx1.mp4")
output_csv_path   = video_path.replace(".mp4", "_trajectories_combined.csv")

# === JSONを置くフォルダ（動画と同じフォルダ） ===
base_dir = os.path.dirname(video_path)

# === シーン番号抽出 ===
scene_match = re.search(r"scene_(\d+)", video_path)
scene_no = scene_match.group(1) if scene_match else "unknown"

# === シーン専用の画像フォルダをループ前に作成 ===
risk_frame_dir = os.path.join(base_dir, f"risk_frames_scene_{scene_no}")
os.makedirs(risk_frame_dir, exist_ok=True)


# ---------- BEV 変換用座標 ----------------------------------------------------
src  = np.load(os.path.join(base_path, "src_points_3_forward.npy"))
dst  = np.load(os.path.join(base_path, "dst_points_3_forward.npy"))
M    = cv2.getPerspectiveTransform(src, dst)
Minv = cv2.getPerspectiveTransform(dst, src)
scale_px_per_m = 50.82

# ---------- 速度 CSV 読み込み -------------------------------------------------
df_speed = pd.read_csv(csv_path, dtype={"frame": int}, low_memory=False)

# ---------- ByteTrack ---------------------------------------------------------
args = types.SimpleNamespace(track_thresh=0.3, match_thresh=0.7,
                             track_buffer=30, frame_rate=15, mot20=False)
tracker = BYTETracker(args)

# ---------- モデル ------------------------------------------------------------
# 4クラス学習済みモデルを1本だけロード
# 最新の学習済みモデル（TLD_yolov8m_4cls_50ep2）
det_model = YOLO(
    r"C:\Users\s1280\PycharmProjects\yolo_classify_project\runs\detect\TLD_yolov8m_4cls_50ep2\weights\best.pt"
).to("cuda")


# クラス辞書を意図クラスに変更
class_names = {0: "brake", 1: "go", 2: "left", 3: "right"}

# ---------- 動画入出力 --------------------------------------------------------
cap = cv2.VideoCapture(video_path)
fps = cap.get(cv2.CAP_PROP_FPS)
w   = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
h   = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
# 修正後（元の動画サイズで出力）：
out = cv2.VideoWriter(output_video_path,
                      cv2.VideoWriter_fourcc(*"mp4v"), fps, (360, 300))

# 分割動画のファイル名から scene_xxx を取得
scene_match = re.search(r"scene_(\d+)", video_path)
scene_no = int(scene_match.group(1)) if scene_match else 0

# CSVの最初のフレーム番号を基準にオフセット決定
csv_first_frame = int(df_speed["frame"].iloc[0])
offset = csv_first_frame
print(f"このシーンの開始オフセット（CSV基準）: {offset} フレーム")

json_records = []   # ← CoVLA 風 JSON のための格納リスト

def draw_text_roboto(cv_img, text, position, font_size=16, color=(255,255,255)):
    """
    RobotoMono フォントでテキストを描画する関数

    Args:
        cv_img: OpenCV画像 (BGR)
        text: 描画するテキスト
        position: (x, y) 座標
        font_size: フォントサイズ
        color: RGB色 (255,255,255) = 白

    Returns:
        テキストが描画されたOpenCV画像
    """
    # OpenCV (BGR) → Pillow (RGB)
    cv_rgb = cv2.cvtColor(cv_img, cv2.COLOR_BGR2RGB)
    pil_img = Image.fromarray(cv_rgb)
    draw = ImageDraw.Draw(pil_img)

    try:
        font = ImageFont.truetype(font_path, font_size)
    except:
        # フォントが見つからない場合はデフォルトフォントを使用
        font = ImageFont.load_default()

    draw.text(position, text, font=font, fill=color)

    # Pillow (RGB) → OpenCV (BGR)
    return cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)

def draw_text_clean(cv_img, text, position, font_size=16, text_color=(255,255,255)):
    """
    背景なしの白文字テキストを描画する関数

    Args:
        cv_img: OpenCV画像 (BGR)
        text: 描画するテキスト
        position: (x, y) 座標
        font_size: フォントサイズ
        text_color: テキスト色 (RGB) - デフォルト白

    Returns:
        テキストが描画されたOpenCV画像
    """
    # OpenCV (BGR) → Pillow (RGB)
    cv_rgb = cv2.cvtColor(cv_img, cv2.COLOR_BGR2RGB)
    pil_img = Image.fromarray(cv_rgb)
    draw = ImageDraw.Draw(pil_img)

    try:
        font = ImageFont.truetype(font_path, font_size)
    except:
        font = ImageFont.load_default()

    # テキストを描画（背景なし）
    draw.text(position, text, font=font, fill=text_color)

    # Pillow (RGB) → OpenCV (BGR)
    return cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)

# ---------- 記号設定 ----------------------------------------------------------
def brake_distance(df, idx, amax=6.0):
    row = df.loc[df["frame"] == idx, "vtti.speed_gps"]
    v   = 0.0 if row.empty else row.values[0]/3.6
    return (v**2)/(2*amax) if v>0 else 1.0

intent_buffer = defaultdict(list)

# フレーム開始前に初期化（ループの外）
llava_line1 = "LLaVA Description:"
llava_line2 = ""
llava_done = False  # 一度だけ実行（必要ならFalseに戻す）
llava_status_map = {}
risk_zone_counter = defaultdict(int)  # メインループ外で一度だけ


# ---------- メインループ -------------------------------------------------------
frame_idx, trajectories, track_history = 0, [], {}
start_time = time.time()
print("▶ 統合処理を開始します...")
try:
    while cap.isOpened():
        risky_found = False  # ←★ ループの先頭で毎フレーム初期化
        ok, frame = cap.read()
        if not ok:
            break

        global_frame_idx = offset + frame_idx

        # === 1) 停止距離ポリゴン ==========================================
        stop_m  = brake_distance(df_speed, global_frame_idx)
        bev_h   = int(10*scale_px_per_m)
        bev_img = cv2.warpPerspective(frame, M, (360, bev_h))

        mask = np.ones((bev_h,360), np.uint8)*255
        cv2.fillPoly(mask,[np.array([[0,bev_h],[90,bev_h],[0,bev_h*0.3]],np.int32)],0)
        cv2.fillPoly(mask,[np.array([[360,bev_h],[270,bev_h],[360,bev_h*0.3]],np.int32)],0)
        gray   = cv2.cvtColor(cv2.bitwise_and(bev_img, bev_img, mask=mask), cv2.COLOR_BGR2GRAY)
        edges  = cv2.Canny(cv2.GaussianBlur(gray,(5,5),0), 15, 30)
        binary = (edges>0).astype(np.uint8)

        hist = np.sum(binary[binary.shape[0]//2:,:], axis=0)
        mid  = hist.shape[0]//2
        l_base = np.argmax(hist[:mid]); r_base = np.argmax(hist[mid:])+mid

        sliding_vis = np.dstack([binary*255]*3)
        n_win, margin, minpix = 30, 40, 30
        win_h = binary.shape[0]//n_win
        nz_y, nz_x = binary.nonzero()
        lx_cur, rx_cur = l_base, r_base
        left_pts, right_pts = [], []
        for win in range(n_win):
            wy_low, wy_high = binary.shape[0]-(win+1)*win_h, binary.shape[0]-win*win_h
            win_l = ((nz_y>=wy_low)&(nz_y<wy_high)&(nz_x>=lx_cur-margin)&(nz_x<lx_cur+margin)).nonzero()[0]
            win_r = ((nz_y>=wy_low)&(nz_y<wy_high)&(nz_x>=rx_cur-margin)&(nz_x<rx_cur+margin)).nonzero()[0]
            if len(win_l)>minpix: lx_cur=int(np.mean(nz_x[win_l]))
            if len(win_r)>minpix: rx_cur=int(np.mean(nz_x[win_r]))
            cy=(wy_low+wy_high)//2
            left_pts.append((lx_cur,cy)); right_pts.append((rx_cur,cy))

        if left_pts and right_pts:
            l_coef=np.polyfit([p[1] for p in left_pts],[p[0] for p in left_pts],2)
            r_coef=np.polyfit([p[1] for p in right_pts],[p[0] for p in right_pts],2)
            stop_px=min(int(stop_m*scale_px_per_m), bev_h)
            y_bot,y_top=bev_h-1, bev_h-stop_px
            def poly_x(y,c): return int(np.polyval(c,y))
            poly_bev=np.array([[l_base,y_bot],[poly_x(y_top,l_coef),y_top],
                               [poly_x(y_top,r_coef),y_top],[r_base,y_bot]],np.float32)
            poly_pts_img=cv2.perspectiveTransform(poly_bev[None], Minv)[0]
        else:
            poly_pts_img=np.array([[0,0]])

        # === 2) 車両検出 & ByteTrack =====================================
        det_res = det_model(frame, verbose=False, conf=0.4)[0]

        dets = []
        cls_list = []
        intent_list = []

        for b in det_res.boxes:
            x1, y1, x2, y2 = b.xyxy[0].tolist()
            conf = float(b.conf[0])
            cls_id = int(b.cls[0])  # 0=brake, 1=go, 2=left, 3=right

            # YOLOから直接intentを取る
            intent = class_names[cls_id]

            dets.append([x1, y1, x2, y2, conf])
            cls_list.append(cls_id)
            intent_list.append(intent)
        if dets:
            tracks = tracker.update(np.array(dets, np.float32), [h, w], [h, w])
        else:
            tracks = []  # 検出なしなら空リストにする

        # --- 車両ごと -----------------------------------------------------
        padding, min_w, min_h = 5, 5, 10
        for i, t in enumerate(tracks):
            tid, x, y, bw, bh = t.track_id, *map(int, t.tlwh)
            x0 = max(0, x - padding)
            y0 = max(0, y - padding)
            x1 = min(x + bw + padding, w - 1)
            y1 = min(y + bh + padding, h - 1)
            bw_c, bh_c = x1 - x0, y1 - y0
            cx = x0 + bw_c // 2
            cy = y1

            # === intentマッチング（まず安全に決める） ===
            if i < len(intent_list):
                matched_intent = intent_list[i]
            else:
                matched_intent = "go"  # デフォルト

            # === 小さすぎるBBoxはスキップ ===
            if bw_c < min_w or bh_c < min_h:
                # ただし、意図履歴だけは更新しておくと安定する
                intent_buffer[tid].append(matched_intent)
                if len(intent_buffer[tid]) > 5:
                    intent_buffer[tid].pop(0)
                continue

            # intent履歴を更新（matched_intentを使用）
            intent_buffer[tid].append(matched_intent)
            if len(intent_buffer[tid]) > 5:
                intent_buffer[tid].pop(0)

            # 直近5フレームのmajority voteで安定化
            final_intent = max(set(intent_buffer[tid]), key=intent_buffer[tid].count)

            # intentそのまま（brake / go / left / right）を描画
            frame = draw_text_clean(frame, final_intent, (x0 + 5, y0 + 5),
                                    font_size=16, text_color=(0, 255, 255))

            # バウンディングボックス描画
            cv2.rectangle(frame, (x0, y0), (x1, y1), (255, 0, 0), 2)

            # トラッキング履歴を残す（線描画）
            track_history.setdefault(tid, []).append((cx, cy))
            for k in range(1, len(track_history[tid])):
                cv2.line(frame, track_history[tid][k - 1], track_history[tid][k], (0, 255, 255), 2)

            # Ego速度取得
            v_row = df_speed.loc[df_speed["frame"] == global_frame_idx, "vtti.speed_gps"]
            v_now = 0.0 if v_row.empty else v_row.values[0]

            # 軌跡データ記録（classはfinal_intentにする）
            trajectories.append({
                "frame": frame_idx,
                "id": tid,
                "x": cx,
                "y": cy,
                "area": bw_c * bh_c,
                "class": final_intent,  # ← ここ修正
                "speed": v_now,
                "risky": int(cv2.pointPolygonTest(poly_pts_img.astype(np.int32), (cx, cy), False) >= 0),
                "intent": final_intent  # ← intentもfinal_intentに統一
            })

            # === ポリゴン内判定 ===
            in_risk_zone = cv2.pointPolygonTest(poly_pts_img.astype(np.int32), (cx, cy), False) >= 0
            risky_found |= in_risk_zone

            # === カウンタ処理（1フレーム以上連続で入ったら生成AI呼ぶ用） ===
            if in_risk_zone:
                risk_zone_counter[tid] += 1
            else:
                risk_zone_counter[tid] = 0  # リセット

            # === 1フレーム以上連続で入っている場合にのみ生成AIを呼び出す ===
            if risk_zone_counter[tid] >= 2:
                if tid not in llava_status_map or not llava_status_map[tid]["active"]:
                    # ここはもうフォルダがある前提
                    risk_frame_filename = f"frame_{global_frame_idx:06d}.jpg"
                    risk_frame_path = os.path.join(risk_frame_dir, risk_frame_filename)
                    cv2.imwrite(risk_frame_path, frame)

                    # === リスクフレーム保存パス ===
                    risk_frame_filename = f"frame_{global_frame_idx:06d}.jpg"
                    risk_frame_path = os.path.join(risk_frame_dir, risk_frame_filename)
                    cv2.imwrite(risk_frame_path, frame)

                    driving_facts = {
                        "frame_idx": global_frame_idx,
                        "tid": tid,
                        "label": final_intent,
                        "v_now": v_now,
                        "intent": final_intent,
                        "cx": cx,
                        "cy": cy,
                        "area": bw_c * bh_c
                    }

                    result = assess_risk_from_image(risk_frame_path, driving_facts)

                    detected_vehicle_list = [
                        {
                            "id": tr["id"],
                            "type": tr["class"],
                            "intent": tr["intent"],
                            "in_risk_zone": (tr["risky"] == 1),
                            "bbox_area": tr["area"]
                        }
                        for tr in trajectories if tr["frame"] == frame_idx
                    ]

                    co_caption = (
                        f"The ego vehicle is driving at {vtti_speed:.1f} km/h "
                        f"with longitudinal acceleration {vtti_accel:.2f} m/s². "
                        f"Detected intent: {final_intent}. "
                        f"Risky zone: YES. "
                        f"AI analysis: {result.get('reason', '')}"
                    )

                    json_record = {
                        "frame_id": int(global_frame_idx),
                        "image_path": risk_frame_path,
                        "ego_vehicle": {
                            "speed_kmh": float(vtti_speed),
                            "accel_mps2": float(vtti_accel),
                            "brake_pressed": (final_intent == "brake"),
                            "left_blinker": (final_intent == "left"),
                            "right_blinker": (final_intent == "right"),
                            "lane_change": (final_intent in ["left", "right"]),
                            "risk_zone": True
                        },
                        "detected_vehicles": {
                            "count": len(detected_vehicle_list),
                            "vehicles": detected_vehicle_list
                        },
                        "risk_assessment": {
                            "ai_risk_score": result.get("risk_probability", None),
                            "ai_lane_change": result.get("lane_change_detected", "unknown"),
                            "suggested_maneuver": result.get("suggested_maneuver", "unknown"),
                            "weather": result.get("weather", "?"),
                            "road_condition": result.get("road_condition", "?"),
                            "reason": result.get("reason", "")
                        },
                        "caption": co_caption
                    }

                    json_records.append(json_record)

                    print("📘 === LLaVA リスク評価スクリプション ===")
                    print(f"🆔 Vehicle ID: {tid}")
                    print(f"📸 Frame: {frame_idx}")
                    print(f"📊 Risk Score: {result.get('risk_probability')}")
                    print(f"🚗 Lane Change: {result.get('lane_change_detected')}")
                    print(f"⚠️ Suggested Maneuver: {result.get('suggested_maneuver')}")
                    print(f"💬 Reason: {result.get('reason')}")
                    print("==========================================")

                    desc_text = result.get("reason", "")
                    llava_status_map[tid] = {
                        "active": True,
                        "line1": "LLaVA Description:",
                        "line2": result.get("reason", ""),
                        "weather": result.get("weather", "?"),
                        "road_condition": result.get("road_condition", "?"),
                        "fog": "1" if "fog" in result.get("weather", "").lower() else "0",
                        "rain": "1" if "rain" in result.get("weather", "").lower() else "0"
                    }
            else:
                if tid in llava_status_map:
                    llava_status_map[tid]["active"] = False  # リスク範囲外に出たらリセット

        # === 左上：SHRP2 情報 - きれいに左上に配置 ================================
        v_row = df_speed.loc[df_speed["frame"] == global_frame_idx]
        vtti_speed = v_row["vtti.speed_gps"].values[0] if not v_row.empty else 0.0
        vtti_accel = v_row["vtti.accel_x"].values[0] if not v_row.empty else 0.0

        info_lines = [
            f"Frame.No: {global_frame_idx}",
            f"FPS: {fps:.3f}",
            f"Ego Speed: {vtti_speed:.1f} km/h",
            f"Accel X: {vtti_accel:.2f} m/s²"
        ]

        # 左上情報をRobotoMonoで描画（フォントサイズ12pxに変更）
        for i, txt in enumerate(info_lines):
            frame = draw_text_clean(frame, txt, (5, 5 + i * 18),
                                   font_size=12, text_color=(255,255,255))

        # === 右上：ターゲット情報（シンプル表示） ===================================
        # 対象選定：リスク対象がいればそれを、なければ最大bbox
        risky_tracks = [tr for tr in trajectories if tr["frame"] == frame_idx and tr["risky"] == 1]
        if risky_tracks:
            target = risky_tracks[0]
        else:
            current_tracks = [tr for tr in trajectories if tr["frame"] == frame_idx]
            target = max(current_tracks, key=lambda t: t["area"]) if current_tracks else None

        if target:
            tid = target["id"]
            intent = target["intent"]

            # 右上に表示するシンプルな情報
            target_info_lines = [
                f"Target ID: {tid}",
                f"Current Action: {intent}",
                f"#Vehicles: {len([tr for tr in trajectories if tr['frame'] == frame_idx])}",
                f"Risk Distance: {int(stop_m)}m"
            ]

            # 右上に描画
            for i, txt in enumerate(target_info_lines):
                frame = draw_text_clean(frame, txt, (w - 170, 5 + i * 16),
                                        font_size=11, text_color=(255, 255, 255))

        # === 3) ポリゴン & 表示 ===========================================
        overlay=frame.copy()
        poly_col=(0,0,255) if risky_found else (0,255,0)
        if len(poly_pts_img)==4:
            cv2.fillPoly(overlay,[poly_pts_img.astype(np.int32)],poly_col)
        cv2.addWeighted(overlay,0.3,frame,0.7,0,frame)

        # === 4)レイアウト & 出力 ====================================
        canny_vis=cv2.cvtColor(binary*255,cv2.COLOR_GRAY2BGR)
        canny_vis=cv2.resize(canny_vis,(360,240))
        sliding_vis=cv2.resize(sliding_vis,(360,240))

        # === 毎フレーム：LLaVA出力用マスク + 描画（RobotoMono使用） =====================
        # 代表的な tid を選ぶ（active なもの）
        representative = next((tid for tid, info in llava_status_map.items() if info.get("active", False)), None)

        if representative is not None:
            info = llava_status_map[representative]
            weather = info.get("weather", "?")
            fog = info.get("fog", "0")
            rain = info.get("rain", "0")
            road = info.get("road_condition", "?")

            # line1: weather 情報（短縮版）
            llava_line1 = f"Weather: {weather}, Fog: {'Yes' if fog=='1' else 'No'}, Rain: {'Yes' if rain=='1' else 'No'}, Road: {road}"


            # line2: reason（長い場合は大幅短縮）
            reason_text = info.get("line2", "")

            # テキストを3行に分割する処理
            max_chars_per_line = 85  # 1行に収めたい最大文字数

            if len(reason_text) > max_chars_per_line * 3:
                llava_line2 = reason_text[:max_chars_per_line]
                llava_line3 = reason_text[max_chars_per_line:max_chars_per_line * 2]
                llava_line4 = reason_text[max_chars_per_line * 2:max_chars_per_line * 3] + "..."
            elif len(reason_text) > max_chars_per_line * 2:
                llava_line2 = reason_text[:max_chars_per_line]
                llava_line3 = reason_text[max_chars_per_line:max_chars_per_line * 2]
                llava_line4 = reason_text[max_chars_per_line * 2:]
            elif len(reason_text) > max_chars_per_line:
                llava_line2 = reason_text[:max_chars_per_line]
                llava_line3 = reason_text[max_chars_per_line:]
                llava_line4 = ""
            else:
                llava_line2 = reason_text
                llava_line3 = ""
                llava_line4 = ""
        else:
            llava_line1 = "LLaVA:"
            llava_line2 = ""
            llava_line3 = ""
            llava_line4 = ""  # ←★ここを忘れずに！

        # === LLaVA表示帯（llava_vis）を生成し、frameにstackする ======================
        band_h = 60
        llava_vis = np.zeros((band_h, 360, 3), dtype=np.uint8)

        font_size = 7
        line_height = 14  # 行間を適度に空ける
        y_base = 6

        # 各行の描画
        llava_vis = draw_text_clean(llava_vis, llava_line1, (5, y_base + 0 * line_height),
                                    font_size=font_size, text_color=(255, 255, 255))

        if llava_line2:
            llava_vis = draw_text_clean(llava_vis, llava_line2, (5, y_base + 1 * line_height),
                                        font_size=font_size, text_color=(255, 255, 255))
        if llava_line3:
            llava_vis = draw_text_clean(llava_vis, llava_line3, (5, y_base + 2 * line_height),
                                        font_size=font_size, text_color=(255, 255, 255))
        if llava_line4:
            llava_vis = draw_text_clean(llava_vis, llava_line4, (5, y_base + 3 * line_height),
                                        font_size=font_size, text_color=(255, 255, 255))

        resized_frame = cv2.resize(frame, (360, 240))
        combined = np.vstack([resized_frame, llava_vis])

        # === 🚨 重要：動画出力（ここでcombinedを書き出す！）===
        out.write(combined)

        # === 進捗ログ (10フレームごと) ===================================
        if frame_idx % 10 == 0:
            elapsed = time.time() - start_time
            pct = (frame_idx + 1) / total_frames * 100
            print(f"[{frame_idx:6d}/{total_frames}] {pct:5.1f}% "
                  f"det:{len(dets):2d} trk:{len(track_history):2d} "
                  f"elapsed:{elapsed:6.1f}s")

        frame_idx += 1

except KeyboardInterrupt:
    print("⏹️ 中断されました。現在の進捗まで保存します...")

except Exception as e:
    print(f"❌ エラーが発生しました: {e}")

finally:
    print("🔄 クリーンアップ中...")

    # 動画キャプチャのリリース
    try:
        cap.release()
    except:
        pass

    # 動画出力のリリース（最重要！）
    try:
        out.release()
        print("✅ 動画ファイルを保存しました")
    except:
        print("⚠️ 動画保存でエラーが発生しました")

    # CSVファイルの保存
    try:
        if trajectories:
            pd.DataFrame(trajectories).to_csv(output_csv_path, index=False)
            print("✅ CSVファイルを保存しました")
    except:
        print("⚠️ CSV保存でエラーが発生しました")


    # === JSON 保存 ===
    output_json_path = os.path.join(base_dir, f"scene_{scene_no}_covla.json")

    if json_records:
        with open(output_json_path, "w", encoding="utf-8") as f:
            json.dump(json_records, f, ensure_ascii=False, indent=2)
        print(f"✅ CoVLA 風 JSON を保存しました → {output_json_path}")

    # 処理完了メッセージ
    total_elapsed = time.time() - start_time
    print(f"\n📊 処理時間: {total_elapsed:.1f}秒")
    print(f"🎬 処理フレーム数: {frame_idx:,}")
    print(f"🎥 出力動画: {output_video_path}")
    print(f"📄 出力CSV: {output_csv_path}")
    print("🏁 完了")