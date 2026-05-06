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
# ---------- パス設定 ----------------------------------------------------------
video_path        = r"C:\Users\s1280\Desktop\SHRP2rawdata\5\new\5_Combined_FaceSizeCentered_bottom_left.mp4"
csv_path          = r"C:\Users\s1280\Desktop\SHRP2rawdata\5\original\5_acc_speed_filled_complete.csv"
base_path         = r"C:\Users\s1280\PycharmProjects\yolo_classify_project\26x"
best_pt_path      = r"C:\Users\s1280\PycharmProjects\yolo_classify_project\runs\detect\train16\weights\best.pt"
font_path         = r"C:\Users\s1280\PycharmProjects\yolo_classify_project\26x\fonts\RobotoMono-Regular.ttf"

output_video_path = video_path.replace(".mp4", "_combined_speed_yolo_vx1.mp4")
output_csv_path   = video_path.replace(".mp4", "_trajectories_combined.csv")

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
det_model   = YOLO("yolov8x.pt").to("cuda")        # 車・トラック
light_model = YOLO(best_pt_path).to("cuda")        # ライト (left_on / right_on / top_on)

target_classes = [2, 7]        # car, truck
class_names    = {2: "car", 7: "truck"}

# ---------- 動画入出力 --------------------------------------------------------
cap = cv2.VideoCapture(video_path)
fps = int(cap.get(cv2.CAP_PROP_FPS))
w   = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
h   = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
# 修正後（元の動画サイズで出力）：
out = cv2.VideoWriter(output_video_path,
                      cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))

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
intent_symbol = {
    "左折": "<-", "右折": "->", "直進": "^",
    "ブレーキ": "B", "夜間走行": "ON"
}

def positional_filter(box_xyxy, labels, scores,
                      W=128, H=128,
                      top_y_max=0.35,
                      left_x_max=0.40,
                      right_x_min=0.60):
    kept = []
    for (x1, y1, x2, y2), lab, sc in zip(box_xyxy, labels, scores):
        cx, cy = (x1 + x2) / 2, (y1 + y2) / 2
        cx_n, cy_n = cx / W, cy / H
        if lab.startswith("top_") and cy_n <= top_y_max:
            kept.append((lab, sc))
        elif lab.startswith("left_") and cx_n <= left_x_max:
            kept.append((lab, sc))
        elif lab.startswith("right_") and cx_n >= right_x_min:
            kept.append((lab, sc))
    side_best = {}
    for lab, sc in kept:
        side = "_".join(lab.split("_")[:2])
        if side not in side_best or sc > side_best[side][1]:
            side_best[side] = (lab, sc)
    return [v[0] for v in side_best.values()]

def classify_intent(lbls):
    s = set(lbls)
    left_on = "left_on" in s
    right_on = "right_on" in s
    top_on = "top_on" in s
    left_off = "left_off" in s
    right_off = "right_off" in s
    top_off = "top_off" in s
    if top_on:
        return "brake"
    if left_on and right_on and top_off and not top_on:
        return "night"
    if left_on and right_off and not right_on:
        return "left"
    if right_on and left_off and not left_on:
        return "right"
    return "straight"

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


# ---------- メインループ -------------------------------------------------------
frame_idx, trajectories, track_history = 0, [], {}
start_time = time.time()
print("▶ 統合処理を開始します...")

while cap.isOpened():
    ok, frame = cap.read()
    if not ok:
        break

    # === 1) 停止距離ポリゴン ==========================================
    stop_m  = brake_distance(df_speed, frame_idx)
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
    det_res=det_model(frame, verbose=False)[0]
    dets, cls_list=[],[]
    for b in det_res.boxes:
        if int(b.cls[0]) in target_classes:
            x1,y1,x2,y2=b.xyxy[0].tolist()
            dets.append([x1,y1,x2,y2,b.conf[0].item()])
            cls_list.append(int(b.cls[0]))
    tracks=[]; risky_found=False
    if dets:
        tracks=tracker.update(np.array(dets,np.float32), [h,w],[h,w])



    # --- 車両ごと -----------------------------------------------------
    padding,min_w,min_h=5,5,10
    for i,t in enumerate(tracks):
        tid,x,y,bw,bh=t.track_id,*map(int,t.tlwh)
        x0=max(0,x-padding); y0=max(0,y-padding)
        x1=min(x+bw+padding,w-1); y1=min(y+bh+padding,h-1)
        bw_c, bh_c = x1-x0, y1-y0
        cx=x0+bw_c//2; cy=y1
        if bw_c < min_w or bh_c < min_h:
            l_lbl = []
        else:
            roi = frame[y0:y1, x0:x1]
            roi128 = cv2.resize(roi, (128, 128), interpolation=cv2.INTER_LINEAR)
            l_res = light_model(roi128, verbose=False, conf=0.4)[0]
            if l_res.boxes:
                boxes = l_res.boxes.xyxy.cpu().numpy()
                labels = [l_res.names[int(c)] for c in l_res.boxes.cls]
                scores = l_res.boxes.conf.cpu().numpy()
                l_lbl = positional_filter(boxes, labels, scores)
            else:
                l_lbl = []

        # Intent バッファと分類
        intent_buffer[tid].append(l_lbl)
        if len(intent_buffer[tid]) > 5:
            intent_buffer[tid].pop(0)
        flat_lbls = [lab for sub in intent_buffer[tid] for lab in sub]
        intent = classify_intent(flat_lbls)

        # RobotoMono フォントで車両情報を描画
        symbol = intent_symbol.get(intent, "↑")
        label = class_names.get(cls_list[i], "unk")

        # バウンディングボックス描画
        cv2.rectangle(frame, (x0, y0), (x1, y1), (255, 0, 0), 2)

        # 意図シンボルを描画（矢印マークを確実に表示）
        frame = draw_text_clean(frame, symbol, (x0+5, y0+5),
                               font_size=16, text_color=(0,255,255))

        # 車両ID とクラス名を描画（フォントサイズ小さめ）
        vehicle_info = f"{tid} {label}"
        frame = draw_text_clean(frame, vehicle_info, (x0+5, y0+25),
                               font_size=12, text_color=(255,255,255))

        # トラッキング履歴の描画
        track_history.setdefault(tid,[]).append((cx,cy))
        for k in range(1,len(track_history[tid])):
            cv2.line(frame, track_history[tid][k-1], track_history[tid][k], (0,255,255),2)

        # 軌跡データ記録
        v_row=df_speed.loc[df_speed["frame"]==frame_idx,"vtti.speed_gps"]
        v_now=0.0 if v_row.empty else v_row.values[0]
        trajectories.append({"frame":frame_idx,"id":tid,"x":cx,"y":cy,
                             "area":bw_c*bh_c,"class":label,"speed":v_now,
                             "risky":int(cv2.pointPolygonTest(poly_pts_img.astype(np.int32),(cx,cy),False)>=0),
                             "intent":intent})

        # === ポリゴン内判定 ===
        in_risk_zone = cv2.pointPolygonTest(poly_pts_img.astype(np.int32), (cx, cy), False) >= 0
        risky_found |= in_risk_zone

        if in_risk_zone:
            if tid not in llava_status_map or not llava_status_map[tid]["active"]:
                risk_frame_path = f"risk_frame_{frame_idx}.jpg"
                cv2.imwrite(risk_frame_path, frame)

                from risk_assessment_api import assess_risk_from_image

                driving_facts = {
                    "frame_idx": frame_idx,
                    "tid": tid,
                    "label": label,
                    "v_now": v_now,
                    "intent": intent,
                    "cx": cx,
                    "cy": cy,
                    "area": bw_c * bh_c
                }

                result = assess_risk_from_image(risk_frame_path, driving_facts)

                print("📘 === LLaVA リスク評価スクリプション ===")
                print(f"🆔 Vehicle ID: {tid}")
                print(f"📸 Frame: {frame_idx}")
                print(f"📊 Risk Score: {result.get('risk_probability')}")
                print(f"🚗 Lane Change: {result.get('lane_change_detected')}")
                print(f"⚠️ Action: {result.get('recommended_action')}")
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
    v_row = df_speed.loc[df_speed["frame"] == frame_idx]
    vtti_speed = v_row["vtti.speed_gps"].values[0] if not v_row.empty else 0.0
    vtti_accel = v_row["vtti.accel_x"].values[0] if not v_row.empty else 0.0

    info_lines = [
        f"Frame.No: {frame_idx}",
        f"FPS: {fps:.3f}",
        f"Ego Speed: {vtti_speed:.1f} km/h",
        f"Accel X: {vtti_accel:.2f} m/s²"
    ]

    # 左上情報をRobotoMonoで描画（フォントサイズ12pxに変更）
    for i, txt in enumerate(info_lines):
        frame = draw_text_clean(frame, txt, (5, 5 + i * 18),
                               font_size=12, text_color=(255,255,255))

    # === 右上：ターゲット情報 =================================================
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
        l_lbls = intent_buffer[tid][-1] if intent_buffer[tid] else []

        def status(label_on, label_off, label_type):
            if label_on in l_lbls:
                return "1"
            elif label_off in l_lbls:
                return "0"
            else:
                return "-"

        top_status = status("top_on", "top_off", "Top")
        right_status = status("right_on", "right_off", "Right")
        left_status = status("left_on", "left_off", "Left")

        # 各ラベルの出現数
        label_counts = {k: 0 for k in ["top_on", "top_off", "right_on", "right_off", "left_on", "left_off"]}
        for lbl in intent_buffer[tid]:
            for l in lbl:
                if l in label_counts:
                    label_counts[l] += 1

        target_info_lines = [
            f"Target ID: {tid}",
            f"Top:{top_status} Right:{right_status} Left:{left_status}",
            f"Current Action: {intent}",
            f"Brake:{label_counts['top_on']} Right:{label_counts['right_on']} Left:{label_counts['left_on']}",
            f"#Front: {len([tr for tr in trajectories if tr['frame'] == frame_idx])}",
            f"#Rear: 0",
            f"Risk Distance: {int(stop_m)}m"
        ]

        # 右上情報をRobotoMonoで描画（右端により近く配置）
        for i, txt in enumerate(target_info_lines):
            frame = draw_text_clean(frame, txt, (w - 170, 5 + i * 16),
                                   font_size=11, text_color=(255,255,255))

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
    band_h = 25
    frame_h, frame_w = frame.shape[:2]

    # 下帯を黒く塗りつぶす（高さそのまま）
    cv2.rectangle(frame, (0, frame_h - band_h), (frame_w, frame_h), (0, 0, 0), -1)

    # 代表的な tid を選ぶ（active なもの）
    representative = next((tid for tid, info in llava_status_map.items() if info.get("active", False)), None)

    if representative is not None:
        info = llava_status_map[representative]
        weather = info.get("weather", "?")
        fog = info.get("fog", "0")
        rain = info.get("rain", "0")
        road = info.get("road_condition", "?")

        # line1: weather 情報
        llava_line1 = f"LLaVA: weather={weather} fog={fog} rain={rain} road={road}"
        # line2: reason
        llava_line2 = info.get("line2", "")
    else:
        llava_line1 = "LLaVA:"
        llava_line2 = ""

    # 帯の中に2行をギリギリで収めて描画（フォント小さめ）
    frame = draw_text_clean(frame, llava_line1, (5, frame_h - 14), font_size=5, text_color=(255, 255, 255))
    frame = draw_text_clean(frame, llava_line2, (5, frame_h - 4), font_size=5, text_color=(255, 255, 255))

    # 動画出力
    out.write(frame)

    # === 進捗ログ (10フレームごと) ===================================
    if frame_idx % 10 == 0:
        elapsed = time.time() - start_time
        pct = (frame_idx+1) / total_frames * 100
        print(f"[{frame_idx:6d}/{total_frames}] {pct:5.1f}% "
              f"det:{len(dets):2d} trk:{len(track_history):2d} "
              f"elapsed:{elapsed:6.1f}s")

    frame_idx += 1

# ---------- 後処理 ------------------------------------------------------------
cap.release(); out.release()
pd.DataFrame(trajectories).to_csv(output_csv_path,index=False)
print(f"✅ 統合完了： 動画 → {output_video_path}\n               CSV  → {output_csv_path}")