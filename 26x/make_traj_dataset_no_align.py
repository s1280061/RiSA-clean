# make_traj_dataset_use_ego_speed_only.py
import os, glob
import pandas as pd
import numpy as np

# ===== 入出力 =====
INPUT_CSVS = [
    r"C:\Users\s1280\Desktop\traj_data\3_Combined_FaceSizeCentered_bottom_left_trajectories_combined.csv",
    r"C:\Users\s1280\Desktop\traj_data\4_Combined_FaceSizeCentered_bottom_left_trajectories_combined.csv",
    r"C:\Users\s1280\Desktop\traj_data\5_Combined_FaceSizeCentered_bottom_left_trajectories_combined.csv",
    r"C:\Users\s1280\Desktop\traj_data\6_Combined_FaceSizeCentered_bottom_left_trajectories_combined.csv",
]
OUTPUT_CSV = r"C:\Users\s1280\Desktop\trajectory_data\traj_dataset_p30_f45_s5_norm360x240_with_ego_speed.csv"

# ===== パラメータ =====
W, H = 360, 240
FPS = 15
H_PAST, H_FUT = 30, 45
SLIDE = 5
ALLOW_GAP = 1
MAX_SAMPLES_PER_TRACK = 200

# 速度の正規化（推論パイプラインと同じ 0–120km/h → 0–1）
SPEED_MIN, SPEED_MAX = 0.0, 120.0
def speed_to_feature(v_kmh: float) -> float:
    v = (v_kmh - SPEED_MIN) / (SPEED_MAX - SPEED_MIN)
    return float(0.0 if v < 0 else 1.0 if v > 1 else v)

def interp_window(fr, xs, ys, start, L, allow_gap=1):
    fwin = fr[start:start+L].astype(np.int64)
    xw   = xs[start:start+L].astype(np.float32).copy()
    yw   = ys[start:start+L].astype(np.float32).copy()
    if np.any(np.diff(fwin) > (1+allow_gap)):  # 大きく欠落はNG
        return None
    f0 = fwin[0]
    fexp = np.arange(f0, f0+L, dtype=np.int64)
    if not np.array_equal(fwin, fexp):         # 小欠落は補間
        xw = np.interp(fexp, fwin, xw)
        yw = np.interp(fexp, fwin, yw)
        fwin = fexp
    return fwin, xw, yw

def compute_vxy(past_xy, fps):
    """対象車の画面内速度ベクトル（そのまま特徴として使う）"""
    v = np.zeros_like(past_xy, dtype=np.float32)
    v[1:-1] = (past_xy[2:] - past_xy[:-2]) * (fps/2.0)
    v[0]    = (past_xy[1] - past_xy[0]) * fps
    v[-1]   = (past_xy[-1] - past_xy[-2]) * fps
    return v

# ====== 自車速度の読み込み設定（あなたの環境に合わせて直す） ======
EGO_SPEED_MAP = {
    # 左：INPUT_CSVS の basename、右：そのソースに対応する scene_* CSV 群
    "3_Combined_FaceSizeCentered_bottom_left_trajectories_combined.csv":
        r"C:\Users\s1280\Desktop\SHRP2rawdata\3\csv_divided\scene_*.csv",
    "4_Combined_FaceSizeCentered_bottom_left_trajectories_combined.csv":
        r"C:\Users\s1280\Desktop\SHRP2rawdata\4\csv_divided\scene_*.csv",
    "5_Combined_FaceSizeCentered_bottom_left_trajectories_combined.csv":
        r"C:\Users\s1280\Desktop\SHRP2rawdata\5\csv_divided\scene_*.csv",
    "6_Combined_FaceSizeCentered_bottom_left_trajectories_combined.csv":
        r"C:\Users\s1280\Desktop\SHRP2rawdata\6\csv_divided\scene_*.csv",
}

def load_ego_speed_series(pattern):
    """scene_*.csv 群から frame -> normalized ego speed の辞書を作る"""
    frames, speeds = [], []
    for p in sorted(glob.glob(pattern)):
        try:
            df = pd.read_csv(p, usecols=["frame", "vtti.speed_gps"])
        except Exception:
            df = pd.read_csv(p)
            if "frame" not in df.columns:
                raise ValueError(f"{p}: 'frame' 列がありません")
            sp_col = "vtti.speed_gps"
            if sp_col not in df.columns:
                cand = [c for c in df.columns if "speed" in c.lower()]
                if not cand:
                    raise ValueError(f"{p}: 自車速度列が見つかりません")
                sp_col = cand[0]
            df = df[["frame", sp_col]].rename(columns={sp_col: "vtti.speed_gps"})
        frames.append(df["frame"].to_numpy(np.int64))
        speeds.append(df["vtti.speed_gps"].to_numpy(np.float32))
    if not frames:
        print(f"⚠️ ego speed CSV が見つからない: {pattern}")
        return {}
    fr = np.concatenate(frames)
    sp = np.concatenate(speeds)
    order = np.argsort(fr)
    fr, sp = fr[order], sp[order]
    d = {}
    for f, v in zip(fr, sp):
        d[int(f)] = speed_to_feature(float(v))
    return d

# ===== 読み込み =====
dfs = []
for p in INPUT_CSVS:
    df = pd.read_csv(p)
    df["source"] = os.path.basename(p)
    dfs.append(df)
df = pd.concat(dfs, ignore_index=True)
df = (df.sort_values(["source","id","frame"])
        .drop_duplicates(subset=["source","id","frame"], keep="last"))

req = {"frame","id","x","y","source"}
miss = req - set(df.columns)
if miss:
    raise ValueError(f"必要カラムなし: {miss}")

# ===== 自車速度を source ごとに辞書化 =====
ego_speed_cache = {}
for src_name in df["source"].unique():
    pat = EGO_SPEED_MAP.get(src_name, None)
    ego_speed_cache[src_name] = load_ego_speed_series(pat) if pat else {}
    if not ego_speed_cache[src_name]:
        print(f"⚠️ {src_name}: 自車速度が見つからないため 0 で埋めます。")

# ===== サンプル生成 =====
cols_past = [f"past_{ax}{i+1}" for i in range(H_PAST) for ax in ("x","y")]
cols_v    = [f"past_v{ax}{i+1}" for i in range(H_PAST) for ax in ("x","y")]
cols_sp   = [f"past_speed{i+1}" for i in range(H_PAST)]  # ← ここが「自車速度」
cols_fut  = [f"fut_{ax}{i+1}"  for i in range(H_FUT)  for ax in ("x","y")]
cols = ["source","id","frame_end"] + cols_past + cols_v + cols_sp + cols_fut
samples = []
L = H_PAST + H_FUT

for (src, tid), g in df.groupby(["source","id"], sort=False):
    if len(g) < L:
        continue
    xs = (g["x"].to_numpy(np.float32) / W)
    ys = (g["y"].to_numpy(np.float32) / H)
    fr = g["frame"].to_numpy(np.int64)

    taken = 0
    for start in range(0, len(g) - L + 1, SLIDE):
        chk = interp_window(fr, xs, ys, start, L, ALLOW_GAP)
        if chk is None:
            continue
        fwin, xw, yw = chk

        past_xy = np.stack([xw[:H_PAST], yw[:H_PAST]], axis=-1)
        fut_xy  = np.stack([xw[H_PAST:H_PAST+H_FUT],
                            yw[H_PAST:H_PAST+H_FUT]], axis=-1)

        # 原点合わせ（過去末点を0,0）
        origin = past_xy[-1].copy()
        past_rel = past_xy - origin
        fut_rel  = fut_xy  - origin

        # 対象車の画面内速度ベクトル（従来どおり）
        v_xy = compute_vxy(past_rel, FPS)

        # ★ 自車速度（frame基準）を past_speed* に入れる
        es = ego_speed_cache.get(src, {})
        past_frames = fwin[:H_PAST]
        sp_ego = np.array([es.get(int(f), np.nan) for f in past_frames], dtype=np.float32)
        # 欠損は補間 + 埋め
        if np.any(~np.isfinite(sp_ego)):
            idx = np.arange(len(sp_ego))
            good = np.isfinite(sp_ego)
            if good.any():
                sp_ego[~good] = np.interp(idx[~good], idx[good], sp_ego[good])
            sp_ego[~np.isfinite(sp_ego)] = 0.0

        row = [src, int(tid), int(past_frames[-1])]
        row += past_rel.reshape(-1).tolist()
        row += v_xy.reshape(-1).tolist()
        row += sp_ego.tolist()          # ← ここが「自車速度（正規化）」だけ
        row += fut_rel.reshape(-1).tolist()
        samples.append(row)

        taken += 1
        if taken >= MAX_SAMPLES_PER_TRACK:
            break

out = pd.DataFrame(samples, columns=cols)
out.to_csv(OUTPUT_CSV, index=False)
print(f"✅ 保存: {OUTPUT_CSV}   サンプル数={len(out)}")

# ===== 80:20 の IDベース分割 =====
RATIO, SEED = 0.8, 42
rng = np.random.default_rng(SEED)

OUTPUT_TRAIN_CSV = r"C:\Users\s1280\Desktop\trajectory_data\traj_dataset_p30_f45_s5_norm360x240_with_ego_speed_train.csv"
OUTPUT_VAL_CSV   = r"C:\Users\s1280\Desktop\trajectory_data\traj_dataset_p30_f45_s5_norm360x240_with_ego_speed_val.csv"

train_parts, val_parts = [], []
for src, g in out.groupby("source", sort=False):
    ids = g["id"].unique()
    rng.shuffle(ids)
    n_train_ids = max(1, int(np.ceil(len(ids) * RATIO)))
    train_ids = set(ids[:n_train_ids])
    train_parts.append(g[g["id"].isin(train_ids)])
    val_parts.append(g[~g["id"].isin(train_ids)])

pd.concat(train_parts, ignore_index=True).to_csv(OUTPUT_TRAIN_CSV, index=False)
pd.concat(val_parts,   ignore_index=True).to_csv(OUTPUT_VAL_CSV,   index=False)
print(f"✅ Train保存: {OUTPUT_TRAIN_CSV}")
print(f"✅  Val 保存: {OUTPUT_VAL_CSV}")
