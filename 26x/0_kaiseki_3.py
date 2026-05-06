# -*- coding: utf-8 -*-
import os, re, json, random
from datetime import datetime
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import matplotlib.pyplot as plt

# ====== パラメータ ======
CSV_PATH   = r"C:\Users\s1280\Desktop\trajectory_data\3_Combined_FaceSizeCentered_bottom_left_trajectories_combined5.csv"
H_PAST     = 20
H_FUT      = 45          # 15fps × 3s
SLIDE      = 5
VAL_SPLIT  = 0.2         # id単位で分割
SEED       = 42
BATCH_SIZE = 256
EPOCHS     = 80
LR         = 1e-3
WD         = 1e-4
HIDDEN     = 160
LAYERS     = 2
DROPOUT    = 0.1
CLIP_NORM  = 1.0
PATIENCE   = 12
# 時間重みと教師強制
TIME_W_END = 2.5
TF_START   = 0.4
TF_END     = 0.15

SAVE_ROOT = os.path.join(os.path.dirname(CSV_PATH),
                         "outputs_traj_cvres_" + datetime.now().strftime("%Y%m%d_%H%M%S"))
os.makedirs(SAVE_ROOT, exist_ok=True)

# ====== 再現性 ======
random.seed(SEED); np.random.seed(SEED); torch.manual_seed(SEED)
if torch.cuda.is_available(): torch.cuda.manual_seed_all(SEED)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ====== 連番チェック ======
def consecutive(frames: np.ndarray) -> bool:
    if len(frames) < 2: return False
    return np.all(np.diff(frames.astype(np.int64)) == 1)

# ====== ウィンドウ生成（オンザフライ） ======
class WindowSet(Dataset):
    def __init__(self, df, ids, h_past=20, h_fut=45, slide=5, build_stats=False):
        self.h_past, self.h_fut, self.slide = h_past, h_fut, slide
        self.samples = []  # (tid, start_idx_in_group)
        self.groups = {}
        for tid, g in df[df["id"].isin(ids)].groupby("id"):
            g = g.sort_values("frame").reset_index(drop=True)
            self.groups[tid] = g
            T = len(g)
            if T < h_past + h_fut: continue
            for s in range(0, T - (h_past + h_fut) + 1, slide):
                past_f = g["frame"].values[s:s+h_past]
                fut_f  = g["frame"].values[s+h_past:s+h_past+h_fut]
                if not (consecutive(past_f) and consecutive(fut_f) and past_f[-1]+1 == fut_f[0]):
                    continue
                self.samples.append((tid, s))

        # 正規化統計（学習セットで作る）
        self.mean_xy = np.zeros(2, np.float32)
        self.std_xy  = np.ones(2, np.float32)
        self.std_vel = np.ones(2, np.float32)
        if build_stats:
            xs, vs = [], []
            for tid, s in self.samples:
                g = self.groups[tid]
                x = g[["x","y"]].values[s:s+h_past].astype(np.float32)          # [T,2]
                x = x - x[-1]  # 原点合わせ（相対化）
                v = np.zeros_like(x); v[1:] = x[1:] - x[:-1]                    # [T,2]
                xs.append(x); vs.append(v)
            X = np.concatenate(xs, axis=0); V = np.concatenate(vs, axis=0)
            self.mean_xy = X.mean(axis=0)
            self.std_xy  = X.std(axis=0); self.std_xy[self.std_xy==0] = 1.0
            self.std_vel = V.std(axis=0); self.std_vel[self.std_vel==0] = 1.0

    def __len__(self): return len(self.samples)

    def __getitem__(self, i):
        tid, s = self.samples[i]
        g = self.groups[tid]
        # 位置（画面座標）→ 相対化
        # まず「生」の past / fut を取って
        past_raw = g[["x", "y"]].values[s:s + self.h_past].astype(np.float32)
        fut_raw = g[["x", "y"]].values[s + self.h_past:s + self.h_past + self.h_fut].astype(np.float32)

        # 原点を先に保存
        origin = past_raw[-1].copy()

        # それから両方を同じ原点で相対化
        past = past_raw - origin
        fut = fut_raw - origin
        # 同じ原点基準

        # 速度
        vel = np.zeros_like(past); vel[1:] = past[1:] - past[:-1]        # [T,2]

        # 等速Δ（将来）
        v_last = past[-1] - past[-2]                                     # [2]
        cv_delta = np.tile(v_last, (self.h_fut,1)).astype(np.float32)    # [F,2]

        # 真の将来Δ
        fut_delta = np.zeros_like(fut)
        fut_delta[0] = fut[0]
        fut_delta[1:] = fut[1:] - fut[:-1]

        # 正規化
        past_n = (past - self.mean_xy) / self.std_xy
        vel_n  = vel / self.std_vel
        fut_n  = (fut  - self.mean_xy) / self.std_xy
        fut_delta_n = fut_delta / self.std_xy
        cv_delta_n  = cv_delta / self.std_xy

        # 入力は [位置,速度]
        past_in = np.concatenate([past_n, vel_n], axis=-1)               # [T,4]

        return (torch.from_numpy(past_in),
                torch.from_numpy(fut_n),
                torch.from_numpy(fut_delta_n),
                torch.from_numpy(cv_delta_n))

# ====== データ読み込み & 分割 ======
df = pd.read_csv(CSV_PATH, usecols=["frame","id","x","y"])
df = df.dropna().sort_values(["id","frame"])
all_ids = df["id"].unique().tolist()
random.shuffle(all_ids)
n_val = max(1, int(len(all_ids)*VAL_SPLIT))
val_ids = set(all_ids[:n_val]); train_ids = set(all_ids[n_val:])

# 学習セットで統計を作る
train_set = WindowSet(df, train_ids, H_PAST, H_FUT, SLIDE, build_stats=True)
val_set   = WindowSet(df, val_ids,   H_PAST, H_FUT, SLIDE, build_stats=False)
# 学習統計を検証にも共有
val_set.mean_xy, val_set.std_xy, val_set.std_vel = train_set.mean_xy, train_set.std_xy, train_set.std_vel

# 保存
with open(os.path.join(SAVE_ROOT, "scaler.json"), "w") as f:
    json.dump({"mean_xy": train_set.mean_xy.tolist(),
               "std_xy":  train_set.std_xy.tolist(),
               "std_vel": train_set.std_vel.tolist()}, f, indent=2)

print(f"train samples: {len(train_set)} | val samples: {len(val_set)}")

train_loader = DataLoader(train_set, batch_size=BATCH_SIZE, shuffle=True,  drop_last=True)
val_loader   = DataLoader(val_set,   batch_size=BATCH_SIZE, shuffle=False, drop_last=False)

# ====== モデル（入力4次元: 位置2+速度2、出力はΔ残差）======
class GRUResidual(nn.Module):
    def __init__(self, in_dim=4, hid=HIDDEN, layers=LAYERS, out_steps=H_FUT, dropout=DROPOUT):
        super().__init__()
        self.enc = nn.GRU(in_dim, hid, num_layers=layers, batch_first=True,
                          dropout=(dropout if layers>1 else 0.0))
        self.dec = nn.GRU(2, hid, num_layers=layers, batch_first=True,
                          dropout=(dropout if layers>1 else 0.0))
        self.head = nn.Linear(hid, 2)
        self.out_steps = out_steps

    def forward(self, past_in, teacher_delta=None, teacher_prob=0.0):
        B = past_in.size(0)
        _, h = self.enc(past_in)
        y_t = torch.zeros(B, 1, 2, device=past_in.device)  # 直前残差Δ
        outs = []
        for t in range(self.out_steps):
            o, h = self.dec(y_t, h)
            d_res = self.head(o)  # 残差Δ
            outs.append(d_res)
            use_tf = (self.training and teacher_delta is not None and
                      torch.rand(1, device=past_in.device).item() < teacher_prob)
            y_t = teacher_delta[:, t:t+1, :] if use_tf else d_res
        return torch.cat(outs, dim=1)  # [B,F,2]

model = GRUResidual().to(device)
optim = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WD)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optim, T_max=EPOCHS, eta_min=LR*0.1)
crit = nn.SmoothL1Loss(reduction="none")

time_w = torch.linspace(1.0, TIME_W_END, steps=H_FUT, device=device).view(1, H_FUT, 1)
y_weight = 1.5  # y方向の感度を少し上げる

def timeweighted_loss(pred_delta, true_delta):
    base = crit(pred_delta, true_delta)       # [B,F,2]
    base[...,1] = base[...,1] * y_weight
    return (base * time_w).mean()

def cumsum_coord(delta):  # Δ列→位置（累積）
    return torch.cumsum(delta, dim=1)

# ====== ループ ======
best_val, no_imp = 1e9, 0
hist = {"train":[], "val":[], "ADE":[],"FDE":[],"CV_ADE":[],"CV_FDE":[]}

for epoch in range(1, EPOCHS+1):
    t = (epoch-1)/max(1,EPOCHS-1)
    tf_prob = TF_START*(1-t) + TF_END*t

    # --- train
    model.train(); tr = 0.0
    for past_in, fut_n, fut_d_n, cv_d_n in train_loader:
        past_in, fut_n, fut_d_n, cv_d_n = past_in.to(device), fut_n.to(device), fut_d_n.to(device), cv_d_n.to(device)
        # 残差を予測
        pred_res = model(past_in, teacher_delta=fut_d_n - cv_d_n, teacher_prob=tf_prob)
        pred_d   = cv_d_n + pred_res
        loss = timeweighted_loss(pred_d, fut_d_n)

        optim.zero_grad(); loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), CLIP_NORM)
        optim.step()
        tr += loss.item()
    tr /= max(1,len(train_loader))

    # --- val
    model.eval(); va = 0.0
    ade_list, fde_list, ade_cv_list, fde_cv_list = [], [], [], []
    with torch.no_grad():
        for past_in, fut_n, fut_d_n, cv_d_n in val_loader:
            past_in, fut_n, fut_d_n, cv_d_n = past_in.to(device), fut_n.to(device), fut_d_n.to(device), cv_d_n.to(device)
            pred_res = model(past_in, teacher_delta=None, teacher_prob=0.0)
            pred_d   = cv_d_n + pred_res
            va += timeweighted_loss(pred_d, fut_d_n).item()

            pred_n = cumsum_coord(pred_d)       # [B,F,2]
            fut_px  = (fut_n.cpu().numpy()  * train_set.std_xy + train_set.mean_xy)
            pred_px = (pred_n.cpu().numpy() * train_set.std_xy + train_set.mean_xy)
            diff = pred_px - fut_px
            dist = np.linalg.norm(diff, axis=-1)
            ade_list.append(dist.mean()); fde_list.append(dist[:,-1].mean())

            # CV baseline
            cv_n = cumsum_coord(cv_d_n)         # [B,F,2]
            cv_px = (cv_n.cpu().numpy() * train_set.std_xy + train_set.mean_xy)
            diff_cv = cv_px - fut_px
            dist_cv = np.linalg.norm(diff_cv, axis=-1)
            ade_cv_list.append(dist_cv.mean()); fde_cv_list.append(dist_cv[:,-1].mean())

    va /= max(1,len(val_loader))
    ADE, FDE   = float(np.mean(ade_list)), float(np.mean(fde_list))
    CV_ADE, CV_FDE = float(np.mean(ade_cv_list)), float(np.mean(fde_cv_list))
    hist["train"].append(tr); hist["val"].append(va)
    hist["ADE"].append(ADE); hist["FDE"].append(FDE)
    hist["CV_ADE"].append(CV_ADE); hist["CV_FDE"].append(CV_FDE)

    print(f"Epoch {epoch:03d}/{EPOCHS} | train {tr:.4f} | val {va:.4f} | "
          f"ADE {ADE:.2f}px | FDE {FDE:.2f}px | CV_ADE {CV_ADE:.2f}px | CV_FDE {CV_FDE:.2f}px | tf={tf_prob:.2f}")

    # save / early stop
    if va < best_val - 1e-4:
        best_val, no_imp = va, 0
        torch.save(model.state_dict(), os.path.join(SAVE_ROOT, "model_best.pt"))
    else:
        no_imp += 1
        if no_imp >= PATIENCE:
            print(f"⏹️ Early stop at epoch {epoch}")
            break
    scheduler.step()

# 保存
torch.save(model.state_dict(), os.path.join(SAVE_ROOT, "model_last.pt"))
pd.DataFrame(hist).to_csv(os.path.join(SAVE_ROOT, "metrics.csv"), index=False)

# ====== 可視化（バリデーション1サンプル）======
try:
    state = torch.load(os.path.join(SAVE_ROOT, "model_best.pt"), map_location=device, weights_only=True)
except TypeError:
    state = torch.load(os.path.join(SAVE_ROOT, "model_best.pt"), map_location=device)
model.load_state_dict(state); model.eval()

if len(val_set) > 0:
    import random
    idx = random.randrange(len(val_set))
    past_in, fut_n, fut_d_n, cv_d_n = val_set[idx]
    with torch.no_grad():
        pred_res = model(past_in.unsqueeze(0).to(device))
        pred_d   = cv_d_n.unsqueeze(0).to(device) + pred_res
        pred_n   = torch.cumsum(pred_d, dim=1).squeeze(0).cpu().numpy()  # [F,2]

    past_n = past_in.numpy()[:, :2]                          # 入力の位置成分
    past_px = past_n * train_set.std_xy + train_set.mean_xy
    fut_px  = fut_n.numpy() * train_set.std_xy + train_set.mean_xy
    pred_px = pred_n       * train_set.std_xy + train_set.mean_xy

    # xだけ
    T, F = H_PAST, H_FUT
    t_p, t_f = np.arange(T), np.arange(T, T+F)
    plt.figure(figsize=(9,5))
    plt.plot(t_p, past_px[:,0], label="past x")
    plt.plot(t_f, fut_px[:,0],  label="true future x")
    plt.plot(t_f, pred_px[:,0], "--", label="pred future x")
    plt.xlabel("frame"); plt.ylabel("x (px)")
    plt.title("Validation sample (x only)")
    plt.grid(True); plt.legend(); plt.tight_layout()
    plt.savefig(os.path.join(SAVE_ROOT, "viz_future_x.png"), dpi=150); plt.close()

    # 2D
    plt.figure(figsize=(5,5))
    plt.plot(past_px[:,0], past_px[:,1], "-o", ms=2, label="past")
    plt.plot(fut_px[:,0],  fut_px[:,1],  "-o", ms=2, label="true")
    plt.plot(pred_px[:,0], pred_px[:,1], "--o", ms=2, label="pred")
    plt.gca().invert_yaxis(); plt.axis("equal"); plt.grid(True); plt.legend()
    plt.tight_layout(); plt.savefig(os.path.join(SAVE_ROOT, "viz_future_xy.png"), dpi=150); plt.close()

print(f"✅ Done. outputs -> {SAVE_ROOT}")
