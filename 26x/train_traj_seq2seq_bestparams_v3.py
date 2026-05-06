import os, math, argparse, random, numpy as np, pandas as pd, torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

# ====== Config (defaults are from best Optuna trial) ======
# ここだけ変更
DEF_TRAIN_CSV = r"C:\Users\s1280\Desktop\trajectory_data\traj_dataset_p30_f45_s5_norm360x240_with_ego_speed_train.csv"
DEF_VAL_CSV   = r"C:\Users\s1280\Desktop\trajectory_data\traj_dataset_p30_f45_s5_norm360x240_with_ego_speed_val.csv"

# NOTE: CSV 名に合わせて既定は p30/f45 に揃える
H_PAST, H_FUT = 30, 45
INPUT_SIZE    = 5      # [x, y, vx, vy, speed] (正規化済み)
OUTPUT_SIZE   = 2      # future (x,y)
# === Best trial params ===
HIDDEN        = 256
N_LAYERS      = 3
DROPOUT       = 0.2053413756886844
BATCH_SIZE    = 128
EPOCHS        = 25
LR            = 0.002448232533731172
WEIGHT_DECAY  = 1.6907970766924026e-05
TF_START      = 0.9179821668608998    # 初期教師強制率
TF_END        = 0.1718641022530241    # 最終教師強制率
SMOOTHL1_BETA = 0.07106471801489683
CLIP_NORM     = 1.0
SEED          = 42
CKPT_DIR      = "./checkpoints_traj"
# 画像サイズ（px指標のため）
DEF_IMG_W, DEF_IMG_H = 360.0, 240.0
W_PX, H_PX = DEF_IMG_W, DEF_IMG_H


# ====== Utils ======
def set_seed(s=SEED):
    random.seed(s); np.random.seed(s); torch.manual_seed(s); torch.cuda.manual_seed_all(s)

def teacher_forcing_ratio(epoch, total, tf_start=TF_START, tf_end=TF_END):
    # 線形に TF を減衰（終盤で下がりきる）
    alpha = max(0.0, min(1.0, 1 - epoch/(total-1 + 1e-9)))
    return tf_end + (tf_start - tf_end) * alpha

# ====== Dataset ======
class TrajDataset(Dataset):
    def __init__(self, csv_path, h_past=H_PAST, h_fut=H_FUT):
        self.h_past, self.h_fut = h_past, h_fut
        df = pd.read_csv(csv_path)

        # 列名をプログラム的に決め打ち
        past_x = [f"past_x{i+1}" for i in range(h_past)]
        past_y = [f"past_y{i+1}" for i in range(h_past)]
        past_vx= [f"past_vx{i+1}" for i in range(h_past)]
        past_vy= [f"past_vy{i+1}" for i in range(h_past)]
        past_sp= [f"past_speed{i+1}" for i in range(h_past)]

        fut_x  = [f"fut_x{i+1}"  for i in range(h_fut)]
        fut_y  = [f"fut_y{i+1}"  for i in range(h_fut)]

        need_cols = past_x + past_y + past_vx + past_vy + past_sp + fut_x + fut_y
        missing = [c for c in need_cols if c not in df.columns]
        if missing:
            raise ValueError(f"CSVに必要列がありません: {missing[:10]} … (h_past={h_past}, h_fut={h_fut})")

        # ndarray 化（float32）
        px  = df[past_x].to_numpy(np.float32)
        py  = df[past_y].to_numpy(np.float32)
        pvx = df[past_vx].to_numpy(np.float32)
        pvy = df[past_vy].to_numpy(np.float32)
        psp = df[past_sp].to_numpy(np.float32)
        fx  = df[fut_x].to_numpy(np.float32)
        fy  = df[fut_y].to_numpy(np.float32)

        # 入力系列: (N, T_past, 5)
        self.X = np.stack([px, py, pvx, pvy, psp], axis=-1)
        # 出力系列: (N, T_fut, 2)
        self.Y = np.stack([fx, fy], axis=-1)

        # 軽いサニティ（NaN/Inf除去）
        bad = ~np.isfinite(self.X).all(axis=(1,2)) | ~np.isfinite(self.Y).all(axis=(1,2))
        if bad.any():
            keep = ~bad
            self.X, self.Y = self.X[keep], self.Y[keep]

    def __len__(self): return len(self.X)
    def __getitem__(self, idx):
        x = torch.from_numpy(self.X[idx])  # (T_past, 5)
        y = torch.from_numpy(self.Y[idx])  # (T_fut, 2)
        return x, y

# ====== Model: GRU Encoder-Decoder ======
class Encoder(nn.Module):
    def __init__(self, input_size=INPUT_SIZE, hidden=HIDDEN, n_layers=N_LAYERS, dropout=DROPOUT):
        super().__init__()
        self.rnn = nn.GRU(input_size, hidden, num_layers=n_layers, batch_first=True,
                          dropout=dropout if n_layers>1 else 0.0)
    def forward(self, x):
        # x: (B, T_past, input_size)
        out, h = self.rnn(x)   # h: (layers, B, hidden)
        return h

class Decoder(nn.Module):
    def __init__(self, hidden=HIDDEN, out_size=OUTPUT_SIZE, n_layers=N_LAYERS, dropout=DROPOUT):
        super().__init__()
        self.rnn = nn.GRU(out_size, hidden, num_layers=n_layers, batch_first=True,
                          dropout=dropout if n_layers>1 else 0.0)
        self.proj = nn.Linear(hidden, out_size)

    def forward(self, h0, tgt=None, steps=H_FUT, tf_ratio=0.0):
        # h0: (layers, B, hidden)
        B = h0.shape[1]
        # 初期入力はゼロ（正規化座標; 将来は相対座標扱いでもOK）
        dec_in = torch.zeros(B, 1, OUTPUT_SIZE, device=h0.device)
        outs = []
        h = h0
        for t in range(steps):
            o, h = self.rnn(dec_in, h)              # (B,1,H), (layers,B,H)
            y = self.proj(o)                        # (B,1,2)
            outs.append(y)
            if (tgt is not None) and (random.random() < tf_ratio):
                dec_in = tgt[:, t:t+1, :]           # 教師強制
            else:
                dec_in = y.detach()                 # 自己回帰
        return torch.cat(outs, dim=1)               # (B,steps,2)

class Seq2Seq(nn.Module):
    def __init__(self, input_size=INPUT_SIZE, hidden=HIDDEN, n_layers=N_LAYERS, dropout=DROPOUT,
                 out_size=OUTPUT_SIZE):
        super().__init__()
        self.encoder = Encoder(input_size=input_size, hidden=hidden, n_layers=n_layers, dropout=dropout)
        self.decoder = Decoder(hidden=hidden, out_size=out_size, n_layers=n_layers, dropout=dropout)
    def forward(self, x, tgt=None, tf_ratio=0.0, steps=H_FUT):
        h = self.encoder(x)
        y = self.decoder(h, tgt=tgt, steps=steps, tf_ratio=tf_ratio)
        return y

# ====== Metrics ======
def ade_fde_norm(pred, gt):
    # pred, gt: (B,T,2) in normalized coords [0,1]
    err = torch.linalg.norm(pred - gt, dim=-1)   # (B,T)
    ade = err.mean()
    fde = err[:, -1].mean()
    return ade, fde

def ade_fde_pixels(pred, gt):
    # 正規化座標を px に戻してから距離（最新の W_PX/H_PX を使用）
    pred_px = pred.clone()
    gt_px   = gt.clone()
    pred_px[..., 0] *= W_PX; pred_px[..., 1] *= H_PX
    gt_px[...,   0] *= W_PX; gt_px[...,   1] *= H_PX
    diff = pred_px - gt_px
    dist = torch.sqrt(diff[...,0]**2 + diff[...,1]**2)
    ade = dist.mean()
    fde = dist[:, -1].mean()
    return ade, fde

# ====== Train/Eval ======
def train_one_epoch(model, loader, opt, scaler, device, epoch, total_epochs, beta=SMOOTHL1_BETA,
                    steps=H_FUT, use_amp=True):
    model.train()
    total_loss = total_ade_n = total_fde_n = total_ade_px = total_fde_px = 0.0
    n = 0
    tf_ratio = teacher_forcing_ratio(epoch, total_epochs)

    for x, y in loader:
        x, y = x.to(device), y.to(device)
        opt.zero_grad(set_to_none=True)
        # AMP: CUDA では fp16、CPU なら bfloat16
        dtype = torch.float16 if (use_amp and device.type == 'cuda') else torch.bfloat16
        with torch.autocast(device_type=device.type, dtype=dtype, enabled=use_amp):
            pred = model(x, tgt=y, tf_ratio=tf_ratio, steps=steps)
            loss = nn.functional.smooth_l1_loss(pred, y, beta=beta)
        # ★ 修正: unscale → クリップ → step
        scaler.scale(loss).backward()
        scaler.unscale_(opt)
        nn.utils.clip_grad_norm_(model.parameters(), CLIP_NORM)
        scaler.step(opt); scaler.update()

        # Metrics (norm + px)
        ade_n, fde_n = ade_fde_norm(pred.detach(), y)
        ade_px, fde_px = ade_fde_pixels(pred.detach(), y)
        bs = x.size(0); n += bs
        total_loss   += loss.item()   * bs
        total_ade_n  += ade_n.item()  * bs
        total_fde_n  += fde_n.item()  * bs
        total_ade_px += ade_px.item() * bs
        total_fde_px += fde_px.item() * bs

    return (total_loss/n, total_ade_n/n, total_fde_n/n, total_ade_px/n, total_fde_px/n, tf_ratio)

@torch.no_grad()
def eval_one_epoch(model, loader, device, beta=SMOOTHL1_BETA, steps=H_FUT):
    model.eval()
    total_loss = total_ade_n = total_fde_n = total_ade_px = total_fde_px = 0.0
    n = 0
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        pred = model(x, tgt=None, tf_ratio=0.0, steps=steps)  # 評価は教師強制なし
        loss = nn.functional.smooth_l1_loss(pred, y, beta=beta)
        ade_n, fde_n = ade_fde_norm(pred, y)
        ade_px, fde_px = ade_fde_pixels(pred, y)
        bs = x.size(0); n += bs
        total_loss   += loss.item()   * bs
        total_ade_n  += ade_n.item()  * bs
        total_fde_n  += fde_n.item()  * bs
        total_ade_px += ade_px.item() * bs
        total_fde_px += fde_px.item() * bs
    return (total_loss/n, total_ade_n/n, total_fde_n/n, total_ade_px/n, total_fde_px/n)

# ====== Main ======
def main():
    global W_PX, H_PX  # ← 最初に宣言
    set_seed()
    parser = argparse.ArgumentParser()
    parser.add_argument("--train_csv", default=DEF_TRAIN_CSV)
    parser.add_argument("--val_csv",   default=DEF_VAL_CSV)
    parser.add_argument("--epochs", type=int, default=EPOCHS)
    parser.add_argument("--batch",  type=int, default=BATCH_SIZE)
    parser.add_argument("--lr",     type=float, default=LR)
    parser.add_argument("--wd",     type=float, default=WEIGHT_DECAY)
    parser.add_argument("--hidden", type=int, default=HIDDEN)
    parser.add_argument("--layers", type=int, default=N_LAYERS)
    parser.add_argument("--dropout",type=float, default=DROPOUT)
    parser.add_argument("--h_past", type=int, default=H_PAST)
    parser.add_argument("--h_fut",  type=int, default=H_FUT)
    parser.add_argument("--tf_start", type=float, default=TF_START)
    parser.add_argument("--tf_end",   type=float, default=TF_END)
    parser.add_argument("--beta",     type=float, default=SMOOTHL1_BETA)
    parser.add_argument("--save_dir", default=CKPT_DIR)
    parser.add_argument("--img_w", type=float, default=DEF_IMG_W)
    parser.add_argument("--img_h", type=float, default=DEF_IMG_H)
    args = parser.parse_args()

    # パース後にグローバルへ反映
    W_PX = float(args.img_w)
    H_PX = float(args.img_h)

    os.makedirs(args.save_dir, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    train_ds = TrajDataset(args.train_csv, args.h_past, args.h_fut)
    val_ds   = TrajDataset(args.val_csv,   args.h_past, args.h_fut)
    train_ld = DataLoader(train_ds, batch_size=args.batch, shuffle=True,  num_workers=4, pin_memory=True, drop_last=True)
    val_ld   = DataLoader(val_ds,   batch_size=args.batch, shuffle=False, num_workers=4, pin_memory=True)

    model = Seq2Seq(input_size=INPUT_SIZE, hidden=args.hidden, n_layers=args.layers, dropout=args.dropout,
                    out_size=OUTPUT_SIZE).to(device)

    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.wd)
    # CosineAnnealing: 最終LRをゼロに貼り付けない
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.epochs, eta_min=1e-6)
    # AMP の新APIに対応
    scaler = torch.amp.GradScaler('cuda', enabled=(device.type == 'cuda'))
    use_amp = True  # CPUでもbfloat16で有効化（上でdtype切替）

    best_val_norm = math.inf
    best_val_px   = math.inf

    for epoch in range(args.epochs):
        tr_loss, tr_ade_n, tr_fde_n, tr_ade_px, tr_fde_px, tf_ratio = \
            train_one_epoch(model, train_ld, opt, scaler, device, epoch, args.epochs, beta=args.beta,
                            steps=args.h_fut, use_amp=use_amp)
        va_loss, va_ade_n, va_fde_n, va_ade_px, va_fde_px = \
            eval_one_epoch(model, val_ld, device, beta=args.beta, steps=args.h_fut)
        sched.step()

        msg = (f"[{epoch+1:02d}/{args.epochs}] "
               f"train loss {tr_loss:.4f} | ADE(n) {tr_ade_n:.4f} | FDE(n) {tr_fde_n:.4f} | "
               f"ADE(px) {tr_ade_px:.2f} | FDE(px) {tr_fde_px:.2f} | "
               f"val loss {va_loss:.4f} | ADE(n) {va_ade_n:.4f} | FDE(n) {va_fde_n:.4f} | "
               f"ADE(px) {va_ade_px:.2f} | FDE(px) {va_fde_px:.2f} | "
               f"tf {tf_ratio:.2f} | lr {sched.get_last_lr()[0]:.2e}")
        print(msg)

        # ベスト保存 (norm 指標)
        if va_ade_n < best_val_norm:
            best_val_norm = va_ade_n
            ckpt = {
                "epoch": epoch+1,
                "state_dict": model.state_dict(),
                "optimizer": opt.state_dict(),
                "scheduler": sched.state_dict(),
                "val_ade_norm": va_ade_n,
                "val_fde_norm": va_fde_n,
                "val_ade_px": va_ade_px,
                "val_fde_px": va_fde_px,
                "cfg": vars(args),
            }
            path = os.path.join(args.save_dir, "best_ade_norm.pt")
            torch.save(ckpt, path)
            print(f"  ✅ Saved best(norm) to: {path} (ADE(n)={va_ade_n:.4f}, FDE(n)={va_fde_n:.4f} | ADE(px)={va_ade_px:.2f}, FDE(px)={va_fde_px:.2f})")

        # pxベースのベストも別名で保存
        if va_ade_px < best_val_px:
            best_val_px = va_ade_px
            ckpt_px = {
                "epoch": epoch+1,
                "state_dict": model.state_dict(),
                "optimizer": opt.state_dict(),
                "scheduler": sched.state_dict(),
                "val_ade_norm": va_ade_n,
                "val_fde_norm": va_fde_n,
                "val_ade_px": va_ade_px,
                "val_fde_px": va_fde_px,
                "cfg": vars(args),
            }
            path_px = os.path.join(args.save_dir, "best_ade_px.pt")
            torch.save(ckpt_px, path_px)
            print(f"  ✅ Saved best(px)  to: {path_px} (ADE(px)={va_ade_px:.2f}, FDE(px)={va_fde_px:.2f})")

    # 最終保存
    final_path = os.path.join(args.save_dir, "last.pt")
    torch.save({"state_dict": model.state_dict()}, final_path)
    print(f"  ✅ Saved last to: {final_path}")

if __name__ == "__main__":
    main()
