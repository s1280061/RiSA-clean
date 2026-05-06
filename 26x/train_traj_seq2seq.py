# train_traj_seq2seq.py
import os, math, argparse, random, numpy as np, pandas as pd, torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

# ====== Config (デフォ) ======
DEF_TRAIN_CSV = r"C:\Users\s1280\Desktop\trajectory_data\traj_dataset_p30_f45_s5_norm360x240_with_speed_train.csv"
DEF_VAL_CSV   = r"C:\Users\s1280\Desktop\trajectory_data\traj_dataset_p30_f45_s5_norm360x240_with_speed_val.csv"
H_PAST, H_FUT = 30, 45
INPUT_SIZE    = 5      # [x, y, vx, vy, speed]
OUTPUT_SIZE   = 2      # future (x,y)
HIDDEN        = 128
N_LAYERS      = 2
DROPOUT       = 0.1
BATCH_SIZE    = 256
EPOCHS        = 50
LR            = 1e-3
WEIGHT_DECAY  = 1e-4
TF_START      = 0.8    # 初期教師強制率
TF_END        = 0.1    # 最終教師強制率
CLIP_NORM     = 1.0
SEED          = 42
CKPT_DIR      = "./checkpoints_traj"

# ====== Utils ======
def set_seed(s=SEED):
    random.seed(s); np.random.seed(s); torch.manual_seed(s); torch.cuda.manual_seed_all(s)

def teacher_forcing_ratio(epoch, total):
    # 線形に TF を減衰
    alpha = max(0.0, min(1.0, 1 - epoch/(total-1 + 1e-9)))
    return TF_END + (TF_START - TF_END) * alpha

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
            raise ValueError(f"CSVに必要列がありません: {missing[:10]} ...")

        # ndarray 化（float32）
        px  = df[past_x].to_numpy(np.float32)
        py  = df[past_y].to_numpy(np.float32)
        pvx = df[past_vx].to_numpy(np.float32)
        pvy = df[past_vy].to_numpy(np.float32)
        psp = df[past_sp].to_numpy(np.float32)
        fx  = df[fut_x].to_numpy(np.float32)
        fy  = df[fut_y].to_numpy(np.float32)

        # 入力系列: (N, T_past, 5)
        self.X = np.stack([px, py, pvx, pvy, psp], axis=-1)  # (N, T, 5)
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
        self.rnn = nn.GRU(input_size, hidden, num_layers=n_layers, batch_first=True, dropout=dropout if n_layers>1 else 0.0)
    def forward(self, x):
        # x: (B, T_past, input_size)
        out, h = self.rnn(x)   # h: (layers, B, hidden)
        return h

class Decoder(nn.Module):
    def __init__(self, hidden=HIDDEN, out_size=OUTPUT_SIZE, n_layers=N_LAYERS, dropout=DROPOUT):
        super().__init__()
        self.rnn = nn.GRU(out_size, hidden, num_layers=n_layers, batch_first=True, dropout=dropout if n_layers>1 else 0.0)
        self.proj = nn.Linear(hidden, out_size)

    def forward(self, h0, tgt=None, steps=H_FUT, tf_ratio=0.0):
        # h0: (layers, B, hidden)
        B = h0.shape[1]
        # 初期入力はゼロ (未来は原点相対の想定なので0始点でもOK)
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
    def __init__(self):
        super().__init__()
        self.encoder = Encoder()
        self.decoder = Decoder()
    def forward(self, x, tgt=None, tf_ratio=0.0):
        h = self.encoder(x)
        y = self.decoder(h, tgt=tgt, steps=x.new_tensor([H_FUT]).long().item(), tf_ratio=tf_ratio)
        return y

# ====== Metrics ======
def ade_fde(pred, gt):
    # pred, gt: (B,T,2)
    err = torch.linalg.norm(pred - gt, dim=-1)   # (B,T)
    ade = err.mean()
    fde = err[:, -1].mean()
    return ade, fde

# ====== Train ======
def train_one_epoch(model, loader, opt, scaler, device, epoch, total_epochs):
    model.train()
    total_loss, total_ade, total_fde, n = 0.0, 0.0, 0.0, 0
    tf_ratio = teacher_forcing_ratio(epoch, total_epochs)
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        opt.zero_grad(set_to_none=True)
        with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=True):
            pred = model(x, tgt=y, tf_ratio=tf_ratio)
            loss = nn.functional.smooth_l1_loss(pred, y, beta=0.02)
        scaler.scale(loss).backward()
        nn.utils.clip_grad_norm_(model.parameters(), CLIP_NORM)
        scaler.step(opt); scaler.update()

        ade, fde = ade_fde(pred.detach(), y)
        bs = x.size(0); n += bs
        total_loss += loss.item() * bs
        total_ade  += ade.item()  * bs
        total_fde  += fde.item()  * bs
    return total_loss/n, total_ade/n, total_fde/n, tf_ratio

@torch.no_grad()
def eval_one_epoch(model, loader, device):
    model.eval()
    total_loss, total_ade, total_fde, n = 0.0, 0.0, 0.0, 0
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        pred = model(x, tgt=None, tf_ratio=0.0)  # 評価は教師強制なし
        loss = nn.functional.smooth_l1_loss(pred, y, beta=0.02)
        ade, fde = ade_fde(pred, y)
        bs = x.size(0); n += bs
        total_loss += loss.item() * bs
        total_ade  += ade.item()  * bs
        total_fde  += fde.item()  * bs
    return total_loss/n, total_ade/n, total_fde/n

# ====== Main ======
def main():
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
    parser.add_argument("--save_dir", default=CKPT_DIR)
    args = parser.parse_args()

    os.makedirs(args.save_dir, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    train_ds = TrajDataset(args.train_csv, H_PAST, H_FUT)
    val_ds   = TrajDataset(args.val_csv,   H_PAST, H_FUT)
    train_ld = DataLoader(train_ds, batch_size=args.batch, shuffle=True,  num_workers=4, pin_memory=True, drop_last=True)
    val_ld   = DataLoader(val_ds,   batch_size=args.batch, shuffle=False, num_workers=4, pin_memory=True)

    model = Seq2Seq()
    # 動的にハイパラ反映（hidden/layers/dropout）
    model.encoder.rnn.hidden_size = args.hidden
    model.decoder.rnn.hidden_size = args.hidden
    # 再構築（簡潔のため）
    model.encoder = Encoder(input_size=INPUT_SIZE, hidden=args.hidden, n_layers=args.layers, dropout=args.dropout)
    model.decoder = Decoder(hidden=args.hidden, out_size=OUTPUT_SIZE, n_layers=args.layers, dropout=args.dropout)
    model.to(device)

    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.wd)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.epochs)
    scaler = torch.cuda.amp.GradScaler(enabled=(device.type == "cuda"))

    best_val = math.inf
    for epoch in range(args.epochs):
        tr_loss, tr_ade, tr_fde, tf_ratio = train_one_epoch(model, train_ld, opt, scaler, device, epoch, args.epochs)
        va_loss, va_ade, va_fde = eval_one_epoch(model, val_ld, device)
        sched.step()

        msg = (f"[{epoch+1:02d}/{args.epochs}] "
               f"train loss {tr_loss:.4f} | ADE {tr_ade:.4f} | FDE {tr_fde:.4f} | "
               f"val loss {va_loss:.4f} | ADE {va_ade:.4f} | FDE {va_fde:.4f} | "
               f"tf {tf_ratio:.2f} | lr {sched.get_last_lr()[0]:.2e}")
        print(msg)

        # ベスト保存 (ADE を主指標)
        if va_ade < best_val:
            best_val = va_ade
            ckpt = {
                "epoch": epoch+1,
                "state_dict": model.state_dict(),
                "optimizer": opt.state_dict(),
                "scheduler": sched.state_dict(),
                "val_ade": va_ade,
                "val_fde": va_fde,
                "cfg": vars(args),
            }
            path = os.path.join(args.save_dir, "best_ade.pt")
            torch.save(ckpt, path)
            print(f"  ✅ Saved best to: {path} (ADE={va_ade:.4f}, FDE={va_fde:.4f})")

    # 最終保存
    final_path = os.path.join(args.save_dir, "last.pt")
    torch.save({"state_dict": model.state_dict()}, final_path)
    print(f"  ✅ Saved last to: {final_path}")

if __name__ == "__main__":
    main()
