# train_traj_seq2seq_px_rel.py  (Best params fixed: Trial 15 + paper artifacts)
import os, math, argparse, random, json, csv
import numpy as np, pandas as pd, torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import matplotlib.pyplot as plt

# ====== Config (px 相対座標用) ======
DEF_TRAIN_CSV = r"C:\Users\s1280\Desktop\trajectory_data\traj_dataset_p30_f45_s5_px_with_ego_speed_train.csv"
DEF_VAL_CSV   = r"C:\Users\s1280\Desktop\trajectory_data\traj_dataset_p30_f45_s5_px_with_ego_speed_val.csv"

H_PAST, H_FUT = 30, 45
INPUT_SIZE    = 5      # [x(px), y(px), vx(px/s), vy(px/s), speed(0..1)]
OUTPUT_SIZE   = 2      # future (x(px), y(px))

# ====== ★ Fixed to Optuna Best (Trial 15) ======
HIDDEN        = 192
N_LAYERS      = 2
DROPOUT       = 0.12687839643883927
BATCH_SIZE    = 128
EPOCHS        = 25
LR            = 0.0011343582910511758
WEIGHT_DECAY  = 2.571461359876629e-06
TF_START      = 0.626337163630613
TF_END        = 0.08255813457084756
SMOOTHL1_BETA = 0.03074052147039913
CLIP_NORM     = 0.8237843140154353
SEED          = 42
CKPT_DIR      = "./checkpoints_traj_px_best15"

# ====== Utils ======
def set_seed(s=SEED):
    random.seed(s); np.random.seed(s); torch.manual_seed(s); torch.cuda.manual_seed_all(s)

def teacher_forcing_ratio(epoch, total, tf_start=TF_START, tf_end=TF_END):
    alpha = max(0.0, min(1.0, 1 - epoch/(total-1 + 1e-9)))
    return tf_end + (tf_start - tf_end) * alpha

# ====== Dataset ======
class TrajDataset(Dataset):
    def __init__(self, csv_path, h_past=H_PAST, h_fut=H_FUT):
        self.h_past, self.h_fut = h_past, h_fut
        df = pd.read_csv(csv_path)

        past_x = [f"past_x{i+1}"  for i in range(h_past)]
        past_y = [f"past_y{i+1}"  for i in range(h_past)]
        past_vx= [f"past_vx{i+1}" for i in range(h_past)]
        past_vy= [f"past_vy{i+1}" for i in range(h_past)]
        past_sp= [f"past_speed{i+1}" for i in range(h_past)]
        fut_x  = [f"fut_x{i+1}"     for i in range(h_fut)]
        fut_y  = [f"fut_y{i+1}"     for i in range(h_fut)]

        need_cols = past_x + past_y + past_vx + past_vy + past_sp + fut_x + fut_y
        missing = [c for c in need_cols if c not in df.columns]
        if missing:
            raise ValueError(f"CSVに必要列がありません: {missing[:10]} … (h_past={h_past}, h_fut={h_fut})")

        px  = df[past_x].to_numpy(np.float32)   # px（相対）
        py  = df[past_y].to_numpy(np.float32)   # px（相対）
        pvx = df[past_vx].to_numpy(np.float32)  # px/s
        pvy = df[past_vy].to_numpy(np.float32)  # px/s
        psp = df[past_sp].to_numpy(np.float32)  # 0..1（自車速度のみ）
        fx  = df[fut_x].to_numpy(np.float32)    # px（相対）
        fy  = df[fut_y].to_numpy(np.float32)    # px（相対）

        self.X = np.stack([px, py, pvx, pvy, psp], axis=-1)   # (N, T_past, 5)
        self.Y = np.stack([fx, fy], axis=-1)                  # (N, T_fut, 2)

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
        _, h = self.rnn(x)   # h: (layers, B, hidden)
        return h

class Decoder(nn.Module):
    def __init__(self, hidden=HIDDEN, out_size=OUTPUT_SIZE, n_layers=N_LAYERS, dropout=DROPOUT):
        super().__init__()
        self.rnn = nn.GRU(out_size, hidden, num_layers=n_layers, batch_first=True,
                          dropout=dropout if n_layers>1 else 0.0)
        self.proj = nn.Linear(hidden, out_size)

    def forward(self, h0, tgt=None, steps=H_FUT, tf_ratio=0.0):
        B = h0.shape[1]
        dec_in = torch.zeros(B, 1, OUTPUT_SIZE, device=h0.device)  # 初期はゼロ（相対px）
        outs, h = [], h0
        for t in range(steps):
            o, h = self.rnn(dec_in, h)      # (B,1,H), (layers,B,H)
            y = self.proj(o)                # (B,1,2)  相対px
            outs.append(y)
            if (tgt is not None) and (random.random() < tf_ratio):
                dec_in = tgt[:, t:t+1, :]   # 教師強制
            else:
                dec_in = y.detach()         # 自己回帰
        return torch.cat(outs, dim=1)       # (B,steps,2)

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

# ====== Metrics (px) ======
def ade_fde_px(pred, gt):
    # pred, gt: (B,T,2) in px（相対）→ L2(px)
    diff = pred - gt
    dist = torch.sqrt(diff[...,0]**2 + diff[...,1]**2)  # (B,T)
    ade = dist.mean()
    fde = dist[:, -1].mean()
    return ade, fde

# ====== Paper Artifacts ======
def save_paper_artifacts(paper_dir, history, best_epoch, best_val_ade, best_val_fde, cfg):
    os.makedirs(paper_dir, exist_ok=True)

    # CSV: metrics_per_epoch.csv
    csv_path = os.path.join(paper_dir, "metrics_per_epoch.csv")
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["epoch","train_loss","train_ADE_px","train_FDE_px","val_loss","val_ADE_px","val_FDE_px","lr","tf_ratio"])
        for e, row in enumerate(history, 1):
            w.writerow([e] + [f"{v:.8f}" for v in row])

    # 図: training_curves.(png|pdf)（loss=左軸、ADE/FDE=右軸）
    epochs = list(range(1, len(history)+1))
    tr_loss   = [h[0] for h in history]
    tr_ade    = [h[1] for h in history]
    tr_fde    = [h[2] for h in history]
    va_loss   = [h[3] for h in history]
    va_ade    = [h[4] for h in history]
    va_fde    = [h[5] for h in history]

    fig, ax1 = plt.subplots(figsize=(5.2, 3.2), dpi=200)
    ax2 = ax1.twinx()
    l1, = ax1.plot(epochs, tr_loss, linestyle="-",  label="Train Loss")
    l2, = ax1.plot(epochs, va_loss, linestyle="--", label="Val Loss")
    l3, = ax2.plot(epochs, tr_ade,  linestyle="-.", label="Train ADE(px)")
    l4, = ax2.plot(epochs, va_ade,  linestyle=":",  label="Val ADE(px)")
    l5, = ax2.plot(epochs, tr_fde,  linestyle="-.", label="Train FDE(px)")
    l6, = ax2.plot(epochs, va_fde,  linestyle=":",  label="Val FDE(px)")

    ax1.set_xlabel("Epoch"); ax1.set_ylabel("Loss"); ax2.set_ylabel("px")
    ax1.grid(True, linewidth=0.3, alpha=0.6)
    lines = [l1,l2,l3,l4,l5,l6]; labels=[ln.get_label() for ln in lines]
    ax1.legend(lines, labels, loc="upper right", fontsize=8, frameon=False)
    if best_epoch is not None:
        ax1.axvline(best_epoch, linestyle=":", linewidth=1.0, alpha=0.8)
    fig.tight_layout()
    fig.savefig(os.path.join(paper_dir, "training_curves.png"))
    fig.savefig(os.path.join(paper_dir, "training_curves.pdf"))
    plt.close(fig)

    # 図: 学習率 & TF 比
    lr      = [h[6] for h in history]
    tf_ratio= [h[7] for h in history]
    fig2, ax1b = plt.subplots(figsize=(5.2, 2.4), dpi=200)
    ax2b = ax1b.twinx()
    l_lr, = ax1b.plot(epochs, lr, linestyle="-",  label="LR")
    l_tf, = ax2b.plot(epochs, tf_ratio, linestyle="--", label="TF Ratio")
    ax1b.set_xlabel("Epoch"); ax1b.set_ylabel("LR"); ax2b.set_ylabel("TF Ratio")
    ax1b.grid(True, linewidth=0.3, alpha=0.6)
    ax1b.legend([l_lr,l_tf], ["LR","TF Ratio"], loc="upper right", fontsize=8, frameon=False)
    fig2.tight_layout()
    fig2.savefig(os.path.join(paper_dir, "lr_tf_schedule.png"))
    plt.close(fig2)

    # JSON: best_summary.json
    best_json = {
        "best_epoch": best_epoch,
        "best_val_ADE_px": float(best_val_ade) if best_val_ade is not None else None,
        "best_val_FDE_px": float(best_val_fde) if best_val_fde is not None else None,
        "config": cfg,
    }
    with open(os.path.join(paper_dir, "best_summary.json"), "w", encoding="utf-8") as f:
        json.dump(best_json, f, ensure_ascii=False, indent=2)

    # LaTeX: best_summary.tex
    tex = r"""\begin{table}[t]
\centering
\small
\begin{tabular}{lcc}
\hline
& ADE (px) & FDE (px) \\
\hline
Best (epoch %d) & %.2f & %.2f \\
\hline
\end{tabular}
\caption{Validation performance with the best configuration.}
\label{tab:best_traj_px}
\end{table}
""" % (best_epoch if best_epoch is not None else -1,
       best_val_ade if best_val_ade is not None else float("nan"),
       best_val_fde if best_val_fde is not None else float("nan"))
    with open(os.path.join(paper_dir, "best_summary.tex"), "w", encoding="utf-8") as f:
        f.write(tex)

    # README
    with open(os.path.join(paper_dir, "README.md"), "w", encoding="utf-8") as f:
        f.write(
            "# Paper Artifacts\n"
            "- metrics_per_epoch.csv: per-epoch metrics\n"
            "- training_curves.(png|pdf): compact curves for paper\n"
            "- lr_tf_schedule.png: learning-rate & teacher-forcing ratio\n"
            "- best_summary.json: best epoch and metrics\n"
            "- best_summary.tex: LaTeX table snippet\n"
        )

# ====== Train/Eval ======
def train_one_epoch(model, loader, opt, scaler, device, epoch, total_epochs, beta=SMOOTHL1_BETA,
                    steps=H_FUT, use_amp=True):
    model.train()
    total_loss = total_ade_px = total_fde_px = 0.0
    n = 0
    tf_ratio = teacher_forcing_ratio(epoch, total_epochs)

    for x, y in loader:
        x, y = x.to(device), y.to(device)
        opt.zero_grad(set_to_none=True)

        enabled_amp = (use_amp and device.type == 'cuda')
        with torch.autocast(device_type='cuda', dtype=torch.float16, enabled=enabled_amp):
            pred = model(x, tgt=y, tf_ratio=tf_ratio, steps=steps)
            loss = nn.functional.smooth_l1_loss(pred, y, beta=beta)

        scaler.scale(loss).backward()
        scaler.unscale_(opt)
        nn.utils.clip_grad_norm_(model.parameters(), CLIP_NORM)
        scaler.step(opt); scaler.update()

        ade_px, fde_px = ade_fde_px(pred.detach(), y)
        bs = x.size(0); n += bs
        total_loss   += loss.item()   * bs
        total_ade_px += ade_px.item() * bs
        total_fde_px += fde_px.item() * bs

    return (total_loss/n, total_ade_px/n, total_fde_px/n, tf_ratio)

@torch.no_grad()
def eval_one_epoch(model, loader, device, beta=SMOOTHL1_BETA, steps=H_FUT):
    model.eval()
    total_loss = total_ade_px = total_fde_px = 0.0
    n = 0
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        pred = model(x, tgt=None, tf_ratio=0.0, steps=steps)  # 教師強制なし
        loss = nn.functional.smooth_l1_loss(pred, y, beta=beta)
        ade_px, fde_px = ade_fde_px(pred, y)
        bs = x.size(0); n += bs
        total_loss   += loss.item()   * bs
        total_ade_px += ade_px.item() * bs
        total_fde_px += fde_px.item() * bs
    return (total_loss/n, total_ade_px/n, total_fde_px/n)

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
    parser.add_argument("--h_past", type=int, default=H_PAST)
    parser.add_argument("--h_fut",  type=int, default=H_FUT)
    parser.add_argument("--tf_start", type=float, default=TF_START)
    parser.add_argument("--tf_end",   type=float, default=TF_END)
    parser.add_argument("--beta",     type=float, default=SMOOTHL1_BETA)
    parser.add_argument("--save_dir", default=CKPT_DIR)
    args = parser.parse_args()

    os.makedirs(args.save_dir, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    train_ds = TrajDataset(args.train_csv, args.h_past, args.h_fut)
    val_ds   = TrajDataset(args.val_csv,   args.h_past, args.h_fut)
    train_ld = DataLoader(train_ds, batch_size=args.batch, shuffle=True,
                          num_workers=4, pin_memory=(device.type=='cuda'), drop_last=True)
    val_ld   = DataLoader(val_ds,   batch_size=args.batch, shuffle=False,
                          num_workers=4, pin_memory=(device.type=='cuda'))

    model = Seq2Seq(input_size=INPUT_SIZE, hidden=args.hidden, n_layers=args.layers,
                    dropout=args.dropout, out_size=OUTPUT_SIZE).to(device)

    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.wd)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.epochs, eta_min=1e-6)
    scaler = torch.amp.GradScaler('cuda', enabled=(device.type == 'cuda'))
    use_amp = True  # AMPはCUDA時のみ有効化される設定

    best_val_px = math.inf
    best_val_fpx = None
    best_epoch = None

    # 履歴 [train_loss, train_ADE, train_FDE, val_loss, val_ADE, val_FDE, lr, tf_ratio]
    history = []

    for epoch in range(args.epochs):
        tr_loss, tr_ade_px, tr_fde_px, tf_ratio = \
            train_one_epoch(model, train_ld, opt, scaler, device, epoch, args.epochs,
                            beta=args.beta, steps=args.h_fut, use_amp=use_amp)
        va_loss, va_ade_px, va_fde_px = \
            eval_one_epoch(model, val_ld, device, beta=args.beta, steps=args.h_fut)
        sched.step()

        lr_now = sched.get_last_lr()[0]
        history.append([tr_loss, tr_ade_px, tr_fde_px, va_loss, va_ade_px, va_fde_px, lr_now, tf_ratio])

        print(f"[{epoch+1:02d}/{args.epochs}] "
              f"train loss {tr_loss:.4f} | ADE(px) {tr_ade_px:.2f} | FDE(px) {tr_fde_px:.2f} | "
              f"val loss {va_loss:.4f} | ADE(px) {va_ade_px:.2f} | FDE(px) {va_fde_px:.2f} | "
              f"tf {tf_ratio:.2f} | lr {lr_now:.2e}")

        # ベスト(px)を保存
        if va_ade_px < best_val_px:
            best_val_px = va_ade_px
            best_val_fpx = va_fde_px
            best_epoch = epoch + 1
            ckpt = {
                "epoch": epoch+1,
                "state_dict": model.state_dict(),
                "optimizer": opt.state_dict(),
                "scheduler": sched.state_dict(),
                "val_ade_px": va_ade_px,
                "val_fde_px": va_fde_px,
                "cfg": vars(args),
            }
            path = os.path.join(args.save_dir, "best_ade_px.pt")
            torch.save(ckpt, path)
            print(f"  ✅ Saved best(px) to: {path} (ADE(px)={va_ade_px:.2f}, FDE(px)={va_fde_px:.2f})")

    # 最終
    final_path = os.path.join(args.save_dir, "last.pt")
    torch.save({"state_dict": model.state_dict(), "cfg": vars(args)}, final_path)
    print(f"  ✅ Saved last to: {final_path}")

    # ★ 論文用アーティファクトの出力
    paper_dir = os.path.join(args.save_dir, "paper_artifacts")
    save_paper_artifacts(
        paper_dir=paper_dir,
        history=history,
        best_epoch=best_epoch,
        best_val_ade=best_val_px,
        best_val_fde=best_val_fpx,
        cfg=vars(args),
    )
    print(f"  📁 Paper artifacts saved to: {paper_dir}")

if __name__ == "__main__":
    main()
