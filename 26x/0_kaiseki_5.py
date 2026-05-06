# train_traj_seq.py  ← 0_kaiseki_5.py にそのまま貼ってOK
# Residual Seq2Seq (20→45), 入力[20,5], 出力[45,2]（相対座標）
import argparse, random, json, math, os
from pathlib import Path
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import matplotlib.pyplot as plt

# ============== 可視化/書き出しユーティリティ ==============
def predict_loader(model, loader, device):
    model.eval(); preds = []
    with torch.no_grad():
        for x, _ in loader:
            x = x.to(device)
            p, _ = model(x, teach=None, tf_ratio=0.0)  # 自己回帰
            preds.append(p.cpu().numpy())
    return np.concatenate(preds, axis=0)

def save_predictions_csv(ds: Dataset, preds: np.ndarray, out_path: Path):
    df = ds.df.copy()
    px = preds[:,:,0]; py = preds[:,:,1]
    for i in range(FUT_STEPS):
        df[f"fut_pred_x{i+1}"] = px[:, i]
        df[f"fut_pred_y{i+1}"] = py[:, i]
    df.to_csv(out_path, index=False)

def per_step_rmse(preds: np.ndarray, gts: np.ndarray):
    diff = preds - gts
    return np.sqrt((diff**2).sum(axis=-1)).mean(axis=0)  # [45]

def viz_trajectories_grid(past: np.ndarray, gt: np.ndarray, pred: np.ndarray, out_path: Path, max_plots=12):
    N = min(len(past), max_plots)
    cols = 4; rows = int(np.ceil(N / cols))
    plt.figure(figsize=(3.2*cols, 3.2*rows))
    for i in range(N):
        ax = plt.subplot(rows, cols, i+1)
        ax.plot(past[i,:,0], past[i,:,1], '-o', lw=1, ms=2, label='past',  color='#1f77b4')
        ax.plot(gt[i,:,0],   gt[i,:,1],   '-o', lw=1, ms=2, label='gt',    color='#2ca02c')
        ax.plot(pred[i,:,0], pred[i,:,1], '-o', lw=1, ms=2, label='pred',  color='#9467bd')
        ax.axhline(0, lw=0.3, color='k'); ax.axvline(0, lw=0.3, color='k')
        ax.set_aspect('equal', adjustable='box')
        ax.set_title(f"sample {i+1}")
        ax.grid(True, lw=0.3, alpha=0.4)
        if i == 0: ax.legend(fontsize=8)
    plt.tight_layout(); plt.savefig(out_path, dpi=150); plt.close()

def reconstruct_past_from_df(df_part: pd.DataFrame):
    pos_x = [f"past_x{i+1}" for i in range(PAST_STEPS)]
    pos_y = [f"past_y{i+1}" for i in range(PAST_STEPS)]
    if "past_x20" in pos_x:
        pos_x.remove("past_x20"); pos_y.remove("past_y20")
    X = df_part[pos_x].values.astype(np.float32)
    Y = df_part[pos_y].values.astype(np.float32)
    past = np.stack([X, Y], axis=-1)                 # [N,19,2]
    zeros = np.zeros((len(df_part), 1, 2), np.float32)
    return np.concatenate([past, zeros], axis=1)     # [N,20,2]

def extract_gt_future(df_part: pd.DataFrame):
    gx = df_part[[f"fut_x{i+1}" for i in range(FUT_STEPS)]].values.astype(np.float32)
    gy = df_part[[f"fut_y{i+1}" for i in range(FUT_STEPS)]].values.astype(np.float32)
    return np.stack([gx, gy], axis=-1)               # [N,45,2]

# ================== AMP 互換シム ==================
def make_grad_scaler(enabled: bool):
    try:    return torch.amp.GradScaler(enabled=enabled)
    except: return torch.cuda.amp.GradScaler(enabled=enabled)

class autocast_ctx:
    def __init__(self, enabled: bool):
        self.enabled = enabled; self.ctx = None
    def __enter__(self):
        try:    self.ctx = torch.amp.autocast(device_type="cuda", enabled=self.enabled)
        except: self.ctx = torch.cuda.amp.autocast(enabled=self.enabled)
        return self.ctx.__enter__()
    def __exit__(self, et, ev, tb): return self.ctx.__exit__(et, ev, tb)

PAST_STEPS = 20
FUT_STEPS  = 45

# ================== Dataset（標準化対応） ==================
class TrajCsvDataset(Dataset):
    def __init__(self, csv_path, norm_stats=None):
        self.df = pd.read_csv(csv_path)
        pos_x = [f"past_x{i+1}" for i in range(PAST_STEPS)]
        pos_y = [f"past_y{i+1}" for i in range(PAST_STEPS)]
        if "past_x20" in pos_x:
            pos_x.remove("past_x20"); pos_y.remove("past_y20")
        vel_x = [f"past_vx{i+1}" for i in range(PAST_STEPS)]
        vel_y = [f"past_vy{i+1}" for i in range(PAST_STEPS)]
        spd   = [f"past_speed{i+1}" for i in range(PAST_STEPS)]
        self.in_cols  = pos_x + pos_y + vel_x + vel_y + spd
        self.out_cols = [f"fut_x{i+1}" for i in range(FUT_STEPS)] + [f"fut_y{i+1}" for i in range(FUT_STEPS)]
        self._pos_len = len(pos_x)  # 19
        self.norm = norm_stats  # dict or None

    def __len__(self): return len(self.df)

    def __getitem__(self, idx):
        r = self.df.iloc[idx]
        vals_in  = r[self.in_cols].values.astype(np.float32)
        vals_out = r[self.out_cols].values.astype(np.float32)
        off = 0
        pos_x = vals_in[off:off+self._pos_len]; off += self._pos_len
        pos_y = vals_in[off:off+self._pos_len]; off += self._pos_len
        pos = np.stack([pos_x, pos_y], axis=1)
        pos = np.vstack([pos, np.zeros((1,2), np.float32)])   # (20,2)
        vel_x = vals_in[off:off+PAST_STEPS]; off += PAST_STEPS
        vel_y = vals_in[off:off+PAST_STEPS]; off += PAST_STEPS
        vel = np.stack([vel_x, vel_y], axis=1)                 # (20,2)
        spd = vals_in[off:off+PAST_STEPS].reshape(PAST_STEPS,1)

        if self.norm is not None:
            pos = (pos - self.norm["pos_mean"]) / self.norm["pos_std"]
            vel = (vel - self.norm["vel_mean"]) / self.norm["vel_std"]
            spd = (spd - self.norm["spd_mean"]) / self.norm["spd_std"]

        x = np.concatenate([pos, vel, spd], axis=1)            # (20,5)
        y = vals_out.reshape(2, FUT_STEPS).T                   # (45,2)
        return torch.from_numpy(x), torch.from_numpy(y)

# ================== Model (Residual Seq2Seq) ==================
class SeqResidual(nn.Module):
    """
    Encoder-GRU → Decoder-GRU（Δを逐次生成）→ cumsum で座標に戻す
    teach: Teacher Forcing 用 GT [B,F,2]
    """
    def __init__(self, in_dim=5, hid=256, layers=2, fut_steps=FUT_STEPS, drop=0.1):
        super().__init__()
        self.enc = nn.GRU(in_dim, hid, num_layers=layers, batch_first=True,
                          dropout=(drop if layers > 1 else 0.0))
        self.dec = nn.GRU(2, hid, num_layers=layers, batch_first=True,
                          dropout=(drop if layers > 1 else 0.0))
        self.head = nn.Linear(hid, 2)
        self.fut_steps = fut_steps

    def forward(self, x, teach=None, tf_ratio: float = 0.0):
        B = x.size(0)
        _, h = self.enc(x)
        y_prev = x.new_zeros(B, 1, 2)     # 始点=0
        outs = []
        for t in range(self.fut_steps):
            o, h = self.dec(y_prev, h)
            d = self.head(o)              # Δ_t
            outs.append(d)
            y_hat = y_prev + d
            use_tf = (teach is not None) and (torch.rand(1).item() < tf_ratio)
            y_prev = teach[:, t:t+1, :] if use_tf else y_hat
        deltas = torch.cat(outs, dim=1)   # [B,F,2]
        preds  = torch.cumsum(deltas, dim=1)
        return preds, deltas

# ================== Utils ==================
def seed_everything(seed=42):
    random.seed(seed); np.random.seed(seed)
    torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = True

@torch.no_grad()
def eval_epoch(model, loader, device, tail_weight=1.0):
    model.eval()
    tot_loss = 0.0; n = 0; ADE = 0.0; FDE = 0.0
    mse = nn.MSELoss(reduction="none")
    w = torch.linspace(1.0, tail_weight, FUT_STEPS, device=device).view(1,-1,1)
    for x, y in loader:
        x = x.to(device); y = y.to(device)
        p, _ = model(x, teach=None, tf_ratio=0.0)
        l = (mse(p, y) * w).mean()  # 重み付きMSE
        tot_loss += l.item() * x.size(0) * FUT_STEPS * 2
        n += y.numel()
        err = torch.linalg.norm(p - y, dim=-1)   # [B,F]
        ADE += err.sum().item()
        FDE += err[:,-1].sum().item()
    loss = tot_loss / n
    ade  = ADE / (len(loader.dataset)*FUT_STEPS)
    fde  = FDE / len(loader.dataset)
    return loss, ade, fde

def compute_norm_stats(train_csv, out_path):
    df = pd.read_csv(train_csv)
    px = [f"past_x{i+1}" for i in range(PAST_STEPS) if i+1 != 20]
    py = [f"past_y{i+1}" for i in range(PAST_STEPS) if i+1 != 20]
    vx = [f"past_vx{i+1}" for i in range(PAST_STEPS)]
    vy = [f"past_vy{i+1}" for i in range(PAST_STEPS)]
    sp = [f"past_speed{i+1}" for i in range(PAST_STEPS)]
    pos = np.stack([df[px].values, df[py].values], axis=-1).astype(np.float32)  # [N,19,2]
    pos = np.concatenate([pos, np.zeros((len(df),1,2), np.float32)], axis=1)    # [N,20,2]
    vel = np.stack([df[vx].values, df[vy].values], axis=-1).astype(np.float32)  # [N,20,2]
    spd = df[sp].values.astype(np.float32).reshape(len(df),PAST_STEPS,1)
    stats = dict(
        pos_mean=pos.mean(axis=(0,)), pos_std=pos.std(axis=(0,))+1e-6,
        vel_mean=vel.mean(axis=(0,)), vel_std=vel.std(axis=(0,))+1e-6,
        spd_mean=spd.mean(axis=(0,)), spd_std=spd.std(axis=(0,))+1e-6,
    )
    np.save(out_path, stats)
    return stats

# 速度ビン評価（ざっくり）
@torch.no_grad()
def eval_by_speedbins(model, ds: TrajCsvDataset, device, bins=(0.0, 0.02, 0.05, 1.0)):
    loader = DataLoader(ds, batch_size=256, shuffle=False, num_workers=0, pin_memory=(device.type=="cuda"))
    # 推定：履歴スピードの平均をビン指標に
    spd_cols = [c for c in ds.in_cols if "past_speed" in c]
    spd_mean = ds.df[spd_cols].mean(axis=1).values
    idxs = [np.where((spd_mean>=bins[i]) & (spd_mean<bins[i+1]))[0] for i in range(len(bins)-1)]
    out = []
    for i, idc in enumerate(idxs):
        if len(idc)==0: out.append((np.nan,np.nan)); continue
        sub = torch.utils.data.Subset(ds, idc.tolist())
        tl, ade, fde = eval_epoch(model, DataLoader(sub, batch_size=256, shuffle=False), device)
        out.append((ade, fde))
    return out

# ================== Training ==================
def train(args):
    seed_everything(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() and not args.cpu else "cpu")
    out_dir = Path(args.out_dir); out_dir.mkdir(parents=True, exist_ok=True)

    # 標準化統計
    norm_path = out_dir / "norm_stats.npy"
    norm = None
    if args.normalize:
        norm = compute_norm_stats(args.train_csv, norm_path)
        print(f"[norm] saved: {norm_path}")
    else:
        print("[norm] disabled")

    train_ds = TrajCsvDataset(args.train_csv, norm)
    val_ds   = TrajCsvDataset(args.val_csv, norm)
    use_pin = (device.type == "cuda")
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True,
                              num_workers=0, pin_memory=use_pin)
    val_loader   = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False,
                              num_workers=0, pin_memory=use_pin)

    model = SeqResidual(in_dim=5, hid=args.hidden, layers=args.layers,
                        fut_steps=FUT_STEPS, drop=args.dropout).to(device)

    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.epochs, eta_min=args.lr*args.lr_floor_ratio)
    amp_enabled = (device.type == "cuda")
    scaler = make_grad_scaler(enabled=amp_enabled)

    # EMA
    ema = torch.optim.swa_utils.AveragedModel(model, multi_avg_fn=torch.optim.swa_utils.get_ema_multi_avg_fn(0.999))

    # 損失部品
    mse_red_none = nn.MSELoss(reduction="none")
    delta_loss = nn.L1Loss()

    hist = {"train_loss": [], "val_loss": [], "val_ADE": [], "val_FDE": []}
    best_monitor = float("inf")
    patience_left = args.early_stopping
    ckpt_w = out_dir / "traj_seq_best_weights.pt"
    ckpt_cfg = out_dir / "traj_seq_best_cfg.json"

    global_step = 0
    for epoch in range(1, args.epochs + 1):
        model.train()
        # Teacher Forcing: 0.7 → 0.1 をコサインで
        t = epoch / args.epochs
        tf_ratio = 0.1 + 0.6 * (0.5 * (1 + math.cos(math.pi * t)))

        run_loss = 0.0; seen = 0
        opt.zero_grad(set_to_none=True)

        # 後半重み
        w = torch.linspace(1.0, args.tail_weight, FUT_STEPS, device=device).view(1,-1,1)

        for x, y in train_loader:
            x = x.to(device); y = y.to(device)

            with autocast_ctx(enabled=amp_enabled):
                y_hat, d_hat = model(x, teach=y, tf_ratio=tf_ratio)
                d_gt = torch.diff(torch.cat([y.new_zeros(y.shape[0], 1, 2), y], dim=1), dim=1)

                # 重み付きMSE（後半ほど重く）
                mse = (mse_red_none(y_hat, y) * w).mean()

                # ΔL1
                d_l1 = delta_loss(d_hat, d_gt)

                # FDE
                fde = torch.linalg.norm(y_hat[:, -1] - y[:, -1], dim=-1).mean()

                # 符号反転ペナルティ（x方向の過度な反転抑制）
                sign_flip = torch.relu(torch.sign(d_hat[:,1:,0] * d_hat[:,:-1,0]) * -1)
                flip_pen = sign_flip.mean() * args.flip_coef

                loss = mse + args.lambda_delta * d_l1 + args.lambda_fde * fde + flip_pen

            scaler.scale(loss / args.grad_accum).backward()
            global_step += 1
            if (global_step % args.grad_accum) == 0:
                nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
                scaler.step(opt); scaler.update()
                opt.zero_grad(set_to_none=True)
                ema.update_parameters(model)

            # ログ（MSE基準）
            run_loss += mse.item() * x.size(0) * FUT_STEPS * 2
            seen += y.numel()

        # ===== 検証（EMA重み） =====
        saved = model.state_dict()
        model.load_state_dict(ema.state_dict(), strict=False)
        val_loss, val_ade, val_fde = eval_epoch(model, val_loader, device, tail_weight=args.tail_weight)
        model.load_state_dict(saved, strict=False)
        sched.step()

        train_loss = run_loss / seen
        print(f"[{epoch:02d}] train {train_loss:.6f} | val {val_loss:.6f} | ADE {val_ade:.4f} | FDE {val_fde:.4f} | tf {tf_ratio:.2f} | lr {opt.param_groups[0]['lr']:.2e}")
        hist["train_loss"].append(train_loss); hist["val_loss"].append(val_loss)
        hist["val_ADE"].append(val_ade);       hist["val_FDE"].append(val_fde)

        # 早期停止（FDE監視）
        monitor = val_fde
        if monitor < best_monitor - args.es_min_delta:
            best_monitor = monitor
            torch.save(ema.state_dict(), ckpt_w)  # EMA重みを保存
            with open(ckpt_cfg, "w", encoding="utf-8") as f:
                json.dump(vars(args), f, ensure_ascii=False, indent=2)
            patience_left = args.early_stopping
            print(f"  ↳ saved (EMA) weights: {ckpt_w}  [best FDE: {best_monitor:.4f}]")
        else:
            patience_left -= 1
            if patience_left <= 0:
                print("Early stopping."); break

    # ===== 可視化とログ =====
    log_df = pd.DataFrame(hist); log_csv = out_dir / "training_log.csv"; log_df.to_csv(log_csv, index=False)
    plt.figure(figsize=(7,4)); plt.plot(hist["train_loss"], label="train_loss"); plt.plot(hist["val_loss"], label="val_loss")
    plt.xlabel("epoch"); plt.ylabel("loss"); plt.legend(); plt.tight_layout()
    loss_png = out_dir / "loss_curve.png"; plt.savefig(loss_png, dpi=150); plt.close()

    plt.figure(figsize=(7,4)); plt.plot(hist["val_ADE"], label="val_ADE"); plt.plot(hist["val_FDE"], label="val_FDE")
    plt.xlabel("epoch"); plt.ylabel("error"); plt.legend(); plt.tight_layout()
    adefde_png = out_dir / "ade_fde_curve.png"; plt.savefig(adefde_png, dpi=150); plt.close()

    print(f"📈 Saved curves:\n  - {loss_png}\n  - {adefde_png}\n📝 Log CSV: {log_csv}")

    # ===== 最終 test =====
    if args.test_csv:
        with open(ckpt_cfg, "r", encoding="utf-8") as f:
            cfg = json.load(f)
        best_model = SeqResidual(in_dim=5, hid=cfg["hidden"], layers=cfg["layers"],
                                 fut_steps=FUT_STEPS, drop=cfg["dropout"]).to(device)
        # EMA重みロード
        try:    state_dict = torch.load(ckpt_w, map_location=device, weights_only=True)
        except TypeError: state_dict = torch.load(ckpt_w, map_location=device)
        best_model.load_state_dict(state_dict, strict=False)

        # 正規化の読み出し
        norm = None
        if args.normalize and (out_dir/"norm_stats.npy").exists():
            norm = np.load(out_dir/"norm_stats.npy", allow_pickle=True).item()

        test_ds = TrajCsvDataset(args.test_csv, norm)
        test_loader = DataLoader(test_ds, batch_size=args.batch_size, shuffle=False,
                                 num_workers=0, pin_memory=(device.type=="cuda"))
        tl, ade, fde = eval_epoch(best_model, test_loader, device, tail_weight=args.tail_weight)
        print(f"[TEST] loss {tl:.6f} | ADE {ade:.4f} | FDE {fde:.4f}")

        # 速度ビン別
        bins = (0.0, 0.02, 0.05, 1.0)
        bybin = eval_by_speedbins(best_model, test_ds, device, bins=bins)
        for i,(a,f) in enumerate(bybin):
            print(f"[TEST speed {bins[i]}–{bins[i+1]}] ADE {a:.4f} | FDE {f:.4f}")

        preds = predict_loader(best_model, test_loader, device)
        pred_csv = out_dir / "predictions_test.csv"; save_predictions_csv(test_ds, preds, pred_csv)
        print(f"📝 Saved predictions: {pred_csv}")

        gts = extract_gt_future(test_ds.df)
        rmse = per_step_rmse(preds, gts)
        plt.figure(figsize=(7,4))
        plt.plot(np.arange(1, FUT_STEPS+1)/15.0, rmse)
        plt.xlabel("seconds into future"); plt.ylabel("RMSE (px in 360x240 space)")
        plt.title("Per-step RMSE (test)"); plt.tight_layout()
        rmse_png = out_dir / "per_step_rmse.png"; plt.savefig(rmse_png, dpi=150); plt.close()
        print(f"📈 Saved per-step RMSE: {rmse_png}")

        np.random.seed(args.seed)
        idx = np.random.choice(len(test_ds), size=min(12, len(test_ds)), replace=False)
        past = reconstruct_past_from_df(test_ds.df.iloc[idx].reset_index(drop=True))
        viz_png = out_dir / "viz_samples.png"
        viz_trajectories_grid(past, gts[idx], preds[idx], viz_png, max_plots=len(idx))
        print(f"🖼️ Saved trajectory montage: {viz_png}")

# ================== CLI ==================
def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--train_csv", default=r"C:\Users\s1280\Desktop\trajectory_data\splits\traj_train.csv")
    ap.add_argument("--val_csv",   default=r"C:\Users\s1280\Desktop\trajectory_data\splits\traj_val.csv")
    ap.add_argument("--test_csv",  default=r"C:\Users\s1280\Desktop\trajectory_data\splits\traj_test.csv")
    ap.add_argument("--out_dir",   default=r"C:\Users\s1280\Desktop\trajectory_data\models")
    ap.add_argument("--hidden", type=int, default=256)
    ap.add_argument("--layers", type=int, default=2)
    ap.add_argument("--dropout", type=float, default=0.1)
    ap.add_argument("--batch_size", type=int, default=128)
    ap.add_argument("--epochs", type=int, default=120)          # ← 増やした
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--lr_floor_ratio", type=float, default=0.05)
    ap.add_argument("--weight_decay", type=float, default=1e-4)
    ap.add_argument("--grad_accum", type=int, default=1)
    ap.add_argument("--early_stopping", type=int, default=25)   # ← 緩め
    ap.add_argument("--es_min_delta", type=float, default=1e-4)
    ap.add_argument("--lambda_delta", type=float, default=0.2)
    ap.add_argument("--lambda_fde",   type=float, default=0.5)  # ← ちょい強め
    ap.add_argument("--tail_weight",  type=float, default=1.5)  # ← 後半重み（1.0=均等）
    ap.add_argument("--flip_coef",    type=float, default=0.05) # ← 符号反転ペナルティ
    ap.add_argument("--normalize", action="store_true")         # ← 使うとz-score
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--cpu", action="store_true")
    return ap.parse_args()

if __name__ == "__main__":
    args = parse_args()
    os.makedirs(args.out_dir, exist_ok=True)
    with open(Path(args.out_dir)/"traj_seq_args.json","w",encoding="utf-8") as f:
        json.dump(vars(args), f, ensure_ascii=False, indent=2)
    train(args)
