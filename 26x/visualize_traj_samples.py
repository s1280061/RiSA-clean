# report_and_visualize_traj.py
import os, re, argparse, random, numpy as np, pandas as pd, torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import matplotlib.pyplot as plt

# ================= Dataset =================
class TrajDataset(Dataset):
    def __init__(self, csv_path, h_past, h_fut):
        self.h_past, self.h_fut = h_past, h_fut
        df = pd.read_csv(csv_path)
        self.df  = df
        self.px  = df[[f"past_x{i+1}"     for i in range(h_past)]].to_numpy(np.float32)
        self.py  = df[[f"past_y{i+1}"     for i in range(h_past)]].to_numpy(np.float32)
        self.pvx = df[[f"past_vx{i+1}"    for i in range(h_past)]].to_numpy(np.float32)
        self.pvy = df[[f"past_vy{i+1}"    for i in range(h_past)]].to_numpy(np.float32)
        self.psp = df[[f"past_speed{i+1}" for i in range(h_past)]].to_numpy(np.float32)
        self.fx  = df[[f"fut_x{i+1}"      for i in range(h_fut)]].to_numpy(np.float32)
        self.fy  = df[[f"fut_y{i+1}"      for i in range(h_fut)]].to_numpy(np.float32)

    def __len__(self): return len(self.df)

    def __getitem__(self, idx):
        X = np.stack([self.px[idx], self.py[idx], self.pvx[idx], self.pvy[idx], self.psp[idx]], axis=-1)  # (T_past,5)
        Y = np.stack([self.fx[idx], self.fy[idx]], axis=-1)                                               # (T_fut,2)
        init_speed = float(self.psp[idx][-1])  # 過去末の速度（正規化）
        # 未来の曲がり具合proxy
        v = Y[1:] - Y[:-1]
        ang = np.arctan2(v[:,1], v[:,0])
        dang = np.abs(np.diff(np.unwrap(ang)))
        curvature = float(np.nansum(dang))
        return torch.from_numpy(X), torch.from_numpy(Y), init_speed, curvature

# ================= Model =================
class Encoder(nn.Module):
    def __init__(self, input_size, hidden, n_layers, dropout):
        super().__init__()
        self.rnn = nn.GRU(input_size, hidden, num_layers=n_layers, batch_first=True,
                          dropout=dropout if n_layers>1 else 0.0)
    def forward(self, x):
        _, h = self.rnn(x); return h

class Decoder(nn.Module):
    def __init__(self, hidden, out_size, n_layers, dropout):
        super().__init__()
        self.rnn = nn.GRU(out_size, hidden, num_layers=n_layers, batch_first=True,
                          dropout=dropout if n_layers>1 else 0.0)
        self.proj = nn.Linear(hidden, out_size)
    def forward(self, h0, steps):
        B = h0.shape[1]
        dec_in = torch.zeros(B,1,2, device=h0.device)
        outs, h = [], h0
        for _ in range(steps):
            o, h = self.rnn(dec_in, h)
            y = self.proj(o)
            outs.append(y)
            dec_in = y
        return torch.cat(outs, dim=1)

class Seq2Seq(nn.Module):
    def __init__(self, input_size, hidden, n_layers, dropout):
        super().__init__()
        self.encoder = Encoder(input_size, hidden, n_layers, dropout)
        self.decoder = Decoder(hidden, 2, n_layers, dropout)
    def forward(self, x, steps):
        return self.decoder(self.encoder(x), steps=steps)

# ================= Metrics =================
def ade_fde_norm(pred, gt):
    err = torch.linalg.norm(pred - gt, dim=-1)  # (B,T)
    ade = err.mean(dim=1)                        # (B,)
    fde = err[:, -1]                             # (B,)
    return ade, fde, err

def to_pixels(xy, W, H):
    out = xy.clone()
    out[...,0] *= W; out[...,1] *= H
    return out

# ================= Viz helpers =================
def compute_global_limits(df, H_PAST, H_FUT, img_w=None, img_h=None, margin=0.10):
    xs_all, ys_all = [], []
    for _, row in df.iterrows():
        past_xy = np.array([[row[f"past_x{i+1}"], row[f"past_y{i+1}"]] for i in range(H_PAST)])
        fut_xy  = np.array([[row[f"fut_x{i+1}"],  row[f"fut_y{i+1}"]]  for i in range(H_FUT)])
        coords = np.vstack([past_xy, fut_xy])
        if img_w and img_h:
            coords[:,0] *= img_w; coords[:,1] *= img_h
        xs_all.extend(coords[:,0]); ys_all.extend(coords[:,1])
    x_min, x_max = min(xs_all), max(xs_all)
    y_min, y_max = min(ys_all), max(ys_all)
    dx = (x_max - x_min) * margin
    dy = (y_max - y_min) * margin
    return (x_min - dx, x_max + dx), (y_min - dy, y_max + dy)

def plot_traj(past_xy, fut_gt, fut_pred, save_path, xlim, ylim, title=None):
    fig = plt.figure(figsize=(5.5, 4.2), dpi=150)
    ax = plt.gca()
    ax.plot(past_xy[:,0], past_xy[:,1], marker='o', linewidth=1, markersize=2, label='past')
    ax.plot([past_xy[-1,0], fut_gt[0,0]], [past_xy[-1,1], fut_gt[0,1]], linewidth=1, alpha=0.5)
    ax.plot(fut_gt[:,0], fut_gt[:,1], linewidth=2, label='future GT')
    ax.plot(fut_pred[:,0], fut_pred[:,1], linewidth=2, linestyle='--', label='future Pred')
    ax.grid(True, alpha=0.3); ax.legend(loc='best')
    ax.set_xlim(*xlim); ax.set_ylim(*ylim)
    ax.set_aspect('equal', adjustable='box')
    if title: ax.set_title(title)
    fig.tight_layout(); fig.savefig(save_path); plt.close(fig)

# ================= Train log parser (optional) =================
LOG_PAT = re.compile(
    r"\[(\d+)/(\d+)\].*?train loss ([\d.]+).*?ADE\(n\) ([\d.]+).*?FDE\(n\) ([\d.]+).*?"
    r"val loss ([\d.]+).*?ADE\(n\) ([\d.]+).*?FDE\(n\) ([\d.]+)", re.I
)

def parse_train_log(path):
    ep, tr_loss, tr_ade, tr_fde, va_loss, va_ade, va_fde = [], [], [], [], [], [], []
    with open(path, "r", encoding="utf-8", errors="ignore") as f:
        for line in f:
            m = LOG_PAT.search(line)
            if m:
                ep.append(int(m.group(1)))
                tr_loss.append(float(m.group(3)))
                tr_ade.append(float(m.group(4)))
                tr_fde.append(float(m.group(5)))
                va_loss.append(float(m.group(6)))
                va_ade.append(float(m.group(7)))
                va_fde.append(float(m.group(8)))
    return {
        "epoch": np.array(ep),
        "train_loss": np.array(tr_loss),
        "train_ADE": np.array(tr_ade),
        "train_FDE": np.array(tr_fde),
        "val_loss": np.array(va_loss),
        "val_ADE": np.array(va_ade),
        "val_FDE": np.array(va_fde),
    }

def plot_training_curves(hist, outdir):
    if len(hist["epoch"]) == 0: return
    # Loss
    plt.figure(dpi=150)
    plt.plot(hist["epoch"], hist["train_loss"], label="train")
    plt.plot(hist["epoch"], hist["val_loss"], label="val")
    plt.xlabel("epoch"); plt.ylabel("loss"); plt.title("Training/Validation Loss")
    plt.grid(True, alpha=0.3); plt.legend(); plt.tight_layout()
    plt.savefig(os.path.join(outdir, "curve_loss.png")); plt.close()
    # ADE
    plt.figure(dpi=150)
    plt.plot(hist["epoch"], hist["train_ADE"], label="train ADE(n)")
    plt.plot(hist["epoch"], hist["val_ADE"], label="val ADE(n)")
    plt.xlabel("epoch"); plt.ylabel("ADE (norm)"); plt.title("ADE over epochs")
    plt.grid(True, alpha=0.3); plt.legend(); plt.tight_layout()
    plt.savefig(os.path.join(outdir, "curve_ADE.png")); plt.close()
    # FDE
    plt.figure(dpi=150)
    plt.plot(hist["epoch"], hist["train_FDE"], label="train FDE(n)")
    plt.plot(hist["epoch"], hist["val_FDE"], label="val FDE(n)")
    plt.xlabel("epoch"); plt.ylabel("FDE (norm)"); plt.title("FDE over epochs")
    plt.grid(True, alpha=0.3); plt.legend(); plt.tight_layout()
    plt.savefig(os.path.join(outdir, "curve_FDE.png")); plt.close()

# ================= Main =================
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt",    required=True)
    ap.add_argument("--val_csv", required=True)
    ap.add_argument("--outdir",  default="report_out")
    ap.add_argument("--num",     type=int, default=12)
    ap.add_argument("--img_w",   type=float, default=360.0)
    ap.add_argument("--img_h",   type=float, default=240.0)
    ap.add_argument("--train_log", type=str, default=None, help="学習ログ（任意）")
    args = ap.parse_args()

    os.makedirs(args.outdir, exist_ok=True)

    # 1) Load checkpoint & cfg
    ckpt = torch.load(args.ckpt, map_location="cpu")  # 既存ckpt互換のため weights_only は使わない
    cfg  = ckpt.get("cfg", {})
    h_past = int(cfg.get("h_past", 30))
    h_fut  = int(cfg.get("h_fut", 45))
    hidden = int(cfg.get("hidden", 256))
    layers = int(cfg.get("layers", 3))
    drop   = float(cfg.get("dropout", 0.2))

    # 2) Dataset & Model
    ds = TrajDataset(args.val_csv, h_past, h_fut)
    ld = DataLoader(ds, batch_size=256, shuffle=False, num_workers=0)
    model = Seq2Seq(input_size=5, hidden=hidden, n_layers=layers, dropout=drop)
    model.load_state_dict(ckpt["state_dict"])
    model.eval()

    # 3) 可視化用：共通スケール（px）
    xlim, ylim = compute_global_limits(ds.df, h_past, h_fut, img_w=args.img_w, img_h=args.img_h, margin=0.10)

    # 4) Val全体での指標
    all_ade_n, all_fde_n, all_ade_px, all_fde_px, ade_t_px = [], [], [], [], []

    with torch.no_grad():
        for X, Y, _, _ in ld:
            pred = model(X, steps=h_fut)
            # norm
            ade_n, fde_n, err_n = ade_fde_norm(pred, Y)
            # px
            pred_px = to_pixels(pred, args.img_w, args.img_h)
            Y_px    = to_pixels(Y,    args.img_w, args.img_h)
            diff = pred_px - Y_px
            dist_px = torch.sqrt(diff[...,0]**2 + diff[...,1]**2)
            ade_px = dist_px.mean(dim=1)
            fde_px = dist_px[:, -1]
            all_ade_n.append(ade_n.cpu()); all_fde_n.append(fde_n.cpu())
            all_ade_px.append(ade_px.cpu()); all_fde_px.append(fde_px.cpu())
            ade_t_px.append(dist_px.mean(dim=0).cpu())  # (T,)

    aden = torch.cat(all_ade_n).numpy()
    fden = torch.cat(all_fde_n).numpy()
    adep = torch.cat(all_ade_px).numpy()
    fdep = torch.cat(all_fde_px).numpy()
    ade_t_px = torch.stack(ade_t_px).numpy().mean(axis=0)  # (T,)

    # 5) per-sample CSV（追加特徴も含めたいので再ループ）
    init_speeds, curvatures = [], []
    for i in range(len(ds)):
        _, _, s, c = ds[i]
        init_speeds.append(float(s)); curvatures.append(float(c))
    per_sample = pd.DataFrame({
        "ADE_norm": aden, "FDE_norm": fden,
        "ADE_px": adep,   "FDE_px": fdep,
        "init_speed_norm": np.array(init_speeds),
        "curvature_proxy": np.array(curvatures),
    })
    per_sample.to_csv(os.path.join(args.outdir, "metrics_per_sample.csv"), index=False)

    # 6) summary CSV + LaTeX
    def stats(a):
        return {
            "mean": float(np.mean(a)),
            "std":  float(np.std(a)),
            "p25":  float(np.percentile(a, 25)),
            "p50":  float(np.percentile(a, 50)),
            "p75":  float(np.percentile(a, 75)),
        }
    summary = pd.DataFrame({
        "ADE_norm": [stats(aden)],
        "FDE_norm": [stats(fden)],
        "ADE_px":   [stats(adep)],
        "FDE_px":   [stats(fdep)],
    }).T
    summary.to_csv(os.path.join(args.outdir, "summary_stats.csv"))

    with open(os.path.join(args.outdir, "table_summary.tex"), "w", encoding="utf-8") as f:
        f.write(r"""\begin{table}[t]
\centering
\begin{tabular}{lrrrrr}
\hline
Metric & Mean & Std & P25 & P50 & P75 \\
\hline
ADE (norm) & %.4f & %.4f & %.4f & %.4f & %.4f \\
FDE (norm) & %.4f & %.4f & %.4f & %.4f & %.4f \\
ADE (px)   & %.2f & %.2f & %.2f & %.2f & %.2f \\
FDE (px)   & %.2f & %.2f & %.2f & %.2f & %.2f \\
\hline
\end{tabular}
\caption{Validation errors (N=%d, horizon=%d).}
\end{table}
""" % (summary.loc["ADE_norm","mean"], summary.loc["ADE_norm","std"], summary.loc["ADE_norm","p25"], summary.loc["ADE_norm","p50"], summary.loc["ADE_norm","p75"],
       summary.loc["FDE_norm","mean"], summary.loc["FDE_norm","std"], summary.loc["FDE_norm","p25"], summary.loc["FDE_norm","p50"], summary.loc["FDE_norm","p75"],
       summary.loc["ADE_px","mean"],   summary.loc["ADE_px","std"],   summary.loc["ADE_px","p25"],   summary.loc["ADE_px","p50"],   summary.loc["ADE_px","p75"],
       summary.loc["FDE_px","mean"],   summary.loc["FDE_px","std"],   summary.loc["FDE_px","p25"],   summary.loc["FDE_px","p50"],   summary.loc["FDE_px","p75"],
       len(ds), h_fut))

    # 7) 図：ヒスト/CDF/時間別ADE/速度×FDE/速度ビン箱ひげ
    def save_hist(arr, title, path, bins=40):
        plt.figure(dpi=150); plt.hist(arr, bins=bins)
        plt.title(title); plt.xlabel(title); plt.ylabel("Count")
        plt.tight_layout(); plt.savefig(path); plt.close()
    def save_cdf(arr, title, path):
        xs = np.sort(arr); ys = np.arange(1, len(xs)+1)/len(xs)
        plt.figure(dpi=150); plt.plot(xs, ys)
        plt.title(title); plt.xlabel(title); plt.ylabel("CDF")
        plt.tight_layout(); plt.savefig(path); plt.close()

    save_hist(adep, "ADE (px)", os.path.join(args.outdir, "hist_ADEpx.png"))
    save_hist(fdep, "FDE (px)", os.path.join(args.outdir, "hist_FDEpx.png"))
    save_cdf(adep,  "ADE (px) CDF", os.path.join(args.outdir, "cdf_ADEpx.png"))
    save_cdf(fdep,  "FDE (px) CDF", os.path.join(args.outdir, "cdf_FDEpx.png"))

    plt.figure(dpi=150); plt.plot(np.arange(1, len(ade_t_px)+1), ade_t_px)
    plt.title("ADE over horizon (px)"); plt.xlabel("future step"); plt.ylabel("ADE_t (px)")
    plt.grid(True, alpha=0.3); plt.tight_layout()
    plt.savefig(os.path.join(args.outdir, "ADE_over_horizon_px.png")); plt.close()

    sp = np.array(init_speeds)
    plt.figure(dpi=150); plt.scatter(sp, fdep, s=8, alpha=0.5)
    plt.title("FDE(px) vs initial speed (norm)"); plt.xlabel("initial speed (norm)"); plt.ylabel("FDE (px)")
    plt.grid(True, alpha=0.3); plt.tight_layout()
    plt.savefig(os.path.join(args.outdir, "FDE_vs_init_speed.png")); plt.close()

    bins = [0, 0.02, 0.05, 0.1, 0.2, 1.0]
    labels = [f"{bins[i]}–{bins[i+1]}" for i in range(len(bins)-1)]
    groups = []
    for i in range(len(bins)-1):
        m = (sp >= bins[i]) & (sp < bins[i+1])
        groups.append(adep[m])
    plt.figure(dpi=150); plt.boxplot(groups, labels=labels, showfliers=False)
    plt.title("ADE(px) by initial speed bin"); plt.xlabel("speed bin (norm)"); plt.ylabel("ADE (px)")
    plt.tight_layout(); plt.savefig(os.path.join(args.outdir, "ADE_box_by_speedbin.png")); plt.close()

    # 8) サンプル可視化（共通スケール、px、Y軸は画像座標そのまま ※下が大きい）
    os.makedirs(os.path.join(args.outdir, "samples"), exist_ok=True)
    idxs = random.sample(range(len(ds)), k=min(args.num, len(ds)))
    for i, idx in enumerate(idxs):
        X, Y, _, _ = ds[idx]
        with torch.no_grad():
            pred = model(X.unsqueeze(0), steps=h_fut).squeeze(0)
        past_xy = X[:, :2]               # (T_past, 2) norm
        fut_gt  = Y                       # (T_fut, 2)  norm
        # to px（画像座標。見やすさ優先で上下反転はしない＝下が大きい）
        past_px = to_pixels(past_xy, args.img_w, args.img_h).numpy()
        gt_px   = to_pixels(fut_gt,  args.img_w, args.img_h).numpy()
        pr_px   = to_pixels(pred,    args.img_w, args.img_h).numpy()
        save_path = os.path.join(args.outdir, "samples", f"sample_{i:02d}.png")
        plot_traj(past_px, gt_px, pr_px, save_path, xlim=xlim, ylim=ylim, title=f"sample_{i:02d}")

    # 9) 学習曲線（任意）
    if args.train_log and os.path.exists(args.train_log):
        hist = parse_train_log(args.train_log)
        plot_training_curves(hist, args.outdir)

    print(f"✅ Done. All outputs saved under: {args.outdir}")

if __name__ == "__main__":
    main()
