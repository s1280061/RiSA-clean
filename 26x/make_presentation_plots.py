# make_presentation_plots.py
import os, argparse, numpy as np, pandas as pd, torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import matplotlib.pyplot as plt

# ====== Dataset（速度などの特徴も返す） ======
class TrajDataset(Dataset):
    def __init__(self, csv_path, h_past, h_fut):
        self.h_past, self.h_fut = h_past, h_fut
        df = pd.read_csv(csv_path)
        self.df = df
        self.px  = df[[f"past_x{i+1}"     for i in range(h_past)]].to_numpy(np.float32)
        self.py  = df[[f"past_y{i+1}"     for i in range(h_past)]].to_numpy(np.float32)
        self.pvx = df[[f"past_vx{i+1}"    for i in range(h_past)]].to_numpy(np.float32)
        self.pvy = df[[f"past_vy{i+1}"    for i in range(h_past)]].to_numpy(np.float32)
        self.psp = df[[f"past_speed{i+1}" for i in range(h_past)]].to_numpy(np.float32)
        self.fx  = df[[f"fut_x{i+1}"      for i in range(h_fut)]].to_numpy(np.float32)
        self.fy  = df[[f"fut_y{i+1}"      for i in range(h_fut)]].to_numpy(np.float32)

    def __len__(self): return len(self.df)
    def __getitem__(self, idx):
        X = np.stack([self.px[idx], self.py[idx], self.pvx[idx], self.pvy[idx], self.psp[idx]], axis=-1)  # (T,5)
        Y = np.stack([self.fx[idx], self.fy[idx]], axis=-1)  # (Tfut,2)
        # 追加特徴
        init_speed = float(self.psp[idx][-1])  # 過去末の速度
        # 曲がり具合（簡易）：未来ベクトルの総曲率 proxy = 前向き差分角度の絶対和
        fut = Y
        v = fut[1:] - fut[:-1]
        ang = np.arctan2(v[:,1], v[:,0])
        dtheta = np.abs(np.unwrap(ang)[1:] - np.unwrap(ang)[:-1])
        curvature_proxy = float(np.nansum(dtheta))
        return torch.from_numpy(X), torch.from_numpy(Y), init_speed, curvature_proxy

# ====== モデル ======
class Encoder(nn.Module):
    def __init__(self, input_size, hidden, n_layers, dropout):
        super().__init__()
        self.rnn = nn.GRU(input_size, hidden, num_layers=n_layers, batch_first=True,
                          dropout=dropout if n_layers>1 else 0.0)
    def forward(self, x): _, h = self.rnn(x); return h

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
            y = self.proj(o); outs.append(y); dec_in = y
        return torch.cat(outs, dim=1)

class Seq2Seq(nn.Module):
    def __init__(self, input_size, hidden, n_layers, dropout):
        super().__init__()
        self.encoder = Encoder(input_size, hidden, n_layers, dropout)
        self.decoder = Decoder(hidden, 2, n_layers, dropout)
    def forward(self, x, steps): return self.decoder(self.encoder(x), steps=steps)

# ====== 指標 ======
def ade_fde_norm(pred, gt):
    err = torch.linalg.norm(pred - gt, dim=-1)   # (B,T)
    ade = err.mean(dim=1)                        # (B,)
    fde = err[:, -1]                             # (B,)
    return ade, fde, err                         # err for ADE_t

def to_pixels(xy, W, H):
    out = xy.clone()
    out[...,0] *= W; out[...,1] *= H
    return out

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--ckpt', required=True)
    ap.add_argument('--val_csv', required=True)
    ap.add_argument('--outdir', default='report_out')
    ap.add_argument('--img_w', type=float, default=360.0)
    ap.add_argument('--img_h', type=float, default=240.0)
    args = ap.parse_args()

    os.makedirs(args.outdir, exist_ok=True)
    ckpt = torch.load(args.ckpt, map_location='cpu')
    cfg  = ckpt.get('cfg', {})
    h_past = int(cfg.get('h_past', 30))
    h_fut  = int(cfg.get('h_fut', 45))
    hidden = int(cfg.get('hidden', 256))
    layers = int(cfg.get('layers', 3))
    drop   = float(cfg.get('dropout', 0.2))

    ds = TrajDataset(args.val_csv, h_past, h_fut)
    ld = DataLoader(ds, batch_size=256, shuffle=False, num_workers=0)
    model = Seq2Seq(input_size=5, hidden=hidden, n_layers=layers, dropout=drop)
    model.load_state_dict(ckpt['state_dict']); model.eval()

    all_aden, all_fden = [], []
    all_adepx, all_fdepx = [], []
    all_errt_px = []  # per-t error px for ADE over horizon
    init_speeds, curvatures = [], []

    with torch.no_grad():
        for X, Y, s0, curv in ld:
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

            all_aden.append(ade_n.cpu()); all_fden.append(fde_n.cpu())
            all_adepx.append(ade_px.cpu()); all_fdepx.append(fde_px.cpu())
            all_errt_px.append(dist_px.mean(dim=0).cpu())  # (T,)

            init_speeds.extend(s0.numpy().tolist())
            curvatures.extend(curv.numpy().tolist())

    import torch as th
    aden = th.cat(all_aden).numpy(); fden = th.cat(all_fden).numpy()
    adep = th.cat(all_adepx).numpy(); fdep = th.cat(all_fdepx).numpy()
    ade_t_px = th.stack(all_errt_px).numpy()      # (B,T) mean over batch already; stack→(N,T)
    ade_t_px = ade_t_px.mean(axis=0)              # (T,) dataset mean

    # ===== CSV: per-sample =====
    per_sample = pd.DataFrame({
        "ADE_norm": aden, "FDE_norm": fden,
        "ADE_px": adep,   "FDE_px": fdep,
        "init_speed_norm": np.array(init_speeds),
        "curvature_proxy": np.array(curvatures),
    })
    per_sample.to_csv(os.path.join(args.outdir, "metrics_per_sample.csv"), index=False)

    # ===== summary stats =====
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

    # ===== 図: Hist / CDF =====
    def save_hist(arr, title, path, bins=40):
        plt.figure(dpi=150); plt.hist(arr, bins=bins)
        plt.title(title); plt.xlabel(title); plt.ylabel("Count"); plt.tight_layout(); plt.savefig(path); plt.close()
    def save_cdf(arr, title, path):
        xs = np.sort(arr); ys = np.arange(1, len(xs)+1)/len(xs)
        plt.figure(dpi=150); plt.plot(xs, ys)
        plt.title(title); plt.xlabel(title); plt.ylabel("CDF"); plt.tight_layout(); plt.savefig(path); plt.close()

    save_hist(adep, "ADE (px)", os.path.join(args.outdir, "hist_ADEpx.png"))
    save_hist(fdep, "FDE (px)", os.path.join(args.outdir, "hist_FDEpx.png"))
    save_cdf(adep,  "ADE (px) CDF", os.path.join(args.outdir, "cdf_ADEpx.png"))
    save_cdf(fdep,  "FDE (px) CDF", os.path.join(args.outdir, "cdf_FDEpx.png"))

    # ===== 図: ADE over horizon（px）=====
    plt.figure(dpi=150); plt.plot(np.arange(1, len(ade_t_px)+1), ade_t_px)
    plt.title("ADE over horizon (px)"); plt.xlabel("t (future step)"); plt.ylabel("ADE_t (px)")
    plt.grid(True, alpha=0.3); plt.tight_layout(); plt.savefig(os.path.join(args.outdir, "ADE_over_horizon_px.png")); plt.close()

    # ===== 図: FDE vs 初期速度 =====
    plt.figure(dpi=150); plt.scatter(init_speeds, fdep, s=6, alpha=0.5)
    plt.title("FDE(px) vs initial speed (norm)"); plt.xlabel("initial speed (norm)"); plt.ylabel("FDE (px)")
    plt.grid(True, alpha=0.3); plt.tight_layout(); plt.savefig(os.path.join(args.outdir, "FDE_vs_init_speed.png")); plt.close()

    # ===== 図: 速度ビン別の ADE boxplot =====
    sp = np.array(init_speeds)
    bins = [0, 0.02, 0.05, 0.1, 0.2, 1.0]  # 正規化速度の区切り（必要なら調整）
    labels = [f"{bins[i]}–{bins[i+1]}" for i in range(len(bins)-1)]
    groups = []
    for i in range(len(bins)-1):
        m = (sp >= bins[i]) & (sp < bins[i+1])
        groups.append(adep[m])
    plt.figure(dpi=150); plt.boxplot(groups, labels=labels, showfliers=False)
    plt.title("ADE(px) by initial speed bin"); plt.xlabel("speed bin (norm)"); plt.ylabel("ADE (px)")
    plt.tight_layout(); plt.savefig(os.path.join(args.outdir, "ADE_box_by_speedbin.png")); plt.close()

    # ===== LaTeX: summary table =====
    with open(os.path.join(args.outdir, "table_summary.tex"), "w", encoding="utf-8") as f:
        f.write(r"""\begin{table}[t]
\centering
\begin{tabular}{lrrrr}
\hline
Metric & Mean & Std & P25 & P50 & P75 \\
\hline
ADE (norm) & %.4f & %.4f & %.4f & %.4f & %.4f \\
FDE (norm) & %.4f & %.4f & %.4f & %.4f & %.4f \\
ADE (px)   & %.2f & %.2f & %.2f & %.2f & %.2f \\
FDE (px)   & %.2f & %.2f & %.2f & %.2f & %.2f \\
\hline
\end{tabular}
\caption{Validation trajectory errors (N=%d, horizon=%d).}
\end{table}
""" % (summary.loc["ADE_norm","mean"], summary.loc["ADE_norm","std"], summary.loc["ADE_norm","p25"], summary.loc["ADE_norm","p50"], summary.loc["ADE_norm","p75"],
       summary.loc["FDE_norm","mean"], summary.loc["FDE_norm","std"], summary.loc["FDE_norm","p25"], summary.loc["FDE_norm","p50"], summary.loc["FDE_norm","p75"],
       summary.loc["ADE_px","mean"],   summary.loc["ADE_px","std"],   summary.loc["ADE_px","p25"],   summary.loc["ADE_px","p50"],   summary.loc["ADE_px","p75"],
       summary.loc["FDE_px","mean"],   summary.loc["FDE_px","std"],   summary.loc["FDE_px","p25"],   summary.loc["FDE_px","p50"],   summary.loc["FDE_px","p75"],
       len(ds), h_fut))
    print(f"Saved all to: {args.outdir}")

if __name__ == '__main__':
    main()
