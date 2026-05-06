import os, re, json, argparse, math, csv, warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

def find_horizons(df):
    """past_* と fut_* の列から H_PAST, H_FUT を自動推定"""
    def max_idx(prefix):
        pat = re.compile(rf"^{re.escape(prefix)}(\d+)$")
        mx = 0
        for c in df.columns:
            m = pat.match(c)
            if m:
                mx = max(mx, int(m.group(1)))
        return mx
    h_past = min(
        max_idx("past_x"),
        max_idx("past_y"),
        max_idx("past_vx"),
        max_idx("past_vy"),
        max_idx("past_speed"),
    )
    h_fut = min(max_idx("fut_x"), max_idx("fut_y"))
    return h_past, h_fut

def basic_stats(df, h_past, h_fut):
    """座標・速度の基本統計を返す"""
    cols_groups = {
        "past_x":   [f"past_x{i+1}" for i in range(h_past)],
        "past_y":   [f"past_y{i+1}" for i in range(h_past)],
        "past_vx":  [f"past_vx{i+1}" for i in range(h_past)],
        "past_vy":  [f"past_vy{i+1}" for i in range(h_past)],
        "past_sp":  [f"past_speed{i+1}" for i in range(h_past)],
        "fut_x":    [f"fut_x{i+1}" for i in range(h_fut)],
        "fut_y":    [f"fut_y{i+1}" for i in range(h_fut)],
    }
    stats = {}
    for k, cols in cols_groups.items():
        cols = [c for c in cols if c in df.columns]
        if not cols:
            stats[k] = {"count": 0}
            continue
        arr = df[cols].to_numpy(dtype=np.float32)
        stats[k] = {
            "count": int(np.isfinite(arr).sum()),
            "nan_ratio": float(np.isnan(arr).mean()),
            "mean": float(np.nanmean(arr)),
            "std": float(np.nanstd(arr)),
            "p5": float(np.nanpercentile(arr, 5)),
            "p50": float(np.nanpercentile(arr, 50)),
            "p95": float(np.nanpercentile(arr, 95)),
            "min": float(np.nanmin(arr)),
            "max": float(np.nanmax(arr)),
        }
    return stats

def ensure_dir(p):
    os.makedirs(p, exist_ok=True)
    return p

def plot_hist(data, title, xlabel, out_png, out_pdf, bins=80):
    fig = plt.figure(figsize=(5.2, 3.2), dpi=200)
    ax = plt.gca()
    ax.hist(data, bins=bins)
    ax.set_title(title, fontsize=10)
    ax.set_xlabel(xlabel)
    ax.set_ylabel("Count")
    ax.grid(True, linewidth=0.3, alpha=0.6)
    fig.tight_layout()
    fig.savefig(out_png)
    fig.savefig(out_pdf)
    plt.close(fig)

def plot_missing_matrix(df, cols, out_png):
    """簡易欠損ヒートマップ（重くならないよう1000件サンプル）"""
    take = min(len(df), 1000)
    mat = (~df[cols].head(take).isna()).astype(int).to_numpy()
    fig = plt.figure(figsize=(6.0, 3.5), dpi=200)
    ax = plt.gca()
    im = ax.imshow(mat.T, aspect="auto", interpolation="nearest")
    ax.set_yticks(np.linspace(0, len(cols)-1, min(10, len(cols))).astype(int))
    ax.set_yticklabels([cols[i] for i in ax.get_yticks().astype(int)], fontsize=7)
    ax.set_xlabel("Sample Index (subset)")
    ax.set_title("Missing Matrix (1=present, 0=NaN)")
    fig.tight_layout()
    fig.savefig(out_png)
    plt.close(fig)

def write_latex_table(out_dir, n_train, n_val):
    ensure_dir(os.path.join(out_dir, "latex"))
    tex = r"""\begin{table}[t]
\centering
\small
\begin{tabular}{lrr}
\hline
Split & \#Samples & Ratio \\
\hline
Train & %d & %.1f\%% \\
Val   & %d & %.1f\%% \\
\hline
Total & %d & 100.0\%% \\
\hline
\end{tabular}
\caption{Dataset size used for training and validation.}
\label{tab:dataset_size}
\end{table}
""" % (n_train, n_train*100.0/(n_train+n_val+1e-9), n_val, n_val*100.0/(n_train+n_val+1e-9), n_train+n_val)
    with open(os.path.join(out_dir, "latex", "dataset_table.tex"), "w", encoding="utf-8") as f:
        f.write(tex)

def write_latex_text(out_dir, h_past, h_fut, train_csv, val_csv):
    txt = (
        "We used two CSV files for trajectory modeling: "
        f"train ({os.path.basename(train_csv)}) and val ({os.path.basename(val_csv)}). "
        f"Each sample contains past {h_past} steps and future {h_fut} steps with px-relative coordinates. "
        "Inputs include $(x,y)$, $(v_x,v_y)$, and normalized ego-speed. "
        "We minized Smooth L1 loss on the future $(x,y)$ sequence and report ADE/FDE in px on the validation split."
    )
    ensure_dir(os.path.join(out_dir, "latex"))
    with open(os.path.join(out_dir, "latex", "dataset_text.tex"), "w", encoding="utf-8") as f:
        f.write(txt)

def count_columns(df):
    """列パターンごとの本数をCSVに出す（確認用）"""
    patterns = [
        "past_x", "past_y", "past_vx", "past_vy", "past_speed", "fut_x", "fut_y"
    ]
    rows = []
    for p in patterns:
        cnt = sum(1 for c in df.columns if c.startswith(p))
        rows.append((p, cnt))
    return rows

def analyze_one_split(df, split_name, out_dir, h_past, h_fut):
    # 基本統計
    stats = basic_stats(df, h_past, h_fut)

    # ヒストグラム用データ
    sp_cols = [c for c in [f"past_speed{i+1}" for i in range(h_past)] if c in df.columns]
    vx_cols = [c for c in [f"past_vx{i+1}" for i in range(h_past)] if c in df.columns]
    vy_cols = [c for c in [f"past_vy{i+1}" for i in range(h_past)] if c in df.columns]
    fx_cols = [c for c in [f"fut_x{i+1}" for i in range(h_fut)] if c in df.columns]
    fy_cols = [c for c in [f"fut_y{i+1}" for i in range(h_fut)] if c in df.columns]

    # speed distribution (0..1 想定)
    if sp_cols:
        sp = df[sp_cols].to_numpy(dtype=np.float32).ravel()
        sp = sp[np.isfinite(sp)]
        plot_hist(sp, f"{split_name} - Past Ego Speed Distribution", "normalized speed",
                  os.path.join(out_dir, f"hist_speed_{split_name}.png"),
                  os.path.join(out_dir, f"hist_speed_{split_name}.pdf"))

    # velocity magnitude distribution
    if vx_cols and vy_cols:
        vx = df[vx_cols].to_numpy(dtype=np.float32)
        vy = df[vy_cols].to_numpy(dtype=np.float32)
        mag = np.sqrt(vx**2 + vy**2).ravel()
        mag = mag[np.isfinite(mag)]
        plot_hist(mag, f"{split_name} - Past Velocity Magnitude", "px/s",
                  os.path.join(out_dir, f"hist_vel_mag_{split_name}.png"),
                  os.path.join(out_dir, f"hist_vel_mag_{split_name}.pdf"))

    # future displacement per step (フレームごとの変位量)
    if fx_cols and fy_cols:
        fx = df[fx_cols].to_numpy(dtype=np.float32)
        fy = df[fy_cols].to_numpy(dtype=np.float32)
        # 0原点からの相対将来座標 -> 1ステップ差分の大きさ
        dfx = np.diff(fx, axis=1)
        dfy = np.diff(fy, axis=1)
        disp = np.sqrt(dfx**2 + dfy**2).ravel()
        disp = disp[np.isfinite(disp)]
        plot_hist(disp, f"{split_name} - Future Step Displacement", "px/step",
                  os.path.join(out_dir, f"hist_disp_fut_{split_name}.png"),
                  os.path.join(out_dir, f"hist_disp_fut_{split_name}.pdf"))

    # 欠損ヒートマップ（主要列のみ）
    core_cols = []
    for base, H in [("past_x", h_past), ("past_y", h_past), ("fut_x", h_fut), ("fut_y", h_fut)]:
        core_cols.extend([f"{base}{i+1}" for i in range(H) if f"{base}{i+1}" in df.columns])
    if core_cols:
        plot_missing_matrix(df, core_cols, os.path.join(out_dir, f"missing_matrix_{split_name}.png"))

    return stats

def main():
    warnings.filterwarnings("ignore", category=RuntimeWarning)
    ap = argparse.ArgumentParser()
    ap.add_argument("--train", required=True)
    ap.add_argument("--val", required=True)
    ap.add_argument("--out", default="./dataset_report_px")
    args = ap.parse_args()

    ensure_dir(args.out)
    latex_dir = ensure_dir(os.path.join(args.out, "latex"))

    # 1) 読み込み
    train_df = pd.read_csv(args.train)
    val_df   = pd.read_csv(args.val)

    # 2) 地平推定（train優先、valで整合確認）
    h_past_tr, h_fut_tr = find_horizons(train_df)
    h_past_va, h_fut_va = find_horizons(val_df)
    if (h_past_tr != h_past_va) or (h_fut_tr != h_fut_va):
        print(f"[WARN] HORIZON mismatch: train({h_past_tr},{h_fut_tr}) vs val({h_past_va},{h_fut_va})")
    H_PAST = min(h_past_tr, h_past_va)
    H_FUT  = min(h_fut_tr, h_fut_va)

    # 3) 件数
    n_train = len(train_df)
    n_val   = len(val_df)

    # 4) 列カウントCSV
    with open(os.path.join(args.out, "column_counts.csv"), "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["split", "pattern", "count"])
        for split, df in [("train", train_df), ("val", val_df)]:
            for pat, cnt in count_columns(df):
                w.writerow([split, pat, cnt])

    # 5) 統計（per split）
    stats_train = analyze_one_split(train_df, "train", args.out, H_PAST, H_FUT)
    stats_val   = analyze_one_split(val_df, "val",   args.out, H_PAST, H_FUT)

    # 6) per_split_stats.csv
    def stats_to_rows(stats, split):
        rows = []
        for k, d in stats.items():
            if "count" not in d: continue
            rows.append([split, k, d.get("count", 0), d.get("nan_ratio", np.nan), d.get("mean", np.nan),
                         d.get("std", np.nan), d.get("p5", np.nan), d.get("p50", np.nan),
                         d.get("p95", np.nan), d.get("min", np.nan), d.get("max", np.nan)])
        return rows

    with open(os.path.join(args.out, "per_split_stats.csv"), "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["split","group","count","nan_ratio","mean","std","p5","p50","p95","min","max"])
        for r in stats_to_rows(stats_train, "train") + stats_to_rows(stats_val, "val"):
            w.writerow(r)

    # 7) summary.json
    summary = {
        "train_csv": args.train,
        "val_csv": args.val,
        "n_train": int(n_train),
        "n_val": int(n_val),
        "h_past": int(H_PAST),
        "h_fut": int(H_FUT),
        "train_columns": list(train_df.columns),
        "val_columns": list(val_df.columns),
    }
    with open(os.path.join(args.out, "summary.json"), "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    # 8) LaTeX テーブル & テキスト
    write_latex_table(args.out, n_train, n_val)
    write_latex_text(args.out, H_PAST, H_FUT, args.train, args.val)

    print(f"✅ Report saved to: {os.path.abspath(args.out)}")
    print(f"- summary.json, per_split_stats.csv, column_counts.csv")
    print(f"- hist_*_(train|val).png/.pdf, missing_matrix_(train|val).png")
    print(f"- latex/dataset_table.tex, latex/dataset_text.tex")

if __name__ == "__main__":
    main()
