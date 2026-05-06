# split_traj_by_id.py
import argparse
from pathlib import Path
import pandas as pd
import numpy as np

def stratified_split_by_source_bins(groups, ratios=(0.70,0.15,0.15), seed=42, bins=(1,10,25,50,1_000_000)):
    """
    groups: DataFrame [source,id,rows]
    - sourceごと、さらにrowsのビンごとに分けて、比率で割り当て
    """
    rng = np.random.RandomState(seed)
    # rowsをビン分け（サイズ偏り対策）
    bin_edges = np.array(bins, dtype=int)
    groups = groups.copy()
    groups["size_bin"] = pd.cut(groups["rows"], bins=np.r_[-1,bin_edges], labels=False, right=True)

    keys = {"train":[], "val":[], "test":[]}
    for src, g_src in groups.groupby("source"):
        for b, g_bin in g_src.groupby("size_bin"):
            idx = g_bin.index.to_numpy()
            rng.shuffle(idx)
            n = len(idx)
            n_train = int(round(n*ratios[0]))
            n_val   = int(round(n*ratios[1]))
            n_test  = n - n_train - n_val
            take_tr = idx[:n_train]
            take_va = idx[n_train:n_train+n_val]
            take_te = idx[n_train+n_val:]
            keys["train"].extend(g_bin.loc[take_tr, ["source","id"]].itertuples(index=False, name=None))
            keys["val"].extend(  g_bin.loc[take_va, ["source","id"]].itertuples(index=False, name=None))
            keys["test"].extend( g_bin.loc[take_te, ["source","id"]].itertuples(index=False, name=None))
    return keys

def downsample_train_rows(df_train, max_rows_per_id=None, seed=42):
    """train内で1つの(id,source)からの行数が多すぎる時に上限でダウンサンプル（多様性確保）"""
    if not max_rows_per_id or max_rows_per_id <= 0:
        return df_train
    rng = np.random.RandomState(seed)
    out_parts = []
    for (src, tid), g in df_train.groupby(["source","id"], sort=False):
        if len(g) > max_rows_per_id:
            out_parts.append(g.sample(n=max_rows_per_id, random_state=int(rng.randint(0,1e9))))
        else:
            out_parts.append(g)
    return pd.concat(out_parts, ignore_index=True)

def main(args):
    in_path = Path(args.input_csv)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(in_path)
    assert {"source","id"}.issubset(df.columns), "CSVにはsource,id列が必要です"

    # グループ化してサイズ算出
    groups = df.groupby(["source","id"]).size().reset_index(name="rows")

    # stratified split（source×size_bin）
    keys = stratified_split_by_source_bins(
        groups,
        ratios=(args.train_ratio, args.val_ratio, 1.0-args.train_ratio-args.val_ratio),
        seed=args.seed,
        bins=tuple(args.bin_edges)
    )
    key_train = set(keys["train"])
    key_val   = set(keys["val"])
    key_test  = set(keys["test"])

    # マスク作成（速い方法）
    pair = list(zip(df["source"].tolist(), df["id"].tolist()))
    m_train = [p in key_train for p in pair]
    m_val   = [p in key_val   for p in pair]
    m_test  = [p in key_test  for p in pair]

    df_train = df[m_train].reset_index(drop=True)
    df_val   = df[m_val].reset_index(drop=True)
    df_test  = df[m_test].reset_index(drop=True)

    # trainの過剰IDダウンサンプル（任意）
    if args.max_rows_per_id_train and args.max_rows_per_id_train > 0:
        df_train = downsample_train_rows(df_train, args.max_rows_per_id_train, seed=args.seed)

    # 保存
    p_tr = out_dir / "traj_train.csv"
    p_va = out_dir / "traj_val.csv"
    p_te = out_dir / "traj_test.csv"
    df_train.to_csv(p_tr, index=False)
    df_val.to_csv(p_va, index=False)
    df_test.to_csv(p_te, index=False)

    # サマリ（グループ・行の両面で出す）
    def stat(df_part, name):
        g = df_part.groupby(["source","id"]).size()
        return {
            "name": name,
            "groups": len(g),
            "rows": int(len(df_part)),
            "per_source_groups": df_part.groupby("source")[["id"]].nunique().to_dict()["id"],
            "per_source_rows": df_part.groupby("source").size().to_dict(),
        }

    summary = {
        "total_groups": int(len(groups)),
        "total_rows": int(len(df)),
        "train": stat(df_train,"train"),
        "val":   stat(df_val,  "val"),
        "test":  stat(df_test, "test"),
        "paths": {"train": str(p_tr), "val": str(p_va), "test": str(p_te)}
    }
    print(summary)


if __name__ == "__main__":
    import sys, argparse
    # 引数が渡されたら argparse、なければ固定パラメータで実行
    if len(sys.argv) > 1:
        ap = argparse.ArgumentParser()
        ap.add_argument("--input_csv", required=True,
                        help="統合済みデータセットCSV（traj_dataset_*.csv）")
        ap.add_argument("--out_dir", default="./splits", help="出力ディレクトリ")
        ap.add_argument("--train_ratio", type=float, default=0.70)
        ap.add_argument("--val_ratio",   type=float, default=0.15)
        ap.add_argument("--seed", type=int, default=42)
        ap.add_argument("--bin_edges", type=int, nargs="+", default=[10, 25, 50, 1000000])
        ap.add_argument("--max_rows_per_id_train", type=int, default=0)
        args = ap.parse_args()
    else:
        class Args: pass
        args = Args()
        args.input_csv = r"C:\Users\s1280\Desktop\trajectory_data\traj_dataset_p20_f45_s5_norm360x240_with_speed.csv"
        args.out_dir = r"C:\Users\s1280\Desktop\trajectory_data\splits"
        args.train_ratio = 0.70
        args.val_ratio = 0.15
        args.seed = 42
        args.bin_edges = [10, 25, 50, 1000000]   # 小/中/大/特大の目安
        args.max_rows_per_id_train = 60          # 同一ID取りすぎ抑制（不要なら0）

    main(args)
