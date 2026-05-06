import os
import torch
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# ========= CONFIG ==========
CSV_PATH = r"C:\Users\s1280\Desktop\trajectory_data\traj_dataset_p30_f45_s5_px_with_ego_speed_val.csv"
CKPT_PATH = r".\checkpoints_traj_px_best15\best_ade_px.pt"

SAVE_X = r".\traj_vis_x.png"
SAVE_Y = r".\traj_vis_y.png"

H_PAST = 30
H_FUT = 45
INPUT_SIZE = 5
OUTPUT_SIZE = 2

# ===== モデル読み込み =====
from train_traj_seq2seq_px_rel import Seq2Seq

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def load_model():
    ckpt = torch.load(CKPT_PATH, map_location=device)
    cfg = ckpt["cfg"]

    model = Seq2Seq(
        input_size=INPUT_SIZE,
        hidden=cfg["hidden"],
        n_layers=cfg["layers"],
        dropout=cfg["dropout"],
        out_size=OUTPUT_SIZE,
    ).to(device)

    model.load_state_dict(ckpt["state_dict"])
    model.eval()
    return model


# ===== データ1サンプル読み込み =====
def load_sample(csv_path):
    df = pd.read_csv(csv_path)
    idx = np.random.randint(len(df))
    row = df.iloc[idx]

    # past
    px = np.array([row[f"past_x{i+1}"] for i in range(H_PAST)], dtype=np.float32)
    py = np.array([row[f"past_y{i+1}"] for i in range(H_PAST)], dtype=np.float32)
    pvx = np.array([row[f"past_vx{i+1}"] for i in range(H_PAST)], dtype=np.float32)
    pvy = np.array([row[f"past_vy{i+1}"] for i in range(H_PAST)], dtype=np.float32)
    sp = np.array([row[f"past_speed{i+1}"] for i in range(H_PAST)], dtype=np.float32)

    # future
    fx = np.array([row[f"fut_x{i+1}"] for i in range(H_FUT)], dtype=np.float32)
    fy = np.array([row[f"fut_y{i+1}"] for i in range(H_FUT)], dtype=np.float32)

    X = np.stack([px, py, pvx, pvy, sp], axis=-1)  # (30,5)
    Y = np.stack([fx, fy], axis=-1)                # (45,2)

    return torch.tensor(X).unsqueeze(0), torch.tensor(Y).unsqueeze(0)


# ===== 可視化：X or Y のみを描画 =====
def plot_axis(past, pred, gt, axis=0, save_path="out.png"):
    """
    axis=0 → x座標
    axis=1 → y座標
    """

    past = past.cpu().numpy()[0][:, axis]   # (30,)
    pred = pred.cpu().numpy()[0][:, axis]   # (45,)
    gt   = gt.cpu().numpy()[0][:, axis]     # (45,)

    # 時間軸 = フレーム番号
    t_past = np.arange(0, len(past))
    t_fut  = np.arange(len(past), len(past) + len(gt))

    plt.figure(figsize=(7, 4), dpi=200)

    # past
    plt.plot(t_past, past, color="blue", label="Input past")

    # future (GT)
    plt.plot(t_fut, gt, color="green", label="True future")

    # future (pred)
    plt.plot(t_fut, pred, "r--", label="Predicted future")

    # 💡 修正：横軸をFrameに
    plt.xlabel("Frame (15 fps)")
    plt.ylabel("X coordinate (px)" if axis == 0 else "Y coordinate (px)")
    plt.title("Input and Future Prediction")

    plt.legend()
    plt.grid(True, alpha=0.4)

    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()

    print(f"Saved → {save_path}")


# ===== MAIN =====
def main():
    model = load_model()

    X, Y = load_sample(CSV_PATH)
    X, Y = X.to(device).float(), Y.to(device).float()

    with torch.no_grad():
        pred = model(X, tgt=None, tf_ratio=0.0, steps=H_FUT)

    past_xy = X[..., :2]  # (1,30,2)

    # X座標の図
    plot_axis(past_xy, pred, Y, axis=0, save_path=SAVE_X)

    # Y座標の図
    plot_axis(past_xy, pred, Y, axis=1, save_path=SAVE_Y)


if __name__ == "__main__":
    main()
