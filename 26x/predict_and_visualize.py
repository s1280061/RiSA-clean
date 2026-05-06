import os
import torch
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# ========= CONFIG ==========
CSV_PATH = r"C:\Users\s1280\Desktop\trajectory_data\traj_dataset_p30_f45_s5_px_with_ego_speed_val.csv"
CKPT_PATH = r".\checkpoints_traj_px_best15\best_ade_px.pt"
SAVE_IMG = r".\traj_vis_example.png"

H_PAST = 30
H_FUT = 45
INPUT_SIZE = 5
OUTPUT_SIZE = 2

# ========= MODEL (same as training) ==========
from train_traj_seq2seq_px_rel import Seq2Seq  # ← あなたのコードから import できる

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


# ========= DATA LOADING (single sample) ==========
def load_sample(csv_path):
    df = pd.read_csv(csv_path)

    idx = np.random.randint(len(df))  # ランダムに1件
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


# ========= VISUALIZATION ==========
def visualize_traj(past, pred, gt, save_path):
    past = past.cpu().numpy()[0]  # (T_past, 2)
    pred = pred.cpu().numpy()[0]  # (T_fut,  2)
    gt   = gt.cpu().numpy()[0]    # (T_fut,  2)

    plt.figure(figsize=(5, 5), dpi=200)

    # past: blue
    plt.plot(past[:,0], past[:,1], "o-", color="blue", label="Past (30)")

    # ground truth: green
    plt.plot(gt[:,0], gt[:,1], "o-", color="green", label="GT Future (45)")

    # predicted: red
    plt.plot(pred[:,0], pred[:,1], "o-", color="red", label="Predicted (45)")

    plt.axhline(0, color="gray", linewidth=0.5)
    plt.axvline(0, color="gray", linewidth=0.5)

    plt.gca().invert_yaxis()  # 座標系を映像のようにする場合

    plt.legend()
    plt.title("Trajectory Prediction (Relative px coordinates)")
    plt.xlabel("x (px)")
    plt.ylabel("y (px)")
    plt.grid(True, alpha=0.4)

    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()
    print(f"Saved visualization → {save_path}")


# ========= MAIN ==========
def main():
    model = load_model()

    X, Y = load_sample(CSV_PATH)
    X, Y = X.to(device).float(), Y.to(device).float()

    with torch.no_grad():
        pred = model(X, tgt=None, tf_ratio=0.0, steps=H_FUT)

    past_xy = X[..., :2]  # (1,30,2)

    visualize_traj(past_xy, pred, Y, SAVE_IMG)


if __name__ == "__main__":
    main()
