import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt

# === パス設定（必要に応じて書き換えてください） ===
X_path = r"C:\Users\s1280\Desktop\SHRP2rawdata\3\LSTMdataset_new_1\lstm_X.npy"
Y_path = r"C:\Users\s1280\Desktop\SHRP2rawdata\3\LSTMdataset_new_1\lstm_Y.npy"
model_path = r"C:\Users\s1280\Desktop\SHRP2rawdata\3\LSTMdataset_new_1\lstm_trajectory_model.pth"

# === データ読み込み ===
X = np.load(X_path)
Y = np.load(Y_path)

# === モデル定義（保存済みと同じ構造にすること）
class LSTMModel(nn.Module):
    def __init__(self, input_size=2, hidden_size=64, output_len=15):
        super().__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_len * 2)
        self.output_len = output_len

    def forward(self, x):
        out, _ = self.lstm(x)
        last_hidden = out[:, -1, :]  # 最後の時刻の出力
        pred = self.fc(last_hidden)
        return pred.view(-1, self.output_len, 2)

# === モデル読み込み ===
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = LSTMModel().to(device)
model.load_state_dict(torch.load(model_path, map_location=device))
model.eval()

# === 1サンプル予測 ===
sample_idx = 0
x_sample = torch.tensor(X[sample_idx:sample_idx+1], dtype=torch.float32).to(device)
y_true = Y[sample_idx]

with torch.no_grad():
    y_pred = model(x_sample).cpu().numpy()[0]

# === 描画 ===
plt.figure(figsize=(6, 6))
plt.plot(X[sample_idx][:, 0], X[sample_idx][:, 1], 'bo-', label='Input (Past)')
plt.plot(y_true[:, 0], y_true[:, 1], 'go--', label='Ground Truth (Future)')
plt.plot(y_pred[:, 0], y_pred[:, 1], 'ro--', label='Predicted (Future)')
plt.legend()
plt.title("Trajectory Prediction Example")
plt.xlabel("x")
plt.ylabel("y")
plt.axis("equal")
plt.grid(True)
plt.tight_layout()
plt.show()

