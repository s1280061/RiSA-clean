# trajectory_predictor_module.py

import torch
import torch.nn as nn
import numpy as np

class LSTMModel(nn.Module):
    def __init__(self, input_size=2, hidden_size=64, output_len=15):
        super().__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_len * 2)
        self.output_len = output_len

    def forward(self, x):
        out, _ = self.lstm(x)
        last_hidden = out[:, -1, :]
        pred = self.fc(last_hidden)
        return pred.view(-1, self.output_len, 2)

def load_trained_model(model_path, device='cpu'):
    model = LSTMModel()
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.to(device)
    model.eval()
    return model


def predict_trajectory(model, past_trajectory, device='cpu', frame_size=None):
    """
    入力: past_trajectory: (10, 2) の np.array（過去10ステップの(x, y)）
    出力: (15, 2) の np.array（将来予測）
    """
    input_tensor = torch.tensor(past_trajectory, dtype=torch.float32).unsqueeze(0).to(device)

    with torch.no_grad():
        pred = model(input_tensor).cpu().numpy()[0]

    # === スケーリング調整（オプション）
    if frame_size is not None:
        fw, fh = frame_size
        pred *= np.array([fw, fh])  # [0,1] 正規化 → ピクセル座標に戻す

    # === デバッグ出力 ===
    print(f"[predict_trajectory] 入力 shape: {past_trajectory.shape}")
    print(f"[predict_trajectory] 出力 shape: {pred.shape}")
    print(f"[predict_trajectory] 予測値:\n{pred}")

    return pred



