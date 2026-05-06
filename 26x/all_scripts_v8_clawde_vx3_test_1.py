import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import os
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

# === ローカルパス設定 ===
X_path = r"C:\Users\s1280\Desktop\SHRP2rawdata\3\LSTMdataset_new_1\lstm_X.npy"
Y_path = r"C:\Users\s1280\Desktop\SHRP2rawdata\3\LSTMdataset_new_1\lstm_Y.npy"
output_model_path = r"C:\Users\s1280\Desktop\SHRP2rawdata\3\LSTMdataset_new_1\lstm_trajectory_model.pth"

# === 学習パラメータ ===
batch_size = 64
epochs = 30
learning_rate = 0.001
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# === データ読み込み ===
X = np.load(X_path)
Y = np.load(Y_path)

# === Dataset定義 ===
class TrajectoryDataset(Dataset):
    def __init__(self, X, Y):
        self.X = torch.tensor(X, dtype=torch.float32)
        self.Y = torch.tensor(Y, dtype=torch.float32)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.Y[idx]

# === train/val 分割
X_train, X_val, Y_train, Y_val = train_test_split(X, Y, test_size=0.2, random_state=42)
train_loader = DataLoader(TrajectoryDataset(X_train, Y_train), batch_size=batch_size, shuffle=True)
val_loader = DataLoader(TrajectoryDataset(X_val, Y_val), batch_size=batch_size)

# === モデル定義（LSTM → Linear）
class LSTMModel(nn.Module):
    def __init__(self, input_size=2, hidden_size=64, output_len=15):
        super().__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_len * 2)
        self.output_len = output_len

    def forward(self, x):
        out, _ = self.lstm(x)         # (B, 10, H)
        last_hidden = out[:, -1, :]   # (B, H)
        pred = self.fc(last_hidden)   # (B, 30)
        return pred.view(-1, self.output_len, 2)

# === 学習ループ
model = LSTMModel().to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
criterion = nn.MSELoss()

for epoch in range(epochs):
    model.train()
    train_loss = 0.0
    for x_batch, y_batch in train_loader:
        x_batch, y_batch = x_batch.to(device), y_batch.to(device)
        pred = model(x_batch)
        loss = criterion(pred, y_batch)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        train_loss += loss.item() * x_batch.size(0)

    train_loss /= len(train_loader.dataset)

    # Validation
    model.eval()
    val_loss = 0.0
    with torch.no_grad():
        for x_batch, y_batch in val_loader:
            x_batch, y_batch = x_batch.to(device), y_batch.to(device)
            pred = model(x_batch)
            loss = criterion(pred, y_batch)
            val_loss += loss.item() * x_batch.size(0)
    val_loss /= len(val_loader.dataset)

    print(f"Epoch {epoch+1}/{epochs} | Train Loss: {train_loss:.6f} | Val Loss: {val_loss:.6f}")

# === モデル保存
torch.save(model.state_dict(), output_model_path)
print(f"✅ モデル保存完了: {output_model_path}")

