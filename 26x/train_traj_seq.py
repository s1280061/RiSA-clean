# train_traj_seq.py  ← 推論用の最小版
import torch
import torch.nn as nn

PAST_STEPS = 20
FUT_STEPS  = 45

class SeqModel(nn.Module):
    def __init__(self, rnn_type="gru", in_dim=5, hid=192, layers=2, fut_steps=FUT_STEPS, drop=0.1):
        super().__init__()
        rnn_cls = nn.GRU if rnn_type.lower()=="gru" else nn.LSTM
        self.rnn = rnn_cls(input_size=in_dim, hidden_size=hid, num_layers=layers,
                           batch_first=True, dropout=drop if layers>1 else 0.0)
        self.head = nn.Sequential(
            nn.Linear(hid, 256),
            nn.ReLU(),
            nn.Linear(256, fut_steps*2)
        )
        self.fut_steps = fut_steps
    def forward(self, x):
        h, _ = self.rnn(x)            # [B,T,H]
        h_last = h[:,-1,:]            # [B,H]
        out = self.head(h_last)       # [B, 2*F]
        return out.view(-1, self.fut_steps, 2)
