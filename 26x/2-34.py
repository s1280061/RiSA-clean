# 事前に pip install torchview graphviz が必要です
from torchview import draw_graph
import torch

# モデルのインスタンス化（学習済み重みは不要）
model = Seq2Seq(input_size=5, hidden=192, n_layers=2, out_size=2)

# ダミー入力の作成
batch_size = 1
h_past = 30
h_fut = 45
input_data = torch.randn(batch_size, h_past, 5)
target_data = torch.randn(batch_size, h_fut, 2) # Teacher Forcing用

# グラフの描画と保存
# forwardの引数に合わせて入力を渡します
model_graph = draw_graph(model, input_data=input_data, tgt=target_data, tf_ratio=0.5, steps=h_fut,
                         graph_name="Seq2Seq_GRU_Model",
                         expand_nested=True, # サブモジュールを展開表示
                         save_graph=True) # 画像ファイルとして保存
model_graph.visual_graph