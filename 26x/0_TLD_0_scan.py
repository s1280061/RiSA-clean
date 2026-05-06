import json
import numpy as np
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from sklearn.cluster import KMeans
from sentence_transformers import SentenceTransformer

# === JSONファイルのパス ===
json_path = r"C:\Users\s1280\Downloads\responses_sadm_72.json"

# === JSON読み込み ===
with open(json_path, "r", encoding="utf-8") as f:
    responses = json.load(f)

# === テキスト＆ラベル準備 ===
texts = []
labels = []  # 1=Slow down, 2=Change lanes
for r in responses:
    if r.get("ethics_answer"):
        texts.append(r["ethics_answer"])
        labels.append(r["ftr_choice"])  # 1 or 2

print(f"✅ Loaded {len(texts)} answers")

# === Sentence-BERTモデルで文章埋め込み ===
model = SentenceTransformer("all-MiniLM-L6-v2")  # 軽量＆高速
embeddings = model.encode(texts)

# === 次元圧縮（t-SNEで2次元化） ===
tsne = TSNE(n_components=2, random_state=42, perplexity=10)
emb_2d = tsne.fit_transform(embeddings)

# === 可視化 ===
plt.figure(figsize=(8,6))

for lbl, color, name in [(1, "blue", "Slow down"), (2, "red", "Change lanes")]:
    idx = [i for i, l in enumerate(labels) if l == lbl]
    plt.scatter(emb_2d[idx,0], emb_2d[idx,1], c=color, label=name, alpha=0.7)

plt.legend()
plt.title("t-SNE embedding of ethics_answer (Slow vs Change lanes)")
plt.xlabel("Dim 1")
plt.ylabel("Dim 2")
plt.tight_layout()
plt.savefig(r"C:\Users\s1280\Desktop\tsne_ethics_answers.png")
plt.show()

# === KMeansクラスタリング（参考） ===
kmeans = KMeans(n_clusters=2, random_state=42)
clusters = kmeans.fit_predict(embeddings)

# クラスタとFTRの一致率チェック
match = sum([1 for i,c in enumerate(clusters) if (labels[i]==2 and c==1) or (labels[i]==1 and c==0)]) / len(labels)
print(f"✅ KMeans cluster vs FTR match rate: {match:.2f}")

