import os
import numpy as np
from sklearn.cluster import KMeans
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
from PIL import Image
import shutil
from tqdm import tqdm

# ===============================
# ① CLIP特徴とパスを読み込み
# ===============================
embeddings = np.load(r"D:\JAAD_collages\jaad_clip_embeddings.npy")
paths = np.load(r"D:\JAAD_collages\jaad_image_paths.npy", allow_pickle=True)

# ===============================
# ② K-meansクラスタリング
# ===============================
n_clusters = 10  # クラスタ数（適宜調整）
print(f"Clustering into {n_clusters} groups...")
kmeans = KMeans(n_clusters=n_clusters, random_state=42)
labels = kmeans.fit_predict(embeddings)
print("✅ K-means done.")

# ===============================
# ③ t-SNEで可視化（全体の分布を確認）
# ===============================
print("Computing t-SNE (this may take a few minutes)...")
tsne = TSNE(n_components=2, random_state=42, perplexity=30)
embeddings_2d = tsne.fit_transform(embeddings)

plt.figure(figsize=(10, 8))
plt.scatter(embeddings_2d[:, 0], embeddings_2d[:, 1], c=labels, cmap='tab10', s=10)
plt.title("CLIP Feature Clustering (JAAD)")
plt.xlabel("t-SNE 1")
plt.ylabel("t-SNE 2")
plt.show()

# ===============================
# ④ クラスタごとに画像をフォルダに分類
# ===============================
output_root = r"D:\JAAD_collages\cluster_results"
os.makedirs(output_root, exist_ok=True)

print("Saving cluster images to folders...")
for cluster_id in range(n_clusters):
    cluster_dir = os.path.join(output_root, f"cluster_{cluster_id}")
    os.makedirs(cluster_dir, exist_ok=True)

    # 現在のクラスタに属する画像インデックス
    idx = np.where(labels == cluster_id)[0]

    # 200枚など制限をかけたい場合（全部コピーすると時間がかかる）
    # idx = idx[:200]

    for i in tqdm(idx, desc=f"Cluster {cluster_id}"):
        src = paths[i]
        fname = os.path.basename(src)
        dst = os.path.join(cluster_dir, fname)
        try:
            shutil.copy(src, dst)
        except Exception as e:
            print(f"Error copying {src}: {e}")

print("✅ All clusters saved to:", output_root)

# ===============================
# ⑤ クラスタ結果の保存（再利用用）
# ===============================
np.save(os.path.join(output_root, "cluster_labels.npy"), labels)
np.save(os.path.join(output_root, "tsne_embeddings.npy"), embeddings_2d)
print("✅ Saved cluster labels and t-SNE embeddings.")
