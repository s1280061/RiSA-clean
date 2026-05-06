import os
import cv2
import torch
import clip
import numpy as np
from PIL import Image
from sklearn.cluster import KMeans
from tqdm import tqdm
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import shutil
import matplotlib.pyplot as plt
import pandas as pd

# === CLIPモデルロード ===
device = "cuda" if torch.cuda.is_available() else "cpu"
model, preprocess = clip.load("ViT-B/32", device=device)

# === 処理するベースパス ===
base_dir = r"C:\Users\s1280\Desktop\SHRP2rawdata"
sub_versions = ["3", "4", "5", "6"]  # 使いたいバージョン

# === キー画像があるフォルダ
frames = []
for ver in sub_versions:
    frame_dir = os.path.join(base_dir, "scene_keyframes", ver)
    if os.path.exists(frame_dir):
        imgs = [os.path.join(frame_dir, f) for f in os.listdir(frame_dir) if f.endswith(".jpg")]
        frames.extend(imgs)
        print(f"[FOUND] {len(imgs)} images from {frame_dir}")
    else:
        print(f"[WARN] {frame_dir} は存在しません")

print(f"✅ 合計 {len(frames)} 枚のキー画像が見つかりました")

# === CLIP埋め込み抽出 ===
embeddings = []
for fpath in tqdm(frames, desc="Extracting CLIP embeddings"):
    image = cv2.imread(fpath)
    if image is None:
        print(f"[SKIP] 読み込み失敗: {fpath}")
        continue

    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    pil_image = preprocess(Image.fromarray(image_rgb)).unsqueeze(0).to(device)

    with torch.no_grad():
        feat = model.encode_image(pil_image)
        feat = feat / feat.norm(dim=-1, keepdim=True)
    embeddings.append(feat.cpu().numpy()[0])

embeddings = np.array(embeddings)
print(f"✅ Embeddings shape: {embeddings.shape}")

# === KMeansクラスタリング ===
NUM_SCENES = 15
kmeans = KMeans(n_clusters=NUM_SCENES, random_state=42, n_init=10)
labels = kmeans.fit_predict(embeddings)
print(f"✅ {NUM_SCENES} クラスタに分類しました")

# === クラスタごとの代表フレーム抽出 ===
unique_frames = []
for cluster_id in range(NUM_SCENES):
    cluster_indices = np.where(labels == cluster_id)[0]
    center = kmeans.cluster_centers_[cluster_id]
    dists = np.linalg.norm(embeddings[cluster_indices] - center, axis=1)
    best_idx = cluster_indices[np.argmin(dists)]
    unique_frames.append(frames[best_idx])

# === t-SNE可視化 ===
pca = PCA(n_components=50)
pca_feats = pca.fit_transform(embeddings)
tsne = TSNE(n_components=2, random_state=42, perplexity=30)
tsne_feats = tsne.fit_transform(pca_feats)

plt.figure(figsize=(10,8))
plt.scatter(tsne_feats[:,0], tsne_feats[:,1], c=labels, cmap="tab20", s=8)
plt.colorbar(label="Cluster ID")
plt.title("t-SNE Visualization of Keyframe Scene Embeddings")
plt.show()

# ========================
# クラスタごとのフォルダ分け
# ========================
print("\n=== クラスタごとのフォルダ分け開始 ===")
cluster_group_dir = os.path.join(base_dir, "cluster_groups")
os.makedirs(cluster_group_dir, exist_ok=True)

cluster_counts = {}
detail_rows = []

for cluster_id in range(NUM_SCENES):
    cluster_indices = np.where(labels == cluster_id)[0]
    cluster_counts[cluster_id] = len(cluster_indices)

    # クラスタフォルダ
    cluster_dir = os.path.join(cluster_group_dir, f"cluster_{cluster_id}")
    os.makedirs(cluster_dir, exist_ok=True)

    for idx in cluster_indices:
        img_path = frames[idx]
        shutil.copy(img_path, cluster_dir)
        # 詳細リストに追加
        detail_rows.append({
            "cluster_id": cluster_id,
            "filename": os.path.basename(img_path),
            "original_path": img_path
        })

# クラスタサイズ集計を保存
summary_df = pd.DataFrame(list(cluster_counts.items()), columns=["cluster_id", "count"])
summary_csv_path = os.path.join(cluster_group_dir, "cluster_summary.csv")
summary_df.to_csv(summary_csv_path, index=False)

# クラスタ詳細リストを保存
detail_df = pd.DataFrame(detail_rows)
detail_csv_path = os.path.join(cluster_group_dir, "cluster_detail.csv")
detail_df.to_csv(detail_csv_path, index=False)

print(f"✅ クラスタごとフォルダ分け完了 → {cluster_group_dir}")
print(f"✅ クラスタ枚数集計CSV → {summary_csv_path}")
print(f"✅ クラスタ詳細CSV → {detail_csv_path}")

# ========================
# 代表フレームだけ集めるフォルダ
# ========================
unique_dir = os.path.join(base_dir, "unique_scenes_all")
os.makedirs(unique_dir, exist_ok=True)

for uf in unique_frames:
    shutil.copy(uf, unique_dir)

print(f"✅ 各クラスタの代表フレームを {unique_dir} に保存しました！")

