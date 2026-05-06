import os
import cv2
import torch
import clip
import numpy as np
from PIL import Image
from tqdm import tqdm
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.metrics import (
    silhouette_score,
    davies_bouldin_score,
    calinski_harabasz_score
)
import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.offsetbox import OffsetImage, AnnotationBbox
import shutil

# === CLIPモデルロード ===
device = "cuda" if torch.cuda.is_available() else "cpu"
model, preprocess = clip.load("ViT-B/16", device=device)

# === 対象となるcentral_framesフォルダ ===
frame_dirs = [
    r"C:\Users\s1280\Desktop\SHRP2rawdata\3\new_divided\central_frames",
    r"C:\Users\s1280\Desktop\SHRP2rawdata\4\new_divided\central_frames",
    r"C:\Users\s1280\Desktop\SHRP2rawdata\5\new_divided\central_frames",
    r"C:\Users\s1280\Desktop\SHRP2rawdata\6\new_divided\central_frames"
]

# === 画像収集 ===
frames = []
for frame_dir in frame_dirs:
    if os.path.exists(frame_dir):
        imgs = [os.path.join(frame_dir, f) for f in os.listdir(frame_dir) if f.endswith(".jpg")]
        frames.extend(imgs)
        print(f"[FOUND] {len(imgs)} images from {frame_dir}")
    else:
        print(f"[SKIP] {frame_dir} は存在しません")

print(f"✅ 合計 {len(frames)} 枚の central_frames が見つかりました")

# === CLIP埋め込み抽出 ===
embeddings = []
for fpath in tqdm(frames, desc="Extracting CLIP embeddings"):
    image = cv2.imread(fpath)
    if image is None:
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
NUM_SCENES = 10  # クラスタ数（必要に応じて調整可）
kmeans = KMeans(n_clusters=NUM_SCENES, random_state=42, n_init=10)
labels = kmeans.fit_predict(embeddings)
print(f"✅ {NUM_SCENES} クラスタに分類しました")

# === クラスタ品質評価 ===
sil_score = silhouette_score(embeddings, labels)
db_score = davies_bouldin_score(embeddings, labels)
print(f"Silhouette Score: {sil_score:.3f}  (1に近いほど良い)")
print(f"Davies-Bouldin Index: {db_score:.3f}  (小さいほど良い)")

# === クラスタ代表フレーム抽出 ===
unique_frames = []
for cluster_id in range(NUM_SCENES):
    cluster_indices = np.where(labels == cluster_id)[0]
    center = kmeans.cluster_centers_[cluster_id]
    dists = np.linalg.norm(embeddings[cluster_indices] - center, axis=1)
    best_idx = cluster_indices[np.argmin(dists)]
    unique_frames.append(frames[best_idx])

# === t-SNE次元削減 ===
pca = PCA(n_components=50)
pca_feats = pca.fit_transform(embeddings)
tsne = TSNE(n_components=2, random_state=42, perplexity=30)
tsne_feats = tsne.fit_transform(pca_feats)

# === 保存先フォルダ ===
save_base_dir = r"C:\Users\s1280\Desktop\SHRP2rawdata\central_frames_clustering"
os.makedirs(save_base_dir, exist_ok=True)

# === t-SNE散布図を保存 ===
def save_tsne_scatter(tsne_feats, labels, save_path):
    plt.figure(figsize=(10, 8))
    plt.scatter(tsne_feats[:, 0], tsne_feats[:, 1], c=labels, cmap="tab20", s=8)
    plt.colorbar(label="Cluster ID")
    plt.title("t-SNE Visualization of Central Frame Scene Embeddings")
    plt.savefig(save_path, dpi=200, bbox_inches="tight")
    plt.close()
    print(f"✅ t-SNE散布図を保存しました → {save_path}")

save_tsne_scatter(tsne_feats, labels, os.path.join(save_base_dir, "tsne_clusters.png"))

# === クラスタ評価グラフ ===
def save_cluster_eval_graph(embeddings, save_dir):
    clusters_range = range(5, 51, 5)
    inertia_values, silhouette_scores, db_scores, ch_scores = [], [], [], []

    print("\n=== クラスタ数ごとの評価 ===")
    for n_clusters in clusters_range:
        print(f"→ k={n_clusters} クラスタ計算中…")
        kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
        labels_k = kmeans.fit_predict(embeddings)
        inertia_values.append(kmeans.inertia_)
        silhouette_scores.append(silhouette_score(embeddings, labels_k))
        db_scores.append(davies_bouldin_score(embeddings, labels_k))
        ch_scores.append(calinski_harabasz_score(embeddings, labels_k))

    fig, axs = plt.subplots(4, 1, figsize=(8, 20))
    axs[0].plot(clusters_range, inertia_values, 'r-o'); axs[0].set_title('Elbow Method (Inertia)')
    axs[1].plot(clusters_range, silhouette_scores, 'b-o'); axs[1].set_title('Silhouette Score (higher better)')
    axs[2].plot(clusters_range, db_scores, 'g-o'); axs[2].set_title('Davies-Bouldin Index (lower better)')
    axs[3].plot(clusters_range, ch_scores, 'm-o'); axs[3].set_title('Calinski-Harabasz (higher better)')
    for ax in axs:
        ax.grid(True)
        ax.set_xlabel('Number of clusters (k)')
    plt.tight_layout()

    save_path = os.path.join(save_dir, "cluster_evaluation_metrics.png")
    plt.savefig(save_path, dpi=200)
    plt.close()
    print(f"✅ クラスタ評価グラフを保存しました → {save_path}")

save_cluster_eval_graph(embeddings, save_base_dir)

# === t-SNE散布図に代表クラスタ画像を追加 ===
def plot_tsne_with_cluster_images(tsne_feats, labels, unique_frames, save_path):
    fig, ax = plt.subplots(figsize=(12, 10))
    scatter = ax.scatter(tsne_feats[:, 0], tsne_feats[:, 1], c=labels, cmap="tab20", s=8)
    plt.colorbar(scatter, label="Cluster ID")
    plt.title("t-SNE with Representative Cluster Images")

    cluster_ids = sorted(np.unique(labels))
    for cid, uf in zip(cluster_ids, unique_frames):
        cluster_idx = np.where(labels == cid)[0]
        cluster_center = tsne_feats[cluster_idx].mean(axis=0)
        img = Image.open(uf)
        img.thumbnail((50, 50))
        imagebox = OffsetImage(img, zoom=0.8)
        ab = AnnotationBbox(imagebox, cluster_center, frameon=True, pad=0.3)
        ax.add_artist(ab)
        ax.text(cluster_center[0], cluster_center[1] + 5, f"Cluster {cid}", fontsize=10, ha="center")
    plt.savefig(save_path, dpi=200, bbox_inches="tight")
    plt.close()
    print(f"✅ クラスタ画像付き散布図を保存しました → {save_path}")

plot_tsne_with_cluster_images(
    tsne_feats, labels, unique_frames,
    save_path=os.path.join(save_base_dir, "tsne_with_cluster_images.png")
)

# === クラスタごとに画像を分けて保存 ===
def save_images_by_cluster(frames, labels, save_dir):
    cluster_base_dir = os.path.join(save_dir, "clustered_images")
    if os.path.exists(cluster_base_dir):
        shutil.rmtree(cluster_base_dir)
    os.makedirs(cluster_base_dir, exist_ok=True)

    unique_labels = sorted(np.unique(labels))
    cluster_counts = {}

    print("\n=== クラスタごとに画像をコピー中 ===")
    for cluster_id in unique_labels:
        cluster_dir = os.path.join(cluster_base_dir, f"cluster_{cluster_id:02d}")
        os.makedirs(cluster_dir, exist_ok=True)
        cluster_counts[cluster_id] = 0

    for i, (frame_path, cluster_id) in enumerate(tqdm(zip(frames, labels), desc="Copying images")):
        if not os.path.exists(frame_path):
            continue
        original_filename = os.path.basename(frame_path)
        name, ext = os.path.splitext(original_filename)
        cluster_dir = os.path.join(cluster_base_dir, f"cluster_{cluster_id:02d}")
        new_filename = f"{name}_cluster{cluster_id:02d}_{cluster_counts[cluster_id]:04d}{ext}"
        dest_path = os.path.join(cluster_dir, new_filename)
        try:
            shutil.copy2(frame_path, dest_path)
            cluster_counts[cluster_id] += 1
        except Exception as e:
            print(f"⚠️ コピー失敗: {frame_path} → {e}")

    print(f"\n✅ クラスタ別画像保存完了 → {cluster_base_dir}")
    print("=== クラスタ別画像数 ===")
    total_saved = 0
    for cluster_id in unique_labels:
        count = cluster_counts[cluster_id]
        total_saved += count
        print(f"Cluster {cluster_id:2d}: {count:4d} 枚")
    print(f"合計保存数: {total_saved} 枚")

    # CSVにクラスタ情報を保存
    cluster_info = []
    for i, (frame_path, cluster_id) in enumerate(zip(frames, labels)):
        cluster_info.append({
            'original_path': frame_path,
            'cluster_id': cluster_id,
            'filename': os.path.basename(frame_path)
        })
    df = pd.DataFrame(cluster_info)
    csv_path = os.path.join(cluster_base_dir, "cluster_assignments.csv")
    df.to_csv(csv_path, index=False, encoding='utf-8-sig')
    print(f"✅ クラスタ割り当てCSVを保存しました → {csv_path}")

save_images_by_cluster(frames, labels, save_base_dir)