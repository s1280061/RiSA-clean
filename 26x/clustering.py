# -*- coding: utf-8 -*-
import os
import json
import cv2
import torch
import clip
import joblib
import random
import shutil
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from PIL import Image
from tqdm import tqdm
from datetime import datetime
from collections import defaultdict
from matplotlib.offsetbox import OffsetImage, AnnotationBbox

from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.metrics import (
    silhouette_score,
    davies_bouldin_score,
    calinski_harabasz_score
)

# =========================
# 再現性確保：乱数固定
# =========================
SEED = 42
def set_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

set_seed(SEED)

# =========================
# 設定
# =========================
CLIP_MODEL_NAME = "ViT-B/16"
NUM_SCENES = 10                # KMeansクラスタ数
PCA_DIM = 50                   # t-SNE前の次元圧縮
TSNE_BASE_PERPLEXITY = 30      # サンプル数に応じて自動調整
EVAL_K_RANGE = list(range(5, 51, 5))  # 評価用のk候補

device = "cuda" if torch.cuda.is_available() else "cpu"
model, preprocess = clip.load(CLIP_MODEL_NAME, device=device)

# === 対象となるcentral_framesフォルダ ===
frame_dirs = [
    r"C:\Users\s1280\Desktop\SHRP2rawdata\3\new_divided\central_frames",
    r"C:\Users\s1280\Desktop\SHRP2rawdata\4\new_divided\central_frames",
    r"C:\Users\s1280\Desktop\SHRP2rawdata\5\new_divided\central_frames",
    r"C:\Users\s1280\Desktop\SHRP2rawdata\6\new_divided\central_frames"
]

# === 出力先 ===
save_base_dir = r"C:\Users\s1280\Desktop\SHRP2rawdata\central_frames_clustering_v7"
os.makedirs(save_base_dir, exist_ok=True)
artifacts_dir = os.path.join(save_base_dir, "artifacts")
plots_dir = os.path.join(save_base_dir, "plots")
clustered_dir = os.path.join(save_base_dir, "clustered_images")
for d in [artifacts_dir, plots_dir, clustered_dir]:
    os.makedirs(d, exist_ok=True)

# =========================
# 実験01: 二値＋conf の環境フラグ定義
# =========================
# 4項目：is_wet_surface / is_snowy_surface / is_dry_surface / is_low_visibility
GT_FLAGS = [
    "is_wet_surface",
    "is_snowy_surface",
    "is_dry_surface",
    "is_low_visibility",
]

def gt_columns():
    """
    GT CSVに入れる列順を返す。
    各フラグは yes(0/1) と conf(0-100) の2列。
    最後に annotator / notes を付与。
    """
    cols = []
    for k in GT_FLAGS:
        cols += [f"{k}_yes", f"{k}_conf"]
    cols += ["annotator", "notes"]
    return cols

# =========================
# 画像収集
# =========================
frames = []
for frame_dir in frame_dirs:
    if os.path.exists(frame_dir):
        imgs = [os.path.join(frame_dir, f)
                for f in os.listdir(frame_dir)
                if f.lower().endswith(".jpg")]
        # 並び安定化（任意）
        imgs = sorted(imgs, key=lambda p: os.path.basename(p).lower())
        frames.extend(imgs)
        print(f"[FOUND] {len(imgs)} images from {frame_dir}")
    else:
        print(f"[SKIP] {frame_dir} は存在しません")

print(f"✅ 合計 {len(frames)} 枚の central_frames が見つかりました")
if len(frames) == 0:
    raise SystemExit("画像が見つからないため終了します。")

# =========================
# CLIP埋め込み抽出
# =========================
embeddings = []
valid_frames = []  # 読み込み成功したフレームのみ対応させる
for fpath in tqdm(frames, desc="Extracting CLIP embeddings"):
    image = cv2.imread(fpath)
    if image is None:
        print(f"[WARN] 読み込み失敗: {fpath}")
        continue
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    pil_image = preprocess(Image.fromarray(image_rgb)).unsqueeze(0).to(device)
    with torch.no_grad():
        feat = model.encode_image(pil_image)
        feat = feat / feat.norm(dim=-1, keepdim=True)
    embeddings.append(feat.cpu().numpy()[0])
    valid_frames.append(fpath)

embeddings = np.array(embeddings, dtype=np.float32)
print(f"✅ Embeddings shape: {embeddings.shape}")

if embeddings.shape[0] < 2:
    raise SystemExit("有効な埋め込みが2枚未満のためクラスタリングできません。")

# 保存：埋め込みとフレーム対応
np.save(os.path.join(artifacts_dir, "embeddings.npy"), embeddings)
pd.DataFrame({
    "index": np.arange(len(valid_frames), dtype=int),
    "path": valid_frames
}).to_csv(os.path.join(artifacts_dir, "frames_list.csv"), index=False, encoding="utf-8-sig")

# CSVで埋め込み（重い場合はnpyのみでOK）
emb_df = pd.DataFrame(embeddings, columns=[f"e{i+1}" for i in range(embeddings.shape[1])])
emb_df.insert(0, "path", valid_frames)
emb_df.to_csv(os.path.join(artifacts_dir, "embeddings.csv"), index=False, encoding="utf-8-sig")

# =========================
# KMeansクラスタリング
# =========================
kmeans = KMeans(n_clusters=NUM_SCENES, random_state=SEED, n_init=10)
labels = kmeans.fit_predict(embeddings)
print(f"✅ {NUM_SCENES} クラスタに分類しました")

# 保存：KMeansモデル・ラベル・クラスタ中心
joblib.dump(kmeans, os.path.join(artifacts_dir, "kmeans_model.pkl"))
np.save(os.path.join(artifacts_dir, "kmeans_centers.npy"), kmeans.cluster_centers_)
np.save(os.path.join(artifacts_dir, "labels.npy"), labels)

assign_df = pd.DataFrame({
    "path": valid_frames,
    "cluster_id": labels
})
assign_df.to_csv(os.path.join(artifacts_dir, "cluster_assignments.csv"), index=False, encoding="utf-8-sig")

# =========================
# クラスタ品質評価
# =========================
if len(np.unique(labels)) > 1:
    sil_score = silhouette_score(embeddings, labels)
else:
    sil_score = np.nan
db_score = davies_bouldin_score(embeddings, labels)
ch_score = calinski_harabasz_score(embeddings, labels)

print(f"Silhouette Score: {sil_score:.3f}  (1に近いほど良い)")
print(f"Davies-Bouldin Index: {db_score:.3f}  (小さいほど良い)")
print(f"Calinski-Harabasz: {ch_score:.3f}  (大きいほど良い)")

# =========================
# 代表フレーム抽出
# =========================
unique_frames = []
for cluster_id in range(NUM_SCENES):
    cluster_indices = np.where(labels == cluster_id)[0]
    center = kmeans.cluster_centers_[cluster_id]
    dists = np.linalg.norm(embeddings[cluster_indices] - center, axis=1)
    best_idx = cluster_indices[np.argmin(dists)]
    unique_frames.append(valid_frames[best_idx])

pd.DataFrame({"cluster_id": list(range(NUM_SCENES)),
              "repr_path": unique_frames}).to_csv(
    os.path.join(artifacts_dir, "representative_frames.csv"), index=False, encoding="utf-8-sig"
)

# =========================
# t-SNE（PCA→t-SNE）
# =========================
pca = PCA(n_components=min(PCA_DIM, embeddings.shape[1]), random_state=SEED)
pca_feats = pca.fit_transform(embeddings)
np.save(os.path.join(artifacts_dir, "pca_features.npy"), pca_feats)
joblib.dump(pca, os.path.join(artifacts_dir, "pca_model.pkl"))

# perplexityは(サンプル数-1)/3 を上限に自動調整
n_samples = pca_feats.shape[0]
max_perp = max(5, (n_samples - 1) // 3)
perplexity = int(min(TSNE_BASE_PERPLEXITY, max_perp))
if perplexity < 5:
    perplexity = 5  # 下限

tsne = TSNE(n_components=2, random_state=SEED, perplexity=perplexity, init="pca")
tsne_feats = tsne.fit_transform(pca_feats)
np.save(os.path.join(artifacts_dir, "tsne_features.npy"), tsne_feats)

# =========================
# 可視化：t-SNE散布図
# =========================
def save_tsne_scatter(tsne_feats, labels, save_path):
    plt.figure(figsize=(10, 8))
    plt.scatter(tsne_feats[:, 0], tsne_feats[:, 1], c=labels, cmap="tab20", s=8)
    plt.colorbar(label="Cluster ID")
    plt.title("t-SNE Visualization of Central Frame Scene Embeddings")
    plt.savefig(save_path, dpi=200, bbox_inches="tight")
    plt.close()
    print(f"✅ t-SNE散布図を保存しました → {save_path}")

save_tsne_scatter(tsne_feats, labels, os.path.join(plots_dir, "tsne_clusters.png"))

# =========================
# kを変えた評価（Elbow/Silhouette/DB/CH）
# =========================
def save_cluster_eval_graph(embeddings, save_dir, ks):
    inertia_values, silhouette_scores, db_scores, ch_scores = [], [], [], []
    rows = []
    print("\n=== クラスタ数ごとの評価 ===")
    for n_clusters in ks:
        print(f"→ k={n_clusters} クラスタ計算中…")
        km = KMeans(n_clusters=n_clusters, random_state=SEED, n_init=10)
        lab_k = km.fit_predict(embeddings)
        inertia_values.append(km.inertia_)
        sil = silhouette_score(embeddings, lab_k) if len(np.unique(lab_k)) > 1 else np.nan
        db  = davies_bouldin_score(embeddings, lab_k)
        ch  = calinski_harabasz_score(embeddings, lab_k)
        silhouette_scores.append(sil)
        db_scores.append(db)
        ch_scores.append(ch)
        rows.append({"k": n_clusters, "inertia": km.inertia_, "silhouette": sil, "davies_bouldin": db, "calinski_harabasz": ch})

    # 保存：CSV
    pd.DataFrame(rows).to_csv(os.path.join(save_dir, "cluster_eval_metrics.csv"),
                              index=False, encoding="utf-8-sig")

    # 保存：図
    fig, axs = plt.subplots(4, 1, figsize=(8, 20))
    axs[0].plot(ks, inertia_values, 'r-o'); axs[0].set_title('Elbow Method (Inertia)')
    axs[1].plot(ks, silhouette_scores, 'b-o'); axs[1].set_title('Silhouette Score (higher better)')
    axs[2].plot(ks, db_scores, 'g-o'); axs[2].set_title('Davies-Bouldin Index (lower better)')
    axs[3].plot(ks, ch_scores, 'm-o'); axs[3].set_title('Calinski-Harabasz (higher better)')
    for ax in axs:
        ax.grid(True)
    for ax in axs:
        ax.set_xlabel('Number of clusters (k)')
    plt.tight_layout()
    save_path = os.path.join(save_dir, "cluster_evaluation_metrics.png")
    plt.savefig(save_path, dpi=200)
    plt.close()
    print(f"✅ クラスタ評価グラフを保存しました → {save_path}")

save_cluster_eval_graph(embeddings, plots_dir, EVAL_K_RANGE)

# =========================
# t-SNE散布図に代表クラスタ画像を配置
# =========================
def plot_tsne_with_cluster_images(tsne_feats, labels, unique_frames, save_path):
    fig, ax = plt.subplots(figsize=(12, 10))
    scatter = ax.scatter(tsne_feats[:, 0], tsne_feats[:, 1],
                         c=labels, cmap="tab20", s=8)
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

    plt.savefig(save_path, dpi=200, bbox_inches="tight")
    plt.close()
    print(f"✅ クラスタ画像付き散布図を保存しました → {save_path}")

plot_tsne_with_cluster_images(
    tsne_feats, labels, unique_frames,
    save_path=os.path.join(plots_dir, "tsne_with_cluster_images.png")
)

# =========================
# クラスタごとに画像をコピー（元名を維持）
# =========================
def save_images_by_cluster(frames, labels, save_dir):
    # クラスタごとに元パスを集約
    groups = defaultdict(list)
    for p, c in zip(frames, labels):
        groups[int(c)].append(p)

    # 出力先を作り直し
    if os.path.exists(save_dir):
        shutil.rmtree(save_dir)
    os.makedirs(save_dir, exist_ok=True)

    assign_rows = []

    print("\n=== クラスタごとに画像をコピー中（元ファイル名を維持） ===")
    for cid in sorted(groups.keys()):
        cluster_dir = os.path.join(save_dir, f"cluster_{cid:02d}")
        os.makedirs(cluster_dir, exist_ok=True)

        # クラスタ内はファイル名で安定ソート
        srcs = sorted(groups[cid], key=lambda p: os.path.basename(p).lower())

        # 同名衝突対策
        seen = defaultdict(int)

        for src in tqdm(srcs, desc=f"cluster_{cid:02d}"):
            base = os.path.basename(src)
            name, ext = os.path.splitext(base)

            candidate = base
            dst_path = os.path.join(cluster_dir, candidate)
            if os.path.exists(dst_path):
                seen[name] += 1
                candidate = f"{name}_dup{seen[name]}{ext}"
                dst_path = os.path.join(cluster_dir, candidate)

            shutil.copy2(src, dst_path)

            assign_rows.append({
                "original_path": src,
                "saved_path": dst_path,
                "filename": candidate,
                "cluster_id": cid
            })

    # ====== CSV保存（通常版） ======
    df = pd.DataFrame(assign_rows)
    csv_path = os.path.join(save_dir, "cluster_assignments.csv")
    df.to_csv(csv_path, index=False, encoding="utf-8-sig")

    # ====== GTテンプレCSV（実験01：4フラグ yes/conf） ======
    df_gt = df.sort_values(["cluster_id", "filename"]).reset_index(drop=True).copy()
    # 空のGT列を追加
    for col in gt_columns():
        df_gt[col] = ""

    # 列順（見やすさ用）
    cols = ["cluster_id", "filename", "saved_path", "original_path"] + gt_columns()
    df_gt = df_gt[cols]

    csv_path_gt = os.path.join(save_dir, "cluster_assignments_gt.csv")
    df_gt.to_csv(csv_path_gt, index=False, encoding="utf-8-sig")

    print(f"\n✅ クラスタ別画像保存完了 → {save_dir}")
    print(f"✅ クラスタ割り当てCSVを保存しました → {csv_path}")
    print(f"✅ GT用テンプレCSVを保存しました → {csv_path_gt}")

save_images_by_cluster(valid_frames, labels, clustered_dir)

# =========================
# 実行構成の保存（再現用）
# =========================
run_config = {
    "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
    "seed": SEED,
    "device": device,
    "clip_model": CLIP_MODEL_NAME,
    "num_scenes": NUM_SCENES,
    "pca_dim": PCA_DIM,
    "tsne_perplexity_used": perplexity,
    "frame_dirs": frame_dirs,
    "save_base_dir": save_base_dir,
    "counts": {
        "total_found_images": len(frames),
        "valid_images": len(valid_frames)
    }
}
with open(os.path.join(artifacts_dir, "run_config.json"), "w", encoding="utf-8") as f:
    json.dump(run_config, f, ensure_ascii=False, indent=2)

print("\n🎉 完了！ 主要成果物は以下に保存されました：")
print(f"- 埋め込み: {os.path.join(artifacts_dir, 'embeddings.npy / embeddings.csv')}")
print(f"- フレーム一覧: {os.path.join(artifacts_dir, 'frames_list.csv')}")
print(f"- KMeans: {os.path.join(artifacts_dir, 'kmeans_model.pkl')}")
print(f"- ラベル: {os.path.join(artifacts_dir, 'labels.npy')}")
print(f"- 代表フレーム: {os.path.join(artifacts_dir, 'representative_frames.csv')}")
print(f"- PCA特徴/モデル: {os.path.join(artifacts_dir, 'pca_features.npy')} / pca_model.pkl")
print(f"- t-SNE特徴: {os.path.join(artifacts_dir, 'tsne_features.npy')}")
print(f"- 可視化: {os.path.join(plots_dir, 'tsne_clusters.png')}, {os.path.join(plots_dir, 'tsne_with_cluster_images.png')}, {os.path.join(plots_dir, 'cluster_evaluation_metrics.png')}")
print(f"- クラスタ別コピー: {clustered_dir}")
print(f"- 実行構成: {os.path.join(artifacts_dir, 'run_config.json')}")

# =========================
# （任意）簡易バリデータ
# =========================
def validate_gt_csv(csv_path: str):
    """
    ルール:
      - *_yes は {0,1,空}、*_conf は [0..100] or 空
      - dry=1 の場合、wet=0 かつ snowy=0 を推奨（矛盾があれば警告）
    """
    df = pd.read_csv(csv_path)
    ok = True
    for idx, row in df.iterrows():
        for k in GT_FLAGS:
            ycol, ccol = f"{k}_yes", f"{k}_conf"
            yv = row.get(ycol, "")
            cv = row.get(ccol, "")
            if yv != "" and yv not in (0, 1):
                print(f"[row {idx}] {ycol} must be 0/1/empty -> got {yv}")
                ok = False
            try:
                if cv != "" and not (0 <= float(cv) <= 100):
                    print(f"[row {idx}] {ccol} must be 0..100/empty -> got {cv}")
                    ok = False
            except Exception:
                print(f"[row {idx}] {ccol} must be numeric/empty -> got {cv}")
                ok = False

        dry = row.get("is_dry_surface_yes", "")
        wet = row.get("is_wet_surface_yes", "")
        snow = row.get("is_snowy_surface_yes", "")
        if dry == 1 and (wet == 1 or snow == 1):
            print(f"[row {idx}] dry=1 with wet/snowy=1 (check)")

    print("✅ GT CSV validation:", "PASS" if ok else "ISSUES FOUND")
    return ok
