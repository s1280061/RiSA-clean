import os
import torch
import clip
from PIL import Image
import numpy as np
from tqdm import tqdm
Image.MAX_IMAGE_PIXELS = None  # Pillowのサイズ制限を解除

device = "cuda" if torch.cuda.is_available() else "cpu"
model, preprocess = clip.load("ViT-B/32", device=device)

base_dir = r"D:\JAAD_crops_collages"
embeddings = []
paths = []

for root, _, files in os.walk(base_dir):
    for f in files:
        if f.lower().endswith(('.png', '.jpg', '.jpeg')):
            path = os.path.join(root, f)
            img = preprocess(Image.open(path).convert("RGB")).unsqueeze(0).to(device)
            with torch.no_grad():
                feat = model.encode_image(img)
                feat = feat / feat.norm(dim=-1, keepdim=True)  # 正規化
            embeddings.append(feat.cpu().numpy())
            paths.append(path)

embeddings = np.vstack(embeddings)
np.save("jaad_clip_embeddings.npy", embeddings)
np.save("jaad_image_paths.npy", np.array(paths))
