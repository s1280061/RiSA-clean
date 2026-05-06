import os, io, base64, re, requests, json, torch
from PIL import Image

# === LLaVAローカルAPI関数 ===
def ask_llava_yesno(image_path, question):
    img = Image.open(image_path).convert("RGB").resize((224, 224))
    buffer = io.BytesIO()
    img.save(buffer, format="JPEG", quality=40)
    img_b64 = base64.b64encode(buffer.getvalue()).decode("utf-8")

    prompt_text = f"{question} Answer only Yes or No."
    payload = {
        "model": "llava:latest",
        "prompt": prompt_text,
        "images": [img_b64],
        "stream": False,
        "options": {"temperature": 0.0}
    }

    response = requests.post("http://localhost:11434/api/generate", json=payload)
    raw_text = response.text

    match = re.search(r'"response"\s*:\s*"([^"]+)"', raw_text)
    if match:
        answer = match.group(1).strip().lower()
        if "yes" in answer:
            return "Yes"
        elif "no" in answer:
            return "No"
    return "Unknown"

# === GPUメモリ使用量表示関数 ===
def print_gpu_memory(prefix=""):
    if torch.cuda.is_available():
        mem = torch.cuda.memory_allocated() / (1024 ** 2)
        mem_reserved = torch.cuda.memory_reserved() / (1024 ** 2)
        print(f"{prefix} GPU Memory: allocated={mem:.1f} MB, reserved={mem_reserved:.1f} MB")

# === コンテキストリスト（24個） ===
contexts = [
    "daytime", "night-time", "twilight", "sunny", "rainy", "snowy", "foggy", "dust/sandstorm",
    "trees overhead", "paved road", "lane markers visible", "off road", "parking lot", "indoors",
    "outdoors", "tunnel", "urban canyon", "rural area", "city", "highway", "construction zone",
    "heavy traffic", "bridge", "underpass"
]

# === ベースディレクトリ & クラスタ一覧 ===
base_dir = r"C:\Users\s1280\Desktop\SHRP2rawdata\central_frames_clustering_v7\clustered_images"
clusters = [f"cluster_{str(i).zfill(2)}" for i in range(10)]  # cluster_00～cluster_09

# === JSON保存用フォルダ ===
save_dir = os.path.join(base_dir, "llava_results")
os.makedirs(save_dir, exist_ok=True)

for cluster_name in clusters:
    cluster_path = os.path.join(base_dir, cluster_name)
    if not os.path.exists(cluster_path):
        print(f"⚠ {cluster_name} は存在しません、スキップ")
        continue

    frames = [os.path.join(cluster_path, f) for f in os.listdir(cluster_path) if f.endswith(".jpg")]
    print(f"\n📂 {cluster_name} 対象画像数: {len(frames)} 枚")

    cluster_results = []

    for idx, img_path in enumerate(frames):
        img_name = os.path.basename(img_path)
        print(f"\n[{idx+1}/{len(frames)}] 🖼 {img_name} → 推論中...")

        qa_list = []
        for ctx in contexts:
            question = f"Is this {ctx}?"
            pred = ask_llava_yesno(img_path, question)
            qa_list.append({"question": question, "pred": pred})
            print(f"   {ctx:20s}: {pred}")

        cluster_results.append({
            "cluster": cluster_name,
            "image": img_name,
            "qa_results": qa_list
        })

        # 各画像ごとにGPUメモリ状況を確認
        print_gpu_memory(prefix="   After this image:")

    # === クラスタごとにJSON保存 ===
    output_json = os.path.join(save_dir, f"llava_{cluster_name}_results.json")
    with open(output_json, "w", encoding="utf-8") as f:
        json.dump(cluster_results, f, ensure_ascii=False, indent=2)

    print(f"\n✅ {cluster_name} のQ&A結果を保存しました → {output_json}")
    print_gpu_memory(prefix="   After cluster:")

print(f"\n🎉 全クラスタの処理が完了しました！\n保存先: {save_dir}")

