import os
import csv
import base64
import json
import requests

# ============================
# 設定
# ============================
OLLAMA_ENDPOINT = "http://localhost:11434/api/generate"
LLAVA_MODEL = "llava:latest"

PROMPT_CSV = r"C:\Users\s1280\Desktop\vlm_prompts_23_en.csv"
IMAGE_ROOT = r"C:\Users\s1280\Desktop\SHRP2_outputs_sorted"

OUTPUT_JSONL = r"C:\Users\s1280\Desktop\SHRP2_outputs_sorted\llava_results.jsonl"

IMAGE_EXT = [".jpg", ".jpeg", ".png"]


# ============================
# Base64 エンコード
# ============================
def encode_image(path):
    with open(path, "rb") as f:
        return base64.b64encode(f.read()).decode("utf-8")


# ============================
# LLAVA（Ollama）問い合わせ
# ============================
def query_llava(image_path, question_text):
    img_b64 = encode_image(image_path)

    forced_prompt = (
        "Answer concisely.\n"
        "Respond ONLY with the required format.\n"
        "Do NOT add explanations.\n"
        f"Question: {question_text}"
    )

    payload = {
        "model": LLAVA_MODEL,
        "prompt": forced_prompt,
        "images": [img_b64],
        "stream": False,
        "options": {
            "temperature": 0,
            "reset": True
        }
    }

    try:
        response = requests.post(OLLAMA_ENDPOINT, json=payload)
        response.raise_for_status()
        data = response.json()
        return data.get("response", "").strip()
    except Exception as e:
        return f"ERROR: {str(e)}"


# ============================
# Resume 用："処理済み image_path" をロード
# ============================
def load_done_images(jsonl_path):
    done = set()
    if not os.path.exists(jsonl_path):
        return done

    with open(jsonl_path, "r", encoding="utf-8") as f:
        for line in f:
            try:
                data = json.loads(line)
                done.add(data["image_path"])
            except:
                pass
    return done


# ============================
# メイン処理（★ Resume 対応版 ★）
# ============================
def main():

    # 23質問読み込み
    with open(PROMPT_CSV, encoding="utf-8-sig") as f:
        reader = csv.DictReader(f)
        questions = list(reader)

    # すでに処理済みの画像一覧をロード
    done_images = load_done_images(OUTPUT_JSONL)
    print(f"Loaded {len(done_images)} processed images.")

    # JSONL 追記モード
    out = open(OUTPUT_JSONL, "a", encoding="utf-8")

    # 全サブフォルダを探索
    for root, dirs, files in os.walk(IMAGE_ROOT):
        for filename in files:

            if not any(filename.lower().endswith(ext) for ext in IMAGE_EXT):
                continue

            image_path = os.path.join(root, filename)

            # ★ Resume: 処理済みならスキップ
            if image_path in done_images:
                print(f"Skip (already done): {image_path}")
                continue

            print(f"\n=== Processing image: {image_path} ===")

            answers = {}

            # 23質問ループ
            for row in questions:
                qid = row["question_id"]
                qtext = row["question_text"]

                print(f"[Q{qid}] {qtext}")

                ans = query_llava(image_path, qtext)

                print(f"Answer: {ans}")

                answers[f"Q{qid}"] = ans

            # JSONL に保存
            record = {
                "image_path": image_path,
                "answers": answers
            }
            out.write(json.dumps(record, ensure_ascii=False) + "\n")

            print(f"--- Saved record for {image_path} ---")

    out.close()

    print("\n============================")
    print(" 完了！JSONL に保存されました →")
    print(" ", OUTPUT_JSONL)
    print("============================")


if __name__ == "__main__":
    main()
