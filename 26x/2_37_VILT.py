import os
import csv
import json
import torch
from PIL import Image
from transformers import (
    ViltForQuestionAnswering,
    ViltFeatureExtractor,
    BertTokenizer
)
# ============================
# 設定
# ============================
PROMPT_CSV = r"C:\Users\s1280\Desktop\vlm_prompts_23_en.csv"
IMAGE_ROOT = r"C:\Users\s1280\Desktop\SHRP2_outputs_sorted"
OUTPUT_JSONL = r"C:\Users\s1280\Desktop\SHRP2_outputs_sorted\vilt_results.jsonl"

IMAGE_EXT = [".jpg", ".jpeg", ".png"]

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

print("Loading ViLT model...")

feature_extractor = ViltFeatureExtractor.from_pretrained(
    "dandelin/vilt-b32-finetuned-vqa"
)

tokenizer = BertTokenizer.from_pretrained(
    "dandelin/vilt-b32-finetuned-vqa"
)

model = ViltForQuestionAnswering.from_pretrained(
    "dandelin/vilt-b32-finetuned-vqa"
).to(DEVICE)

model.eval()
print("ViLT loaded successfully.")


# ============================
# ViLT 推論
# ============================
def query_vilt(image_path, question_text):
    try:
        image = Image.open(image_path).convert("RGB")

        encoding = feature_extractor(
            images=image,
            return_tensors="pt"
        )

        text_inputs = tokenizer(
            question_text,
            return_tensors="pt"
        )

        inputs = {
            "input_ids": text_inputs["input_ids"].to(DEVICE),
            "attention_mask": text_inputs["attention_mask"].to(DEVICE),
            "pixel_values": encoding["pixel_values"].to(DEVICE),
        }

        with torch.no_grad():
            outputs = model(**inputs)

        logits = outputs.logits
        probs = torch.softmax(logits, dim=-1)
        pred_id = probs.argmax(-1).item()

        answer = model.config.id2label[pred_id]
        confidence = probs[0, pred_id].item()

        return {
            "answer": answer,
            "confidence": round(confidence, 4)
        }

    except Exception as e:
        return {
            "answer": "ERROR",
            "confidence": 0.0,
            "error": str(e)
        }

# ============================
# Resume 用：処理済み画像ロード
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
# メイン処理
# ============================
def main():

    # 質問CSV読み込み
    with open(PROMPT_CSV, encoding="utf-8-sig") as f:
        reader = csv.DictReader(f)
        questions = list(reader)

    # Resume
    done_images = load_done_images(OUTPUT_JSONL)
    print(f"Loaded {len(done_images)} processed images.")

    out = open(OUTPUT_JSONL, "a", encoding="utf-8")

    # 画像探索
    for root, dirs, files in os.walk(IMAGE_ROOT):
        for filename in files:

            if not any(filename.lower().endswith(ext) for ext in IMAGE_EXT):
                continue

            image_path = os.path.join(root, filename)

            if image_path in done_images:
                print(f"Skip (already done): {image_path}")
                continue

            print(f"\n=== Processing image: {image_path} ===")

            answers = {}

            for row in questions:
                qid = row["question_id"]
                qtext = row["question_text"]

                print(f"[Q{qid}] {qtext}")

                result = query_vilt(image_path, qtext)

                print(
                    f"Answer: {result['answer']} "
                    f"(conf={result['confidence']})"
                )

                answers[f"Q{qid}"] = result

            record = {
                "image_path": image_path,
                "answers": answers
            }

            out.write(json.dumps(record, ensure_ascii=False) + "\n")
            out.flush()

            print(f"--- Saved record for {image_path} ---")

    out.close()

    print("\n============================")
    print(" 完了！JSONL に保存されました →")
    print(" ", OUTPUT_JSONL)
    print("============================")

if __name__ == "__main__":
    main()
