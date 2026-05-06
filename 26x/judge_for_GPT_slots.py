# -*- coding: utf-8 -*-
"""
Run ChatGPT judge on 50 slots packed under for_GPT/
- Input layout (per slot): <slot>/<original_image_name>.jpg, context.json, llava.json, meta.json
- Model default: gpt-5-mini (change with --model)
- Outputs:
    <slot>/<model>_judge.json
    for_GPT/results_<model>.csv (append/resume-safe)
"""

import os, json, base64, argparse, csv, time
from glob import glob
from datetime import datetime
from openai import OpenAI

SYSTEM_PROMPT = """You are an impartial judge specialized in traffic safety.
Given RiSA outputs (context JSON, LLaVA JSON, and an image with trajectory),
score the following using a 5-point Likert scale:

1) situation_accuracy: Accuracy of situation recognition
2) advice_appropriateness: Appropriateness of the driving advice
3) safety_risk_calibration: Calibration of safety vs. risk (not overly aggressive nor overly conservative)

Likert scale:
1 = Completely inappropriate / incorrect
3 = Partially appropriate / uncertain
5 = Completely appropriate / ideal

Return ONLY a strict JSON object:
{
  "situation_accuracy": <int>,
  "advice_appropriateness": <int>,
  "safety_risk_calibration": <int>,
  "overall": <float>,  // average of the three, rounded to 1 decimal place
  "comment": "<short justification in English>"
}
"""

# Optional price table (USD per 1M tokens). Adjust to your account if needed.
PRICE_TABLE = {
    "gpt-5-nano": {"input": 0.15, "output": 0.60},
    "gpt-5-mini": {"input": 0.60, "output": 2.40},   # <- ここは必要なら更新してください
    "gpt-5":      {"input": 3.00, "output": 12.00},
}

def b64_image(path: str) -> str:
    with open(path, "rb") as f:
        return base64.b64encode(f.read()).decode("utf-8")

def load_json(path: str):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)

def judge_one(client, model, ctx_path, llv_path, img_path):
    img_b64 = b64_image(img_path)
    messages = [
        {"role":"system","content": SYSTEM_PROMPT},
        {"role":"user","content":[
            {"type":"text","text":"This is the RiSA context JSON:"},
            {"type":"text","text":json.dumps(load_json(ctx_path), ensure_ascii=False)},
            {"type":"text","text":"This is the LLaVA JSON:"},
            {"type":"text","text":json.dumps(load_json(llv_path), ensure_ascii=False)},
            {"type":"text","text":"This is the image with trajectory:"},
            {"type":"image_url","image_url":{"url":f"data:image/jpeg;base64,{img_b64}"}}]}
    ]
    resp = client.chat.completions.create(
        model=model,
        messages=messages,
        response_format={"type":"json_object"},
    )
    result = json.loads(resp.choices[0].message.content)
    result["overall"] = round(float(result["overall"]), 1)

    usage = resp.usage
    pt, ct, tt = usage.prompt_tokens, usage.completion_tokens, usage.total_tokens

    prices = PRICE_TABLE.get(model)
    if prices:
        usd = (pt/1_000_000)*prices["input"] + (ct/1_000_000)*prices["output"]
    else:
        usd = 0.0

    result["token_usage"] = {
        "prompt_tokens": pt,
        "completion_tokens": ct,
        "total_tokens": tt,
        "cost_usd": round(usd, 6)
    }
    return result

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", default=r"C:\Users\s1280\Desktop\SHRP2rawdata\for_GPT")
    ap.add_argument("--model", default="gpt-5-mini")
    ap.add_argument("--csv_name", default=None, help="results CSV filename (default: results_<model>.csv)")
    ap.add_argument("--resume", action="store_true", help="skip slots that already have <model>_judge.json")
    args = ap.parse_args()

    client = OpenAI()  # uses OPENAI_API_KEY

    root = args.dir
    csv_path = os.path.join(root, args.csv_name or f"results_{args.model}.csv")
    new_csv = not os.path.exists(csv_path)
    if new_csv:
        with open(csv_path, "w", newline="", encoding="utf-8") as f:
            w = csv.writer(f)
            w.writerow([
                "timestamp","slot","image","model",
                "situation_accuracy","advice_appropriateness","safety_risk_calibration",
                "overall","comment","prompt_tokens","completion_tokens","total_tokens","cost_usd"
            ])

    slots = sorted([p for p in glob(os.path.join(root, "*")) if os.path.isdir(p)])
    print(f"===== for_GPT | Slots: {len(slots)} | Model: {args.model} =====")

    done = 0
    for slot in slots:
        # files
        ctx = os.path.join(slot, "context.json")
        llv = os.path.join(slot, "llava.json")
        # 画像（context/llava/meta 以外の jpg を拾う）
        imgs = [p for p in glob(os.path.join(slot, "*.jpg")) if os.path.basename(p).lower() not in {"context.jpg","llava.jpg"}]
        if not (os.path.isfile(ctx) and os.path.isfile(llv) and imgs):
            continue
        img = imgs[0]
        out_json = os.path.join(slot, f"{args.model}_judge.json")
        if args.resume and os.path.exists(out_json):
            continue

        try:
            res = judge_one(client, args.model, ctx, llv, img)
        except Exception as e:
            msg = str(e)
            print(f"  ✖ {os.path.basename(slot)}: API error -> {msg}")
            if "insufficient_quota" in msg or "You exceeded your current quota" in msg:
                print("🛑 Quota exceeded. Stop here.")
                break
            # 軽いレート制限などはワンショット待機
            if "rate limit" in msg.lower() or "429" in msg:
                time.sleep(5)
            continue

        # save per-slot JSON
        with open(out_json, "w", encoding="utf-8") as f:
            json.dump(res, f, ensure_ascii=False, indent=2)

        # append CSV
        with open(csv_path, "a", newline="", encoding="utf-8") as f:
            w = csv.writer(f)
            w.writerow([
                datetime.now().isoformat(timespec="seconds"),
                os.path.basename(slot),
                os.path.basename(img),
                args.model,
                res["situation_accuracy"], res["advice_appropriateness"], res["safety_risk_calibration"],
                res["overall"], res["comment"].replace("\n"," "),
                res["token_usage"]["prompt_tokens"], res["token_usage"]["completion_tokens"],
                res["token_usage"]["total_tokens"], res["token_usage"]["cost_usd"]
            ])

        done += 1
        print(f"  ✓ {os.path.basename(slot)}/{os.path.basename(img)} -> {os.path.basename(out_json)}")

    print(f"✅ Completed slots: {done}")
    print(f"📄 CSV: {csv_path}")

if __name__ == "__main__":
    main()
