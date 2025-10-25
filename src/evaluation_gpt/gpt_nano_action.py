# -*- coding: utf-8 -*-
"""
Batch ChatGPT Judge for RiSA (Step2)
- Targets: ...\<base>\new_divided\scene_XXX_fire\*pre_tid*.jpg only
- Skips: folders with no pre_tid images; files containing 'post'
- Bases: 3,4,5,6 (default)
- Uses: scene_XXX_context.json / scene_XXX_llava.json in new_divided/
- Model: gpt-5-nano (default)
- Outputs:
    <base>\new_divided\judge_results\<scene>\ <image>.json
    <base>\new_divided\judge_results\all_results.csv (append)
"""

import os, re, json, csv, base64, argparse
from glob import glob
from datetime import datetime
from openai import OpenAI

# ------- Config (pricing, prompts) -------
PRICE_TABLE = {
    "gpt-5-nano":  {"input": 0.15, "output": 0.60},
    "gpt-5-mini":  {"input": 0.60, "output": 2.40},
    "gpt-4o-mini": {"input": 0.15, "output": 0.60},
    "gpt-4o":      {"input": 2.50, "output": 10.00},
    "gpt-5":       {"input": 3.00, "output": 12.00},
}

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

# ------- Helpers -------
def b64_image(path: str) -> str:
    with open(path, "rb") as f:
        return base64.b64encode(f.read()).decode("utf-8")

def load_json(path: str):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)

def ensure_dir(p: str):
    if p and not os.path.exists(p):
        os.makedirs(p, exist_ok=True)

def judge_one(client, model, context_json, llava_json, image_path):
    # Build messages
    img_b64 = b64_image(image_path)
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": [
            {"type": "text", "text": "This is the RiSA context JSON:"},
            {"type": "text", "text": json.dumps(load_json(context_json), ensure_ascii=False)},
            {"type": "text", "text": "This is the LLaVA JSON:"},
            {"type": "text", "text": json.dumps(load_json(llava_json), ensure_ascii=False)},
            {"type": "text", "text": "This is the image with trajectory:"},
            {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{img_b64}"}},
        ]},
    ]

    resp = client.chat.completions.create(
        model=model,
        messages=messages,
        response_format={"type": "json_object"},
    )
    result = json.loads(resp.choices[0].message.content)
    # round overall
    result["overall"] = round(float(result["overall"]), 1)

    usage = resp.usage
    pt, ct, tt = usage.prompt_tokens, usage.completion_tokens, usage.total_tokens
    prices = PRICE_TABLE.get(model, {"input": 0.0, "output": 0.0})
    usd = (pt/1_000_000)*prices["input"] + (ct/1_000_000)*prices["output"]
    result["token_usage"] = {
        "prompt_tokens": pt, "completion_tokens": ct, "total_tokens": tt,
        "cost_usd": round(usd, 6)
    }
    return result

# ------- Main -------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default=r"C:\Users\s1280\Desktop\SHRP2rawdata")
    ap.add_argument("--bases", default="3,4,5,6", help="comma-separated base dirs under root")
    ap.add_argument("--model", default="gpt-5-nano")
    ap.add_argument("--out_name", default="all_results.csv")  # CSV file name inside each base/new_divided/judge_results
    ap.add_argument("--max_scenes", type=int, default=0, help="limit number of scenes per base (0 = no limit)")
    args = ap.parse_args()

    client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
    bases = [b.strip() for b in args.bases.split(",") if b.strip()]

    scene_pat = re.compile(r"^scene_(\d{3})_fire$", re.IGNORECASE)

    for base in bases:
        base_dir = os.path.join(args.root, base, "new_divided")
        if not os.path.isdir(base_dir):
            print(f"⚠️  Skip base {base}: not found -> {base_dir}")
            continue

        out_root = os.path.join(base_dir, "judge_results")
        ensure_dir(out_root)
        csv_path = os.path.join(out_root, args.out_name)

        # prepare CSV (append mode)
        new_csv = not os.path.exists(csv_path)
        with open(csv_path, "a", newline="", encoding="utf-8") as fcsv:
            w = csv.writer(fcsv)
            if new_csv:
                # CSV ヘッダ（model 列を追加）
                w.writerow([
                    "timestamp", "base", "scene", "image", "model",  # ← model 追加
                    "situation_accuracy", "advice_appropriateness", "safety_risk_calibration",
                    "overall", "comment", "prompt_tokens", "completion_tokens", "total_tokens", "cost_usd"
                ])

            scenes = sorted([d for d in os.listdir(base_dir) if os.path.isdir(os.path.join(base_dir, d)) and scene_pat.match(d)])
            if args.max_scenes > 0:
                scenes = scenes[:args.max_scenes]

            print(f"\n===== Base {base} | Scenes: {len(scenes)} =====")
            for scene_dirname in scenes:
                m = scene_pat.match(scene_dirname)
                scene_num = m.group(1)
                scene_dir = os.path.join(base_dir, scene_dirname)

                # find images: *pre_tid*.jpg and not containing 'post'
                imgs = sorted([
                    p for p in glob(os.path.join(scene_dir, "*.jpg"))
                    if ("pre_tid" in os.path.basename(p).lower()) and ("post" not in os.path.basename(p).lower())
                ])

                if not imgs:
                    # do not call API when no pre_tid images
                    # print(f"  - {scene_dirname}: no pre_tid images -> skip API")
                    continue

                # JSON paths
                context_json = os.path.join(base_dir, f"scene_{scene_num}_context.json")
                llava_json   = os.path.join(base_dir, f"scene_{scene_num}_llava.json")

                if not (os.path.isfile(context_json) and os.path.isfile(llava_json)):
                    print(f"  - {scene_dirname}: missing JSON(s) -> skip API")
                    continue

                # per-scene output dir
                scene_out_dir = os.path.join(out_root, scene_dirname, args.model)  # ← ここを変更
                ensure_dir(scene_out_dir)

                for img in imgs:
                    img_name = os.path.splitext(os.path.basename(img))[0]
                    out_json = os.path.join(scene_out_dir, f"{img_name}.json")
                    if os.path.exists(out_json):
                        # already evaluated
                        continue

                    try:
                        res = judge_one(client, args.model, context_json, llava_json, img)
                    except Exception as e:
                        print(f"  ✖ {scene_dirname}/{img_name}: API error -> {e}")
                        continue

                    # save JSON
                    with open(out_json, "w", encoding="utf-8") as fj:
                        json.dump(res, fj, ensure_ascii=False, indent=2)

                    # append CSV row
                    # CSV 追記（model を出力）
                    w.writerow([
                        datetime.now().isoformat(timespec="seconds"),
                        base, scene_dirname, img_name + ".jpg", args.model,  # ← model 追加
                        res["situation_accuracy"], res["advice_appropriateness"], res["safety_risk_calibration"],
                        res["overall"], res["comment"].replace("\n", " "),
                        res["token_usage"]["prompt_tokens"], res["token_usage"]["completion_tokens"],
                        res["token_usage"]["total_tokens"], res["token_usage"]["cost_usd"]
                    ])

                    print(f"  ✓ {scene_dirname}/{img_name}.jpg -> saved")

        print(f"✅ Base {base}: results saved to {csv_path}")

if __name__ == "__main__":
    main()