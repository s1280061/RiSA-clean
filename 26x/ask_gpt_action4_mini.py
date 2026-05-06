# -*- coding: utf-8 -*-
"""
Ask GPT-5-mini to choose a driving action (Action-4) from an image only.

- Scans *.jpg/*.png under --images_dir
- Sends one image per request (no context), asks for STRICT JSON:
    {"action": "<KEEP_SPEED|DECELERATE|CHANGE_LEFT|CHANGE_RIGHT>", "reason": "..."}
- Normalizes synonyms to Action-4 and writes a CSV (resume-safe).
"""

import os, json, base64, argparse, csv, time
from glob import glob
from datetime import datetime
from pathlib import Path
from openai import OpenAI

ALLOWED = ["KEEP_SPEED","DECELERATE","CHANGE_LEFT","CHANGE_RIGHT"]

SYSTEM_PROMPT = """You are a careful driving action classifier.
You must look ONLY at the provided ego-vehicle camera image with trajectory overlays.
Choose exactly ONE action from:
- KEEP_SPEED
- DECELERATE
- CHANGE_LEFT
- CHANGE_RIGHT

Trajectory overlay legend:
- Yellow lines: past movement history of each detected vehicle.
- Red lines: predicted future trajectories (uncertain, indicative only).
Use these lines to infer motion trends, but do not assume predictions are guaranteed.

Rules:
- No other labels. Pick one of the four only.
- Choose the most likely action based solely on the image, even if uncertain.
- Return a STRICT JSON object with keys: action, reason (short, <= 20 words).
"""


USER_INSTRUCTION = """Decide the driving action from this single image only.
Return STRICT JSON:
{"action":"<KEEP_SPEED|DECELERATE|CHANGE_LEFT|CHANGE_RIGHT>","reason":"<short english>"}
"""

def b64_image(path: str) -> str:
    with open(path, "rb") as f:
        return base64.b64encode(f.read()).decode("utf-8")

def normalize_action(s: str):
    if not s: return None
    m = str(s).strip().upper().replace("-", " ").replace("_", " ")
    if m in ALLOWED: return m
    keep_kw  = ["KEEP SPEED","MAINTAIN SPEED","STAY IN LANE","KEEP LANE","HOLD SPEED","CONTINUE STRAIGHT"]
    decel_kw = ["DECELERATE","SLOW DOWN","REDUCE SPEED","BRAKE","LOWER SPEED","SLOW YOUR VEHICLE"]
    left_kw  = ["CHANGE LANE LEFT","MERGE LEFT","MOVE LEFT","SHIFT LEFT","GO LEFT","SWITCH LEFT","TURN LEFT"]
    right_kw = ["CHANGE LANE RIGHT","MERGE RIGHT","MOVE RIGHT","SHIFT RIGHT","GO RIGHT","SWITCH RIGHT","TURN RIGHT"]
    def any_in(text, patterns): return any(p in text for p in patterns)
    if any_in(m, keep_kw):  return "KEEP_SPEED"
    if any_in(m, decel_kw): return "DECELERATE"
    if any_in(m, left_kw):  return "CHANGE_LEFT"
    if any_in(m, right_kw): return "CHANGE_RIGHT"
    if "KEEP" in m: return "KEEP_SPEED"
    if "LEFT" in m: return "CHANGE_LEFT"
    if "RIGHT" in m: return "CHANGE_RIGHT"
    if "SLOW" in m or "DECEL" in m or "BRAKE" in m: return "DECELERATE"
    return None

def ask_one(client, model, img_path):
    img_b64 = b64_image(img_path)
    messages = [
        {"role":"system","content": SYSTEM_PROMPT},
        {"role":"user","content":[
            {"type":"text","text": USER_INSTRUCTION},
            {"type":"image_url","image_url":{"url": f"data:image/jpeg;base64,{img_b64}"}}
        ]}
    ]
    resp = client.chat.completions.create(
        model=model,
        messages=messages,
        response_format={"type":"json_object"},
    )
    content = resp.choices[0].message.content
    try:
        obj = json.loads(content)
    except Exception:
        obj = json.loads(content.strip().strip("`"))
    action_raw = obj.get("action")
    reason = (obj.get("reason","") or "").replace("\n"," ").strip()
    action_norm = normalize_action(action_raw)
    usage = getattr(resp, "usage", None)
    return {
        "action_raw": action_raw,
        "action_norm": action_norm,
        "reason": reason,
        "prompt_tokens": getattr(usage, "prompt_tokens", None),
        "completion_tokens": getattr(usage, "completion_tokens", None),
        "total_tokens": getattr(usage, "total_tokens", None),
    }

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--images_dir", default=r"C:\Users\s1280\Desktop\SHRP2rawdata\project_root\eval_images")
    ap.add_argument("--out_csv",    default=r"C:\Users\s1280\Desktop\SHRP2rawdata\project_root\model\gpt5mini_action4.csv")
    ap.add_argument("--model",      default="gpt-5-mini")
    ap.add_argument("--resume",     action="store_true", help="skip images already recorded in out_csv")
    ap.add_argument("--limit",      type=int, default=None, help="optional cap on number of images")
    ap.add_argument("--pattern",    default="*.jpg", help="glob: *.jpg / *.png")
    args = ap.parse_args()

    client = OpenAI()  # requires OPENAI_API_KEY

    images = sorted(glob(os.path.join(args.images_dir, args.pattern)))
    if args.pattern == "*.jpg":  # also scan PNGs by default
        images += sorted(glob(os.path.join(args.images_dir, "*.png")))
    if args.limit:
        images = images[:args.limit]
    print(f"Images found: {len(images)} (dir={args.images_dir})")

    out = Path(args.out_csv)
    out.parent.mkdir(parents=True, exist_ok=True)

    seen = set()
    if out.exists() and args.resume:
        try:
            with open(out, "r", encoding="utf-8") as f:
                for row in csv.DictReader(f):
                    seen.add(row["image"])
        except Exception:
            pass

    new_file = not out.exists()
    done = 0
    with open(out, "a", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        if new_file:
            w.writerow(["timestamp","image","model","action_raw","action_norm","reason",
                        "prompt_tokens","completion_tokens","total_tokens"])
        for img in images:
            name = os.path.basename(img)
            if args.resume and name in seen:
                continue
            try:
                res = ask_one(client, args.model, img)
            except Exception as e:
                msg = str(e)
                print(f"  ✖ {name}: {msg}")
                # 軽いレート制限のリトライ
                if "429" in msg or "rate limit" in msg.lower():
                    time.sleep(5)
                    try:
                        res = ask_one(client, args.model, img)
                    except Exception as e2:
                        print(f"    retry failed: {e2}")
                        continue
                else:
                    continue

            w.writerow([
                datetime.now().isoformat(timespec="seconds"),
                name, args.model,
                res["action_raw"], res["action_norm"], res["reason"],
                res["prompt_tokens"], res["completion_tokens"], res["total_tokens"]
            ])
            f.flush()
            done += 1
            print(f"  ✓ {name} -> {res['action_norm']} | {res['reason'][:60]}")

    print(f"✅ Completed: {done}")
    print(f"📄 CSV: {out}")

if __name__ == "__main__":
    main()
