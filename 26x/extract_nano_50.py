# -*- coding: utf-8 -*-
"""
Extract nano (gpt-5-nano) results for the same 50 cases listed in for_GPT/manifest.csv.
- Reads nano CSVs from <root>\<base>\new_divided\judge_results\all_results.csv (base=3,4,5,6)
- Matches by (base, scene_dirname, image_filename) derived from manifest.orig_image
- Writes for_GPT/results_gpt-5-nano.csv
- Optionally copies per-image nano JSON to each slot as nano_judge.json

Usage:
  python extract_nano_50.py --root "C:\\Users\\s1280\\Desktop\\SHRP2rawdata" --for_gpt_dir "C:\\Users\\s1280\\Desktop\\SHRP2rawdata\\for_GPT" --copy_json
"""

import os, csv, argparse, shutil
from glob import glob
from datetime import datetime

def load_nano_rows(root, bases=("3","4","5","6")):
    """Return dict keyed by (base, scene_dirname, image_filename_lower) -> row dict."""
    keymap = {}
    for base in bases:
        csv_path = os.path.join(root, base, "new_divided", "judge_results", "all_results.csv")
        if not os.path.isfile(csv_path):
            print(f"⚠ nano CSV not found: {csv_path}")
            continue
        with open(csv_path, "r", encoding="utf-8") as f:
            rd = csv.DictReader(f)
            for r in rd:
                scene = (r.get("scene") or "").strip()          # e.g., scene_019_fire
                image = (r.get("image") or "").strip()          # e.g., frame_008732_pre_tid2.jpg
                b     = (r.get("base")  or "").strip()          # e.g., 5
                if not (scene and image and b):
                    continue
                key = (b, scene, image.lower())
                keymap[key] = r
    return keymap

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default=r"C:\Users\s1280\Desktop\SHRP2rawdata")
    ap.add_argument("--for_gpt_dir", default=r"C:\Users\s1280\Desktop\SHRP2rawdata\for_GPT")
    ap.add_argument("--out_csv", default=None, help="Default: for_GPT\\results_gpt-5-nano.csv")
    ap.add_argument("--copy_json", action="store_true", help="Copy per-image nano JSON into each slot as nano_judge.json")
    args = ap.parse_args()

    manifest = os.path.join(args.for_gpt_dir, "manifest.csv")
    if not os.path.isfile(manifest):
        print(f"❌ manifest not found: {manifest}")
        return

    # index all nano rows
    m = load_nano_rows(args.root, bases=("3","4","5","6"))
    if not m:
        print("❌ No nano rows indexed. Check base CSVs.")
        return

    out_csv = args.out_csv or os.path.join(args.for_gpt_dir, "results_gpt-5-nano.csv")
    new_csv = not os.path.exists(out_csv)
    with open(out_csv, "a", newline="", encoding="utf-8") as fo:
        wr = csv.writer(fo)
        if new_csv:
            wr.writerow([
                "timestamp","slot","image","model",
                "situation_accuracy","advice_appropriateness","safety_risk_calibration",
                "overall","comment","prompt_tokens","completion_tokens","total_tokens","cost_usd"
            ])

        # walk manifest slots
        with open(manifest, "r", encoding="utf-8") as fm:
            rd = csv.DictReader(fm)
            found, missing = 0, 0
            for row in rd:
                slot = row["idx"].strip()
                orig_image = row["orig_image"].strip()
                base = row["base"].strip()
                # derive scene dirname and image filename from orig_image path
                image_filename = os.path.basename(orig_image)
                scene_dirname  = os.path.basename(os.path.dirname(orig_image))  # e.g., scene_019_fire
                key = (base, scene_dirname, image_filename.lower())

                nano = m.get(key)
                if not nano:
                    missing += 1
                    print(f"⚠ missing in nano CSV: slot {slot} -> {key}")
                    continue

                # write CSV line in the same format as mini/gpt-5 results
                wr.writerow([
                    datetime.now().isoformat(timespec="seconds"),
                    slot,
                    image_filename,
                    "gpt-5-nano",
                    nano.get("situation_accuracy",""),
                    nano.get("advice_appropriateness",""),
                    nano.get("safety_risk_calibration",""),
                    nano.get("overall",""),
                    (nano.get("comment","") or "").replace("\n"," "),
                    nano.get("prompt_tokens",""),
                    nano.get("completion_tokens",""),
                    nano.get("total_tokens",""),
                    nano.get("cost_usd",""),
                ])
                found += 1

                # optional: copy JSON to slot
                if args.copy_json:
                    slot_dir = os.path.join(args.for_gpt_dir, f"{int(slot):04d}")
                    os.makedirs(slot_dir, exist_ok=True)
                    src_json = os.path.join(args.root, base, "new_divided", "judge_results", scene_dirname, os.path.splitext(image_filename)[0] + ".json")
                    if os.path.isfile(src_json):
                        dst_json = os.path.join(slot_dir, "gpt-5-nano_judge.json")  # 命名をモデル揃え
                        shutil.copy2(src_json, dst_json)

    print(f"✅ Done. Wrote nano 50-case CSV → {out_csv}")

if __name__ == "__main__":
    main()
