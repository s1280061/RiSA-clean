# -*- coding: utf-8 -*-
r"""
Select representative samples and pack them with images.

Modes:
  - balanced      : 既存の配分（Env=25, Turn=15, Brake=10）を満たす方式
  - turn_lowconf  : 景色条件を外し、ウィンカー低信頼度優先で --total 件を抽出
  - turn_highconf : 景色条件を外し、ウィンカー高信頼度優先で --total 件を抽出

Source:
  C:\Users\s1280\Desktop\SHRP2rawdata\
    ├─3\new_divided\scene_***_{context,llava}.json
    ├─4\new_divided\...
    ├─5\new_divided\...
    └─6\new_divided\...

Common filtering (all modes):
  - image_path endswith .jpg
  - image_path contains 'pre_tid'
  - image_path NOT contains 'post'
  - perception.turn_signal.final in {left, off, right} with conf
  - perception.brake.final in {go, brake} with conf

Priority:
  - turn_lowconf : sort by ( turn_conf asc , brake_conf asc , image_path asc )
  - turn_highconf: sort by ( turn_conf desc, brake_conf desc, image_path asc )

Output:
  C:\Users\s1280\Desktop\SHRP2rawdata\for_GPT\
    ├─0001\{original_image_name}.jpg
    │     context.json
    │     llava.json
    │     meta.json
    ├─0002\...
    └─manifest.csv
"""

import os, re, json, csv, argparse, shutil
from glob import glob

ENV_KEYS = ["rural_area", "city", "snowy", "sunny", "rainy"]
TURN_CLASSES = ["left", "off", "right"]
BRAKE_CLASSES = ["go", "brake"]

def load_json(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)

def gather_items(root, bases=("3","4","5","6"), require_env=True):
    """
    Scan scene_*_context.json (and remember its paired llava.json path).
    Return list of frame-level dicts:
      {
        "image_path", "envs", "turn", "turn_conf", "brake", "brake_conf",
        "context_json", "llava_json", "scene", "base"
      }

    require_env=True  : stage1_env YES が少なくとも1つ必要（balanced 用）
    require_env=False : stage1_env が空でも通す（turn_*conf 用）
    """
    items = []
    for base in bases:
        base_dir = os.path.join(root, base, "new_divided")
        if not os.path.isdir(base_dir):
            continue
        ctx_files = sorted(glob(os.path.join(base_dir, "scene_*_context.json")))
        for ctx_path in ctx_files:
            m = re.search(r"scene_(\d{3})_context\.json$", os.path.basename(ctx_path), re.IGNORECASE)
            if not m:
                continue
            scene_id = m.group(1)
            llava_path = os.path.join(base_dir, f"scene_{scene_id}_llava.json")
            if not os.path.isfile(llava_path):
                continue

            try:
                frames = load_json(ctx_path)
            except Exception:
                continue

            for fr in frames:
                img = (fr.get("image_path") or "").strip()
                name = os.path.basename(img).lower()
                if (not name.endswith(".jpg")) or ("pre_tid" not in name) or ("post" in name):
                    continue

                env = fr.get("stage1_env", {}) or {}
                env_yes = [k for k in ENV_KEYS if str(env.get(k, "NO")).upper() == "YES"]
                if require_env and not env_yes:
                    continue

                perc = fr.get("perception", {}) or {}
                turn_info = (perc.get("turn_signal", {}) or {})
                brake_info = (perc.get("brake", {}) or {})
                turn = (turn_info.get("final") or "").lower()
                turn_conf = turn_info.get("final_conf_pct", None)
                brake = (brake_info.get("final") or "").lower()
                brake_conf = brake_info.get("final_conf_pct", None)

                if turn not in TURN_CLASSES or turn_conf is None:
                    continue
                if brake not in BRAKE_CLASSES or brake_conf is None:
                    continue

                try:
                    t_conf = float(turn_conf)
                    b_conf = float(brake_conf)
                except Exception:
                    continue

                items.append({
                    "image_path": img,
                    "envs": env_yes,  # turn_*conf では空の可能性あり
                    "turn": turn,
                    "turn_conf": t_conf,
                    "brake": brake,
                    "brake_conf": b_conf,
                    "context_json": ctx_path,
                    "llava_json": llava_path,
                    "scene": f"scene_{scene_id}",
                    "base": base,
                })
    return items

def select_50(items, env_quota=None, turn_quota=None, brake_quota=None):
    """Greedy fill by env -> turn -> brake with low-conf priority (balanced mode)."""
    if env_quota is None:
        env_quota = {k: 5 for k in ENV_KEYS}            # 5*5=25
    if turn_quota is None:
        turn_quota = {k: 5 for k in TURN_CLASSES}       # 5*3=15
    if brake_quota is None:
        brake_quota = {k: 5 for k in BRAKE_CLASSES}     # 5*2=10
    target_total = sum(env_quota.values()) + sum(turn_quota.values()) + sum(brake_quota.values())

    items_sorted = sorted(items, key=lambda d: (d["turn_conf"], d["brake_conf"], d["image_path"]))
    buck_env = {k: [] for k in env_quota}
    buck_turn = {k: [] for k in turn_quota}
    buck_brake = {k: [] for k in brake_quota}
    for it in items_sorted:
        for e in it["envs"]:
            if e in buck_env:
                buck_env[e].append(it)
        if it["turn"] in buck_turn:
            buck_turn[it["turn"]].append(it)
        if it["brake"] in buck_brake:
            buck_brake[it["brake"]].append(it)

    selected, used = [], set()

    def take_from(bucket, label, n):
        cnt = 0
        for it in bucket.get(label, []):
            k = it["image_path"]
            if k in used:
                continue
            selected.append(it)
            used.add(k)
            cnt += 1
            if cnt >= n:
                break
        return cnt

    for env, q in env_quota.items():
        take_from(buck_env, env, q)

    for t, q in turn_quota.items():
        have = sum(1 for it in selected if it["turn"] == t)
        left = q - have
        if left > 0:
            take_from(buck_turn, t, left)

    for b, q in brake_quota.items():
        have = sum(1 for it in selected if it["brake"] == b)
        left = q - have
        if left > 0:
            take_from(buck_brake, b, left)

    if len(selected) < target_total:
        for it in items_sorted:
            if len(selected) >= target_total:
                break
            if it["image_path"] in used:
                continue
            selected.append(it); used.add(it["image_path"])

    return selected[:target_total]

def select_by_low_turn_conf(items, total=50):
    """turn_conf 昇順 → brake_conf 昇順 → image_path 昇順。重複は image_path で除外。"""
    items_sorted = sorted(
        items, key=lambda d: (d["turn_conf"], d["brake_conf"], d["image_path"])
    )
    selected, used = [], set()
    for it in items_sorted:
        k = it["image_path"]
        if k in used:
            continue
        selected.append(it)
        used.add(k)
        if len(selected) >= total:
            break
    return selected

def select_by_high_turn_conf(items, total=50):
    """turn_conf 降順 → brake_conf 降順 → image_path 昇順。重複は image_path で除外。"""
    items_sorted = sorted(
        items, key=lambda d: (-d["turn_conf"], -d["brake_conf"], d["image_path"])
    )
    selected, used = [], set()
    for it in items_sorted:
        k = it["image_path"]
        if k in used:
            continue
        selected.append(it)
        used.add(k)
        if len(selected) >= total:
            break
    return selected

def pack_output(selected, out_dir):
    os.makedirs(out_dir, exist_ok=True)
    manifest_path = os.path.join(out_dir, "manifest.csv")

    with open(manifest_path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow([
            "idx","image_rel","context_rel","llava_rel",
            "envs","turn","turn_conf","brake","brake_conf",
            "orig_image","orig_context","orig_llava","base","scene"
        ])

        count_written = 0
        for i, it in enumerate(selected, start=1):
            slot = f"{i:04d}"
            slot_dir = os.path.join(out_dir, slot)
            os.makedirs(slot_dir, exist_ok=True)

            src_img = it["image_path"]
            src_ctx = it["context_json"]
            src_llv = it["llava_json"]

            if not os.path.isfile(src_img):
                continue

            orig_name = os.path.basename(src_img)
            dst_img = os.path.join(slot_dir, orig_name)
            dst_ctx = os.path.join(slot_dir, "context.json")
            dst_llv = os.path.join(slot_dir, "llava.json")

            shutil.copy2(src_img, dst_img)
            shutil.copy2(src_ctx, dst_ctx)
            shutil.copy2(src_llv, dst_llv)

            meta = {
                "envs": it["envs"],
                "turn": it["turn"],
                "turn_conf": it["turn_conf"],
                "brake": it["brake"],
                "brake_conf": it["brake_conf"],
                "base": it["base"],
                "scene": it["scene"],
                "orig_image": src_img,
                "orig_context": src_ctx,
                "orig_llava": src_llv,
            }
            with open(os.path.join(slot_dir, "meta.json"), "w", encoding="utf-8") as mf:
                json.dump(meta, mf, ensure_ascii=False, indent=2)

            w.writerow([
                slot,
                os.path.relpath(dst_img, out_dir).replace("\\","/"),
                os.path.relpath(dst_ctx, out_dir).replace("\\","/"),
                os.path.relpath(dst_llv, out_dir).replace("\\","/"),
                "|".join(it["envs"]), it["turn"], it["turn_conf"], it["brake"], it["brake_conf"],
                src_img, src_ctx, src_llv, it["base"], it["scene"]
            ])
            count_written += 1

    print(f"✅ Packed {count_written} slots → {out_dir}")
    print(f"📄 Manifest: {manifest_path}")

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default=r"C:\Users\s1280\Desktop\SHRP2rawdata")
    ap.add_argument("--out_dir", default=r"C:\Users\s1280\Desktop\SHRP2rawdata\for_GPT")

    # モードと件数
    ap.add_argument("--mode", choices=["balanced", "turn_lowconf", "turn_highconf"], default="balanced",
                    help="balanced: 配分充足 / turn_lowconf: 低信頼度優先 / turn_highconf: 高信頼度優先")
    ap.add_argument("--total", type=int, default=50,
                    help="mode=turn_lowconf/turn_highconf の抽出件数")

    # balanced 用クォータ（互換）
    ap.add_argument("--env_quota", default="rural_area=5,city=5,snowy=5,sunny=5,rainy=5")
    ap.add_argument("--turn_quota", default="left=5,off=5,right=5")
    ap.add_argument("--brake_quota", default="go=5,brake=5")
    args = ap.parse_args()

    require_env = (args.mode == "balanced")
    items = gather_items(args.root, bases=("3","4","5","6"), require_env=require_env)
    if not items:
        print("No candidates found. Check JSON structure and paths.")
        return

    if args.mode == "balanced":
        def parse_kv(s):
            out = {}
            if not s:
                return out
            for part in s.split(","):
                k, v = part.split("=")
                out[k].strip() if False else None  # no-op to avoid linter complaints
                out[k.strip()] = int(v)
            return out
        env_q = parse_kv(args.env_quota)
        turn_q = parse_kv(args.turn_quota)
        brake_q = parse_kv(args.brake_quota)
        selected = select_50(items, env_q, turn_q, brake_q)
    elif args.mode == "turn_lowconf":
        selected = select_by_low_turn_conf(items, total=args.total)
        if len(selected) < args.total:
            print(f"Warning: requested {args.total} but only {len(selected)} candidates available.")
    else:  # turn_highconf
        selected = select_by_high_turn_conf(items, total=args.total)
        if len(selected) < args.total:
            print(f"Warning: requested {args.total} but only {len(selected)} candidates available.")

    pack_output(selected, args.out_dir)

if __name__ == "__main__":
    main()
