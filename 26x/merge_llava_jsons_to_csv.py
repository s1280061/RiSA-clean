# -*- coding: utf-8 -*-
import argparse, json, re
from pathlib import Path
import pandas as pd

def extract_frame_core(s: str):
    if not isinstance(s, str): return None
    s = s.strip()
    pats = [r"frame[_\- ]+(\d{3,})", r"[\\/](\d{3,})[_\.]"]
    for pat in pats:
        m = re.search(pat, s, flags=re.IGNORECASE)
        if m:
            return str(int(m.group(1)))  # drop leading zeros
    return None

def get_maneuver(d: dict):
    # 探索順：risk_assessment.* -> 直下候補
    keys_nested = [("risk_assessment","suggested_maneuver"),
                   ("advice","suggested_maneuver")]
    for k1,k2 in keys_nested:
        v = (d.get(k1, {}) or {}).get(k2)
        if v: return v
    for k in ["suggested_maneuver","maneuver","recommended_action","decision"]:
        v = d.get(k)
        if v: return v
    return None

def get_lane_change(d: dict):
    for k1,k2 in [("risk_assessment","lane_change_detected"),
                  ("advice","lane_change_detected")]:
        v = (d.get(k1, {}) or {}).get(k2)
        if v: return v
    return d.get("lane_change_detected")

def get_reason(d: dict):
    for k1,k2 in [("risk_assessment","reason"),
                  ("advice","reason")]:
        v = (d.get(k1, {}) or {}).get(k2)
        if v: return v
    return d.get("reason")

def normalize_action4(m):
    if m is None: return None
    m = str(m).lower().strip().replace("-", " ").replace("_", " ")
    keep_kw  = ["keep speed","maintain speed","stay in lane","keep lane","hold speed","continue straight","maintain current speed"]
    decel_kw = ["decelerate","slow down","reduce speed","brake","lower speed","slow your vehicle"]
    left_kw  = ["change lane left","change to the left","lane change left","merge left","move left","shift left","go left","switch left","turn left"]
    right_kw = ["change lane right","change to the right","lane change right","merge right","move right","shift right","go right","switch right","turn right"]
    def any_in(text, pats): return any(p in text for p in pats)
    if any_in(m, keep_kw):  return "KEEP_SPEED"
    if any_in(m, decel_kw): return "DECELERATE"
    if any_in(m, left_kw):  return "CHANGE_LEFT"
    if any_in(m, right_kw): return "CHANGE_RIGHT"
    # フォールバック
    if "keep" in m: return "KEEP_SPEED"
    if "left" in m: return "CHANGE_LEFT"
    if "right" in m: return "CHANGE_RIGHT"
    if "slow" in m or "decel" in m or "brake" in m: return "DECELERATE"
    return None

def load_all(llava_dir: Path):
    rows, empty_files = [], []
    for p in sorted(llava_dir.glob("*_llava.json")):
        m = re.match(r"^([3456])_", p.name)   # ex) 3_scene_000_llava.json -> 3
        source_group = m.group(1) if m else ""
        try:
            data = json.loads(p.read_text(encoding="utf-8"))
        except Exception as e:
            print(f"[WARN] JSON読込失敗: {p} ({e})")
            continue

        if isinstance(data, list):
            if len(data) == 0:
                empty_files.append(p.name)
                continue
            iters = data
        elif isinstance(data, dict):
            iters = [data]
        else:
            empty_files.append(p.name)
            continue

        for item in iters:
            if not isinstance(item, dict): continue
            fid   = item.get("frame_id")
            ipath = item.get("image_path") or item.get("img_path") or item.get("filename") or ""
            lc    = get_lane_change(item)
            man   = get_maneuver(item)
            reason= get_reason(item)
            core  = extract_frame_core(str(ipath))
            rows.append({
                "scene_file": p.name,
                "source_group": source_group,        # 3/4/5/6
                "frame_id": fid,
                "image_path": ipath,
                "frame_core": core,
                "lane_change_detected": lc,
                "suggested_maneuver": man,
                "suggested_maneuver_action4": normalize_action4(man),
                "reason": reason,
            })
    return pd.DataFrame(rows), empty_files

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--llava_dir", required=True, help=r"例: C:\Users\s1280\Desktop\SHRP2rawdata\project_root\model")
    ap.add_argument("--out_csv",   required=True, help=r"例: C:\Users\s1280\Desktop\SHRP2rawdata\project_root\results\llava_merged_all.csv")
    args = ap.parse_args()

    llava_dir = Path(args.llava_dir)
    out_csv   = Path(args.out_csv)
    out_csv.parent.mkdir(parents=True, exist_ok=True)

    df, empties = load_all(llava_dir)
    if df.empty:
        print(f"[ERROR] レコードが0件でした（空配列のみ or 読み取り失敗）。dir={llava_dir}")
        return

    # 同じ frame_core が複数ある場合は最後を採用
    if "frame_core" in df.columns:
        df = df.dropna(subset=["frame_core"])
        df = df.drop_duplicates(subset=["frame_core"], keep="last")

    df.to_csv(out_csv, index=False, encoding="utf-8-sig")
    print(f"[INFO] merged rows: {len(df)} -> {out_csv}")

    if empties:
        empty_csv = out_csv.with_name(out_csv.stem + "_emptyjson_files.csv")
        pd.DataFrame({"empty_json_file": empties}).to_csv(empty_csv, index=False, encoding="utf-8-sig")
        print(f"[INFO] 空配列だったJSON一覧: {empty_csv} (n={len(empties)})")

if __name__ == "__main__":
    main()
