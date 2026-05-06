# -*- coding: utf-8 -*-
"""
Merge nano CSVs (base3/4/5/6) into a single file for plotting and analysis.
Input files (in for_GPT):
  - nano_base3_results.csv
  - nano_base4_results.csv
  - nano_base5_results.csv
  - nano_base6_results.csv

Output files:
  - nano_all_results.csv          (row-wise concatenation)
  - nano_summary_by_base.csv      (count & mean scores per base)
"""

import os, csv, argparse
from statistics import mean

def read_csv_rows(path):
    with open(path, "r", encoding="utf-8") as f:
        rd = csv.DictReader(f)
        header = rd.fieldnames
        rows = [r for r in rd]
    return header, rows

def write_csv(path, header, rows):
    with open(path, "w", newline="", encoding="utf-8") as f:
        wr = csv.DictWriter(f, fieldnames=header)
        wr.writeheader()
        wr.writerows(rows)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", default=r"C:\Users\s1280\Desktop\SHRP2rawdata\for_GPT")
    ap.add_argument("--outsuffix", default="nano_all_results.csv")
    args = ap.parse_args()

    root = args.dir
    inputs = [
        os.path.join(root, "nano_base3_results.csv"),
        os.path.join(root, "nano_base4_results.csv"),
        os.path.join(root, "nano_base5_results.csv"),
        os.path.join(root, "nano_base6_results.csv"),
    ]
    merged = []
    header_ref = None

    for p in inputs:
        if not os.path.exists(p):
            print(f"⚠ Missing: {p}")
            continue
        header, rows = read_csv_rows(p)
        if header_ref is None:
            header_ref = header
        else:
            # 最低限の列一致チェック（足りない列は空で埋める）
            for h in header_ref:
                if h not in header:
                    for r in rows:
                        r[h] = ""
        merged.extend(rows)
        print(f"  ✓ loaded {p}: {len(rows)} rows")

    if not merged:
        print("❌ No rows to merge.")
        return

    out_csv = os.path.join(root, args.outsuffix)
    write_csv(out_csv, header_ref, merged)
    print(f"✅ merged -> {out_csv} (rows={len(merged)})")

    # ベース別サマリ（count/mean）
    # 期待ヘッダ: timestamp,base,scene,image,situation_accuracy,advice_appropriateness,
    #             safety_risk_calibration,overall,comment,prompt_tokens,completion_tokens,total_tokens,cost_usd
    by_base = {}
    for r in merged:
        b = (r.get("base") or "").strip()
        if b not in by_base:
            by_base[b] = {"rows": []}
        by_base[b]["rows"].append(r)

    summary_rows = []
    for b, obj in sorted(by_base.items(), key=lambda x: x[0]):
        rows = obj["rows"]
        def fcol(name):
            vals = []
            for r in rows:
                try:
                    vals.append(float(r.get(name, "") or "nan"))
                except:
                    pass
            vals = [v for v in vals if v == v]  # drop NaN
            return mean(vals) if vals else ""
        summary_rows.append({
            "base": b,
            "count": len(rows),
            "mean_situation_accuracy": fcol("situation_accuracy"),
            "mean_advice_appropriateness": fcol("advice_appropriateness"),
            "mean_safety_risk_calibration": fcol("safety_risk_calibration"),
            "mean_overall": fcol("overall"),
            "mean_cost_usd": fcol("cost_usd"),
            "mean_prompt_tokens": fcol("prompt_tokens"),
            "mean_completion_tokens": fcol("completion_tokens"),
            "mean_total_tokens": fcol("total_tokens"),
        })
    sum_csv = os.path.join(root, "nano_summary_by_base.csv")
    write_csv(sum_csv, list(summary_rows[0].keys()), summary_rows)
    print(f"📄 summary -> {sum_csv}")

if __name__ == "__main__":
    main()
