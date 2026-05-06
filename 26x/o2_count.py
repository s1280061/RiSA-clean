# -*- coding: utf-8 -*-
import os
import pandas as pd

# 入力ファイル
files = [
    r"C:\Users\s1280\Desktop\SHRP2rawdata\3\new_divided\judge_results\results_gpt-5-mini.csv",
    r"C:\Users\s1280\Desktop\SHRP2rawdata\4\new_divided\judge_results\results_gpt-5-mini.csv",
    r"C:\Users\s1280\Desktop\SHRP2rawdata\5\new_divided\judge_results\results_gpt-5-mini.csv",
    r"C:\Users\s1280\Desktop\SHRP2rawdata\6\new_divided\judge_results\results_gpt-5-mini.csv",
]

# 出力先
out_dir = r"C:\Users\s1280\Desktop\SHRP2rawdata\for_GPT"
os.makedirs(out_dir, exist_ok=True)
out_path = os.path.join(out_dir, "results_gpt-5-mini_all.csv")

# CSVを読み込んで結合
dfs = [pd.read_csv(f) for f in files]
df_all = pd.concat(dfs, ignore_index=True)

# 保存
df_all.to_csv(out_path, index=False, encoding="utf-8-sig")

# 平均を計算（小数1位で四捨五入）
mean_vals = df_all[["situation_accuracy", "advice_appropriateness",
                    "safety_risk_calibration", "overall"]].mean().round(1)

print("=== 全体平均 ===")
print(mean_vals)
