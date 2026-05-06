# -*- coding: utf-8 -*-
# Experiment 01 – Selected contexts → CSV / TeX / PNG

import os
import textwrap
import pandas as pd
import matplotlib.pyplot as plt

# ==== Output folder ====
OUT_DIR = r"C:\Users\s1280\Desktop\SHRP2rawdata\pipeline_set"
os.makedirs(OUT_DIR, exist_ok=True)

# ==== Table data (あなたの表そのまま) ====
data = [
    {"Context Category": "Rural area", "Primary Safety Concern": "Wildlife sudden entries",
     "Accuracy(%)": 92.0, "Ground Truth Counts(Yes/No)": "721/127"},
    {"Context Category": "City",       "Primary Safety Concern": "Dense pedestrian traffic",
     "Accuracy(%)": 86.9, "Ground Truth Counts(Yes/No)": "76/772"},
    {"Context Category": "Snowy",      "Primary Safety Concern": "Very low grip",
     "Accuracy(%)": 86.4, "Ground Truth Counts(Yes/No)": "41/807"},
    {"Context Category": "Sunny",      "Primary Safety Concern": "Glare and shadows",
     "Accuracy(%)": 86.2, "Ground Truth Counts(Yes/No)": "147/701"},
    {"Context Category": "Rainy",      "Primary Safety Concern": "Low tire grip",
     "Accuracy(%)": 86.0, "Ground Truth Counts(Yes/No)": "368/480"},
]
df = pd.DataFrame(data)

# ==== 1) CSV ====
csv_path = os.path.join(OUT_DIR, "exp01_selected_contexts.csv")
df.to_csv(csv_path, index=False, encoding="utf-8-sig")

# ==== 2) TeX (table 環境つき) ====
latex_cols = df.rename(columns={
    "Accuracy(%)": "Accuracy (\\%)",
    "Ground Truth Counts(Yes/No)": "Ground Truth Counts (Yes/No)",
})
latex_inner = latex_cols.to_latex(
    index=False,
    escape=True,
    column_format="p{3.0cm}p{6.0cm}p{2.0cm}p{3.0cm}",  # 幅はお好みで微調整
)
latex_full = r"""\begin{table}[t]
\centering
\caption{Selected context categories and performance for Experiment~01.}
\label{tab:exp01_contexts}
""" + latex_inner + r"""
\end{table}
"""
tex_path = os.path.join(OUT_DIR, "exp01_selected_contexts.tex")
with open(tex_path, "w", encoding="utf-8") as f:
    f.write(latex_full)

# ==== 3) PNG（表そのものを画像化：自動折り返し＋余白小） ====
df_wrapped = df.copy()
df_wrapped["Primary Safety Concern"] = df_wrapped["Primary Safety Concern"].apply(
    lambda s: "\n".join(textwrap.wrap(str(s), width=28))
)

fig, ax = plt.subplots(figsize=(10, 3.8), dpi=300)  # 横長で行数5に最適化
ax.axis("off")
plt.subplots_adjust(left=0, right=1, top=1, bottom=0)

col_widths = [0.25, 0.45, 0.15, 0.15]  # 合計≈1.0
table_data = [df_wrapped.columns.tolist()] + df_wrapped.values.tolist()
tab = ax.table(cellText=table_data, colWidths=col_widths, loc="upper left", cellLoc="left")
tab.auto_set_font_size(False)
tab.set_fontsize(12)
tab.scale(1, 1.6)  # 行間
for (r, c), cell in tab.get_celld().items():
    cell.set_linewidth(0.9 if r == 0 else 0.5)
    if r == 0:
        cell.set_text_props(weight="bold")

png_path = os.path.join(OUT_DIR, "exp01_selected_contexts.png")
plt.savefig(png_path, dpi=300, bbox_inches="tight", pad_inches=0.02)
plt.close()

print("Saved:")
print("  CSV:", csv_path)
print("  TeX:", tex_path)
print("  PNG:", png_path)
