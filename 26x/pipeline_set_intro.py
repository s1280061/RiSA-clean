# -*- coding: utf-8 -*-
# RiSA – Experiment 01: 3-word safety concerns (CSV / TeX / PNG)

import os
import textwrap
import pandas as pd
import matplotlib.pyplot as plt

# ===== Output directory =====
OUT_DIR = r"C:\Users\s1280\Desktop\SHRP2rawdata\pipeline_set"
os.makedirs(OUT_DIR, exist_ok=True)

COL_CATEGORY = "Context Category"
COL_CONCERN  = "Primary Safety Concern"  # ≈3 words each

# ===== Contexts (from the right-hand table) =====
categories = [
    "Daytime", "Night-time", "Twilight",
    "Sunny", "Rainy", "Snowy", "Foggy", "Dust/Sandstorm",
    "Trees Overhead", "Lane Markers Visible", "Paved Road", "Off Road", "Parking lot",
    "Indoors", "Outdoors", "Tunnel", "Urban Canyon", "Rural area", "City",
    "Highway", "Construction Zone", "Heavy Traffic", "Bridge", "Underpass"
]

# ===== 3-word concerns (no abbreviations) =====
concern = {
    "Daytime": "High speed crossings",
    "Night-time": "Short sight distance",
    "Twilight": "Low sun glare",
    "Sunny": "Glare and shadows",
    "Rainy": "Low tire grip",
    "Snowy": "Very low grip",
    "Foggy": "Severely reduced visibility",
    "Dust/Sandstorm": "Airborne dust occlusion",
    "Trees Overhead": "Shadows hide lines",
    "Lane Markers Visible": "Lane marking loss",
    "Paved Road": "Surface traction change",
    "Off Road": "Unmarked irregular obstacles",
    "Parking lot": "Hidden reversing pedestrians",
    "Indoors": "Tight space reflections",
    "Outdoors": "Strong light changes",
    "Tunnel": "Sudden light transition",
    "Urban Canyon": "Occlusion by buildings",
    "Rural area": "Wildlife sudden entries",
    "City": "Dense pedestrian traffic",
    "Highway": "High speed merging",
    "Construction Zone": "Temporary lane changes",
    "Heavy Traffic": "Short gaps cutins",
    "Bridge": "Crosswinds and icing",
    "Underpass": "Sudden darkness pooling",
}

# ===== Build dataframe =====
df = pd.DataFrame({
    COL_CATEGORY: categories,
    COL_CONCERN:  [concern.get(c, "") for c in categories],
})

# ===== Save CSV =====
csv_path = os.path.join(OUT_DIR, "context_risa_3words.csv")
df.to_csv(csv_path, index=False, encoding="utf-8-sig")

# ===== Save TeX (two-column compact table; fixed column widths) =====
latex_inner = df.rename(columns={
    COL_CATEGORY: "Context Category",
    COL_CONCERN:  "Primary Safety Concern"
}).to_latex(index=False, escape=True, column_format="p{4.0cm}p{9.5cm}")
latex_full = r"""\begin{table}[t]
\centering
\caption{RiSA context categories and primary safety concerns (3-word form).}
\label{tab:risa_contexts_3words}
\setlength{\tabcolsep}{6pt}
""" + latex_inner + r"""
\end{table}
"""
tex_path = os.path.join(OUT_DIR, "context_risa_3words.tex")
with open(tex_path, "w", encoding="utf-8") as f:
    f.write(latex_full)

# ===== Save PNG (table image; minimal margins; wrapping just in case) =====
def wrap_series(series, width):
    return series.apply(lambda s: "\n".join(textwrap.wrap(str(s), width=width)))

WRAP_CONCERN = 28    # small wrap; most rows are 3 words anyway
FONT_SIZE    = 13
ROW_SCALE    = 1.8
FIGSIZE      = (7.5, 12.5)  # portrait; compact margins

df_wrapped = df.copy()
df_wrapped[COL_CONCERN] = wrap_series(df_wrapped[COL_CONCERN], WRAP_CONCERN)

fig, ax = plt.subplots(figsize=FIGSIZE, dpi=300)
ax.axis("off")
plt.subplots_adjust(left=0, right=1, top=1, bottom=0)
ax.margins(0)

col_widths = [0.40, 0.60]  # sum ≈ 1.0
table_data = [df_wrapped.columns.tolist()] + df_wrapped.values.tolist()

tab = ax.table(cellText=table_data, colWidths=col_widths, loc="upper left", cellLoc="left")
tab.auto_set_font_size(False)
tab.set_fontsize(FONT_SIZE)
tab.scale(1, ROW_SCALE)
for (r, c), cell in tab.get_celld().items():
    cell.set_linewidth(0.9 if r == 0 else 0.5)
    if r == 0:
        cell.set_text_props(weight="bold")

png_path = os.path.join(OUT_DIR, "context_risa_3words.png")
plt.savefig(png_path, dpi=300, bbox_inches="tight", pad_inches=0.02)
plt.close()

print("Saved:")
print(" CSV:", csv_path)
print(" TeX:", tex_path)
print(" PNG:", png_path)
