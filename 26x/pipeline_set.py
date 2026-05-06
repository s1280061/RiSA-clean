# -*- coding: utf-8 -*-


import os
import textwrap
import pandas as pd
import matplotlib.pyplot as plt

# ========= Output directory =========

OUT_DIR = r"C:\Users\s1280\Desktop\SHRP2rawdata\pipeline_set"
os.makedirs(OUT_DIR, exist_ok=True)

# ========= Source data =========
rows = [
    {"Context Category": "Rural area",
     "Most Important Risk Scenario Perception": "Few signals/lamps; wildlife or farm vehicles; delayed emergency response",
     "Accuracy": 92.0, "Y": 721, "N": 127,
     "Comment": "High risk & high accuracy"},
    {"Context Category": "City",
     "Most Important Risk Scenario Perception": "Pedestrians, cyclists, dense signals, congestion, complex intersections",
     "Accuracy": 86.9, "Y": 76, "N": 772,
     "Comment": "High risk; good rural vs city contrast"},
    {"Context Category": "Rainy",
     "Most Important Risk Scenario Perception": "Lower tire-road friction, visibility loss, hydroplaning risk",
     "Accuracy": 86.0, "Y": 368, "N": 480,
     "Comment": "High risk & balanced counts"},
    {"Context Category": "Sunny",
     "Most Important Risk Scenario Perception": "Glare, strong shadows, contrast shifts that reduce conspicuity",
     "Accuracy": 86.2, "Y": 147, "N": 701,
     "Comment": "Medium risk; keeps illumination diversity"},
    {"Context Category": "Snowy",
     "Most Important Risk Scenario Perception": "Major friction loss, visibility reduction, lane-marking degradation",
     "Accuracy": 86.4, "Y": 41, "N": 807,
     "Comment": "Very high risk but few Yes samples"},
    {"Context Category": "Highway",
     "Most Important Risk Scenario Perception": "High-speed merges and overtakes; severe-accident potential",
     "Accuracy": 99.5, "Y": 848, "N": 0,
     "Comment": "No-case absent; unsuitable for Yes/No test"},
]

# ========= Build dataframe =========
df = pd.DataFrame(rows)
df["Yes/No"] = df["Y"].astype(int).astype(str) + " / " + df["N"].astype(int).astype(str)
df = df[[
    "Context Category",
    "Most Important Risk Scenario Perception",
    "Accuracy",
    "Yes/No",
    "Y", "N", "Comment"
]]

# ========= Save CSV =========
csv_path = os.path.join(OUT_DIR, "context_categories_table.csv")
df.to_csv(csv_path, index=False, encoding="utf-8-sig")

# ========= Save LaTeX (table environment included) =========
latex_tab = df[[
    "Context Category",
    "Most Important Risk Scenario Perception",
    "Accuracy",
    "Yes/No"
]].rename(columns={
    "Accuracy": "Accuracy (\\%)",
    "Yes/No": "Yes/No (Y/N)",
})
# basic to_latex (keep it simple; journal macros can style it)
latex_inner = latex_tab.to_latex(index=False, escape=True, column_format="llll")
latex_full = r"""\begin{table}[t]
\centering
\caption{Context categories with accuracy and risk perception.}
\label{tab:context_categories}
""" + latex_inner + r"""
\end{table}
"""
tex_path = os.path.join(OUT_DIR, "context_categories_table.tex")
with open(tex_path, "w", encoding="utf-8") as f:
    f.write(latex_full)

# ========= Save accuracy bar chart (matplotlib, no color specified) =========
png_acc_path = os.path.join(OUT_DIR, "context_categories_accuracy.png")
plt.figure(figsize=(10, 5))
plt.bar(df["Context Category"], df["Accuracy"])
plt.title("Accuracy by Context Category")
plt.ylabel("Accuracy (%)")
plt.ylim(0, 100)
plt.xticks(rotation=30, ha="right")
plt.tight_layout()
plt.savefig(png_acc_path, dpi=200, bbox_inches="tight")
plt.close()

# ========= Save the table itself as PNG with wrapped text =========
# Wrap settings (adjust if needed)
wrap_width_desc = 38   # for "Most Important Risk Scenario Perception"
wrap_width_comment = 28  # for "Comment"

df_wrapped = df.copy()
df_wrapped["Most Important Risk Scenario Perception"] = df_wrapped[
    "Most Important Risk Scenario Perception"
].apply(lambda s: "\n".join(textwrap.wrap(str(s), width=wrap_width_desc)))
df_wrapped["Comment"] = df_wrapped["Comment"].apply(
    lambda s: "\n".join(textwrap.wrap(str(s), width=wrap_width_comment))
)

# Build matplotlib table
png_table_path = os.path.join(OUT_DIR, "context_categories_table.png")
fig, ax = plt.subplots(figsize=(14, 5.2))  # wider for readability
ax.axis("off")

# Choose only the columns we want to show in the PNG table
show_cols = [
    "Context Category",
    "Most Important Risk Scenario Perception",
    "Accuracy",
    "Yes/No",
    "Comment"
]
table_data = [show_cols] + df_wrapped[show_cols].values.tolist()

# Wider columns for the two text-heavy fields
col_widths = [0.14, 0.45, 0.10, 0.10, 0.21]  # must sum ~1.0

tab = ax.table(cellText=table_data, colWidths=col_widths, loc="center", cellLoc="left")
tab.auto_set_font_size(False)
tab.set_fontsize(8)
tab.scale(1, 1.6)  # increase row height for wrapped lines

# Make header bold
for (row, col), cell in tab.get_celld().items():
    if row == 0:
        cell.set_text_props(weight='bold')
        cell.set_linewidth(1.0)
    else:
        cell.set_linewidth(0.6)

plt.tight_layout()
plt.savefig(png_table_path, dpi=300, bbox_inches="tight")
plt.close()

print("Saved:")
print(" CSV :", csv_path)
print(" TeX :", tex_path)
print(" PNG (accuracy):", png_acc_path)
print(" PNG (table)   :", png_table_path)
