import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# ===============================
# ★ フォント設定（最重要）
# ===============================
plt.rcParams.update({
    "font.family": "serif",
    "font.serif": ["Times New Roman", "Times"],
    "mathtext.fontset": "cm",
    "axes.unicode_minus": False,
})

sns.set_theme(style="white", font="serif")

# ===============================
# Confusion matrix (counts)
# ===============================
cm = np.array([
    [285, 125, 492],  # Keep
    [3,   2,   15],   # Left
    [8,   9,   43],   # Right
])

labels = ["Keep", "Left", "Right"]

# ===============================
# Row-normalization
# ===============================
cm_norm = cm / cm.sum(axis=1, keepdims=True)

df_cm = pd.DataFrame(
    cm_norm,
    index=labels,
    columns=labels
)

# ===============================
# Plot
# ===============================
fig, ax = plt.subplots(figsize=(7.5, 6.5))

sns.heatmap(
    df_cm,
    annot=True,
    fmt=".2f",
    cmap="Blues",
    vmin=0.0,
    vmax=1.0,
    linewidths=0.6,
    annot_kws={
        "size": 24,
        "family": "serif"   # ★ セル内数値も Times 系に
    },
    cbar_kws={"label": "Row-normalized probability"},
    ax=ax
)

# Axis labels & title
ax.set_xlabel("Lane-change detected by LLaVA", fontsize=26)
ax.set_ylabel("Ground-truth lane-change", fontsize=26)
fig.suptitle(
    "Confusion Matrix of Lane-Change Recognition",
    fontsize=28,
    y=0.98
)

# Tick labels
ax.tick_params(axis="both", labelsize=24)

# Colorbar font control
cbar = ax.collections[0].colorbar
cbar.ax.tick_params(labelsize=20)
cbar.set_label("Row-normalized probability", fontsize=26)

plt.tight_layout()
plt.savefig(
    r"C:\Users\s1280\Desktop\confusion_matrix_stageA_row_normalized.pdf"
)
plt.savefig(
    r"C:\Users\s1280\Desktop\confusion_matrix_stageA_row_normalized.png",
    dpi=300
)
plt.show()
