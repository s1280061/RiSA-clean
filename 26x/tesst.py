import json
import os
import pandas as pd
import matplotlib.pyplot as plt

# ===== Base folder =====
base_dir = r"C:\Users\s1280\Desktop\SHRP2rawdata\central_frames_clustering_10\clustered_images"
gt_file = os.path.join(base_dir, "image_level_gt.json")

# ===== All 24 questions =====
all_questions = [
    "outdoors", "paved road", "highway", "rural area",
    "daytime", "twilight", "sunny", "foggy",
    "trees overhead", "rainy", "construction zone", "city",
    "bridge", "underpass", "night-time", "indoors",
    "tunnel", "urban canyon", "off road", "lane markers visible",
    "dust/sandstorm", "heavy traffic", "snowy", "parking lot"
]

# === Statistics containers ===
label_stats = {q: {"adopted": 0, "excluded": 0} for q in all_questions}
total_adopted = 0
total_excluded = 0

# === Load GT ===
with open(gt_file, "r", encoding="utf-8") as f:
    gt_data = json.load(f)

gt_map = {}
for item in gt_data:
    img = item["image"]
    gt_map[img] = {}
    for q, v in item["labels"].items():
        if v in ["yes", "no"]:
            gt_map[img][q] = v
            label_stats[q]["adopted"] += 1
            total_adopted += 1
        else:
            label_stats[q]["excluded"] += 1
            total_excluded += 1

print(f"✅ GT images: {len(gt_map)}")
print(f"✅ Adopted labels (yes/no): {total_adopted}")
print(f"✅ Excluded labels (ambiguous): {total_excluded}")
print(f"✅ Total labels: {total_adopted + total_excluded}")

# === Create a DataFrame for category stats ===
rows = []
for q in all_questions:
    a = label_stats[q]["adopted"]
    e = label_stats[q]["excluded"]
    total_q = a + e
    adoption_ratio = (a / total_q * 100) if total_q > 0 else 0
    rows.append([q, a, e, total_q, round(adoption_ratio, 1)])

df_stats = pd.DataFrame(rows, columns=[
    "Category", "Adopted", "Excluded", "Total", "Adoption Rate (%)"
])

# === Save as CSV ===
csv_path = os.path.join(base_dir, "label_adoption_stats.csv")
df_stats.to_csv(csv_path, index=False)
print(f"✅ Saved category-wise stats to CSV: {csv_path}")

# === Show in console as table ===
print("\n=== Category-wise Adopted/Excluded ===")
print(df_stats.to_string(index=False))

# === Bar plot for adoption rate ===
plt.figure(figsize=(10,6))
plt.barh(df_stats["Category"], df_stats["Adoption Rate (%)"], color='skyblue')
plt.xlabel("Adoption Rate (%)")
plt.title("Category-wise Adoption Rate (YES≥90% / NO≤10%)")
plt.gca().invert_yaxis()
plt.tight_layout()
plt.savefig(os.path.join(base_dir, "label_adoption_ratio.png"))
plt.close()
print("✅ Saved adoption rate bar chart: label_adoption_ratio.png")

# === Optional: Save table as image ===
fig, ax = plt.subplots(figsize=(10, 8))
ax.axis('off')
table_img = ax.table(
    cellText=df_stats.values,
    colLabels=df_stats.columns,
    loc='center',
    cellLoc='center'
)
table_img.auto_set_font_size(False)
table_img.set_fontsize(8)
table_img.scale(1.2, 1.2)
plt.title("Category-wise Adopted/Excluded Labels")
plt.savefig(os.path.join(base_dir, "label_adoption_stats_table.png"))
plt.close()
print("✅ Saved category-wise table image: label_adoption_stats_table.png")
