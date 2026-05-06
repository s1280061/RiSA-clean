# Scene Understanding Experiment

This repository contains the results of **Scene Understanding Experiment (Cluster-based analysis of SHRP2 driving scenes)** using **LLaVA (Vision-Language Model)**.

---

## 📂 Folder Structure

scene_understanding_experiment/
│
├── 00_raw_data/ # Original clustered images
│ ├── cluster_00/
│ ├── cluster_01/
│ └── ...
│
├── 01_inference_results/ # LLaVA inference results
│ ├── llava_cluster_00_results.json
│ ├── llava_cluster_01_results.json
│ └── ...
│
├── 02_pseudo_gt/ # Weak pseudo Ground Truth (GT)
│ ├── image_level_gt.json
│ ├── label_adoption_stats.csv
│ ├── label_adoption_ratio.png
│ └── label_adoption_stats_table.png
│
├── 03_evaluation/ # Evaluation metrics
│ ├── confusion_matrix.png
│ ├── evaluation_metrics.png
│ └── cluster_evaluation_metrics.png
│
├── 04_visualization/ # Visualization
│ ├── longtail_distribution_contexts_final.png
│ ├── yes_counts_cluster_00.png
│ ├── tsne_clusters.png
│ ├── tsne_with_cluster_images.png
│ └── ...
│
└── README.md


---

## 📝 Experiment Summary

- **Goal**  
  Evaluate the ability of **Vision-Language Models (LLaVA)** to recognize **24 driving context attributes** in SHRP2 dataset.

- **Data**  
  ~6,800 clustered images from SHRP2 driving scenes  
  - Clustered into 10 groups (cluster_00 – cluster_09)
  - 24 binary attributes (e.g., `OUTDOORS`, `HIGHWAY`, `RAINY`, `CITY`, ...)

- **Method**  
  1. LLaVA answered **Yes/No questions** like *“Is this highway?”*  
  2. Created **Pseudo-Ground Truth (weak labels)**  
     - ≥90% YES → auto-labeled YES  
     - ≤10% YES → auto-labeled NO  
     - others → ambiguous (excluded from eval)  
  3. Evaluated only **high-confidence labels**  
     - Accuracy, Precision, Recall, F1-score  

---

## ✅ Results

- **Evaluation on high-confidence subset**


Accuracy = 0.987
Precision = 0.960
Recall = 0.988
F1-score = 0.974

![Confusion Matrix](03_evaluation/confusion_matrix.png)

- **Label adoption ratio**
![Label Adoption Ratio](02_pseudo_gt/label_adoption_ratio.png)

- **Long-tail context distribution**
![Long-tail](04_visualization/longtail_distribution_contexts_final.png)

- **Cluster visualization (t-SNE)**
![t-SNE](04_visualization/tsne_clusters.png)

---

## 🔖 Key Points

- **Weak pseudo-GT** was generated automatically using thresholding  
- Ambiguous labels were excluded from evaluation  
- Clear **long-tail distribution** observed in context attributes  
- High accuracy achieved on **high-confidence subset**

---

## 🚩 Future Work

- Save **embeddings** for reproducibility and new experiments  
- Refine **cluster semantic labeling** (e.g., Cluster 0 = “Daytime Highway”)  
- Extend to **Phase 02: Action/Intention recognition**

---

## 📌 Notes

- Even without original embeddings, all results (metrics, visualization) are **finalized and reproducible** from this repo.
- For academic presentation, **t-SNE visualization and representative cluster images** are sufficient.

---

**Contact:** *(Kaito ASAI/ humanophilic lab)*  
