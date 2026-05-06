import pandas as pd
import matplotlib.pyplot as plt

def save_score_table_as_figure(csv_path, out_path="table_scores.png"):
    df = pd.read_csv(csv_path)

    # 列名を整形
    df = df.rename(columns={
        "mean_situation_accuracy": "Sit. Acc.",
        "mean_advice_appropriateness": "Advice",
        "mean_safety_risk_calibration": "Risk Calib.",
        "mean_overall": "Overall",
    })

    # 残したいスコア列だけを抽出
    keep_cols = ["Model", "Samples", "Sit. Acc.", "Advice", "Risk Calib.", "Overall"]
    df = df[[c for c in keep_cols if c in df.columns]]

    fig, ax = plt.subplots(figsize=(8, 2 + len(df) * 0.4))
    ax.axis('off')

    # テーブル描画
    table = ax.table(
        cellText=df.values,
        colLabels=df.columns,
        cellLoc='center',
        loc='center'
    )
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1.2, 1.2)

    plt.savefig(out_path, dpi=300, bbox_inches='tight')
    print(f"✅ Saved score-only table figure: {out_path}")


# 使い方
csv_path = r"C:\Users\s1280\Desktop\SHRP2rawdata\for_GPT\paper_summary_by_model.csv"
save_score_table_as_figure(csv_path, out_path=r"C:\Users\s1280\Desktop\SHRP2rawdata\for_GPT\table_paper_scores.png")
