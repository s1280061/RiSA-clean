import pandas as pd
import numpy as np
import statsmodels.api as sm
import matplotlib.pyplot as plt

# =====================================================
# 1. CSV 読み込み
# =====================================================
CSV_PATH = r"D:\merged_action4_paper_ready_with_json.csv"
df_raw = pd.read_csv(CSV_PATH)

# =====================================================
# 2. 設定
# =====================================================
SCORE_COLS = {
    "nano": "overall",
    "mini": "overall_mini"
}

X_COLS = [
    "env_rainy",
    "brake_final",
    "lane_change_detected_llava",
    "detected_vehicle_count",
    "ego_speed_kmh"
]

# =====================================================
# 3. 前処理
# =====================================================
def preprocess_dataframe(df):
    df = df[X_COLS + list(SCORE_COLS.values())].copy()

    for c in ["env_rainy", "brake_final", "lane_change_detected_llava"]:
        df[c] = (
            df[c]
            .astype(str)
            .str.strip()
            .str.lower()
        )

    # unknown → keep に統合
    df["lane_change_detected_llava"] = (
        df["lane_change_detected_llava"]
        .replace("unknown", "keep")
    )

    df["detected_vehicle_count"] = pd.to_numeric(
        df["detected_vehicle_count"], errors="coerce"
    )

    df["ego_speed_kmh"] = pd.to_numeric(
        df["ego_speed_kmh"], errors="coerce"
    )

    return df


def make_binary_label(x):
    if pd.isna(x):
        return np.nan
    if x >= 4.0:
        return 1
    elif x <= 2.0:
        return 0
    else:
        return np.nan


def build_design_matrix(df):
    X = df[X_COLS].copy()

    X.loc[:, "env_rainy"] = (X["env_rainy"] == "yes").astype(int)
    X.loc[:, "brake_final"] = (X["brake_final"] == "brake").astype(int)

    X = pd.get_dummies(
        X,
        columns=["lane_change_detected_llava"],
        drop_first=True
    )

    return X.astype(float)


# =====================================================
# 4. 正則化ロジスティック回帰
# =====================================================
def run_regularized_logit(df, model_name):
    y = df[f"{model_name}_y"]
    X = build_design_matrix(df)

    data = pd.concat([y, X], axis=1).dropna()
    y_clean = data[f"{model_name}_y"].astype(int)
    X_clean = data.drop(columns=[f"{model_name}_y"])

    X_clean = sm.add_constant(X_clean)

    logit = sm.Logit(y_clean, X_clean)
    result = logit.fit_regularized(
        method="l1",
        alpha=1.0,
        disp=False
    )

    return result


# =====================================================
# 5. 可視化
# =====================================================
def plot_odds_ratio_bar(results):
    rows = []
    for model, res in results.items():
        odds = np.exp(res.params)
        odds = odds.drop("const", errors="ignore")
        for k, v in odds.items():
            rows.append({
                "model": model,
                "factor": k,
                "odds_ratio": v
            })

    df_plot = pd.DataFrame(rows)

    pivot = df_plot.pivot(
        index="model",
        columns="factor",
        values="odds_ratio"
    )

    pivot.plot(
        kind="bar",
        logy=True,
        figsize=(10, 4)
    )

    plt.axhline(1.0, color="black", linestyle="--", linewidth=1)
    plt.ylabel("Odds Ratio (log scale)")
    plt.title("Regularized logistic regression (odds ratios)")
    plt.xticks(rotation=0)
    plt.tight_layout()
    plt.show()


def plot_coefficients(result, title, top_k=8):
    coef = result.params.drop("const", errors="ignore")
    df = pd.DataFrame({"coef": coef})
    df["abs"] = df["coef"].abs()
    df = df.sort_values("abs").tail(top_k)

    y = np.arange(len(df))

    fig, ax = plt.subplots(
        figsize=(7.0, max(3, len(df) * 0.45))
    )

    ax.scatter(
        df["coef"],
        y,
        color="black",
        zorder=3,
        clip_on=False
    )

    ax.axvline(0, color="red", linestyle="--", linewidth=1)

    xmin = df["coef"].min()
    xmax = df["coef"].max()
    margin = (xmax - xmin) * 0.25 + 0.2
    ax.set_xlim(xmin - margin, xmax + margin)

    ax.set_yticks(y)
    ax.set_yticklabels(df.index)
    ax.set_xlabel("Log-odds coefficient")
    ax.set_title(title)

    ax.grid(axis="x", linestyle=":", alpha=0.5)

    fig.tight_layout()
    plt.show()


# =====================================================
# 6. 実行
# =====================================================
df = preprocess_dataframe(df_raw)

for model, col in SCORE_COLS.items():
    df[f"{model}_y"] = df[col].apply(make_binary_label)

results = {}

for model in SCORE_COLS.keys():
    print(f"\n===== {model.upper()} REGULARIZED LOGISTIC =====")
    res = run_regularized_logit(df, model)
    results[model] = res

    odds = np.exp(res.params).sort_values(ascending=False)
    print("\nOdds Ratios:")
    print(odds)

    plot_coefficients(
        res,
        title=f"Logistic regression coefficients ({model.capitalize()} model)"
    )

plot_odds_ratio_bar(results)
