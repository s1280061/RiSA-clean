import pandas as pd
import numpy as np
import statsmodels.api as sm

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

BASE_X_COLS = [
    "env_rainy",
    "brake_final",
    "lane_change_detected_llava",
    "detected_vehicle_count"
]

TS_COL = "ts_final"

# =====================================================
# 3. 前処理
# =====================================================
def preprocess(df):
    cols = BASE_X_COLS + [TS_COL] + list(SCORE_COLS.values())
    df = df[cols].copy()

    for c in ["env_rainy", "brake_final", "lane_change_detected_llava", TS_COL]:
        df[c] = (
            df[c]
            .astype(str)
            .str.strip()
            .str.lower()
        )

    df["detected_vehicle_count"] = pd.to_numeric(
        df["detected_vehicle_count"], errors="coerce"
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


def build_X(df, use_ts=False):
    X_cols = BASE_X_COLS.copy()
    if use_ts:
        X_cols.append(TS_COL)

    X = df[X_cols].copy()

    # binary
    X.loc[:, "env_rainy"] = (X["env_rainy"] == "yes").astype(int)
    X.loc[:, "brake_final"] = (X["brake_final"] == "brake").astype(int)

    # categorical → dummy
    cat_cols = ["lane_change_detected_llava"]
    if use_ts:
        cat_cols.append(TS_COL)

    X = pd.get_dummies(
        X,
        columns=cat_cols,
        drop_first=True
    )

    return X.astype(float)


def run_regularized_logit(df, model_name, use_ts=False):
    y = df[f"{model_name}_y"]
    X = build_X(df, use_ts=use_ts)

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

    return np.exp(result.params)  # Odds Ratio


# =====================================================
# 4. 実行：ts_final あり / なし
# =====================================================
df = preprocess(df_raw)

for model, col in SCORE_COLS.items():
    df[f"{model}_y"] = df[col].apply(make_binary_label)

print("\n========== Sensitivity Analysis: ts_final ==========")

for model in ["nano", "mini"]:
    print(f"\n--- {model.upper()} ---")

    odds_without_ts = run_regularized_logit(df, model, use_ts=False)
    odds_with_ts = run_regularized_logit(df, model, use_ts=True)

    compare = pd.DataFrame({
        "without_ts": odds_without_ts,
        "with_ts": odds_with_ts
    }).fillna(1.0)

    # 見やすく：主要因子だけ表示
    key_factors = [
        "brake_final",
        "env_rainy",
        "detected_vehicle_count"
    ]
    compare = compare.loc[
        compare.index.intersection(key_factors)
    ]

    print(compare)
