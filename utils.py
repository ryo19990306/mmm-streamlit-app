# utils.py

import pandas as pd
import numpy as np
import jpholiday
from sklearn.linear_model import ElasticNet
from sklearn.metrics import r2_score, mean_absolute_percentage_error, mean_squared_error
from scipy.optimize import minimize
import matplotlib.pyplot as plt

# ▼ Adstock変換
def apply_adstock(x, beta):
    x = np.array(x)
    result = np.zeros_like(x)
    result[0] = x[0]
    for t in range(1, len(x)):
        result[t] = x[t] + beta * result[t - 1]
    return result

# ▼ Saturation変換
def saturation_transform(x, alpha):
    x = np.maximum(x, 0)
    return np.power(x, alpha)

# ▼ 時系列特徴量生成
def create_time_features(df_dates, base_date_min, extra_cols=None):
    df = df_dates.copy()
    df["weekday"] = df["Date"].dt.weekday
    weekday_dummies = pd.get_dummies(df["weekday"], prefix="wd", drop_first=True)
    month_dummies = pd.get_dummies(df["Date"].dt.month, prefix="month", drop_first=True)
    holidays = df["Date"].map(jpholiday.is_holiday)
    weekends = df["Date"].dt.weekday >= 5
    df["is_holiday"] = (holidays | weekends).astype(int)
    df["trend"] = (df["Date"] - base_date_min).dt.days

    extra_df = pd.concat([weekday_dummies, month_dummies, df[["is_holiday", "trend"]]], axis=1)
    if extra_cols is not None:
        extra_df = extra_df.reindex(columns=extra_cols, fill_value=0)
    return extra_df

# ▼ モデル学習（GridSearchCVは外で実施済み前提）
def train_model(df_raw, model=None):
    df = df_raw.copy()
    df.columns = df.columns.str.strip()
    df["Date"] = pd.to_datetime(df["Date"], errors="coerce")
    df["Sales"] = pd.to_numeric(df["Sales"], errors="coerce")
    X_raw = df.drop(columns=["Date", "Sales"], errors="ignore")
    numeric_cols = X_raw.select_dtypes(include=[np.number]).columns.tolist()
    X = X_raw.copy()
    X[numeric_cols] = X[numeric_cols].apply(pd.to_numeric, errors="coerce")
    valid_idx = X.dropna().index.intersection(df["Sales"].dropna().index)
    X = X.loc[valid_idx]
    y = df.loc[valid_idx, "Sales"]
    df = df.dropna(subset=["Date"])
    df["Date"] = pd.to_datetime(df["Date"])

    extra_features = create_time_features(df[["Date"]], df["Date"].min())
    extra_features = extra_features.loc[X.index]
    X = pd.concat([X, extra_features], axis=1)

    extra_feature_cols = list(extra_features.columns)
    media_cols = [col for col in X.columns if col not in extra_feature_cols]

    # α・β仮固定（0.5）
    alphas = [0.5] * len(media_cols)
    betas = [0.5] * len(media_cols)

    # Adstock + Saturation 変換
    X_transformed = []
    for i, col in enumerate(media_cols):
        ad = apply_adstock(X[col].values, betas[i])
        sat = saturation_transform(ad, alphas[i])
        X_transformed.append(sat)
    X_media = np.array(X_transformed).T
    X_extra = X.drop(columns=media_cols).values
    X_all = np.concatenate([X_media, X_extra], axis=1)

    # ElasticNet モデル学習
    if model is None:
        model = ElasticNet(alpha=1.0, l1_ratio=0.5, positive=True, max_iter=10000)
        model.fit(X_all, y)
    else:
        model.fit(X_all, y)

    pred = model.predict(X_all)

    return {"model": model, "alphas": alphas, "betas": betas, "columns": media_cols, "extra_cols": extra_feature_cols}, pd.Series(pred, index=X.index)

# ▼ 評価関数
def evaluate_model(df_pred):
    r2 = r2_score(df_pred["Actual_Sales"], df_pred["Predicted_Sales"])
    rmse = np.sqrt(mean_squared_error(df_pred["Actual_Sales"], df_pred["Predicted_Sales"]))
    mape = mean_absolute_percentage_error(df_pred["Actual_Sales"], df_pred["Predicted_Sales"])

    metrics = pd.DataFrame({
        "Metric": ["R_squared", "MAPE", "RMSE"],
        "Value": [round(r2, 4), round(mape, 4), round(rmse, 4)]
    })

    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(df_pred["Date"], df_pred["Actual_Sales"], label="Actual")
    ax.plot(df_pred["Date"], df_pred["Predicted_Sales"], label="Predicted")
    ax.set_title("Actual vs Predicted Sales")
    ax.set_xlabel("Date")
    ax.set_ylabel("Sales")
    ax.legend()
    plt.xticks(rotation=15)
    plt.tight_layout()

    return metrics, fig

# ▼ パターンA
def generate_optimal_allocation(model_info, budget, start_date, end_date, constraints={}, disp=False):
    import streamlit as st
    days = pd.date_range(start=start_date, end=end_date)
    n_days = len(days)
    if n_days == 0:
        raise ValueError("指定された期間の日数が0です。start_dateとend_dateを確認してください。")

    n_channels = len(model_info["columns"])
    coefs = np.abs(model_info["model"].coef_[:n_channels])
    coefs = np.where(coefs == 0, 1e-6, coefs)
    weights = coefs / np.sum(coefs)
    init_alloc = np.tile(weights * (budget / n_days), n_days)

    def predict_sales(alloc_matrix):
        transformed = []
        for i in range(n_channels):
            ad = apply_adstock(alloc_matrix[:, i], model_info["betas"][i])
            sat = saturation_transform(ad, model_info["alphas"][i])
            transformed.append(sat)
        X_media = np.array(transformed).T
        df_days = pd.DataFrame({"Date": days})
        extra_df = create_time_features(df_days, df_days["Date"].min(), model_info["extra_cols"])
        X_all = np.concatenate([X_media, extra_df.values], axis=1)
        return model_info["model"].predict(X_all)

    def objective(flat_alloc):
        alloc_matrix = flat_alloc.reshape(n_days, n_channels)
        pred = predict_sales(alloc_matrix)
        return -np.sum(pred)

    constraints_list = [{"type": "eq", "fun": lambda x: np.sum(x) - budget}]
    for j, col in enumerate(model_info["columns"]):
        min_val, max_val = constraints.get(col, (0, budget))
        constraints_list.append({
            "type": "ineq",
            "fun": lambda x, j=j, min_val=min_val: np.sum(x.reshape((n_days, n_channels), order="C")[:, j]) - min_val
        })
        constraints_list.append({
            "type": "ineq",
            "fun": lambda x, j=j, max_val=max_val: max_val - np.sum(x.reshape((n_days, n_channels), order="C")[:, j])
        })

    bounds = [(0, budget) for _ in range(n_days * n_channels)]

    result = minimize(
        objective,
        x0=init_alloc,
        method="SLSQP",
        bounds=bounds,
        constraints=constraints_list,
        options={"disp": disp, "maxiter": 500}
    )

    # ▼▼▼ 最適化結果ログ出力 ▼▼▼
    print("Optimization success:", result.success)
    print("Optimization status message:", result.message)
    print("Optimization objective value:", -result.fun)
    st.write("🔍 **最適化結果**")
    st.write("✅ Optimization success:", result.success)
    st.write("💬 Optimization message:", result.message)
    st.write("🎯 Objective (total predicted sales):", round(-result.fun, 2))

    opt_alloc_matrix = result.x.reshape(n_days, n_channels)
    forecast_df = pd.DataFrame({"Date": days})
    forecast_df["Predicted_Sales"] = predict_sales(opt_alloc_matrix)

    alloc_df = pd.DataFrame(opt_alloc_matrix, columns=model_info["columns"])
    alloc_df.insert(0, "Date", days)

    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(forecast_df["Date"], forecast_df["Predicted_Sales"], label="Predicted Sales")
    ax.set_title("Optimal Allocation Forecast (Pattern A)")
    ax.set_xlabel("Date")
    ax.set_ylabel("Sales")
    ax.legend()
    plt.xticks(rotation=15)
    plt.tight_layout()

    return forecast_df, alloc_df, fig

# ▼ パターンB（変更なし）
def predict_from_uploaded_plan(model_info, df_plan):
    if "Date" in df_plan.columns:
        dates = pd.to_datetime(df_plan["Date"])
        df_plan = df_plan.drop(columns=["Date"])
    else:
        dates = pd.date_range(start=pd.Timestamp.today(), periods=len(df_plan))

    media_cols = model_info["columns"]
    alphas = model_info["alphas"]
    betas = model_info["betas"]
    extra_cols = model_info["extra_cols"]

    transformed = []
    for i, col in enumerate(media_cols):
        ad = apply_adstock(df_plan[col].values, betas[i])
        sat = saturation_transform(ad, alphas[i])
        transformed.append(sat)
    X_media = np.array(transformed).T

    df_days = pd.DataFrame({"Date": dates})
    extra_df = create_time_features(df_days, df_days["Date"].min(), extra_cols)
    X_all = np.concatenate([X_media, extra_df.values], axis=1)

    pred = model_info["model"].predict(X_all)

    forecast_df = pd.DataFrame({"Date": dates, "Predicted_Sales": pred})

    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(forecast_df["Date"], forecast_df["Predicted_Sales"], label="Predicted Sales")
    ax.set_title("Forecast from Uploaded Plan (Pattern B)")
    ax.set_xlabel("Date")
    ax.set_ylabel("Sales")
    ax.legend()
    plt.xticks(rotation=15)
    plt.tight_layout()

    return forecast_df, fig
