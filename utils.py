# force update to clear streamlit cache
import pandas as pd
import numpy as np
import jpholiday
from sklearn.linear_model import Ridge
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
    return result.tolist()

# ▼ Saturation変換
def saturation_transform(x, alpha):
    return [max(i, 0)**alpha for i in x]

# ▼ 最適化目的関数（R²最大化）
def objective_alpha_beta(params, trainX, y, media_cols):
    alphas = params[:len(media_cols)]
    betas = params[len(media_cols):]

    X_transformed = []
    for i, col in enumerate(media_cols):
        ad = apply_adstock(trainX[col].values, betas[i])
        sat = saturation_transform(ad, alphas[i])
        X_transformed.append(sat)
    X_media = np.array(X_transformed).T
    X_extra = trainX.drop(columns=media_cols).values
    X_all = np.concatenate([X_media, X_extra], axis=1)

    model = Ridge(alpha=1.0).fit(X_all, y)
    pred = model.predict(X_all)
    return -r2_score(y, pred)

# ▼ モデル学習：train_model()
def train_model(df_raw):
    df = df_raw.copy()
    df.columns = df.columns.str.strip()
    df["Date"] = pd.to_datetime(df["Date"], errors="coerce")
    df["Sales"] = pd.to_numeric(df["Sales"], errors="coerce")

    X = df.drop(columns=["Date", "Sales"], errors="ignore").apply(pd.to_numeric, errors="coerce")
    valid_idx = X.dropna().index & df["Sales"].dropna().index
    X = X.loc[valid_idx]
    y = df.loc[valid_idx, "Sales"]
    df = df.dropna(subset=["Date"])
    df["Date"] = pd.to_datetime(df["Date"])

    # extra features
    df["weekday"] = df["Date"].dt.weekday
    weekday_dummies = pd.get_dummies(df["weekday"], prefix="wd", drop_first=True)
    month_dummies = pd.get_dummies(df["Date"].dt.month, prefix="month", drop_first=True)
    df["is_holiday"] = df["Date"].apply(lambda x: jpholiday.is_holiday(x) or x.weekday() >= 5).astype(int)
    df["trend"] = (df["Date"] - df["Date"].min()).dt.days
    extra_features = pd.concat([weekday_dummies, month_dummies, df[["is_holiday", "trend"]]], axis=1)
    extra_features = extra_features.loc[X.index]
    X = pd.concat([X, extra_features], axis=1)

    non_media_prefixes = ["wd_", "month_", "trend", "is_holiday"]
    media_cols = [col for col in X.columns if not any(col.startswith(p) for p in non_media_prefixes)]

    n_media = len(media_cols)
    init_params = [0.5] * n_media * 2
    bounds = [(0.05, 0.95)] * n_media * 2

    res = minimize(
        objective_alpha_beta,
        x0=init_params,
        args=(X, y, media_cols),
        bounds=bounds,
        method="L-BFGS-B"
    )

    alphas = res.x[:n_media]
    betas = res.x[n_media:]

    X_transformed = []
    for i, col in enumerate(media_cols):
        ad = apply_adstock(X[col].values, betas[i])
        sat = saturation_transform(ad, alphas[i])
        X_transformed.append(sat)
    X_media = np.array(X_transformed).T
    X_extra = X.drop(columns=media_cols).values
    X_all = np.concatenate([X_media, X_extra], axis=1)

    model = Ridge(alpha=1.0).fit(X_all, y)
    pred = model.predict(X_all)

    df_pred = pd.DataFrame({
        "Date": df.loc[X.index, "Date"],
        "Actual_Sales": y,
        "Predicted_Sales": pred
    })

    model_info = {
        "model": model,
        "alphas": alphas,
        "betas": betas,
        "columns": media_cols,
        "extra_cols": X.drop(columns=media_cols).columns.tolist()
    }

    return model_info, df_pred

# ▼ モデル評価：evaluate_model()
def evaluate_model(df_raw, df_pred):
    r2 = r2_score(df_pred["Actual_Sales"], df_pred["Predicted_Sales"])
    rmse = np.sqrt(mean_squared_error(df_pred["Actual_Sales"], df_pred["Predicted_Sales"]))
    mape = mean_absolute_percentage_error(df_pred["Actual_Sales"], df_pred["Predicted_Sales"]) * 100

    metrics = pd.DataFrame({
        "指標": ["R_squared", "MAPE", "RMSE"],
        "値": [round(r2, 4), round(mape, 4), round(rmse, 4)]
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



from datetime import timedelta

def generate_optimal_allocation(model_info, budget, start_date, end_date):
    days = pd.date_range(start=start_date, end=end_date)
    n_days = len(days)
    n_channels = len(model_info["columns"])

    # 1日あたりの全体予算 → 均等配分
    daily_budget = budget / n_days
    daily_per_channel = daily_budget / n_channels

    # 日×施策で予算を構築（配列）
    alloc_matrix = np.full((n_days, n_channels), daily_per_channel)

    alphas = model_info["alphas"]
    betas = model_info["betas"]
    columns = model_info["columns"]
    extra_cols = model_info["extra_cols"]

    # Adstock + Saturation
    transformed = []
    for i in range(n_channels):
        ad = apply_adstock(alloc_matrix[:, i], betas[i])
        sat = saturation_transform(ad, alphas[i])
        transformed.append(sat)
    X_media = np.array(transformed).T

    # Extra features: 曜日・月・祝日・トレンドなど
    df_days = pd.DataFrame({"Date": days})
    df_days["weekday"] = df_days["Date"].dt.weekday
    weekday_dummies = pd.get_dummies(df_days["weekday"], prefix="wd", drop_first=True)
    month_dummies = pd.get_dummies(df_days["Date"].dt.month, prefix="month", drop_first=True)
    df_days["is_holiday"] = df_days["Date"].apply(lambda x: jpholiday.is_holiday(x) or x.weekday() >= 5).astype(int)
    df_days["trend"] = (df_days["Date"] - df_days["Date"].min()).dt.days

    extra_df = pd.concat([weekday_dummies, month_dummies, df_days[["is_holiday", "trend"]]], axis=1)
    # 必要な列だけ取り出す（学習時と同じ構成に合わせる）
    extra_df = extra_df.reindex(columns=extra_cols, fill_value=0)
    X_extra = extra_df.values

    # 合体 → 予測
    X_all = np.concatenate([X_media, X_extra], axis=1)
    pred = model_info["model"].predict(X_all)

    # 出力データ
    forecast_df = pd.DataFrame({"Date": days, "Predicted_Sales": pred})
    alloc_df = pd.DataFrame(alloc_matrix, columns=columns)
    alloc_df["Date"] = days

    # グラフ出力
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(forecast_df["Date"], forecast_df["Predicted_Sales"], label="Predicted Sales")
    ax.set_title("Future Sales Forecast (Pattern A)")
    ax.set_xlabel("Date")
    ax.set_ylabel("Sales")
    ax.legend()
    plt.xticks(rotation=15)
    plt.tight_layout()

    return forecast_df, alloc_df, fig



def predict_from_uploaded_plan(model_info, df_plan):
    df_plan = df_plan.copy()
    if "Date" in df_plan.columns:
        df_plan["Date"] = pd.to_datetime(df_plan["Date"])
        dates = df_plan["Date"]
        df_plan = df_plan.drop(columns=["Date"])
    else:
        dates = pd.date_range(start=pd.Timestamp.today(), periods=len(df_plan))

    media_cols = model_info["columns"]
    alphas = model_info["alphas"]
    betas = model_info["betas"]
    extra_cols = model_info["extra_cols"]

    # Adstock + Saturation
    transformed = []
    for i, col in enumerate(media_cols):
        ad = apply_adstock(df_plan[col].values, betas[i])
        sat = saturation_transform(ad, alphas[i])
        transformed.append(sat)
    X_media = np.array(transformed).T

    # Extra features（Dateがある場合）
    df_days = pd.DataFrame({"Date": dates})
    df_days["weekday"] = df_days["Date"].dt.weekday
    weekday_dummies = pd.get_dummies(df_days["weekday"], prefix="wd", drop_first=True)
    month_dummies = pd.get_dummies(df_days["Date"].dt.month, prefix="month", drop_first=True)
    df_days["is_holiday"] = df_days["Date"].apply(lambda x: jpholiday.is_holiday(x) or x.weekday() >= 5).astype(int)
    df_days["trend"] = (df_days["Date"] - df_days["Date"].min()).dt.days

    extra_df = pd.concat([weekday_dummies, month_dummies, df_days[["is_holiday", "trend"]]], axis=1)
    extra_df = extra_df.reindex(columns=extra_cols, fill_value=0)
    X_extra = extra_df.values

    X_all = np.concatenate([X_media, X_extra], axis=1)
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
