
import pandas as pd
import numpy as np
from sklearn.linear_model import Ridge
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_percentage_error
import matplotlib.pyplot as plt

# --- Adstock変換 ---
def apply_adstock(x, beta):
    x = np.array(x)
    result = np.zeros_like(x)
    result[0] = x[0]
    for t in range(1, len(x)):
        result[t] = x[t] + beta * result[t - 1]
    return result

# --- Saturation変換 ---
def saturation_transform(x, alpha):
    return np.power(x, alpha)

# --- α・βの仮パラメータ（本来は最適化で算出） ---
def default_alpha_beta(n):
    alphas = np.full(n, 0.8)
    betas = np.full(n, 0.5)
    return alphas, betas

# --- モデル学習 ---
def train_model(df_raw):
    df = df_raw.copy()
    y = df["Sales"].values
    X = df.drop(columns=["Date", "Sales"])
    alphas, betas = default_alpha_beta(X.shape[1])

    transformed = []
    for i, col in enumerate(X.columns):
        ad = apply_adstock(X[col].values, betas[i])
        sat = saturation_transform(ad, alphas[i])
        transformed.append(sat)
    X_trans = np.array(transformed).T

    model = Ridge(alpha=1.0)
    model.fit(X_trans, y)
    pred = model.predict(X_trans)

    df_pred = pd.DataFrame({"Date": df["Date"], "Actual_Sales": y, "Predicted_Sales": pred})
    model_info = {
        "model": model,
        "alphas": alphas,
        "betas": betas,
        "columns": X.columns.tolist()
    }
    return model_info, df_pred

# --- モデル評価 ---
def evaluate_model(df_raw, df_pred):
    r2 = r2_score(df_pred["Actual_Sales"], df_pred["Predicted_Sales"])
    rmse = np.sqrt(mean_squared_error(df_pred["Actual_Sales"], df_pred["Predicted_Sales"]))
    mape = mean_absolute_percentage_error(df_pred["Actual_Sales"], df_pred["Predicted_Sales"]) * 100

    metrics = pd.DataFrame({
        "指標": ["R_squared", "MAPE", "RMSE"],
        "値": [r2, mape, rmse]
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

# --- パターンA：予算入力 → 最適配分 & 予測 ---
def generate_optimal_allocation(model_info, total_budget, start_date, end_date):
    days = pd.date_range(start=start_date, end=end_date)
    n_days = len(days)
    n_channels = len(model_info["columns"])
    daily_budget = total_budget / n_days
    budget_per_channel = daily_budget / n_channels

    alloc_matrix = np.full((n_days, n_channels), budget_per_channel)
    alphas = model_info["alphas"]
    betas = model_info["betas"]

    transformed = []
    for i in range(n_channels):
        ad = apply_adstock(alloc_matrix[:, i], betas[i])
        sat = saturation_transform(ad, alphas[i])
        transformed.append(sat)
    X_new = np.array(transformed).T

    pred = model_info["model"].predict(X_new)
    forecast_df = pd.DataFrame({"Date": days, "Predicted_Sales": pred})
    alloc_df = pd.DataFrame(alloc_matrix, columns=model_info["columns"])
    alloc_df["Date"] = days

    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(forecast_df["Date"], forecast_df["Predicted_Sales"], label="Predicted Sales")
    ax.set_title("Future Sales Forecast (Pattern A)")
    ax.set_xlabel("Date")
    ax.set_ylabel("Sales")
    ax.legend()
    plt.xticks(rotation=15)
    plt.tight_layout()

    return forecast_df, alloc_df, fig

# --- パターンB：将来予算ファイル → 予測 ---
def predict_from_uploaded_plan(model_info, df_plan):
    df_plan = df_plan.copy()
    df_plan = df_plan.drop(columns=["Date"]) if "Date" in df_plan.columns else df_plan

    alphas = model_info["alphas"]
    betas = model_info["betas"]
    transformed = []
    for i, col in enumerate(model_info["columns"]):
        ad = apply_adstock(df_plan[col].values, betas[i])
        sat = saturation_transform(ad, alphas[i])
        transformed.append(sat)
    X_new = np.array(transformed).T

    pred = model_info["model"].predict(X_new)
    dates = pd.date_range(start=pd.Timestamp.today(), periods=len(df_plan))
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
