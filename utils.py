
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
    return result

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

# ▼ train_model関数
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

    # ▼ extra features
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
    bounds = [(0.01, 1.0)] * n_media * 2

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

    df_pred = pd.DataFrame({"Date": df.loc[X.index, "Date"], "Actual_Sales": y, "Predicted_Sales": pred})
    model_info = {
        "model": model,
        "alphas": alphas,
        "betas": betas,
        "columns": media_cols,
        "extra_cols": X.drop(columns=media_cols).columns.tolist()
    }
    return model_info, df_pred

# ▼ モデル評価関数
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
