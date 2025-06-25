# app.py（Functional Formula 表示付き 完全版）

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from matplotlib.ticker import FuncFormatter
from utils import (
    train_model,
    evaluate_model,
    apply_adstock,
    saturation_transform
)

# ページ設定
st.set_page_config(page_title="MMM Simulation", layout="wide")
st.title("📊 Marketing Mix Modeling Simulator")

# ファイルアップロード
uploaded_file = st.file_uploader("📎 Upload Raw Data (CSV or Excel)", type=["csv", "xlsx"])

if uploaded_file:
    if uploaded_file.name.endswith(".csv"):
        df_raw = pd.read_csv(uploaded_file)
    else:
        df_raw = pd.read_excel(uploaded_file)

    st.subheader("📄 Uploaded Raw Data")
    st.dataframe(df_raw.head())

    # モデル学習・評価
    model, X, y, y_pred, adstocked_df, saturated_df, coefficients, alphas_betas = train_model(df_raw)
    r2, rmse, mape = evaluate_model(y, y_pred)

    st.subheader("📈 Model Evaluation Metrics")
    st.markdown(f"- R²: **{r2:.3f}**")
    st.markdown(f"- RMSE: **{rmse:.2f}**")
    st.markdown(f"- MAPE: **{mape:.2f}%**")

    # コストの範囲を統一して処理
    max_cost = df_raw.drop(columns=['date', 'sales']).max().max()
    cost_range = np.linspace(0, max_cost, 100)

    # 1. 回帰係数なしの変換後Xプロット
    fig1, ax1 = plt.subplots()
    for col in df_raw.columns:
        if col in ['date', 'sales']:
            continue
        alpha, beta = alphas_betas[col]
        adstocked = [cost_range[0]]
        for t in range(1, len(cost_range)):
            ad_val = cost_range[t] + alpha * adstocked[-1]
            adstocked.append(ad_val)
        adstocked = np.array(adstocked)
        transformed = np.power(adstocked * beta + cost_range, 1 - beta)
        ax1.plot(cost_range, transformed, label=f"{col} (α={alpha:.2f}, β={beta:.2f})")

    ax1.set_title("Transformed Sales Driver by Channel (X without Coefficient)")
    ax1.set_xlabel("Cost (JPY)")
    ax1.set_ylabel("Transformed Variable (Unscaled)")
    ax1.yaxis.set_major_formatter(FuncFormatter(lambda x, _: f"¥{x:,.0f}"))
    ax1.xaxis.set_major_formatter(FuncFormatter(lambda x, _: f"¥{x:,.0f}"))
    ax1.legend(loc='upper center', bbox_to_anchor=(0.5, -0.15), ncol=2)
    st.pyplot(fig1)

    # 2. 回帰係数ありの貢献度プロット
    fig2, ax2 = plt.subplots()
    for col in df_raw.columns:
        if col in ['date', 'sales']:
            continue
        alpha, beta = alphas_betas[col]
        coef = coefficients[col]
        adstocked = [cost_range[0]]
        for t in range(1, len(cost_range)):
            ad_val = cost_range[t] + alpha * adstocked[-1]
            adstocked.append(ad_val)
        adstocked = np.array(adstocked)
        transformed = np.power(adstocked * beta + cost_range, 1 - beta)
        contribution = transformed * coef
        ax2.plot(cost_range, contribution,
                 label=f"{col} (α={alpha:.2f}, β={beta:.2f}, Coef={coef:.2f})")

    ax2.set_title("Predicted Contribution by Channel (A × X)")
    ax2.set_xlabel("Cost (JPY)")
    ax2.set_ylabel("Contribution to Sales")
    ax2.yaxis.set_major_formatter(FuncFormatter(lambda x, _: f"¥{x:,.0f}"))
    ax2.xaxis.set_major_formatter(FuncFormatter(lambda x, _: f"¥{x:,.0f}"))
    ax2.legend(loc='upper center', bbox_to_anchor=(0.5, -0.15), ncol=2)
    st.pyplot(fig2)

    # 3. 数式を表示
    st.subheader("🧮 Functional Formulas per Channel")
    for col in df_raw.columns:
        if col in ['date', 'sales']:
            continue
        alpha, beta = alphas_betas[col]
        coef = coefficients[col]
        st.markdown(
            f"**{col}**: {coef:.3f} × (Adstock(t−1)×{alpha:.3f} + Cost(t))^{1 - beta:.3f}"
        )