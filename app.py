import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
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

    # モデル学習 & 評価
    model_info, df_pred = train_model(df_raw)

    # 評価指標を表示
    st.subheader("📈 Model Evaluation Metrics")
    eval_metrics, eval_plot = evaluate_model(df_raw, df_pred)
    st.pyplot(eval_plot)
    st.dataframe(eval_metrics)

    # モデル情報の展開
    model = model_info["model"]
    media_cols = model_info["columns"]
    alphas = model_info["alphas"]
    betas = model_info["betas"]
    coefs = model.coef_
    coefficients = dict(zip(media_cols, coefs))
    alphas_betas = dict(zip(media_cols, zip(alphas, betas)))

    # 共通X軸（全チャネル最大コストを元に）
    max_cost = max([df_raw[col].max() for col in media_cols])
    cost_range = np.linspace(0, max_cost, 300)

    # Transformed Variable Curve（回帰係数なし）
    st.subheader("📉 Transformed Variable Curve (Adstock + Saturation, no Coefficient)")
    fig1, ax1 = plt.subplots()
    for col in media_cols:
        alpha, beta = alphas_betas[col]
        adstocked = apply_adstock(cost_range, beta)
        saturated = saturation_transform(adstocked, alpha)
        ax1.plot(cost_range, saturated, label=f"{col} (α={alpha:.2f}, β={beta:.2f})")
    ax1.set_title("Transformed Sales Driver by Channel")
    ax1.set_xlabel("Cost (JPY)")
    ax1.set_ylabel("Transformed Value")
    ax1.xaxis.set_major_formatter(ticker.FuncFormatter(lambda x, _: f"¥{x:,.0f}"))
    ax1.yaxis.set_major_formatter(ticker.FuncFormatter(lambda x, _: f"¥{x:,.0f}"))
    ax1.legend(loc="upper center", bbox_to_anchor=(0.5, -0.25), ncol=2)
    st.pyplot(fig1)

    # Functional Curve（回帰係数あり）
    st.subheader("📈 Predicted Contribution Curve (Adstock + Saturation × Coefficient)")
    fig2, ax2 = plt.subplots()
    for col in media_cols:
        alpha, beta = alphas_betas[col]
        coef = coefficients[col]
        adstocked = apply_adstock(cost_range, beta)
        saturated = saturation_transform(adstocked, alpha)
        contribution = np.array(saturated) * coef
        ax2.plot(cost_range, contribution, label=f"{col} (Coef={coef:.2f})")
    ax2.set_title("Predicted Contribution by Channel")
    ax2.set_xlabel("Cost (JPY)")
    ax2.set_ylabel("Sales Contribution (JPY)")
    ax2.xaxis.set_major_formatter(ticker.FuncFormatter(lambda x, _: f"¥{x:,.0f}"))
    ax2.yaxis.set_major_formatter(ticker.FuncFormatter(lambda x, _: f"¥{x:,.0f}"))
    ax2.legend(loc="upper center", bbox_to_anchor=(0.5, -0.25), ncol=2)
    st.pyplot(fig2)

    # Functional Formula 表示
    st.subheader("🧾 Functional Formulas per Channel")
    for col in media_cols:
        alpha, beta = alphas_betas[col]
        coef = coefficients[col]
        formula = f"{coef:.2f} × (Adstock(t-1)×{beta:.3f} + Cost(t))^{alpha:.3f}"
        st.markdown(f"**{col}**: {formula}")