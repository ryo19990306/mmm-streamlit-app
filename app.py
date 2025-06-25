import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from matplotlib.ticker import ScalarFormatter
from utils import train_model, evaluate_model, apply_adstock, saturation_transform

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

    # 学習 & 評価
    model_info, df_pred = train_model(df_raw)
    model = model_info["model"]
    coefficients = model_info["coefficients"]
    alphas_betas = model_info["alphas_betas"]

    # 評価指標表示
    st.subheader("📈 Model Evaluation")
    st.write(f"R² Score: {model_info['r2']:.4f}")

    # グラフ最大コスト取得（全チャネル共通X軸にする）
    cost_cols = model_info["media_columns"]
    max_cost = df_raw[cost_cols].max().max()
    cost_range = np.linspace(0, max_cost, 100)

    # レスポンス曲線（回帰係数なし）
    st.subheader("🧮 Transformed Variable Curve (Adstock + Saturation, no Coefficient)")
    fig, ax1 = plt.subplots()
    for col in cost_cols:
        alpha, beta = alphas_betas[col]
        response = saturation_transform(apply_adstock(cost_range, alpha), beta)
        ax1.plot(cost_range, response, label=f"{col} (α={alpha:.2f}, β={beta:.2f})")
    ax1.set_xlabel("Cost (JPY)")
    ax1.set_ylabel("Transformed Variable (Unscaled)")
    ax1.set_title("Transformed Sales Driver by Channel (X without Coefficient)")
    ax1.legend(loc="upper center", bbox_to_anchor=(0.5, -0.25), ncol=2)
    ax1.xaxis.set_major_formatter(ticker.FuncFormatter(lambda x, _: f"¥{x:,.0f}"))
    st.pyplot(fig)

    # レスポンス曲線（回帰係数あり）
    st.subheader("🧮 Predicted Contribution Curve (Adstock + Saturation × Coefficient)")
    fig2, ax2 = plt.subplots()
    for col in cost_cols:
        alpha, beta = alphas_betas[col]
        coef = coefficients[col]
        response = coef * saturation_transform(apply_adstock(cost_range, alpha), beta)
        ax2.plot(cost_range, response, label=f"{col} (α={alpha:.2f}, β={beta:.2f}, Coef={coef:.2f})")
    ax2.set_xlabel("Cost (JPY)")
    ax2.set_ylabel("Contribution to Sales")
    ax2.set_title("Predicted Contribution by Channel (A × X)")
    ax2.legend(loc="upper center", bbox_to_anchor=(0.5, -0.25), ncol=2)
    ax2.xaxis.set_major_formatter(ticker.FuncFormatter(lambda x, _: f"¥{x:,.0f}"))
    st.pyplot(fig2)

    # 数式の表示
    st.subheader("🧾 Functional Formulas per Channel")
    for col in cost_cols:
        alpha, beta = alphas_betas[col]
        coef = coefficients[col]
        formula = f"{coef:.2f} × (Adstock(t-1)×{beta:.3f} + Cost(t))^{alpha:.3f}"
        st.markdown(f"**{col}**: {formula}")
