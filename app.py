import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from matplotlib.ticker import ScalarFormatter
from utils import (
    train_model, evaluate_model,
    apply_adstock, saturation_transform
)

# ページ設定
st.set_page_config(page_title="MMM Simulation", layout="wide")
st.title("📊 Marketing Mix Modeling Simulator")

# ファイルアップロード
uploaded_file = st.file_uploader("📤 Upload Raw Data (CSV or Excel)", type=["csv", "xlsx"])

if uploaded_file:
    if uploaded_file.name.endswith(".csv"):
        df_raw = pd.read_csv(uploaded_file)
    else:
        df_raw = pd.read_excel(uploaded_file)

    st.success("✅ Data loaded successfully")
    st.dataframe(df_raw.head())

    # モデル学習
    with st.spinner("🔄 Training model..."):
        model_info, df_pred = train_model(df_raw)
    st.success("✅ Model training completed!")

    # モデル評価（実績 vs 予測）
    st.subheader("📈 Actual vs Predicted Sales")
    eval_metrics, eval_plot = evaluate_model(df_raw, df_pred)
    st.pyplot(eval_plot)
    st.dataframe(eval_metrics)

    # 媒体ごとの α・β 表示
    st.subheader("📋 Optimized Parameters per Channel")
    df_params = pd.DataFrame({
        "Channel": model_info["columns"],
        "α (Saturation)": np.round(model_info["alphas"], 4),
        "β (Adstock)": np.round(model_info["betas"], 4)
    })
    st.dataframe(df_params)

    # ▼ 最大コスト（外れ値除去済み）＋ 2つのスライダー（最大値+1000万まで拡張）
    raw_costs = df_raw[model_info["columns"]].values.flatten()
    default_max = int(np.percentile(raw_costs, 95))
    max_limit = int(np.max(raw_costs)) + 10_000_000

    # ▼ 1. 構造分析グラフ（Saturation のみ、回帰係数・Adstockなし）
    st.subheader("📊 Transformed Variable Curve (Saturation only, no Adstock / Coefficient)")

    x_max_sat = st.number_input("🖊 SaturationグラフのMaxCost(¥1,000単位)", min_value=1_000, max_value=max_limit, value=default_max, step=1_000)

    cost_vals_sat = np.linspace(0, x_max_sat, 1000)

    fig1, ax1 = plt.subplots(figsize=(10, 5))
    for i, col in enumerate(model_info["columns"]):
        alpha = np.clip(model_info["alphas"][i], 0.05, 0.95)
        y_vals = np.power(cost_vals_sat, alpha)
        ax1.plot(cost_vals_sat, y_vals, label=f"{col} (α={alpha:.2f})")

    ax1.set_title("Transformed Sales Driver by Channel (Saturation Only, no Coefficient)")
    ax1.set_xlabel("Cost (JPY)")
    ax1.set_ylabel("Transformed Variable (Unscaled)")
    ax1.xaxis.set_major_formatter(ticker.FuncFormatter(lambda x, _: f"¥{x:,.0f}"))
    ax1.yaxis.set_major_formatter(ScalarFormatter(useOffset=False))
    ax1.ticklabel_format(style='plain', axis='y')
    ax1.legend(loc="upper center", bbox_to_anchor=(0.5, -0.15), ncol=2)
    st.pyplot(fig1)

    # ▼ 2. 売上貢献グラフ（回帰係数あり）
    st.subheader("📊 Contribution Curve (Adstock + Saturation × Coefficient)")

   x_max_contrib = st.number_input("🖊 貢献グラフのMaxCost(¥1,000単位)", min_value=1_000, max_value=max_limit, value=default_max, step=1_000)

    cost_vals_contrib = np.linspace(0, x_max_contrib, 1000)

    fig2, ax2 = plt.subplots(figsize=(10, 5))
    for i, col in enumerate(model_info["columns"]):
        alpha = np.clip(model_info["alphas"][i], 0.05, 0.95)
        beta = np.clip(model_info["betas"][i], 0.05, 0.95)
        coef = model_info["model"].coef_[i]
        adstock_vals = apply_adstock(cost_vals_contrib, beta)
        sat_vals = saturation_transform(adstock_vals, alpha)
        y_vals = np.array(sat_vals) * coef
        ax2.plot(cost_vals_contrib, y_vals, label=f"{col} (α={alpha:.2f}, β={beta:.2f}, Coef={coef:.2f})")

    ax2.set_title("Predicted Contribution by Channel (A × X)")
    ax2.set_xlabel("Cost (JPY)")
    ax2.set_ylabel("Contribution to Sales")
    ax2.xaxis.set_major_formatter(ticker.FuncFormatter(lambda x, _: f"¥{x:,.0f}"))
    ax2.yaxis.set_major_formatter(ScalarFormatter(useOffset=False))
    ax2.ticklabel_format(style='plain', axis='y')
    ax2.legend(loc="upper center", bbox_to_anchor=(0.5, -0.15), ncol=2)
    st.pyplot(fig2)

    # ▼ 3. 数式表示（補助的に）
    st.subheader("🧮 Functional Formulas per Channel")
    for i, col in enumerate(model_info["columns"]):
        alpha = np.round(model_info["alphas"][i], 3)
        beta = np.round(model_info["betas"][i], 3)
        coef = np.round(model_info["model"].coef_[i], 3)
        formula = f"{coef} × (Adstock(t-1)×{beta} + Cost(t))^{alpha}"
        st.markdown(f"**{col}**: {formula}")
