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
    st.info("🔄 Training model...")
    model_info, df_pred = train_model(df_raw)

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

    # 全チャネル共通の最大コスト設定（比較可能なX軸）
    global_max_cost = df_raw[model_info["columns"]].max().max() + 1_000_000
    cost_vals = np.linspace(0, global_max_cost, 300)

    # ▼ 1. 構造分析グラフ（回帰係数なし）＝変換後のXそのもの
    st.subheader("📊 Transformed Variable Curve (Adstock + Saturation, no Coefficient)")

    fig1, ax1 = plt.subplots(figsize=(10, 5))
    for i, col in enumerate(model_info["columns"]):
        alpha = np.clip(model_info["alphas"][i], 0.05, 0.95)
        beta = np.clip(model_info["betas"][i], 0.05, 0.95)
        adstock_vals = apply_adstock(cost_vals, beta)
        sat_vals = saturation_transform(adstock_vals, alpha)
        ax1.plot(cost_vals, sat_vals, label=f"{col} (α={alpha:.2f}, β={beta:.2f})")

    ax1.set_title("Transformed Sales Driver by Channel (X without Coefficient)")
    ax1.set_xlabel("Cost (JPY)")
    ax1.set_ylabel("Transformed Variable (Unscaled)")
    ax1.xaxis.set_major_formatter(ticker.FuncFormatter(lambda x, _: f"¥{x:,.0f}"))
    ax1.yaxis.set_major_formatter(ScalarFormatter(useOffset=False))
    ax1.ticklabel_format(style='plain', axis='y')
    ax1.legend(loc="upper center", bbox_to_anchor=(0.5, -0.15), ncol=2)
    st.pyplot(fig1)

    st.markdown("""
    📌 このグラフはチャネルごとの反応構造（Adstock + Saturation後）を示しています。  
    回帰係数は含まれていないため、チャネルがどのように売上ドライバーとして反応するか（構造的な変換効率）を比較できます。
    """)

    # ▼ 2. 売上貢献グラフ（回帰係数あり）＝Ax（貢献）
    st.subheader("📊 Contribution Curve (Adstock + Saturation × Coefficient)")

    fig2, ax2 = plt.subplots(figsize=(10, 5))
    for i, col in enumerate(model_info["columns"]):
        alpha = np.clip(model_info["alphas"][i], 0.05, 0.95)
        beta = np.clip(model_info["betas"][i], 0.05, 0.95)
        coef = model_info["model"].coef_[i]
        adstock_vals = apply_adstock(cost_vals, beta)
        sat_vals = saturation_transform(adstock_vals, alpha)
        y_vals = np.array(sat_vals) * coef
        ax2.plot(cost_vals, y_vals, label=f"{col} (α={alpha:.2f}, β={beta:.2f}, Coef={coef:.2f})")

    ax2.set_title("Predicted Contribution by Channel (A × X)")
    ax2.set_xlabel("Cost (JPY)")
    ax2.set_ylabel("Contribution to Sales")
    ax2.xaxis.set_major_formatter(ticker.FuncFormatter(lambda x, _: f"¥{x:,.0f}"))
    ax2.yaxis.set_major_formatter(ScalarFormatter(useOffset=False))
    ax2.ticklabel_format(style='plain', axis='y')
    ax2.legend(loc="upper center", bbox_to_anchor=(0.5, -0.15), ncol=2)
    st.pyplot(fig2)

    st.markdown("""
    📌 このグラフはチャネルごとの売上貢献度（回帰係数 × 変換構造）を示しています。  
    モデルが学習した回帰係数を反映しており、売上への実際の寄与を可視化しています。
    """)

    # ▼ 数式表示（補助的に）
    st.subheader("🧮 Functional Formulas per Channel")
    for i, col in enumerate(model_info["columns"]):
        alpha = np.round(model_info["alphas"][i], 3)
        beta = np.round(model_info["betas"][i], 3)
        coef = np.round(model_info["model"].coef_[i], 3)
        formula = f"{coef} × (Adstock(t-1)×{beta} + Cost(t))^{alpha}"
        st.markdown(f"**{col}**: {formula}")