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

#ページの基本設定
st.set_page_config(page_title="MMM Simulation", layout="wide")
st.title("📊 Marketing Mix Modeling Simulator")

#ファイルのアップロード受付（CSVまたはExcel）
uploaded_file = st.file_uploader("📤 Upload Raw Data (CSV or Excel)", type=["csv", "xlsx"])

if uploaded_file:
    #アップロードされたファイルの読み込み
    if uploaded_file.name.endswith(".csv"):
        df_raw = pd.read_csv(uploaded_file)
    else:
        df_raw = pd.read_excel(uploaded_file)

    #読み込み成功メッセージと冒頭データの表示
    st.success("✅ Data loaded successfully")
    st.dataframe(df_raw.head())

    #モデルの学習処理（内部でデータ整形・ダミー変数追加などを実施）
    st.info("🔄 Training model...")
    model_info, df_pred = train_model(df_raw)

    #実績と予測の比較グラフを描画 & 評価指標を表形式で表示
    st.subheader("📈 Actual vs Predicted Sales")
    eval_metrics, eval_plot = evaluate_model(df_raw, df_pred)
    st.pyplot(eval_plot)
    st.dataframe(eval_metrics)

    #最適化された α・β パラメータを媒体ごとに表示
    st.subheader("📋 Optimized Parameters per Channel")
    df_params = pd.DataFrame({
        "Channel": model_info["columns"],
        "α (Saturation)": np.round(model_info["alphas"], 4),
        "β (Adstock)": np.round(model_info["betas"], 4)
    })
    st.dataframe(df_params)

    #各媒体のレスポンス曲線（Adstock → Saturation）を描画
    st.subheader("📊 Response Curves (Adstock → Saturation)")
    fig1, ax1 = plt.subplots(figsize=(10, 5))
    for i, col in enumerate(model_info["columns"]):
        alpha = np.clip(model_info["alphas"][i], 0.05, 0.95)
        beta = np.clip(model_info["betas"][i], 0.05, 0.95)

        global_max_cost = df_raw[model_info["columns"]].max().max() + 1_000_000
        cost_vals = np.linspace(0, global_max_cost, 300)

        #Adstock → Saturation変換
        adstock_vals = apply_adstock(cost_vals, beta)
        sat_vals = saturation_transform(adstock_vals, alpha)

        #曲線の描画
        ax1.plot(cost_vals, sat_vals, label=f"{col} (α={alpha:.2f}, β={beta:.2f})")
    
    ax1.set_title("Response Curve by Channel (Adstock → Saturation)")
    ax1.set_xlabel("Cost (JPY)")
    ax1.set_ylabel("Response (Unscaled)")
    ax1.yaxis.set_major_formatter(ScalarFormatter(useOffset=False))
    ax1.ticklabel_format(style='plain', axis='y')
    ax1.xaxis.set_major_formatter(ticker.FuncFormatter(lambda x, _: f"¥{x:,.0f}"))
    ax1.legend(loc="upper center", bbox_to_anchor=(0.5, -0.15), ncol=2)
    st.pyplot(fig1)

    #各媒体の貢献度カーブ（Saturation × 回帰係数）を描画
    st.subheader("📊 Functional Curve (Adstock + Saturation × Coefficient)")
    fig2, ax2 = plt.subplots(figsize=(10, 5))
    for i, col in enumerate(model_info["columns"]):
        alpha = np.clip(model_info["alphas"][i], 0.05, 0.95)
        beta = np.clip(model_info["betas"][i], 0.05, 0.95)
        coef = model_info["model"].coef_[i]

        global_max_cost = df_raw[model_info["columns"]].max().max() + 1_000_000
        cost_vals = np.linspace(0, global_max_cost, 300)

        #Adstock → Saturation → Contribution変換
        adstock_vals = apply_adstock(cost_vals, beta)
        sat_vals = saturation_transform(adstock_vals, alpha)
        contribution_vals = np.array(sat_vals) * coef

        #曲線の描画
        ax2.plot(cost_vals, contribution_vals, label=f"{col} (α={alpha:.2f}, β={beta:.2f})")
    
    ax2.set_title("Functional Curve by Channel (Response × Coefficient)")
    ax2.set_xlabel("Cost (JPY)")
    ax2.set_ylabel("Contribution (Scaled)")
    ax2.yaxis.set_major_formatter(ScalarFormatter(useOffset=False))
    ax2.ticklabel_format(style='plain', axis='y')
    ax2.xaxis.set_major_formatter(ticker.FuncFormatter(lambda x, _: f"¥{x:,.0f}"))
    ax2.legend(loc="upper center", bbox_to_anchor=(0.5, -0.15), ncol=2)
    st.pyplot(fig2)

    #各媒体の関数数式（人間が理解しやすい形で）を表示
    st.subheader("🧮 Functional Formulas per Channel")
    for i, col in enumerate(model_info["columns"]):
        alpha = np.round(model_info["alphas"][i], 3)
        beta = np.round(model_info["betas"][i], 3)
        coef = np.round(model_info["model"].coef_[i], 3)
        formula = f"{coef} × (Adstock(t-1)×{beta} + Cost(t))^{alpha}"
        st.markdown(f"**{col}**: {formula}")
