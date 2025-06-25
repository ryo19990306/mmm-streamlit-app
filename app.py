import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from matplotlib.ticker import ScalarFormatter
from utils import (
    train_model, evaluate_model,
    apply_adstock, saturation_transform,
    generate_optimal_allocation, predict_from_uploaded_plan
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

     #分岐選択（A or B）
    option = st.radio("🛠 パターン選択", ["パターンA：予算最適化（期間＋予算）", "パターンB：日別予算アップロード"])

    if option == "パターンA：予算最適化（期間＋予算）":
        st.header("🅰 期間・予算を指定して最適予算配分")

        start_date = st.date_input("開始日")
        end_date = st.date_input("終了日")
        budget = st.number_input("💰 総予算（円）", min_value=1000, step=10000)

        st.markdown("🔧 媒体ごとの下限〜上限予算を入力（任意）")
        constraints = {}
        for col in model_info["columns"]:
            col1, col2 = st.columns(2)
            with col1:
                min_val = st.number_input(f"{col} の下限", min_value=0, step=1000, value=0)
            with col2:
                max_val = st.number_input(f"{col} の上限", min_value=0, step=1000, value=budget)
            constraints[col] = (min_val, max_val)

        if st.button("🚀 最適予算配分を実行"):
            forecast_df, alloc_df, fig = generate_optimal_allocation(
                model_info, budget, start_date, end_date, constraints
            )
            st.subheader("📊 売上予測グラフ")
            st.pyplot(fig)

            st.subheader("📄 売上予測テーブル")
            st.dataframe(forecast_df)
            st.download_button("📥 売上予測をダウンロード", forecast_df.to_csv(index=False), "forecast.csv", "text/csv")

            st.subheader("📄 施策別予算配分テーブル")
            st.dataframe(alloc_df)
            st.download_button("📥 配分予算をダウンロード", alloc_df.to_csv(index=False), "allocation.csv", "text/csv")

    elif option == "パターンB：日別予算アップロード":
        st.header("🅱 アップロードした予算に基づく売上予測")

        uploaded_plan = st.file_uploader("📤 予算ファイルをアップロード（CSV/Excel）", type=["csv", "xlsx"])
        if uploaded_plan:
            if uploaded_plan.name.endswith(".csv"):
                df_plan = pd.read_csv(uploaded_plan)
            else:
                df_plan = pd.read_excel(uploaded_plan)

            st.success("✅ 予算データを読み込みました")
            st.dataframe(df_plan.head())

            forecast_df, fig = predict_from_uploaded_plan(model_info, df_plan)

            st.subheader("📊 売上予測グラフ")
            st.pyplot(fig)

            st.subheader("📄 売上予測テーブル")
            st.dataframe(forecast_df)
            st.download_button("📥 売上予測をダウンロード", forecast_df.to_csv(index=False), "forecast_b.csv", "text/csv")
