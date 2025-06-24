import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from utils import (
    train_model, evaluate_model,
    generate_optimal_allocation, predict_from_uploaded_plan,
    apply_adstock, saturation_transform
)

st.set_page_config(page_title="MMM予測シミュレーション", layout="wide")
st.title("📊 MMM予算シミュレーション（パターン選択＋関数可視化対応）")

uploaded_file = st.file_uploader("📤 Rawデータをアップロード（CSVまたはExcel）", type=["csv", "xlsx"])

if uploaded_file:
    if uploaded_file.name.endswith(".csv"):
        df_raw = pd.read_csv(uploaded_file)
    else:
        df_raw = pd.read_excel(uploaded_file)

    st.success("✅ データ読み込み成功")
    st.dataframe(df_raw.head())

    st.info("🔄 モデル学習中...")
    model_info, df_pred = train_model(df_raw)

    st.subheader("📈 実績 vs 予測 売上")
    eval_metrics, eval_plot = evaluate_model(df_raw, df_pred)
    st.pyplot(eval_plot)
    st.dataframe(eval_metrics)

    st.subheader("📋 媒体別 最適化パラメータ（α・β）")
    df_params = pd.DataFrame({
        "施策": model_info["columns"],
        "α（飽和度）": np.round(model_info["alphas"], 4),
        "β（広告効果の遅延）": np.round(model_info["betas"], 4)
    })
    st.dataframe(df_params)

    st.subheader("📊 各施策の関数構造グラフ（Adstock + Saturation）")

    x_vals = np.linspace(0, 20, 100)
    fig, ax = plt.subplots(figsize=(8, 4))

    for i, col in enumerate(model_info["columns"]):
        alpha = max(0.1, min(model_info["alphas"][i], 0.9))
        beta = max(0.1, min(model_info["betas"][i], 0.9))
        coef = model_info["model"].coef_[i]

        adstock_vals = x_vals  # 仮想Adstock値
        sat_vals = np.power(np.maximum(adstock_vals, 0), alpha)
        y_vals = sat_vals * coef

        ax.plot(x_vals, y_vals, label=f"{col} (α={alpha:.2f}, β={beta:.2f})")

    ax.set_title("各施策の反応曲線")
    ax.legend()
    st.pyplot(fig)

    st.subheader("📐 各施策の数式")
    for i, col in enumerate(model_info["columns"]):
        alpha = max(0.1, min(model_info["alphas"][i], 0.9))
        beta = max(0.1, min(model_info["betas"][i], 0.9))
        coef = model_info["model"].coef_[i]

        st.markdown(f"### 🔹 {col}")
        st.latex(f"\\text{{貢献}} = ( {col}(t-1) \\times {beta:.3f} + \\text{{Spent}}(t) )^{{{alpha:.3f}}} \\times {coef:.3f}")

    st.subheader("🧩 パターン選択")
    pattern = st.radio("予測パターンを選択してください", ["パターンA：予算と期間を指定", "パターンB：予算配分ファイルをアップロード"])

    if pattern == "パターンA：予算と期間を指定":
        budget = st.number_input("📌 予算（万円）", min_value=1000, max_value=100000, value=10000, step=100)
        start_date = st.date_input("📅 予測開始日")
        end_date = st.date_input("📅 終了日を予測")

        if st.button("🚀 シミュレーション実行"):
            forecast_df, alloc_df, fig = generate_optimal_allocation(model_info, budget, start_date, end_date)
            st.success("✅ シミュレーション成功")
            st.pyplot(fig)
            st.dataframe(forecast_df)
            st.download_button("📥 売上予測をCSVでダウンロード", forecast_df.to_csv(index=False), file_name="forecast_patternA.csv", mime="text/csv")

    elif pattern == "パターンB：予算配分ファイルをアップロード":
        uploaded_plan = st.file_uploader("📤 予算配分ファイル（CSV）", type=["csv"], key="plan_upload")

        if uploaded_plan:
            df_plan = pd.read_csv(uploaded_plan)
            st.dataframe(df_plan.head())

            if st.button("🚀 シミュレーション実行", key="run_patternB"):
                forecast_df, fig = predict_from_uploaded_plan(model_info, df_plan)
                st.success("✅ シミュレーション成功")
                st.pyplot(fig)
                st.dataframe(forecast_df)
                st.download_button("📥 売上予測をCSVでダウンロード", forecast_df.to_csv(index=False), file_name="forecast_patternB.csv", mime="text/csv")
