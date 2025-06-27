import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from matplotlib.ticker import ScalarFormatter
import matplotlib

# ▼ 日本語フォント設定（IPAexGothic例）
matplotlib.rcParams['font.family'] = 'IPAexGothic'

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

def read_uploaded_file(uploaded_file):
    if uploaded_file.name.endswith(".csv"):
        try:
            df_raw = pd.read_csv(uploaded_file, encoding="utf-8")
        except Exception:
            uploaded_file.seek(0)
            df_raw = pd.read_csv(uploaded_file, encoding="shift_jis")
    else:
        df_raw = pd.read_excel(uploaded_file, sheet_name=0)
        if isinstance(df_raw, dict):
            df_raw = df_raw[list(df_raw.keys())[0]]
    return df_raw

if uploaded_file:
    df_raw = read_uploaded_file(uploaded_file)
    st.success("✅ Data loaded successfully")
    st.dataframe(df_raw.head())

    # モデル学習
    with st.spinner("🔄 Training model..."):
        model_info, pred = train_model(df_raw)
    st.success("✅ Model training completed!")

    # 予測結果DataFrame作成
    df_pred = df_raw.copy()
    df_pred = df_pred.loc[pred.index]
    df_pred["Predicted_Sales"] = pred
    df_pred.rename(columns={"Sales": "Actual_Sales"}, inplace=True)

    # モデル評価
    st.subheader("📈 Actual vs Predicted Sales")
    eval_metrics, eval_plot = evaluate_model(df_pred)
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

    # ▼ 最大コスト（95パーセンタイルをデフォルト、最大値＋10M）
    raw_costs = df_raw[model_info["columns"]].values.flatten()
    default_max = int(np.percentile(raw_costs, 95))
    max_limit = int(np.max(raw_costs)) + 10_000_000

    # ▼ SaturationグラフのX軸最大値入力
    st.subheader("🖊 SaturationグラフのMaxCost設定")
    x_max_sat = st.number_input(
        "SaturationグラフのX軸最大値 (Cost)",
        min_value=1_000,
        max_value=max_limit,
        value=default_max,
        step=10_000
    )
    cost_vals_sat = np.linspace(0, x_max_sat, 1000)

    # ▼ Saturation構造分析グラフ
    st.subheader("📊 Transformed Variable Curve (Saturation Only, no Coefficient / Adstock)")
    fig1, ax1 = plt.subplots(figsize=(10, 5))
    for i, col in enumerate(model_info["columns"]):
        alpha = model_info["alphas"][i]
        y_vals = np.power(cost_vals_sat, alpha)
        ax1.plot(cost_vals_sat, y_vals, label=f"{col} (α={alpha:.2f})")
    ax1.set_title("Saturation Only")
    ax1.set_xlabel("Cost (JPY)")
    ax1.set_ylabel("Transformed Variable")
    ax1.xaxis.set_major_formatter(ticker.FuncFormatter(lambda x, _: f"¥{x:,.0f}"))

    # ▼ 凡例を下に配置
    ax1.legend(loc="upper center", bbox_to_anchor=(0.5, -0.15), ncol=2)

    st.pyplot(fig1)

    # ▼ ContributionグラフのX軸最大値入力
    st.subheader("🖊 ContributionグラフのMaxCost設定")
    x_max_contrib = st.number_input(
        "ContributionグラフのX軸最大値 (Cost)",
        min_value=1_000,
        max_value=max_limit,
        value=default_max,
        step=10_000
    )
    cost_vals_contrib = np.linspace(0, x_max_contrib, 1000)

    # ▼ 貢献曲線グラフ
    st.subheader("📊 Contribution Curve (Adstock + Saturation × Coefficient)")
    fig2, ax2 = plt.subplots(figsize=(10, 5))
    for i, col in enumerate(model_info["columns"]):
        alpha = model_info["alphas"][i]
        coef = model_info["model"].coef_[i]
        y_vals = np.power(cost_vals_contrib, alpha) * coef
        ax2.plot(cost_vals_contrib, y_vals, label=f"{col} (α={alpha:.2f}, Coef={coef:.2f})")
    ax2.set_title("Contribution Curve")
    ax2.set_xlabel("Cost (JPY)")
    ax2.set_ylabel("Contribution to Sales")
    ax2.xaxis.set_major_formatter(ticker.FuncFormatter(lambda x, _: f"¥{x:,.0f}"))

    # ▼ 凡例を下に配置
    ax2.legend(loc="upper center", bbox_to_anchor=(0.5, -0.15), ncol=2)

    st.pyplot(fig2)

    # ▼ 数式表示
    st.subheader("🧮 Functional Formulas per Channel")
    for i, col in enumerate(model_info["columns"]):
        alpha = np.round(model_info["alphas"][i], 3)
        beta = np.round(model_info["betas"][i], 3)
        coef = np.round(model_info["model"].coef_[i], 3)
        formula = f"{coef} × (Adstock × β={beta})^{alpha}"
        st.markdown(f"**{col}**: {formula}")

    # ▼ パターン分岐
    option = st.radio("🛠 パターン選択", ["パターンA：予算最適化（期間＋予算）", "パターンB：日別予算アップロード"])

    if option == "パターンA：予算最適化（期間＋予算）":
        st.header("🅰 期間・予算を指定して最適予算配分")

        # 期間選択
        start_date = st.date_input("開始日")
        end_date = st.date_input("終了日")
        budget = st.number_input("総予算", min_value=0, step=1000, value=1_000_000)

        if start_date > end_date:
            st.error("開始日は終了日以前を指定してください。")
            st.stop()

        days = pd.date_range(start=start_date, end=end_date)
        n_days = len(days)
        if n_days == 0:
            st.error("指定された期間の日数が0です。正しい日付範囲を選択してください。")
            st.stop()

        # 媒体ごとの制約
        st.markdown("🔧 媒体ごとの下限〜上限予算（任意）")
        constraints = {}
        for col in model_info["columns"]:
            col1, col2 = st.columns(2)
            with col1:
                min_val = st.number_input(f"{col} の下限", min_value=0, max_value=budget, step=1000, value=0)
            with col2:
                max_val = st.number_input(f"{col} の上限", min_value=0, max_value=budget, step=1000, value=budget)
            constraints[col] = (min_val, max_val)

        if st.button("🚀 最適予算配分を実行"):
            forecast_df, alloc_df, fig = generate_optimal_allocation(
                model_info, budget, start_date, end_date, constraints
            )
            st.pyplot(fig)
            st.subheader("📄 売上予測テーブル")
            st.dataframe(forecast_df)
            st.download_button("📥 売上予測をダウンロード", forecast_df.to_csv(index=False), "forecast.csv", "text/csv")

            st.subheader("📄 施策別予算配分テーブル")
            st.dataframe(alloc_df)
            st.download_button("📥 配分予算をダウンロード", alloc_df.to_csv(index=False), "allocation.csv", "text/csv")

    elif option == "パターンB：日別予算アップロード":
        st.header("🅱 日別予算アップロードによる予測")
        uploaded_plan = st.file_uploader("📤 Upload Plan Data (CSV or Excel)", type=["csv", "xlsx"], key="plan_upload")
        if uploaded_plan:
            df_plan = read_uploaded_file(uploaded_plan)
            st.success("✅ 予算データを読み込みました")
            st.dataframe(df_plan.head())

            forecast_df, fig = predict_from_uploaded_plan(model_info, df_plan)
            st.pyplot(fig)
            st.subheader("📄 売上予測テーブル")
            st.dataframe(forecast_df)
            st.download_button("📥 売上予測をダウンロード", forecast_df.to_csv(index=False), "forecast_b.csv", "text/csv")
