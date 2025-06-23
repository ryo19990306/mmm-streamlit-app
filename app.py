
import streamlit as st
import pandas as pd
from utils import (
    train_model, evaluate_model,
    generate_optimal_allocation, predict_from_uploaded_plan
)

st.set_page_config(page_title="MMM予算シミュレーション", layout="wide")
st.title("📊 MMM予算シミュレーション（パターン選択対応）")

# ▼ データアップロード
uploaded_file = st.file_uploader("📤 Rawデータをアップロード（CSVまたはExcel）", type=["csv", "xlsx"])

if uploaded_file:
    # データ読み込み
    if uploaded_file.name.endswith(".csv"):
        df_raw = pd.read_csv(uploaded_file)
    else:
        df_raw = pd.read_excel(uploaded_file)

    st.success("✅ データ読み込み成功")
    st.dataframe(df_raw.head())

    # ▼ モデル学習（バックエンド処理）
    st.info("🔄 モデル学習中...")
    model, pred_df = train_model(df_raw)

    # ▼ モデル評価・グラフ
    st.subheader("📈 モデル評価（実績 vs 予測）")
    eval_metrics, eval_plot = evaluate_model(df_raw, pred_df)
    st.pyplot(eval_plot)
    st.dataframe(eval_metrics)

    # ▼ 各施策ごとの数式・グラフ
    st.subheader("📊 各媒体（施策）の貢献度")
    # ここに施策ごとの係数/グラフ出力処理を組み込む予定（後で追加）

    st.markdown("---")

    # ▼ パターン選択
    pattern = st.radio("🧭 分析パターンを選択してください", ["パターンA：予算と期間を選ぶ", "パターンB：将来予算をアップロード"])

    if pattern == "パターンA：予算と期間を選ぶ":
        st.subheader("💡 パターンA：最適予算配分からの売上予測")

        budget = st.number_input("💰 予算（一円単位）", min_value=0, max_value=100000000, value=10000000, step=1000, format="%d")
        start_date = st.date_input("📅 予測開始日")
        end_date = st.date_input("📅 予測終了日")

        if st.button("🚀 シミュレーション実行"):
            forecast_df, alloc_df, plot = generate_optimal_allocation(model, budget, start_date, end_date)
            st.pyplot(plot)
            st.dataframe(forecast_df)
            st.download_button("📥 予測結果CSVダウンロード", forecast_df.to_csv(index=False).encode("utf-8"),
                               file_name="forecast_patternA.csv", mime="text/csv")

    else:
        st.subheader("💡 パターンB：予算配分表からの売上予測")

        plan_file = st.file_uploader("📤 将来予算の配分表をアップロード（CSVまたはExcel）", type=["csv", "xlsx"])
        if plan_file:
            if plan_file.name.endswith(".csv"):
                df_plan = pd.read_csv(plan_file)
            else:
                df_plan = pd.read_excel(plan_file)

            st.dataframe(df_plan.head())

            if st.button("🚀 予測実行（アップロード配分）"):
                forecast_df, plot = predict_from_uploaded_plan(model, df_plan)
                st.pyplot(plot)
                st.dataframe(forecast_df)
                st.download_button("📥 予測結果CSVダウンロード", forecast_df.to_csv(index=False).encode("utf-8"),
                                   file_name="forecast_patternB.csv", mime="text/csv")
else:
    st.info("👈 左からRawデータファイルをアップロードしてください。")
