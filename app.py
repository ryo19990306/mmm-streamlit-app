
import streamlit as st
from utils import run_forecast
import pandas as pd

st.set_page_config(page_title="MMM予測シミュレーション", layout="wide")

st.title("📊 MMM 予算シミュレーション（将来予測付き）")

# ユーザー入力
budget = st.slider("📌 予算（万円）", min_value=1000, max_value=100000, step=1000, value=10000)
start_date = st.date_input("🗓 予測開始日")
end_date = st.date_input("🗓 予測終了日")

# 実行
if st.button("🚀 シミュレーション実行"):
    with st.spinner("シミュレーション実行中..."):
        result_df, evaluation_df, image_path = run_forecast(budget, start_date, end_date)
        st.success("シミュレーション完了！")

        st.subheader("📈 売上予測グラフ")
        st.image(image_path, use_column_width=True)

        st.subheader("📋 売上予測データ")
        st.dataframe(result_df)

        st.download_button(
            label="📥 結果CSVをダウンロード",
            data=result_df.to_csv(index=False).encode("utf-8"),
            file_name="forecast_result.csv",
            mime="text/csv"
        )

        st.subheader("📊 モデル評価指標")
        st.dataframe(evaluation_df)
