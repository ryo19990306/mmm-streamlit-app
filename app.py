
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from utils import (
    train_model, evaluate_model,
    generate_optimal_allocation, predict_from_uploaded_plan,
    apply_adstock, saturation_transform
)

st.set_page_config(page_title="MMM Simulation", layout="wide")
st.title("📊 Marketing Mix Modeling Simulator")

uploaded_file = st.file_uploader("📤 Upload Raw Data (CSV or Excel)", type=["csv", "xlsx"])

if uploaded_file:
    if uploaded_file.name.endswith(".csv"):
        df_raw = pd.read_csv(uploaded_file)
    else:
        df_raw = pd.read_excel(uploaded_file)

    st.success("✅ Data loaded successfully")
    st.dataframe(df_raw.head())

    st.info("🔄 Training model...")
    model_info, df_pred = train_model(df_raw)

    st.subheader("📈 Actual vs Predicted Sales")
    eval_metrics, eval_plot = evaluate_model(df_raw, df_pred)
    st.pyplot(eval_plot)
    st.dataframe(eval_metrics)

    st.subheader("📋 Optimized Parameters per Channel")
    df_params = pd.DataFrame({
        "Channel": model_info["columns"],
        "α (Saturation)": np.round(model_info["alphas"], 4),
        "β (Adstock)": np.round(model_info["betas"], 4)
    })
    st.dataframe(df_params)

    st.subheader("📊 Response Curves (Adstock → Saturation)")
    fig1, ax1 = plt.subplots(figsize=(8, 4))
    for i, col in enumerate(model_info["columns"]):
        alpha = max(0.05, min(model_info["alphas"][i], 0.95))
        beta = max(0.05, min(model_info["betas"][i], 0.95))
        max_raw = df_raw[col].dropna().max() if col in df_raw.columns else 0
        max_cost = max_raw + 1_000_000
        cost_vals = np.linspace(0, max_cost, 300)
        adstock_vals = apply_adstock(cost_vals, beta)
        sat_vals = saturation_transform(adstock_vals, alpha)
        ax1.plot(cost_vals, sat_vals, label=f"{col} (α={alpha:.2f}, β={beta:.2f})")
    ax1.set_title("Response Curve by Channel (Adstock → Saturation)")
    ax1.set_xlabel("Cost (JPY)")
    ax1.set_ylabel("Response (Unscaled)")
    ax1.ticklabel_format(style="plain", axis="x")
    ax1.xaxis.set_major_formatter(ticker.FuncFormatter(lambda x, _: f"¥{x:,.0f}"))
    ax1.legend()
    st.pyplot(fig1)

    st.subheader("📊 Functional Curve (Adstock + Saturation × Coefficient)")
    fig2, ax2 = plt.subplots(figsize=(8, 4))
    for i, col in enumerate(model_info["columns"]):
        alpha = max(0.05, min(model_info["alphas"][i], 0.95))
        beta = max(0.05, min(model_info["betas"][i], 0.95))
        coef = model_info["model"].coef_[i]
        max_raw = df_raw[col].dropna().max() if col in df_raw.columns else 0
        max_cost = max_raw + 1_000_000
        cost_vals = np.linspace(0, max_cost, 300)
        adstock_vals = apply_adstock(cost_vals, beta)
        sat_vals = saturation_transform(adstock_vals, alpha)
        y_vals = np.array(sat_vals) * coef
        ax2.plot(cost_vals, y_vals, label=f"{col} (α={alpha:.2f}, β={beta:.2f})")
    ax2.set_title("Functional Curve by Channel (Response × Coefficient)")
    ax2.set_xlabel("Cost (JPY)")
    ax2.set_ylabel("Contribution (Scaled)")
    ax2.ticklabel_format(style="plain", axis="x")
    ax2.xaxis.set_major_formatter(ticker.FuncFormatter(lambda x, _: f"¥{x:,.0f}"))
    ax2.legend()
    st.pyplot(fig2)

    st.subheader("📐 Contribution Formula per Channel")
    for i, col in enumerate(model_info["columns"]):
        alpha = max(0.05, min(model_info["alphas"][i], 0.95))
        beta = max(0.05, min(model_info["betas"][i], 0.95))
        coef = model_info["model"].coef_[i]
        st.markdown(f"### 🔹 {col}")
        st.latex(f"\text{{Contribution}} = ( {col}(t-1) \times {beta:.3f} + \text{{Spent}}(t) )^{{{alpha:.3f}}} \times {coef:.3f}")

    st.subheader("🧩 Forecast Options")
    pattern = st.radio("Choose forecast pattern", ["Pattern A: Budget and Duration", "Pattern B: Upload Budget Allocation File"])

    if pattern == "Pattern A: Budget and Duration":
        budget = st.number_input("📌 Total Budget (JPY)", min_value=1_000_000, max_value=100_000_000, value=10_000_000, step=1_000_000)
        start_date = st.date_input("📅 Forecast Start Date")
        end_date = st.date_input("📅 Forecast End Date")
        if st.button("🚀 Run Forecast"):
            forecast_df, alloc_df, fig = generate_optimal_allocation(model_info, budget, start_date, end_date)
            st.success("✅ Forecast completed")
            st.pyplot(fig)
            st.dataframe(forecast_df)
            st.download_button("📥 Download Forecast CSV", forecast_df.to_csv(index=False), file_name="forecast_patternA.csv", mime="text/csv")

    elif pattern == "Pattern B: Upload Budget Allocation File":
        uploaded_plan = st.file_uploader("📤 Upload Budget Allocation (CSV)", type=["csv"], key="plan_upload")
        if uploaded_plan is not None:
            df_plan = pd.read_csv(uploaded_plan)
            st.dataframe(df_plan.head())
            if st.button("🚀 Run Forecast", key="run_patternB"):
                forecast_df, fig = predict_from_uploaded_plan(model_info, df_plan)
                st.success("✅ Forecast completed")
                st.pyplot(fig)
                st.dataframe(forecast_df)
                st.download_button("📥 Download Forecast CSV", forecast_df.to_csv(index=False), file_name="forecast_patternB.csv", mime="text/csv")
