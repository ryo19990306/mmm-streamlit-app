import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from utils import (
    train_model, evaluate_model,
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

    # 📉 Transformed Variable Curve（実データベース、係数なし）
    st.subheader("📉 Transformed Variable Curve (Adstock + Saturation, no Coefficient)")
    fig1, ax1 = plt.subplots(figsize=(10, 5))

    for i, col in enumerate(model_info["columns"]):
        alpha = np.clip(model_info["alphas"][i], 0.05, 0.95)
        beta = np.clip(model_info["betas"][i], 0.05, 0.95)

        cost_series = df_raw[col].fillna(0).values
        adstock_vals = apply_adstock(cost_series, beta)
        sat_vals = saturation_transform(adstock_vals, alpha)

        ax1.plot(range(len(sat_vals)), sat_vals, label=f"{col} (α={alpha:.2f}, β={beta:.2f})")

    ax1.set_title("Transformed Sales Driver by Channel (X without Coefficient)")
    ax1.set_xlabel("Time (Index)")
    ax1.set_ylabel("Transformed Variable (Unscaled)")
    ax1.ticklabel_format(style="plain", axis="y")
    ax1.legend(loc="lower center", bbox_to_anchor=(0.5, -0.35), ncol=3, fontsize="small")
    st.pyplot(fig1)

    # 📉 Functional Curve（実データベース、係数あり）
    st.subheader("📉 Predicted Contribution by Channel (A × X)")
    fig2, ax2 = plt.subplots(figsize=(10, 5))

    for i, col in enumerate(model_info["columns"]):
        alpha = np.clip(model_info["alphas"][i], 0.05, 0.95)
        beta = np.clip(model_info["betas"][i], 0.05, 0.95)
        coef = model_info["model"].coef_[i]

        cost_series = df_raw[col].fillna(0).values
        adstock_vals = apply_adstock(cost_series, beta)
        sat_vals = saturation_transform(adstock_vals, alpha)
        contribution_vals = np.array(sat_vals) * coef

        ax2.plot(range(len(contribution_vals)), contribution_vals, label=f"{col} (α={alpha:.2f}, β={beta:.2f}, Coef={coef:.2f})")

    ax2.set_title("Predicted Contribution by Channel (A × X)")
    ax2.set_xlabel("Time (Index)")
    ax2.set_ylabel("Contribution to Sales")
    ax2.ticklabel_format(style="plain", axis="y")
    ax2.legend(loc="lower center", bbox_to_anchor=(0.5, -0.35), ncol=3, fontsize="small")
    st.pyplot(fig2)
