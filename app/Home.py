import streamlit as st
import pandas as pd

st.set_page_config(
    page_title="Isolation Forest Fraud Detection",
    page_icon="🚨",
    layout="wide"
)

st.title("🚨 Isolation Forest Fraud Detection")

st.sidebar.header("Dataset Upload")

uploaded_file = st.sidebar.file_uploader(
    "Upload Credit Card Dataset",
    type=["csv"]
)

if uploaded_file is not None:

    df = pd.read_csv(uploaded_file)

    st.session_state["df"] = df

    st.sidebar.success(
        f"Dataset Loaded Successfully ({df.shape[0]} rows)"
    )

st.markdown("""
### Project Overview

This project performs:

- Exploratory Data Analysis
- Anomaly Detection using Isolation Forest
- Fraud Exploration Dashboard

Upload your dataset from the sidebar to begin.
""")