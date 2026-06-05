import streamlit as st
import pandas as pd

st.set_page_config(
    page_title="Isolation Forest",
    page_icon="🚨",
    layout="wide"
)

st.title("🚨 Isolation Forest Anomaly Detection")

uploaded_file = st.sidebar.file_uploader(
    "Upload Dataset",
    type=["csv"]
)

if uploaded_file is not None:

    df = pd.read_csv(
        uploaded_file
    )

    st.session_state["df"] = df

    st.sidebar.success(
        f"Dataset Loaded ({df.shape[0]} rows)"
    )

st.markdown("""
### Features

✔ Upload Any CSV Dataset

✔ EDA Dashboard

✔ Isolation Forest Detection

✔ Download Results
""")