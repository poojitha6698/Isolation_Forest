import streamlit as st
import pandas as pd

from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import IsolationForest

st.title("🚨 Anomaly Detection Dashboard")

if "df" not in st.session_state:

    st.warning(
        "Please upload dataset from Home page."
    )

    st.stop()

df = st.session_state["df"]

st.subheader("Run Isolation Forest")

if st.button("Detect Anomalies"):

    X = df.drop(
        columns=["Class"],
        errors="ignore"
    )

    scaler = StandardScaler()

    X_scaled = scaler.fit_transform(X)

    model = IsolationForest(
        contamination=0.002,
        random_state=42,
        n_estimators=200
    )

    predictions = model.fit_predict(
        X_scaled
    )

    df["Anomaly"] = predictions

    anomaly_count = (
        df["Anomaly"] == -1
    ).sum()

    st.success(
        f"{anomaly_count} anomalies detected"
    )

    st.metric(
        "Anomalies",
        anomaly_count
    )

    st.dataframe(
        df[df["Anomaly"] == -1].head(50)
    )

    csv = df.to_csv(index=False)

    st.download_button(
        "Download Results",
        csv,
        file_name="anomaly_results.csv",
        mime="text/csv"
    )