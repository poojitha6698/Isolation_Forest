import streamlit as st
import pandas as pd

from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import IsolationForest

st.set_page_config(
    page_title="Anomaly Detection",
    page_icon="🚨",
    layout="wide"
)

st.title("🚨 Anomaly Detection Dashboard")

# ----------------------------------------------------
# Check Dataset
# ----------------------------------------------------

if "df" not in st.session_state:

    st.warning(
        "Please upload dataset from Home Page."
    )

    st.stop()

df = st.session_state["df"]

st.subheader("Dataset Preview")

st.dataframe(df.head())

# ----------------------------------------------------
# Detect Anomalies
# ----------------------------------------------------

if st.button("Run Isolation Forest"):

    try:

        target_columns = [
            "Class",
            "class",
            "target",
            "Target",
            "is_fraud",
            "fraud"
        ]

        existing_targets = [
            col
            for col in target_columns
            if col in df.columns
        ]

        X = df.drop(
            columns=existing_targets,
            errors="ignore"
        )

        # Keep only numeric columns

        X = X.select_dtypes(
            include=["number"]
        )

        if X.shape[1] == 0:

            st.error(
                "No numerical columns found."
            )

            st.stop()

        st.subheader(
            "Numerical Features Used"
        )

        st.write(
            list(X.columns)
        )

        # Missing Values

        imputer = SimpleImputer(
            strategy="median"
        )

        X_imputed = imputer.fit_transform(X)

        # Scaling

        scaler = StandardScaler()

        X_scaled = scaler.fit_transform(
            X_imputed
        )

        # Isolation Forest

        model = IsolationForest(
            contamination=0.02,
            random_state=42,
            n_estimators=200
        )

        predictions = model.fit_predict(
            X_scaled
        )

        result_df = df.copy()

        result_df["Anomaly"] = predictions

        anomaly_count = (
            result_df["Anomaly"] == -1
        ).sum()

        normal_count = (
            result_df["Anomaly"] == 1
        ).sum()

        # KPIs

        col1, col2, col3 = st.columns(3)

        with col1:
            st.metric(
                "Total Records",
                len(result_df)
            )

        with col2:
            st.metric(
                "Normal Records",
                normal_count
            )

        with col3:
            st.metric(
                "Anomalies",
                anomaly_count
            )

        st.success(
            f"{anomaly_count} anomalies detected successfully."
        )

        # Show anomalies

        st.subheader(
            "Detected Anomalies"
        )

        anomalies = result_df[
            result_df["Anomaly"] == -1
        ]

        st.dataframe(
            anomalies.head(100)
        )

        # Download

        csv = result_df.to_csv(
            index=False
        )

        st.download_button(
            label="Download Results",
            data=csv,
            file_name="anomaly_results.csv",
            mime="text/csv"
        )

    except Exception as e:

        st.error(
            f"Error: {str(e)}"
        )