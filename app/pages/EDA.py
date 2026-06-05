import streamlit as st
import pandas as pd
import sys
from pathlib import Path

root_path = Path(__file__).resolve().parents[2]
sys.path.append(str(root_path))

from src.visualization.eda_plots import EDAPlots

st.set_page_config(
    page_title="EDA Dashboard",
    page_icon="📊",
    layout="wide"
)

st.title("📊 Exploratory Data Analysis")

# ---------------------------------------
# Dataset Check
# ---------------------------------------

if "df" not in st.session_state:

    st.warning(
        "Please upload dataset from Home page."
    )

    st.stop()

df = st.session_state["df"]

# ---------------------------------------
# Dataset Overview
# ---------------------------------------

st.subheader("Dataset Overview")

col1, col2, col3 = st.columns(3)

with col1:
    st.metric(
        "Rows",
        df.shape[0]
    )

with col2:
    st.metric(
        "Columns",
        df.shape[1]
    )

with col3:
    st.metric(
        "Missing Values",
        int(df.isna().sum().sum())
    )

st.dataframe(df.head())

# ---------------------------------------
# Data Types
# ---------------------------------------

st.subheader("Column Information")

info_df = pd.DataFrame({
    "Column": df.columns,
    "Datatype": df.dtypes.astype(str),
    "Missing Values": df.isna().sum().values
})

st.dataframe(info_df)

# ---------------------------------------
# Missing Values
# ---------------------------------------

st.subheader("Missing Values")

missing_df = pd.DataFrame({
    "Column": df.columns,
    "Missing Count": df.isna().sum().values
})

missing_df = missing_df[
    missing_df["Missing Count"] > 0
]

if len(missing_df) > 0:

    st.dataframe(missing_df)

else:

    st.success(
        "No missing values found."
    )

# ---------------------------------------
# Numerical Columns
# ---------------------------------------

numeric_cols = df.select_dtypes(
    include=["number"]
).columns.tolist()

if len(numeric_cols) == 0:

    st.error(
        "No numerical columns found."
    )

    st.stop()

# ---------------------------------------
# Correlation Heatmap
# ---------------------------------------

st.subheader("Correlation Heatmap")

heatmap = EDAPlots.correlation_heatmap(
    df[numeric_cols]
)

st.pyplot(heatmap)

# ---------------------------------------
# Feature Distribution
# ---------------------------------------

st.subheader("Feature Distribution")

selected_feature = st.selectbox(
    "Select Numerical Feature",
    numeric_cols
)

st.plotly_chart(
    EDAPlots.feature_histogram(
        df,
        selected_feature
    ),
    use_container_width=True
)

# ---------------------------------------
# Feature Boxplot
# ---------------------------------------

st.subheader("Feature Boxplot")

st.plotly_chart(
    EDAPlots.feature_boxplot(
        df,
        selected_feature
    ),
    use_container_width=True
)

# ---------------------------------------
# Summary Statistics
# ---------------------------------------

st.subheader("Summary Statistics")

st.dataframe(
    df[numeric_cols].describe()
)

# ---------------------------------------
# Correlation Ranking
# ---------------------------------------

st.subheader(
    "Top Correlated Features"
)

corr_matrix = (
    df[numeric_cols]
    .corr()
    .abs()
)

corr_pairs = (
    corr_matrix.unstack()
    .sort_values(ascending=False)
)

corr_pairs = corr_pairs[
    corr_pairs < 1
]

st.dataframe(
    corr_pairs.head(20)
)

# ---------------------------------------
# Dataset Download
# ---------------------------------------

csv = df.to_csv(
    index=False
)

st.download_button(
    "Download Dataset",
    csv,
    file_name="dataset.csv",
    mime="text/csv"
)