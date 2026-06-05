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

st.title(
    "📊 Exploratory Data Analysis"
)

df = pd.read_csv(
    "artifacts/train.csv"
)

# -------------------------------------
# Dataset Overview
# -------------------------------------

st.subheader(
    "Dataset Overview"
)

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
        "Fraud Cases",
        df["Class"].sum()
    )

st.dataframe(
    df.head()
)

# -------------------------------------
# Fraud Distribution
# -------------------------------------

st.subheader(
    "Fraud Distribution"
)

st.plotly_chart(
    EDAPlots.fraud_distribution(df),
    use_container_width=True
)

# -------------------------------------
# Amount Distribution
# -------------------------------------

st.subheader(
    "Transaction Amount Distribution"
)

st.plotly_chart(
    EDAPlots.amount_distribution(df),
    use_container_width=True
)

# -------------------------------------
# Amount by Class
# -------------------------------------

st.subheader(
    "Amount by Fraud Class"
)

st.plotly_chart(
    EDAPlots.amount_by_class(df),
    use_container_width=True
)

# -------------------------------------
# Time Distribution
# -------------------------------------

st.subheader(
    "Transaction Time Distribution"
)

st.plotly_chart(
    EDAPlots.time_distribution(df),
    use_container_width=True
)

# -------------------------------------
# Correlation Heatmap
# -------------------------------------

st.subheader(
    "Correlation Heatmap"
)

heatmap = EDAPlots.correlation_heatmap(
    df
)

st.pyplot(
    heatmap
)

# -------------------------------------
# Top Correlations
# -------------------------------------

st.subheader(
    "Top Fraud Indicators"
)

st.plotly_chart(
    EDAPlots.top_correlated_features(df),
    use_container_width=True
)

# -------------------------------------
# Feature Explorer
# -------------------------------------

st.subheader(
    "Feature Explorer"
)

feature = st.selectbox(
    "Select Feature",
    [
        col for col in df.columns
        if col != "Class"
    ]
)

tab1, tab2 = st.tabs(
    [
        "Histogram",
        "Boxplot"
    ]
)

with tab1:

    st.plotly_chart(
        EDAPlots.feature_histogram(
            df,
            feature
        ),
        use_container_width=True
    )

with tab2:

    st.plotly_chart(
        EDAPlots.feature_boxplot(
            df,
            feature
        ),
        use_container_width=True
    )

# -------------------------------------
# Pair Plot
# -------------------------------------

st.subheader(
    "Scatter Matrix"
)

st.plotly_chart(
    EDAPlots.pair_plot(df),
    use_container_width=True
)

# -------------------------------------
# Time vs Amount
# -------------------------------------

st.subheader(
    "Time vs Amount"
)

st.plotly_chart(
    EDAPlots.fraud_amount_scatter(df),
    use_container_width=True
)