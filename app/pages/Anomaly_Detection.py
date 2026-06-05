import streamlit as st
import pandas as pd

from src.pipeline.prediction_pipeline import PredictPipeline

st.title(
    "🚨 Anomaly Detection"
)

amount = st.number_input(
    "Transaction Amount",
    value=100.0
)

time = st.number_input(
    "Transaction Time",
    value=1000.0
)

if st.button("Predict"):

    df = pd.DataFrame(
        [[time,amount]+[0]*28]
    )

    pipe = PredictPipeline()

    result = pipe.predict(df)

    if result[0] == -1:
        st.error(
            "Fraudulent Transaction Detected"
        )
    else:
        st.success(
            "Normal Transaction"
        )