import pandas as pd

from src.utils import load_object

class PredictPipeline:

    def predict(
        self,
        features
    ):

        scaler = load_object(
            "artifacts/scaler.pkl"
        )

        model = load_object(
            "artifacts/model.pkl"
        )

        scaled = scaler.transform(
            features
        )

        pred = model.predict(
            scaled
        )

        return pred