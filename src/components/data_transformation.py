import pandas as pd

from sklearn.preprocessing import StandardScaler

from src.utils import save_object

class DataTransformation:

    def initiate_data_transformation(
        self,
        path
    ):

        df = pd.read_csv(path)

        X = df.drop(
            columns=["Class"]
        )

        scaler = StandardScaler()

        X_scaled = scaler.fit_transform(X)

        save_object(
            "artifacts/scaler.pkl",
            scaler
        )

        return X_scaled