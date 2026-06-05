import pandas as pd
import os

class DataIngestion:

    def initiate_data_ingestion(self):

        df = pd.read_csv(
            "dataset/creditcard.csv"
        )

        os.makedirs(
            "artifacts",
            exist_ok=True
        )

        df.to_csv(
            "artifacts/train.csv",
            index=False
        )

        return "artifacts/train.csv"