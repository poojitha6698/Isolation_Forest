from sklearn.ensemble import IsolationForest

from src.utils import save_object

class ModelTrainer:

    def initiate_model_training(
        self,
        X
    ):

        model = IsolationForest(
            contamination=0.0017,
            n_estimators=200,
            random_state=42
        )

        model.fit(X)

        save_object(
            "artifacts/model.pkl",
            model
        )

        return model