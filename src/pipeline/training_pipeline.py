from src.components.data_ingestion import DataIngestion

from src.components.data_transformation import DataTransformation

from src.components.model_trainer import ModelTrainer

class TrainingPipeline:

    def start_training(self):

        ingestion = DataIngestion()

        path = ingestion.initiate_data_ingestion()

        transformation = DataTransformation()

        X = transformation.initiate_data_transformation(
            path
        )

        trainer = ModelTrainer()

        trainer.initiate_model_training(X)