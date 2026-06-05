from src.pipeline.training_pipeline import TrainingPipeline

if __name__ == "__main__":

    pipe = TrainingPipeline()

    pipe.start_training()

    print("Training Complete")