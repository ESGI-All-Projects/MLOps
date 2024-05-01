from zenml import pipeline, step

from src.data.make_dataset import load_data
from src.models.predict_model import predict

@pipeline
def predict_pipeline():
    """Process data and predict."""
    df = load_data("data/interim/adoption_animal_real_time.csv")
    predictions = predict(df)

if __name__ == "__main__":
    run = predict_pipeline()

