from src.data.make_dataset import load_data
from src.features.build_features import build_features
from src.models.train_model import train_model, visualize_metrics
from src.models.predict_model import predict
from zenml import pipeline, step
from zenml.client import Client


@pipeline
def model_pipeline():
    """Process data and train model + save."""
    df = load_data("data/raw/adoption_animal.csv")
    df = build_features(df)
    model = train_model(df, type_model='xgboost')



@pipeline
def predict_pipeline():
    """Process data and predict."""
    df = load_data("data/raw/adoption_animal.csv")
    predict(df)


if __name__ == "__main__":
    run = model_pipeline()
    run = predict_pipeline()
