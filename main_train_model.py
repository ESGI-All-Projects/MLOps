from src.data.make_dataset import load_data
from src.features.build_features import build_features
from src.models.train_model import train_model, model_evaluator
from src.models.predict_model import predict
from zenml import pipeline, step
from zenml.client import Client


@pipeline
def train_model_pipeline():
    """Process data and train model + save."""
    df_train = load_data("data/interim/adoption_animal_train.csv")
    df_test = load_data("data/interim/adoption_animal_test.csv")
    df_train = build_features(df_train)
    df_test = build_features(df_test)
    model = train_model(df_train)
    model_evaluator(model, df_test)


if __name__ == "__main__":
    run = train_model_pipeline()
