import joblib
import numpy as np
import pandas as pd
from zenml import pipeline, step
from zenml.client import Client

from src.features.build_features import build_features

def load_model(model_path):
    return joblib.load(model_path)

@step
def predict(df: pd.DataFrame) -> np.ndarray:
    df = build_features(df)
    df = df.drop(['vitesse_adoption'], axis=1)
    # model = load_model(f"models/{type_model}.pkl")
    artifact = Client().get_artifact_version('0ba7ac39-3044-4a0b-9908-08387ca80d95')
    model = artifact.load()

    return model.predict(df)
