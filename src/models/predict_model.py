import numpy as np
import pandas as pd
from zenml import pipeline, step
from zenml.client import Client

from src.features.build_features import build_features


@step
def predict(df: pd.DataFrame) -> np.ndarray:
    df = build_features(df)
    df = df.drop(['vitesse_adoption'], axis=1)
    artifact = Client().get_artifact_version('3d412c10-1e67-4bd1-97c7-a5cbdc2a465b')
    model = artifact.load()

    return model.predict(df)
