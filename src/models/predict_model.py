import joblib

from src.features.build_features import build_features

def load_model(model_path):
    return joblib.load(model_path)
def predict(df, type_model):
    df = build_features(df)
    model = load_model(f"models/{type_model}.pkl")

    return model.predict(df)
