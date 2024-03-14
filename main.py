from src.data.make_dataset import load_data
from src.features.build_features import build_features
from src.models.train_model import train_model

df = load_data("data/raw/adoption_animal.csv")
df = build_features(df)
train_model(df, type_model='xgboost')

