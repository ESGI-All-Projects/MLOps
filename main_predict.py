from src.data.make_dataset import load_data
from src.models.predict_model import predict

df = load_data("data/raw/adoption_animal.csv")
df = df[df.index < 10]
df = df.drop(['vitesse_adoption'], axis=1)
pred = predict(df, type_model='xgboost')
print(pred)

