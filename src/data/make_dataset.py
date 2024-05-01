import pandas as pd
from zenml import step
from typing_extensions import Annotated
from sklearn.model_selection import train_test_split


@step
def load_data(file_path: str) -> Annotated[pd.DataFrame, "process_df"]:
    df = pd.read_csv(file_path, sep=",")
    return df

def split_data(file_path: str):
    """
    Sépare les données pour former le jeu d'entrainement, de test et celui à utiliser en temps réel qui n'a pas de label
    Sauvegarde dans le dossier data/interim
    """
    df = pd.read_csv(file_path, sep=",")

    # On récupère les données sans label
    df_real_time = df[df['vitesse_adoption'].isna()]

    # On split le reste des données avec un ratio 80/20
    df = df.dropna(subset=['vitesse_adoption'])
    df_train, df_test = train_test_split(df, test_size=0.2, random_state=42)

    # sauvegarde
    df_real_time.to_csv("data/interim/adoption_animal_real_time.csv", index=False)
    df_train.to_csv("data/interim/adoption_animal_train.csv", index=False)
    df_test.to_csv("data/interim/adoption_animal_test.csv", index=False)
