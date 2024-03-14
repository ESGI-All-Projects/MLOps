import pandas as pd
from zenml import step
from typing_extensions import Annotated


@step
def build_features(df: pd.DataFrame) -> Annotated[pd.DataFrame, "df"]:
    column_to_remove = ['nom_animal',
                        'race_primaire',
                        'race_secondaire',
                        'region_de_malasie',
                        'id_secouriste',
                        'description',
                        'id_animal']

    # categories = ['type_animal', 'genre', 'couleur_1', 'couleur_2', 'couleur_3', 'taille', 'niveau_pilosite', 'vaccin', 'traitement_vermifuge',
    # 'sterilisation', 'sante']

    df = df.drop(column_to_remove, axis=1)
    # df = pd.get_dummies(df, columns=categories)
    return df
