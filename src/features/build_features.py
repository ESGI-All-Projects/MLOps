import pandas as pd

def build_features(df):
    column_to_remove = ['nom_animal',
                        'race_primaire',
                        'race_secondaire',
                        'region_de_malasie',
                        'id_secouriste',
                        'description',
                        'id_animal']

    df = df.drop(column_to_remove, axis=1)
    df = df.dropna(subset=['vitesse_adoption'])
    return df
