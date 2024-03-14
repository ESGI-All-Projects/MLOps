import pandas as pd

def load_data(file_path):
    df = pd.read_csv(file_path, sep=",")
    df = df.dropna(subset=['vitesse_adoption'])
    return df