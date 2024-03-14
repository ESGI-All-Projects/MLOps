import pandas as pd
from zenml import step
from typing_extensions import Annotated


@step
def load_data(file_path: str) -> Annotated[pd.DataFrame, "df"]:
    df = pd.read_csv(file_path, sep=",")
    df = df.dropna(subset=['vitesse_adoption'])
    return df
