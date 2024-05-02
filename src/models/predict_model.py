import numpy as np
import pandas as pd
import json
import requests
from zenml import pipeline, step
from zenml.client import Client
from zenml.integrations.mlflow.model_deployers.mlflow_model_deployer import (
    MLFlowModelDeployer,
)

from src.features.build_features import build_features


@step
def predict(df: pd.DataFrame) -> np.ndarray:
    df = build_features(df)
    df = df.drop(['vitesse_adoption'], axis=1)
    artifact = Client().get_artifact_version('3d412c10-1e67-4bd1-97c7-a5cbdc2a465b')
    model = artifact.load()

    return model.predict(df)

@step(enable_cache=False)
def prediction_service_loader(
    df: pd.DataFrame,
    pipeline_name: str,
    pipeline_step_name: str,
    model_name: str = "trained_model",
) -> None:
    """Get the prediction service started by the deployment pipeline.

    Args:
        pipeline_name: name of the pipeline that deployed the MLflow prediction
            server
        step_name: the name of the step that deployed the MLflow prediction
            server
        running: when this flag is set, the step only returns a running service
        model_name: the name of the model that is deployed
    """
    df = build_features(df)
    df = df.drop(['vitesse_adoption'], axis=1)
    # get the MLflow model deployer stack component
    model_deployer = MLFlowModelDeployer.get_active_model_deployer()

    # fetch existing services with same pipeline name, step name and model name
    existing_services = model_deployer.find_model_server(
        pipeline_name=pipeline_name,
        pipeline_step_name=pipeline_step_name,
        model_name=model_name,
    )

    if not existing_services:
        raise RuntimeError(
            f"No MLflow prediction service deployed by step "
            f"'{pipeline_step_name}' in pipeline '{pipeline_name}' with name "
            f"'{model_name}' is currently running."
        )

    service = existing_services[0]

    # Let's try run a inference request against the prediction service
    #TODO
    payload = json.dumps(
        {
            "inputs": {"messages": [{"role": "user", "content": "Tell a joke!"}]},
            "params": {
                "temperature": 0.5,
                "max_tokens": 20,
            },
        }
    )
    response = requests.post(
        url=service.get_prediction_url(),
        data=payload,
        headers={"Content-Type": "application/json"},
    )

    response.json()