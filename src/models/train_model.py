from typing import Optional

import pandas as pd
import numpy as np
from xgboost import XGBClassifier
from sklearn.metrics import classification_report, accuracy_score, f1_score
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from zenml import pipeline, step, get_step_context
from typing_extensions import Annotated
import mlflow
from zenml.client import Client
from mlflow.tracking import MlflowClient, artifact_utils
from zenml.integrations.mlflow.services import MLFlowDeploymentService, MLFlowDeploymentConfig

experiment_tracker = Client().active_stack.experiment_tracker

@step(experiment_tracker=experiment_tracker.name)
def train_model(df_train: pd.DataFrame) -> Annotated[XGBClassifier, "trained_model"]:
    """
    Entraine un modèle xgboost avec une gridsearch pour finetune les hyperparamètres
    :param df: Data d'entrainement
    :return: modele entrainé
    """
    X_train = df_train.drop('vitesse_adoption', axis=1)
    y_train = df_train['vitesse_adoption']

    param_grid = {
        'max_depth': [3, 4, 5],
        'learning_rate': [0.1, 0.01, 0.001],
        'n_estimators': [100, 200, 300]
    }

    # param_grid = {
    #     'max_depth': [3],
    #     'learning_rate': [0.1],
    #     'n_estimators': [100]
    # }

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    xgb_classifier = XGBClassifier()

    grid_search = GridSearchCV(estimator=xgb_classifier, param_grid=param_grid, cv=cv, scoring='f1_macro', verbose=1,
                               n_jobs=-1)
    grid_search.fit(X_train, y_train)
    best_params = grid_search.best_params_

    # enable autologging
    mlflow.sklearn.autolog()

    model = XGBClassifier(**best_params)
    model.fit(X_train, y_train)

    return model

@step(experiment_tracker=experiment_tracker.name)
def model_evaluator(model: XGBClassifier, df_test: pd.DataFrame):
    X_test = df_test.drop('vitesse_adoption', axis=1)
    y_test = df_test['vitesse_adoption']

    labels = y_test.astype('int').unique()

    y_pred_model = model.predict(X_test)
    y_pred_random = np.random.choice(labels, len(X_test))

    metrics_report_model = classification_report(np.array(y_test).astype('int'), y_pred_model,
                                                 target_names=list(map(str, labels)))
    metrics_report_random = classification_report(np.array(y_test).astype('int'), y_pred_random,
                                                  target_names=list(map(str, labels)))

    accuracy = accuracy_score(np.array(y_test).astype('int'), y_pred_model)
    f1_macro = f1_score(np.array(y_test).astype('int'), y_pred_model, average='macro')
    mlflow.log_metric("Accuracy", accuracy)
    mlflow.log_metric("f1_macro", f1_macro)

    print("Random metrics :")
    print(metrics_report_random)
    print("Model metrics :")
    print(metrics_report_model)

@step
def deploy_model() -> Optional[MLFlowDeploymentService]:
    # Deploy a model using the MLflow Model Deployer
    zenml_client = Client()
    model_deployer = zenml_client.active_stack.model_deployer
    experiment_tracker = zenml_client.active_stack.experiment_tracker
    # Let's get the run id of the current pipeline
    mlflow_run_id = experiment_tracker.get_run_id(
        experiment_name=get_step_context().pipeline_name,
        run_name=get_step_context().run_name,
    )
    # Once we have the run id, we can get the model URI using mlflow client
    experiment_tracker.configure_mlflow()
    client = MlflowClient()
    model_name = "model" # set the model name that was logged
    model_uri = artifact_utils.get_artifact_uri(
        run_id=mlflow_run_id, artifact_path=model_name
    )
    mlflow_deployment_config = MLFlowDeploymentConfig(
        name="mlflow-model-deployment-example",
        description="An example of deploying a model using the MLflow Model Deployer",
        pipeline_name=get_step_context().pipeline_name,
        pipeline_step_name=get_step_context().step_name,
        model_uri=model_uri,
        model_name=model_name,
        workers=1,
        mlserver=False,
        timeout=300,
    )
    service = model_deployer.deploy_model(mlflow_deployment_config)
    return service