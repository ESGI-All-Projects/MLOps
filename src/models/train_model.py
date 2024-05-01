import pandas as pd
import numpy as np
from xgboost import XGBClassifier
from sklearn.metrics import classification_report
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from zenml import step
from typing_extensions import Annotated


@step
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

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    xgb_classifier = XGBClassifier()

    grid_search = GridSearchCV(estimator=xgb_classifier, param_grid=param_grid, cv=cv, scoring='f1_macro', verbose=1,
                               n_jobs=-1)
    grid_search.fit(X_train, y_train)
    best_params = grid_search.best_params_

    model = XGBClassifier(**best_params)
    model.fit(X_train, y_train)

    return model

@step
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
    print("Random metrics :")
    print(metrics_report_random)
    print("Model metrics :")
    print(metrics_report_model)