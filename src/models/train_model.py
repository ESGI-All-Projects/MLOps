import pandas as pd
import numpy as np
import joblib
from xgboost import XGBClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, classification_report
import matplotlib.pyplot as plt
import seaborn as sns
from zenml import step


@step
def train_model(df: pd.DataFrame, type_model='linear') -> None:
    X = df.drop('vitesse_adoption', axis=1)
    Y = df['vitesse_adoption']

    labels = Y.astype('int').unique()

    X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.2, random_state=42)
    if type_model == 'linear':
        model = LogisticRegression()
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
    elif type_model == 'xgboost':
        model = XGBClassifier()
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
    elif type_model == 'random':
        y_pred = np.random.choice(labels, len(X_test))

    joblib.dump(model, f'models/{type_model}.pkl')
    #visualize_metrics(np.array(y_test).astype('int'), y_pred, labels)


@step
def visualize_metrics(y_test, y_pred, labels):
    # evaluate precision/recall
    print(classification_report(y_test, y_pred, target_names=list(map(str, labels))))

    # Afficher la matrice de confusion
    cm = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=labels, yticklabels=labels)
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.title('Confusion Matrix')
    plt.show()
