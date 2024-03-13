import pandas as pd
import numpy as np
import joblib
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix, roc_auc_score
import matplotlib.pyplot as plt
import seaborn as sns
def train_model(df):
    X = df.drop('vitesse_adoption', axis=1)
    Y = df['vitesse_adoption']

    labels = Y.unique()

    X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.2, random_state=42)
    model = XGBClassifier()
    model.fit(X_train, y_train)
    joblib.dump(model, 'models/xgboost.pkl')

    y_pred = model.predict(X_test)
    visualize_metrics(np.array(y_test).astype('int'), y_pred, labels)


def visualize_metrics(y_test, y_pred, labels):
    # evaluate predictions
    accuracy = accuracy_score(y_test, y_pred)
    print("Accuracy: %.2f%%" % (accuracy * 100.0))

    unique_class = labels

    auc_scores = roc_auc_score(y_test, y_pred, multi_class='ovr')

    # Afficher la matrice de confusion
    cm = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=labels, yticklabels=labels)
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.title('Confusion Matrix')
    plt.show()





