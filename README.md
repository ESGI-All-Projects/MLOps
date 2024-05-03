# Projet MLOps : Prédiction de l'Adoption des Animaux

## Description
Ce projet utilise des techniques de Machine Learning pour prédire la vitesse d'adoption des animaux à partir d'un jeu de données recueilli sur une plateforme de bien-être animal en Malaisie. Ce jeu de données comprend des informations détaillées sur plus de 150 000 animaux et vise à améliorer le taux d'adoption en identifiant les facteurs qui accélèrent ou ralentissent leur adoption.

## Structure du Projet
Le projet est structuré comme suit:
- `data/`: Contient les fichiers de données brutes et les scripts pour la préparation des données.
- `src/models/`: Contient les scripts pour entraîner et prédire les modèles.
- `src/`: Contient les scripts source pour le traitement des données, la construction des caractéristiques, etc.
- `tests/`: Contient les tests pour les modules du projet.

### Fichiers de Données
- `adoption_animal.csv`: Données principales incluant la vitesse d'adoption et autres caractéristiques des animaux.
- `categorie_race.csv`: Informations sur les races des animaux.
- `categorie_couleur.csv`: Informations sur les couleurs des animaux.
- `categorie_region.csv`: Informations sur les régions géographiques des données.

## Dépendances
Ce projet utilise Python avec plusieurs bibliothèques spécialisées en sciences des données. Les dépendances exactes sont listées dans le fichier `requirements.txt`.

## Installation
Pour installer les dépendances nécessaires, exécutez:
```bash
pip install -r requirements.txt
```
## Utilisation
Pour exécuter les pipelines de formation et de prédiction, les scripts principaux sont:

- main_train_model.py: Pour entraîner le modèle.
- main_predict.py: Pour exécuter des prédictions à partir du modèle entraîné.

## Gestion des expériences avec ZenML
ZenML est utilisé dans ce projet pour orchestrer les pipelines de Machine Learning, gérer les expériences, et suivre les performances des modèles. Voici les commandes principales de ZenML utilisées et leurs explications :

```bash 
zenml init 
```

Cette commande initialise un répertoire de projet ZenML, créant une configuration de base pour gérer les pipelines et les artefacts de manière organisée.

```bash 
zenml integration install mlflow -y
```
Installe l'intégration MLflow automatiquement. MLflow est utilisé pour le suivi des expériences, la gestion des modèles, et la visualisation des métriques.

```bash 
zenml experiment-tracker register mlflow_experiment_tracker --flavor=mlflow
```
Enregistre un tracker d'expériences utilisant MLflow. Ce suiveur permet de documenter, comparer et visualiser les performances de diverses expériences et configurations de modèle.

```bash 
zenml stack register mlflow_stack -e mlflow_experiment_tracker -a default -o default --set
```
Crée et configure une stack ZenML nommée mlflow_stack avec MLflow comme tracker d'expériences, utilisant les paramètres par défaut pour l'orchestrateur et le dépôt d'artefacts. Cette stack est ensuite définie comme la configuration active pour le projet.

```bash
zenml model-deployer register mlflow_deployer --flavor=mlflow
```
Permet de définir mlflow comme deployer pour effectuer des predictions via l'api en local

```bash
mlflow ui --backend-store-uri <TRACKING_URL>
```
Permet de lancer le serveur Mlflow en local. Il faut aller récupérer l'url dans les métadonnés d'une des steps qui utilise mlflow
```

```bash
zenml up --blocking
```
Permet de lancer zenml sur window
## Auteurs

- Charles Logeais
- Julian Alizay
- Duncan Lopes