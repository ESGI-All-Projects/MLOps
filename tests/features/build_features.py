import unittest
from unittest.mock import patch
import pandas as pd

from src.features.build_features import build_features


class TestBuildFeatures(unittest.TestCase):
    def test_build_features(self):
        # Création d'un DataFrame de test incluant les colonnes à supprimer et d'autres colonnes
        test_data = pd.DataFrame({
            'nom_animal': ['Rex', 'Buddy'],
            'race_primaire': ['Labrador', 'Beagle'],
            'race_secondaire': ['Poodle', 'Boxer'],
            'region_de_malasie': ['North', 'South'],
            'id_secouriste': [101, 102],
            'description': ['Big', 'Small'],
            'id_animal': [1, 2],
            'age': [5, 2],
            'poids': [20, 10]
        })

        # Colonnes attendues après suppression
        expected_columns = ['age', 'poids']

        # Appel de la fonction build_features
        result = build_features(test_data)

        # Vérifier que le DataFrame résultant contient uniquement les colonnes attendues
        self.assertListEqual(list(result.columns), expected_columns,
                             "Les colonnes résultantes ne correspondent pas aux attentes")


if __name__ == '__main__':
    unittest.main(verbosity=2)
