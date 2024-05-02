import unittest
from unittest.mock import patch, call
import pandas as pd


from src.data.make_dataset import load_data, split_data


class TestMakeDataset(unittest.TestCase):
    def test_load_data(self):
        # Création d'un DataFrame de test
        test_data = pd.DataFrame({
            'column1': [1, 2, 3],
            'column2': ['a', 'b', 'c']
        })

        # Utilisation de mock pour simuler pd.read_csv
        with patch('pandas.read_csv', return_value=test_data) as mocked_read_csv:
            result = load_data("dummy_path.csv")
            mocked_read_csv.assert_called_once_with("dummy_path.csv", sep=",")
            self.assertTrue(result.equals(test_data), "Le DataFrame retourné devrait correspondre au DataFrame de test")

    def test_split_data(self):
        # Création d'un DataFrame de test
        test_data = pd.DataFrame({
            'vitesse_adoption': [1, 2, None, 4]
        })

        # Configuration du mock pour read_csv
        with patch('pandas.read_csv', return_value=test_data) as mocked_read_csv:
            with patch('pandas.DataFrame.to_csv') as mocked_to_csv:
                split_data("dummy_path.csv")

                # Vérifier que read_csv est appelé correctement
                mocked_read_csv.assert_called_once_with("dummy_path.csv", sep=",")

                # Vérifier que to_csv est appelé correctement
                self.assertEqual(mocked_to_csv.call_count, 3,
                                 "to_csv doit être appelé trois fois pour les fichiers de sortie")
                # Pour des détails plus précis, vous pouvez également inspecter les arguments avec lesquels to_csv est appelé
                expected_calls = [
                    call("data/interim/adoption_animal_real_time.csv", index=False),
                    call("data/interim/adoption_animal_train.csv", index=False),
                    call("data/interim/adoption_animal_test.csv", index=False)
                ]
                mocked_to_csv.assert_has_calls(expected_calls, any_order=True)


if __name__ == '__main__':
    unittest.main(verbosity=2)
