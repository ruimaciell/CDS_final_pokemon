
import pandas as pd
import unittest
import pytest
import library_final
from library_final.pokedex import  DataLoader as ld
from library_final.pokedex.DataPreprocessor import PokemonBattleProcessor 

combats_data = pd.read_csv("./CDS_final_pokemon/Notebook/raw_data/combats.csv")
pokemon_data = pd.read_csv("./CDS_final_pokemon/Notebook/raw_data/pokemon.csv")
processed_data = pd.read_csv("./Notebook/raw_data/ProcessedData.csv")


class TestPokemonBattleProcessor(unittest.TestCase):
    def setUp(self):
         self.processor = PokemonBattleProcessor(pokemon_data, combats_data)

    def test_process_battle_data(self):
        self.processor.process_battle_data()

        # Check if the PokemonBattleProcessor updates the pokemon_data correctly
        expected_columns = ['#', 'Name', 'Victory_Rate', 'Total_Battles', 'Victory_Counts']
        self.assertListEqual(list(self.processor.pokemon_data.columns), expected_columns)

        # Check specific values based on the provided mock data
        self.assertEqual(self.processor.pokemon_data.loc[0, 'Victory_Rate'], 0.5)
        self.assertEqual(self.processor.pokemon_data.loc[1, 'Victory_Rate'], 0.0)
        self.assertEqual(self.processor.pokemon_data.loc[2, 'Victory_Rate'], 1.0)

    def test_get_processed_data(self):
        processed_data = self.processor.get_processed_data()

        # Check if get_processed_data returns the correct DataFrame
        self.assertIsInstance(processed_data, pd.DataFrame)
        self.assertListEqual(list(processed_data.columns), list(self.processor.pokemon_data.columns))

    def test_display_data(self):
        # This is just to test that the method doesn't raise an error
        self.processor.display_data()

if __name__ == '__main__':
    unittest.main()
