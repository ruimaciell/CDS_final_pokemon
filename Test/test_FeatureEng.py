import pandas as pd
import unittest
from library_final.pokedex import DataLoader as ld
from library_final.pokedex import DataPreprocessor as dp

loader = ld.loader_vscode()
combats_data, pokemon_data, pokemon_id_each_team, team_combat, process_data = loader.load_datasets()

class TestPokemonSkills(unittest.TestCase):
    def setUp(self):
        # Loading Data
        loader = ld.loader_vscode()
        self.combats_data, self.pokemon_data, _, _, self.process_data = loader.load_datasets()

        # Initialize PokemonBattleProcessor for testing
        self.processor = dp.PokemonBattleProcessor(self.pokemon_data, self.combats_data)
        self.processor.process_battle_data()

    # Test for PokemonBattleProcessor.process_battle_data
    def test_process_battle_data(self):
        self.assertIsInstance(self.processor.pokemon_data, pd.DataFrame)
        self.assertIn('Total_Battles', self.processor.pokemon_data.columns)
        self.assertIn('Victory_Counts', self.processor.pokemon_data.columns)
        self.assertIn('Victory_Rate', self.processor.pokemon_data.columns)

    # Test for PokemonBattleProcessor.get_processed_data
    def test_get_processed_data(self):
        processed_data = self.processor.get_processed_data()
        self.assertIsInstance(processed_data, pd.DataFrame)

    # Test for DataProcessor.__init__
    def test_data_processor_init(self):
        processor = dp.DataProcessor(self.process_data)
        self.assertTrue(processor.df.equals(self.process_data))

    # Test for OffensivePowerCalculator.calculate
    def test_offensive_power_calculate(self):
        calculator = dp.OffensivePowerCalculator(self.process_data)
        calculator.calculate()
        self.assertIn('Offensive_Power', calculator.pokemon_data.columns)

    # Test for DefensivePowerCalculator.calculate
    def test_defensive_power_calculate(self):
        calculator = dp.DefensivePowerCalculator(self.process_data)
        calculator.calculate()
        self.assertIn('Defensive_Power', calculator.pokemon_data.columns)

    # Test for SpeedToPowerRatioCalculator.calculate
    def test_speed_to_power_ratio_calculate(self):
        calculator = dp.SpeedToPowerRatioCalculator(self.process_data)
        calculator.calculate()
        self.assertIn('Speed_to_Power_Ratio', calculator.pokemon_data.columns)

if __name__ == '__main__':
    unittest.main()
