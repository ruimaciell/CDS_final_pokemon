import pandas as pd
import library_final
import unittest
import pytest
from library_final.pokedex import  DataLoader as ld
from library_final.pokedex import  DataPreprocessor as dp



###################################################0. Loading Data
loader = ld.loader_vscode()
combats_data, pokemon_data, pokemon_id_each_team, team_combat, process_data = loader.load_datasets()


class TestPokemonSkills(unittest.TestCase):
    # Test for PokemonBattleProcessor.process_battle_data
    def test_process_battle_data():
        processor = dp.PokemonBattleProcessor(pokemon_data, combats_data)
        processor.process_battle_data()
        assert isinstance(processor.pokemon_data, pd.DataFrame)
        assert 'Total_Battles' in processor.pokemon_data.columns
        assert 'Victory_Counts' in processor.pokemon_data.columns
        assert 'Victory_Rate' in processor.pokemon_data.columns

    # Test for PokemonBattleProcessor.get_processed_data
    def test_get_processed_data():
        processor = dp.PokemonBattleProcessor(pokemon_data, combats_data)
        processor.process_battle_data()
        processed_data = processor.get_processed_data()
        assert isinstance(processed_data, pd.DataFrame)

    # Test for DataProcessor.__init__
    def test_data_processor_init():
        processor = dp.DataProcessor(process_data)
        assert processor.df.equals(process_data)


    #################################################2. Testing Pokemon skill functions

    # Test for OffensivePowerCalculator.calculate
    def test_offensive_power_calculate():
        calculator = dp.OffensivePowerCalculator(process_data)
        calculator.calculate()
        assert 'Offensive_Power' in calculator.pokemon_data.columns

    # Test for DefensivePowerCalculator.calculate
    def test_defensive_power_calculate():
        calculator = dp.DefensivePowerCalculator(process_data)
        calculator.calculate()
        assert 'Defensive_Power' in calculator.pokemon_data.columns

    # Test for SpeedToPowerRatioCalculator.calculate
    def test_speed_to_power_ratio_calculate():
        calculator = dp.SpeedToPowerRatioCalculator(process_data)
        calculator.calculate()
        assert 'Speed_to_Power_Ratio' in calculator.pokemon_data.columns



if __name__ == '__main__':
    unittest.main()