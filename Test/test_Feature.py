"""Testing"""

#import pytest
import pandas as pd
# from your_module import PokemonBattleProcessor, DataProcessor, NumericDataAnalyzer, CategoricalDataAnalyzer
# Sample data for testing (You should replace this with the actual data or mock data)
# pokemon_data = pd.DataFrame({'#': [1, 2], 'Name': ['Bulbasaur', 'Ivysaur']})
# combats_data = pd.DataFrame({'First_pokemon': [1, 2], 'Second_pokemon': [2, 1], 'Winner': [1, 2]})


pokemon_data = pd.read_csv('/Users/ruimaciel/Desktop/Barcelona/Computing_for_Data_Science/CDS_final_pokemon/raw_data/pokemon.csv')
combats_data = pd.read_csv('/Users/ruimaciel/Desktop/Barcelona/Computing_for_Data_Science/CDS_final_pokemon/raw_data/combats.csv')
process_data = pd.read_csv('/Users/ruimaciel/Desktop/Barcelona/Computing_for_Data_Science/CDS_final_pokemon/raw_data/ProcessedData.csv')
# Test for PokemonBattleProcessor.process_battle_data
def test_process_battle_data():
    processor = PokemonBattleProcessor(pokemon_data, combats_data)
    processor.process_battle_data()
    assert isinstance(processor.pokemon_data, pd.DataFrame)
    assert 'Total_Battles' in processor.pokemon_data.columns
    assert 'Victory_Counts' in processor.pokemon_data.columns
    assert 'Victory_Rate' in processor.pokemon_data.columns

# Test for PokemonBattleProcessor.get_processed_data
def test_get_processed_data():
    processor = PokemonBattleProcessor(pokemon_data, combats_data)
    processor.process_battle_data()
    processed_data = processor.get_processed_data()
    assert isinstance(processed_data, pd.DataFrame)

# Test for DataProcessor.__init__
def test_data_processor_init():
    processor = DataProcessor(process_data)
    assert processor.df.equals(process_data)

# Test for NumericDataAnalyzer.calculate_statistics
def test_calculate_statistics():
    numeric_analyzer = NumericDataAnalyzer(process_data)
    stats = numeric_analyzer.calculate_statistics()
    assert isinstance(stats, pd.DataFrame)
    assert 'Mean' in stats.columns
    assert 'Mode' in stats.columns

# Test for CategoricalDataAnalyzer.calculate_statistics
def test_categorical_statistics():
    categorical_analyzer = CategoricalDataAnalyzer(process_data)
    stats = categorical_analyzer.calculate_statistics()
    assert isinstance(stats, pd.DataFrame)
    assert 'Variable' in stats.columns
    assert 'Number of Unique Values' in stats.columns

# Test for MissingDataVisualizer.visualize_missing_data
def test_visualize_missing_data():
    visualizer = MissingDataVisualizer(process_data)
    assert visualizer.visualize_missing_data() is None

# Test for CorrelationMatrixVisualizer
class TestCorrelationMatrixVisualizer:
    def test_select_numerical_columns(self):
        visualizer = CorrelationMatrixVisualizer(process_data)
        visualizer.select_numerical_columns()
        assert all(column in process_data.columns for column in visualizer.numerical_columns)

    def test_calculate_correlation_matrix(self):
        visualizer = CorrelationMatrixVisualizer(process_data)
        visualizer.select_numerical_columns()
        visualizer.calculate_correlation_matrix()
        assert isinstance(visualizer.correlation_matrix, pd.DataFrame)

    def test_plot_heatmap(self):
        visualizer = CorrelationMatrixVisualizer(process_data)
        visualizer.select_numerical_columns()
        visualizer.calculate_correlation_matrix()
        assert visualizer.plot_heatmap() is None

# Test for OffensivePowerCalculator.calculate
def test_offensive_power_calculate():
    calculator = OffensivePowerCalculator(process_data)
    calculator.calculate()
    assert 'Offensive_Power' in calculator.pokemon_data.columns

# Test for DefensivePowerCalculator.calculate
def test_defensive_power_calculate():
    calculator = DefensivePowerCalculator(process_data)
    calculator.calculate()
    assert 'Defensive_Power' in calculator.pokemon_data.columns

# Test for SpeedToPowerRatioCalculator.calculate
def test_speed_to_power_ratio_calculate():
    calculator = SpeedToPowerRatioCalculator(process_data)
    calculator.calculate()
    assert 'Speed_to_Power_Ratio' in calculator.pokemon_data.columns

# Test for PokemonTypeEncoder.one_hot_encode_type1
def test_one_hot_encode_type1():
    encoder = PokemonTypeEncoder(sample_data)
    encoder.one_hot_encode_type1()
    assert 'Grass' in encoder.pokemon_data.columns
    assert 'Fire' in encoder.pokemon_data.columns
    assert 'Water' in encoder.pokemon_data.columns

# Test for PokemonTypeEncoder.encode_type2
def test_encode_type2():
    encoder = PokemonTypeEncoder(sample_data)
    encoder.one_hot_encode_type1()
    encoder.encode_type2()
    # Check if 'Poison' and 'Flying' are in the columns and 'None' is not a column
    assert 'Poison' in encoder.pokemon_data.columns
    assert 'Flying' in encoder.pokemon_data.columns
    assert None not in encoder.pokemon_data.columns

# Test for PokemonTypeEncoder.get_updated_dataframe
def test_get_updated_dataframe():
    encoder = PokemonTypeEncoder(sample_data)
    encoder.one_hot_encode_type1()
    encoder.encode_type2()
    updated_df = encoder.get_updated_dataframe()
    assert isinstance(updated_df, pd.DataFrame)
    assert 'Grass' in updated_df.columns
    assert 'Fire' in updated_df.columns

# Test for Pokemon_Dummy_Legendary_Encoder.encode_legendary
def test_encode_legendary():
    legendary_encoder = Pokemon_Dummy_Legendary_Encoder(sample_data)
    legendary_encoder.encode_legendary()
    assert 'Legendary' in legendary_encoder.pokemon_data.columns
    assert all(isinstance(x, int) for x in legendary_encoder.pokemon_data['Legendary'])

# Tests for PokemonVisualizations methods
# Note: These tests check if the methods run without errors. Testing visual outputs is beyond the scope.
@pytest.mark.parametrize("visualization_method", [
    'box_plot_by_generation',
    'violin_plot_by_type_columns',
    'box_plot_legendary_vs_non_legendary',
    'compare_victory_rate_legendary_vs_non_legendary'
])
def test_visualizations(visualization_method):
    visualizer = PokemonVisualizations(sample_data)
    visualization_func = getattr(visualizer, visualization_method)
    assert visualization_func() is None