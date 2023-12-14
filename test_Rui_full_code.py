# 1. Import Libraries - goest to
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import statistics
import pytest

"""data_loader.py"""
# 2. Load Datasets

pokemon_data = pd.read_csv('/Users/ruimaciel/Desktop/Barcelona/Computing_for_Data_Science/CDS_final_pokemon/raw_data/pokemon.csv')
combats_data = pd.read_csv('/Users/ruimaciel/Desktop/Barcelona/Computing_for_Data_Science/CDS_final_pokemon/raw_data/combats.csv')

# 3. Class Definitions

class PokemonBattleProcessor:
    def __init__(self, pokemon_data, combats_data):
        self.pokemon_data = pokemon_data
        self.combats_data = combats_data

    def process_battle_data(self):
        # Counting victories and battles
        victory_counts = self.combats_data['Winner'].value_counts()
        total_battles = (self.combats_data['First_pokemon'].value_counts() +
                         self.combats_data['Second_pokemon'].value_counts())

        # Calculating victory rate
        victory_rate = victory_counts / total_battles
        victory_rate_df = pd.DataFrame({'#': victory_rate.index, 'Victory_Rate': victory_rate.values})

        # Merging data
        self.pokemon_data = pd.merge(self.pokemon_data, victory_rate_df, on='#', how='left')
        self.pokemon_data['Total_Battles'] = self.pokemon_data['#'].map(total_battles)
        self.pokemon_data['Victory_Counts'] = self.pokemon_data['#'].map(victory_counts)
        self.pokemon_data.fillna({'Victory_Rate': 0, 'Total_Battles': 0, 'Victory_Counts': 0}, inplace=True)

    def get_processed_data(self):
        return self.pokemon_data

    def display_data(self, rows=5):
        print(self.pokemon_data.head(rows))


"""pre_and_post_EDA_functions.py"""

class DataProcessor:
    def __init__(self, df):
        self.df = df
        pd.set_option('display.float_format', lambda x: '{:.1f}'.format(x) if x % 1 else '{:.0f}'.format(x))

class NumericDataAnalyzer(DataProcessor):
    def calculate_statistics(self):
        numeric_columns = self.df.select_dtypes(include=['int', 'float']).columns
        stats_dict = {'Variable': numeric_columns, 'Mean': [], 'Mode': [], 'Median': [], 'Standard Deviation': [], 'Minimum': [], 'Maximum': [], 'Count': [], 'IQR': [], 'Skewness': [], 'Range': []}
        for column in numeric_columns:
            stats_dict['Mean'].append(self.df[column].mean())
            stats_dict['Mode'].append(statistics.mode(self.df[column].dropna()))
            stats_dict['Median'].append(self.df[column].median())
            stats_dict['Standard Deviation'].append(self.df[column].std())
            stats_dict['Minimum'].append(self.df[column].min())
            stats_dict['Maximum'].append(self.df[column].max())
            stats_dict['Count'].append(self.df[column].count())
            stats_dict['IQR'].append(np.percentile(self.df[column].dropna(), 75) - np.percentile(self.df[column].dropna(), 25))
            stats_dict['Skewness'].append(self.df[column].skew())
            stats_dict['Range'].append(stats_dict['Maximum'][-1] - stats_dict['Minimum'][-1])
        return pd.DataFrame(stats_dict)

class CategoricalDataAnalyzer(DataProcessor):
    def calculate_statistics(self):
        categorical_columns = self.df.select_dtypes(include=['object']).columns
        stats_dict = {'Variable': categorical_columns, 'Number of Unique Values': [], 'Top 5 Most Frequent Values': []}
        for column in categorical_columns:
            stats_dict['Number of Unique Values'].append(self.df[column].nunique())
            stats_dict['Top 5 Most Frequent Values'].append(self.df[column].value_counts().index[:5].tolist())
        return pd.DataFrame(stats_dict)

class MissingDataVisualizer(DataProcessor):
    def visualize_missing_data(self):
            missing_percentages_df = (self.df.isnull().mean() * 100).round(2).reset_index()
            missing_percentages_df.columns = ['Column Name', 'Percentage Missing']

            # Convert 'Column Name' to string type
            missing_percentages_df['Column Name'] = missing_percentages_df['Column Name'].astype(str)

            plt.figure(figsize=(20, 9))
            plt.bar(missing_percentages_df['Column Name'], missing_percentages_df['Percentage Missing'], color='skyblue')
            plt.xlabel('Column Name')
            plt.ylabel('Percentage Missing')
            plt.title('Percentage of Missing Values by Column')
            plt.xticks(rotation=45, ha='right')
            plt.tight_layout(pad=0.0001)
            plt.show()

class CorrelationMatrixVisualizer:
    def __init__(self, df):
        self.df = df

    def select_numerical_columns(self):
        self.numerical_columns = [col for col in self.df.select_dtypes(include='number').columns]

    def calculate_correlation_matrix(self):
        self.selected_df = self.df[self.numerical_columns]
        self.correlation_matrix = self.selected_df.corr()

    def plot_heatmap(self, figsize=(20, 16)):
        matrix = np.triu(self.correlation_matrix)
        plt.figure(figsize=figsize)
        sns.heatmap(self.correlation_matrix, annot=True, cmap='coolwarm', fmt=".1f", linewidths=0.5, mask=matrix)
        plt.title("Correlation Matrix Heatmap (Numerical Variables Only)")
        plt.show()

"""feature_engineering.py"""

class PokemonDataCalculator:
    def __init__(self, df):
        self.pokemon_data = df

    def normalize(self, column):
        return (self.pokemon_data[column] - self.pokemon_data[column].min()) / (self.pokemon_data[column].max() - self.pokemon_data[column].min())

    def check_columns(self, columns):
        missing_columns = [col for col in columns if col not in self.pokemon_data.columns]
        if missing_columns:
            raise ValueError(f"Missing columns in the dataset: {missing_columns}")

class OffensivePowerCalculator(PokemonDataCalculator):
    def calculate(self):
        self.check_columns(['Sp. Atk', 'Attack', 'Speed'])
        normalized_sp_atk = self.normalize('Sp. Atk')
        normalized_attack = self.normalize('Attack')
        normalized_speed = self.normalize('Speed')
        self.pokemon_data['Offensive_Power'] = (normalized_sp_atk + normalized_attack + 2 * normalized_speed) / 4

class DefensivePowerCalculator(PokemonDataCalculator):
    def calculate(self):
        self.check_columns(['Sp. Def', 'Defense', 'HP'])
        normalized_sp_def = self.normalize('Sp. Def')
        normalized_defense = self.normalize('Defense')
        normalized_hp = self.normalize('HP')
        self.pokemon_data['Defensive_Power'] = (normalized_sp_def + normalized_defense + 2 * normalized_hp) / 4

class SpeedToPowerRatioCalculator(PokemonDataCalculator):
    def calculate(self):
        self.check_columns(['Speed', 'Attack', 'Sp. Atk'])
        normalized_speed = self.normalize('Speed')
        normalized_attack = self.normalize('Attack')
        normalized_sp_atk = self.normalize('Sp. Atk')
        total_attack_power = normalized_sp_atk + normalized_attack
        self.pokemon_data['Speed_to_Power_Ratio'] = normalized_speed / (total_attack_power + 0.001)  # Adding a small constant to avoid division by zero

class PokemonTypeEncoder:
    def __init__(self, df):
        self.original_df = df  # Store the original DataFrame
        self.pokemon_data = df.copy()  # Create a copy to avoid modifying the original DataFrame directly
        self.type_columns = []  # To store the names of type columns

    def one_hot_encode_type1(self):
        # One-hot encode 'Type 1'
        type1_dummies = pd.get_dummies(self.pokemon_data['Type 1'], dummy_na=False)
        self.type_columns = type1_dummies.columns.tolist()
        self.pokemon_data = pd.concat([self.pokemon_data, type1_dummies], axis=1)

    def encode_type2(self):
        # For each type in 'Type 2', if it's a type column, set it to 1
        for index, row in self.pokemon_data.iterrows():
            type2 = row['Type 2']
            if type2 and type2 in self.type_columns:
                self.pokemon_data.at[index, type2] = 1
            elif type2:
                # If it's a new type, add it to the type_columns list and create a new column
                self.type_columns.append(type2)
                self.pokemon_data[type2] = 0
                self.pokemon_data.at[index, type2] = 1

    def get_updated_dataframe(self):
        # Reset indexes before concatenation
        original_df_reset = self.original_df.reset_index(drop=True)
        pokemon_data_reset = self.pokemon_data.reset_index(drop=True)

        # Merge the modified DataFrame with the original DataFrame
        merged_df = pd.concat([original_df_reset, pokemon_data_reset], axis=1)

        # Convert True/False values to 1/0 in the columns created during one-hot encoding
        for column in self.type_columns:
            merged_df[column] = merged_df[column].astype(int)

        # Remove duplicate columns after merging
        merged_df = merged_df.loc[:,~merged_df.columns.duplicated()]

        return merged_df


class Pokemon_Dummy_Legendary_Encoder:
    def __init__(self, df):
        self.pokemon_data = df

    def encode_legendary(self):
        self.pokemon_data['Legendary'] = self.pokemon_data['Legendary'].astype(int)

"""pre_and_post_EDA_functions.py"""

class PokemonVisualizations:
    def __init__(self, df):
        self.df = df

    def box_plot_by_generation(self):
        plt.figure(figsize=(12, 6))
        sns.boxplot(data=self.df, x='Generation', y='Victory_Rate')
        plt.title('Box Plot of Victory Rate by Generation')
        plt.xlabel('Generation')
        plt.ylabel('Victory Rate')
        plt.xticks(rotation=45)
        plt.show()

    def violin_plot_by_type_columns(self):
        # Hardcode the type columns you want to use
        type_columns = ['Bug', 'Dark', 'Dragon', 'Electric', 'Fairy',
                        'Fighting', 'Fire', 'Flying', 'Ghost', 'Grass',
                        'Ground', 'Ice', 'Normal', 'Poison', 'Psychic',
                        'Rock', 'Steel', 'Water']

        # Create a new column 'Type' that combines all type columns
        self.df['Type'] = self.df[type_columns].idxmax(axis=1)

        plt.figure(figsize=(12, 6))
        sns.violinplot(data=self.df, x='Type', y='Victory_Rate')
        plt.title('Violin Plot of Victory Rate by Type')
        plt.xlabel('Type')
        plt.ylabel('Victory Rate')
        plt.xticks(rotation=45)
        plt.show()

    def box_plot_legendary_vs_non_legendary(self):
        plt.figure(figsize=(8, 6))
        sns.boxplot(data=self.df, x='Legendary', y='Victory_Rate')
        plt.title('Box Plot of Victory Rate for Legendary vs. Non-Legendary Pokemon')
        plt.xlabel('Legendary')
        plt.ylabel('Victory Rate')
        plt.xticks([0, 1], ['Non-Legendary', 'Legendary'])
        plt.show()

    def compare_victory_rate_legendary_vs_non_legendary(self):
        mean_victory_rate_legendary = self.df[self.df['Legendary'] == 1]['Victory_Rate'].mean()
        mean_victory_rate_non_legendary = self.df[self.df['Legendary'] == 0]['Victory_Rate'].mean()

        print("Mean Victory Rate for Legendary Pokemon:", mean_victory_rate_legendary)
        print("Mean Victory Rate for Non-Legendary Pokemon:", mean_victory_rate_non_legendary)




"""1. Load Data"""
# 4. Data Processing
battle_processor = PokemonBattleProcessor(pokemon_data, combats_data)
battle_processor.process_battle_data()
df = battle_processor.get_processed_data()


"""2. New File EDA"""
# 5. Basic Data Checks and EDA
numeric_analyzer = NumericDataAnalyzer(df)
numeric_stats = numeric_analyzer.calculate_statistics()
print(numeric_stats)

categorical_analyzer = CategoricalDataAnalyzer(df)
categorical_stats = categorical_analyzer.calculate_statistics()
print(categorical_stats)

missing_data_visualizer = MissingDataVisualizer(df)
missing_data_visualizer.visualize_missing_data()

visualizer = CorrelationMatrixVisualizer(df)
visualizer.select_numerical_columns()
visualizer.calculate_correlation_matrix()
visualizer.plot_heatmap()

# 6. Feature Engineering
offensive_calculator = OffensivePowerCalculator(df)
defensive_calculator = DefensivePowerCalculator(df)
speed_power_ratio_calculator = SpeedToPowerRatioCalculator(df)

offensive_calculator.calculate()
defensive_calculator.calculate()
speed_power_ratio_calculator.calculate()

type_encoder = PokemonTypeEncoder(df)
type_encoder.one_hot_encode_type1()
type_encoder.encode_type2()
df = type_encoder.get_updated_dataframe()

legendary_encoder = Pokemon_Dummy_Legendary_Encoder(df)
legendary_encoder.encode_legendary()

""""Now it runs back the old EDA to make sure everything is fine"""
# 7. EDA Post Feature Engineering
numeric_analyzer = NumericDataAnalyzer(df)
numeric_stats = numeric_analyzer.calculate_statistics()
print(numeric_stats)

categorical_analyzer = CategoricalDataAnalyzer(df)
categorical_stats = categorical_analyzer.calculate_statistics()
print(categorical_stats)

missing_data_visualizer = MissingDataVisualizer(df)
missing_data_visualizer.visualize_missing_data()

visualizer = CorrelationMatrixVisualizer(df)
visualizer.select_numerical_columns()
visualizer.calculate_correlation_matrix()
visualizer.plot_heatmap()

"""These are the new visualizations created post engineering"""
# 8. Additional Visualizations
pokemon_visualizations = PokemonVisualizations(df)
pokemon_visualizations.box_plot_by_generation()
pokemon_visualizations.violin_plot_by_type_columns()
pokemon_visualizations.box_plot_legendary_vs_non_legendary()
pokemon_visualizations.compare_victory_rate_legendary_vs_non_legendary()





"""Testing"""

#import pytest
#import pandas as pd
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