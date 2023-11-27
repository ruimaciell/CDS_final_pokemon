# 1. Import Libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import statistics

# 2. Load Datasets
pokemon_data = pd.read_csv('raw_data/pokemon.csv')
combats_data = pd.read_csv('raw_data/combats.csv')

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
        plt.figure(figsize=(20, 9))
        plt.bar(missing_percentages_df['Column Name'], missing_percentages_df['Percentage Missing'], color='skyblue')
        plt.xlabel('Column Name')
        plt.ylabel('Percentage Missing')
        plt.title('Percentage of Missing Values by Column')
        plt.xticks(rotation=45, ha='right')
        plt.tight_layout(pad=0.0001)
        plt.show()

class PokemonDataCalculator:
    def __init__(self, df):
        self.pokemon_data = df

    def normalize(self, column):
        return (self.pokemon_data[column] - self.pokemon_data[column].min()) / (self.pokemon_data[column].max() - self.pokemon_data[column].min())

class OffensivePowerCalculator(PokemonDataCalculator):
    def calculate(self):
        normalized_sp_atk = self.normalize('Sp. Atk')
        normalized_attack = self.normalize('Attack')
        normalized_speed = self.normalize('Speed')
        self.pokemon_data['Offensive_Power'] = (normalized_sp_atk + normalized_attack + 2 * normalized_speed) / 4

class DefensivePowerCalculator(PokemonDataCalculator):
    def calculate(self):
        normalized_sp_def = self.normalize('Sp. Def')
        normalized_defense = self.normalize('Defense')
        normalized_hp = self.normalize('HP')
        self.pokemon_data['Defensive_Power'] = (normalized_sp_def + normalized_defense + 2 * normalized_hp) / 4

class SpeedToPowerRatioCalculator(PokemonDataCalculator):
    def calculate(self):
        normalized_speed = self.normalize('Speed')
        normalized_attack = self.normalize('Attack')
        normalized_sp_atk = self.normalize('Sp. Atk')
        total_attack_power = normalized_sp_atk + normalized_attack  
        self.pokemon_data['Speed_to_Power_Ratio'] = normalized_speed / total_attack_power  # Adjusted calculation

# 4. Data Processing
battle_processor = PokemonBattleProcessor(pokemon_data, combats_data)
battle_processor.process_battle_data()
df = battle_processor.get_processed_data()


# 6. Power Calculations
offensive_calculator = OffensivePowerCalculator(df)
offensive_calculator.calculate()

defensive_calculator = DefensivePowerCalculator(df)
defensive_calculator.calculate()

speed_ratio_calculator = SpeedToPowerRatioCalculator(df)
speed_ratio_calculator.calculate()

# 7. Final Checks
print("Sp. Atk length:", len(df['Sp. Atk']))
print("Attack length:", len(df['Attack']))
print("Speed length:", len(df['Speed']))

print("Sp. Atk missing values:", df['Sp. Atk'].isna().sum())
print("Attack missing values:", df['Attack'].isna().sum())
print("Speed missing values:", df['Speed'].isna().sum())

# 5. Data Analysis
numeric_analyzer = NumericDataAnalyzer(df)
numeric_stats = numeric_analyzer.calculate_statistics()
print(numeric_stats)

categorical_analyzer = CategoricalDataAnalyzer(df)
categorical_stats = categorical_analyzer.calculate_statistics()
print(categorical_stats)

missing_data_visualizer = MissingDataVisualizer(df)
missing_data_visualizer.visualize_missing_data()