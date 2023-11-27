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

# 4. Data Processing
battle_processor = PokemonBattleProcessor(pokemon_data, combats_data)
battle_processor.process_battle_data()
df = battle_processor.get_processed_data()

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

# 8. Additional Visualizations
pokemon_visualizations = PokemonVisualizations(df)
pokemon_visualizations.box_plot_by_generation()
pokemon_visualizations.violin_plot_by_type_columns()
pokemon_visualizations.box_plot_legendary_vs_non_legendary()
pokemon_visualizations.compare_victory_rate_legendary_vs_non_legendary()

