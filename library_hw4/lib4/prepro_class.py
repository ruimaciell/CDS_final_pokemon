import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import statistics

# Loading the datasets
pokemon_data = pd.read_csv('raw_data/pokemon.csv')
combats_data = pd.read_csv('raw_data/combats.csv')
import pandas as pd

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

# Usage example, assuming pokemon_data and combats_data are your DataFrames
processor = PokemonBattleProcessor(pokemon_data, combats_data)
processor.process_battle_data()
df = processor.get_processed_data()
processor.display_data()


## IT WORKKKKKKKKS. Now lets do basic things to see the state of the data
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


numeric_analyzer = NumericDataAnalyzer(df)
numeric_stats = numeric_analyzer.calculate_statistics()

categorical_analyzer = CategoricalDataAnalyzer(df)
categorical_stats = categorical_analyzer.calculate_statistics()

missing_data_visualizer = MissingDataVisualizer(df)
missing_data_visualizer.visualize_missing_data()

"""Pokemon Type Encoder"""
class PokemonTypeEncoder:
    def __init__(self, df):
        self.pokemon_data = df
        self.type_dummies = None  # Added to store type dummies

    def one_hot_encode_types(self):
        self.type_dummies = pd.get_dummies(self.pokemon_data[['Type 1', 'Type 2']].stack()).groupby(level=0).max()
        self.pokemon_data = pd.concat([self.pokemon_data, self.type_dummies], axis=1)

    def verify_encoding(self):
        self.pokemon_data['Calculated_Type_Count'] = self.pokemon_data[['Type 1', 'Type 2']].notnull().sum(axis=1)
        self.pokemon_data['Sum_One_Hot_Types'] = self.pokemon_data[self.type_dummies.columns].sum(axis=1)
        self.pokemon_data['Sums_Match'] = self.pokemon_data['Calculated_Type_Count'] == self.pokemon_data['Sum_One_Hot_Types']
        matching_percentage = self.pokemon_data['Sums_Match'].mean() * 100
        print(f"Percentage of rows where the sums match: {matching_percentage:.2f}%")

    def clean_data(self):
        self.pokemon_data.drop(['Calculated_Type_Count', 'Sum_One_Hot_Types', 'Sums_Match'], axis=1, inplace=True)


# Usage example assuming df is your DataFrame
processor = Pokemon_type_Encoder(df)
processor.one_hot_encode_types()
processor.verify_encoding()
processor.clean_data()

# Selecting numerical columns except those to be excluded
numerical_columns = [col for col in df.select_dtypes(include='number').columns]

# Creating a new dataframe with only the selected numerical columns
selected_df = df[numerical_columns]

# Correlation matrix 
correlation_matrix = selected_df.corr()
# Getting the Upper Triangle of the co-relation matrix
matrix = np.triu(correlation_matrix)
# Create a heatmap
plt.figure(figsize=(20, 16)) 
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt=".1f", linewidths=0.5, mask=matrix)
plt.title("Correlation Matrix Heatmap (Numerical Variables Only)")
plt.show()