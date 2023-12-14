import pandas as pd
import matplotlib as plt
import seaborn as sns
import numpy as np
import statistics


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

            missing_percentages_df['Column Name'] = missing_percentages_df['Column Name'].astype(str)

            plt.figure(figsize=(10, 9))
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
        type_columns = ['Bug', 'Dark', 'Dragon', 'Electric', 'Fairy',
                        'Fighting', 'Fire', 'Flying', 'Ghost', 'Grass',
                        'Ground', 'Ice', 'Normal', 'Poison', 'Psychic',
                        'Rock', 'Steel', 'Water']

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