import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import statistics


import pandas as pd

# Loading the datasets
pokemon_data = pd.read_csv('raw_data/pokemon.csv')
combats_data = pd.read_csv('raw_data/combats.csv')

# Counting the number of victories for each Pokémon
victory_counts = combats_data['Winner'].value_counts()

# Counting the total number of battles for each Pokémon
battle_counts_first = combats_data['First_pokemon'].value_counts()
battle_counts_second = combats_data['Second_pokemon'].value_counts()
total_battles = battle_counts_first.add(battle_counts_second, fill_value=0)

# Calculating the victory rate for each Pokémon
victory_rate = victory_counts / total_battles

# Creating a DataFrame for the victory rates
victory_rate_df = pd.DataFrame({'#': victory_rate.index, 'Victory_Rate': victory_rate.values})

# Merging the victory rates with the Pokémon details
pokemon_victory_data = pd.merge(pokemon_data, victory_rate_df, on='#', how='left')

# Adding Total Battles and Victory Counts to the DataFrame
pokemon_victory_data['Total_Battles'] = pokemon_victory_data['#'].map(total_battles)
pokemon_victory_data['Victory_Counts'] = pokemon_victory_data['#'].map(victory_counts)

# Filling NaN values in Victory_Rate, Total_Battles, and Victory_Counts with 0 for Pokémon that have not battled
pokemon_victory_data['Victory_Rate'] = pokemon_victory_data['Victory_Rate'].fillna(0)
pokemon_victory_data['Total_Battles'] = pokemon_victory_data['Total_Battles'].fillna(0)
pokemon_victory_data['Victory_Counts'] = pokemon_victory_data['Victory_Counts'].fillna(0)

# Displaying the first few rows of the merged data (optional)
print(pokemon_victory_data.head())

## IT WORKKKKKKKKS. Now lets do basic things to see the state of the data

df=pokemon_victory_data

# Set display option to avoid scientific notation and limit decimals
pd.set_option('display.float_format', lambda x: '{:.1f}'.format(x) if x % 1 else '{:.0f}'.format(x))

def calculate_column_statistics_for_numeric_variables(df):
    # Filter columns with numeric (int or float) data types
    numeric_columns = df.select_dtypes(include=['int', 'float']).columns

    # Get all column names
    all_columns = df.columns
    total_columns = len(all_columns)

    num_numeric_columns = len(numeric_columns)

    # Find columns with categorical values
    categorical_columns = [col for col in all_columns if df[col].dtype == 'object']

    print(f"Total number of columns: {total_columns}")
    print(f"Number of numeric columns (int or float): {num_numeric_columns}")

    # Create a dictionary to store statistics
    stats_dict = {
        'Variable': numeric_columns,
        'Mean': [],
        'Mode': [],
        'Median': [],
        'Standard Deviation': [],
        'Minimum': [],
        'Maximum': [],
        'Count': [],
        'IQR': [],
        'Skewness': [],
        'Range': []
    }

    for column in numeric_columns:
        mean = df[column].mean()
        mode = statistics.mode(df[column].dropna())  # Handle potential multiple modes
        median = df[column].median()
        std_dev = df[column].std()
        min_val = df[column].min()
        max_val = df[column].max()
        count = df[column].count()
        iqr = np.percentile(df[column].dropna(), 75) - np.percentile(df[column].dropna(), 25)
        skew = df[column].skew()
        column_range = max_val - min_val

        stats_dict['Mean'].append(mean)
        stats_dict['Mode'].append(mode)
        stats_dict['Median'].append(median)
        stats_dict['Standard Deviation'].append(std_dev)
        stats_dict['Minimum'].append(min_val)
        stats_dict['Maximum'].append(max_val)
        stats_dict['Count'].append(count)
        stats_dict['IQR'].append(iqr)
        stats_dict['Skewness'].append(skew)
        stats_dict['Range'].append(column_range)

    # Create a DataFrame from the dictionary
    stats_df = pd.DataFrame(stats_dict)

    return stats_df


calculate_column_statistics_for_numeric_variables(df)

numeric_columns = df.select_dtypes(include=['int', 'float']).columns
numeric_columns

# List of variables to plot
variables = numeric_columns

# Setting up the subplots
fig, axes = plt.subplots(15, 3, figsize=(50, 250))

# Creating histograms for each variable
for ax, var in zip(axes.flatten(), variables):
    sns.histplot(df[var], ax=ax, color="skyblue", edgecolor='black', kde=True)
    ax.set_title(f'Distribution of {var}', fontsize = 30)
    ax.set_xlabel(var)
    ax.set_ylabel('Frequency')

# Adjusting the layout
plt.tight_layout()
plt.show()

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


def calculate_column_statistics_for_categorical_variables(df):
    # Filter columns with categorical data types
    categorical_columns = df.select_dtypes(include=['object']).columns

    # Get all column names
    all_columns = df.columns
    total_columns = len(all_columns)

    num_categorical_columns = len(categorical_columns)

    # Find columns with numeric values
    numeric_columns = [col for col in all_columns if df[col].dtype == 'int' or df[col].dtype == 'float']

    print(f"Total number of columns: {total_columns}")
    print(f"Number of categorical columns: {num_categorical_columns}")

    # Set the max_colwidth option to control the width of the displayed column
    pd.set_option('max_colwidth', 100)  # Adjust the width as needed

    # Create a dictionary to store statistics
    stats_dict = {
        'Variable': categorical_columns,
        'Number of Unique Values': [],
        'Top 5 Most Frequent Values': []
    }

    for column in categorical_columns:
        num_unique_values = df[column].nunique()
        top_5_frequent_values = df[column].value_counts().index[:5].tolist()  # Get the top 5 values as a list

        stats_dict['Number of Unique Values'].append(num_unique_values)
        stats_dict['Top 5 Most Frequent Values'].append(top_5_frequent_values)

    # Create a DataFrame from the dictionary
    stats_df = pd.DataFrame(stats_dict)

    # Sort the DataFrame downward by the number of unique values
    stats_df = stats_df.sort_values(by='Number of Unique Values', ascending=False)
    stats_df = stats_df.reset_index(drop=True)

    return stats_df

calculate_column_statistics_for_categorical_variables(df)

## Encoding categorical variables
type1_dummies = pd.get_dummies(pokemon_victory_data['Type 1'], prefix='Type1')
type2_dummies = pd.get_dummies(pokemon_victory_data['Type 2'], prefix='Type2')

combined_types = type1_dummies.add(type2_dummies, fill_value=0).clip(upper=1)

print(combined_types.head(20))


import pandas as pd

# Load the datasets (assuming they are already loaded as pokemon_victory_data)
# pokemon_data = pd.read_csv('path_to_pokemon.csv')
# combats_data = pd.read_csv('path_to_combats.csv')

# One-Hot Encode 'Type 1' and 'Type 2'
type1_dummies = pd.get_dummies(pokemon_victory_data['Type 1'])
type2_dummies = pd.get_dummies(pokemon_victory_data['Type 2'])

# Ensuring both DataFrames have the same columns
all_types = type1_dummies.columns.union(type2_dummies.columns)
type1_dummies = type1_dummies.reindex(columns=all_types, fill_value=0)
type2_dummies = type2_dummies.reindex(columns=all_types, fill_value=0)

# Combining the one-hot encoded data
combined_types = type1_dummies.add(type2_dummies, fill_value=0).clip(upper=1)

# List of one-hot encoded type columns
one_hot_type_columns = combined_types.columns.tolist()

# Concatenating the combined one-hot encoded types with the original data
pokemon_data_encoded = pd.concat([pokemon_victory_data, combined_types], axis=1)

# Calculating the number of types per Pokémon
pokemon_data_encoded['Calculated_Type_Count'] = pokemon_data_encoded['Type 1'].notnull().astype(int) + pokemon_data_encoded['Type 2'].notnull().astype(int)

# Summing the one-hot encoded type columns using the list
pokemon_data_encoded['Sum_One_Hot_Types'] = pokemon_data_encoded[one_hot_type_columns].sum(axis=1)

# Comparing the sums
pokemon_data_encoded['Sums_Match'] = pokemon_data_encoded['Calculated_Type_Count'] == pokemon_data_encoded['Sum_One_Hot_Types']

# Calculating the percentage of matching rows
matching_percentage = pokemon_data_encoded['Sums_Match'].mean() * 100

print(f"Percentage of rows where the sums match: {matching_percentage:.2f}%")