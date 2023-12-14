import pandas as pd

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
                self.pokemon_data.at[index, type2] = int(1)
            elif type2:
                # If it's a new type, add it to the type_columns list and create a new column
                self.type_columns.append(type2)
                self.pokemon_data[type2] = int(0)
                self.pokemon_data.at[index, type2] = int(1)

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