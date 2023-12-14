import pandas as pd

class PokemonDataCalculator:
    def __init__(self, df):
        self.pokemon_data = df

    def normalize(self, column):
        return (self.pokemon_data[column] - self.pokemon_data[column].min()) / (self.pokemon_data[column].max() - self.pokemon_data[column].min())

    def check_columns(self, cols):
        missing_columns = [col for col in cols if col not in self.pokemon_data.columns]
        if missing_columns:
            raise ValueError(f"Missing columns in the dataset: {missing_columns}")

class OffensivePowerCalculator(PokemonDataCalculator):
    def __init__(self,df):
        super().__init__(df)
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