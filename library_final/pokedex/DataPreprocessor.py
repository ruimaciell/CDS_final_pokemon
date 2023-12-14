import pandas as pd


# Process combat Data
class PokemonBattleProcessor:
    def __init__(self, pokemon_data, combats_data):
        self.pokemon_data = pokemon_data
        self.combats_data = combats_data

    def process_battle_data(self):
        # Counting victories and battles
        victory_counts = self.combats_data['winner'].value_counts()
        total_battles = (self.combats_data['first_pokemon'].value_counts() +
                         self.combats_data['second_pokemon'].value_counts())

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

    def display_data(self):
        print(self.pokemon_data)



# Calculating Pokemon Skill Level

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