import os
import json
import psycopg2
import pandas.io.sql as sqlio
from sqlalchemy import create_engine
import pandas as pd
from sklearn.model_selection import train_test_split

#### Loading datasets from Database
# call database credentials from json and save them in variable
def import_credentials():
        path = os.getcwd()
        with open(path+'/query_credentials.json') as f: 
            parms = json.load(f) 
        return parms
    
class loader_database():
    def __init__(self):
        self.credentials = import_credentials()
        self.engine = create_engine(
            f"postgresql://{self.credentials['user']}:{self.credentials['password']}@{self.credentials['host']}:{self.credentials['port']}/{self.credentials['database']}")

    def load_datasets(self):
        sql_1 = "SELECT * FROM public.combats;"
        sql_2 = "SELECT * FROM public.pokemon;"
        sql_3 = "SELECT * FROM public.pokemon_id_each_team;"
        sql_4 = "SELECT * FROM public.team_combat;"
        sql_5 = "SELECT * FROM public.Processed;"
        combats = sqlio.read_sql_query(sql_1, self.engine)
        pokemon = sqlio.read_sql_query(sql_2, self.engine)
        pokemon_id_each_team = sqlio.read_sql_query(sql_3, self.engine)
        team_combat = sqlio.read_sql_query(sql_4, self.engine)
        ProcessedData = sqlio.read_sql_query(sql_5, self.engine)
         
        #self.engine.dispose()
        return combats, pokemon, pokemon_id_each_team, team_combat, ProcessedData
    
#### Loading datasets using Jupyter Notebook
class loader_jupyter():
    def __init__(self):
        self.file_combats = os.getcwd()+'/raw_data/combats.csv'
        self.file_pokemon = os.getcwd()+'/raw_data/pokemon.csv'
        self.file_pokemon_id_each_team = os.getcwd()+'/raw_data/pokemon_id_each_team.csv'
        self.file_team_combat = os.getcwd()+'/raw_data/team_combat.csv'
        self.file_Processed = os.getcwd()+'/raw_data/ProcessedData.csv'

    def load_datasets(self):
        combats = pd.read_csv(self.file_combats)
        pokemon = pd.read_csv(self.file_pokemon)
        pokemon_id_each_team = pd.read_csv(self.file_pokemon_id_each_team)
        team_combat = pd.read_csv(self.file_team_combat)
        Processed = pd.read_csv(self.file_Processed)
        return combats, pokemon, pokemon_id_each_team, team_combat, Processed 

#### Loading datasets Using Vscode
class loader_vscode():
    def __init__(self):
        self.file_combats = os.getcwd()+'/Notebooks/raw_data/combats.csv'
        self.file_pokemon = os.getcwd()+'/Notebooks/raw_data/pokemon.csv'
        self.file_pokemon_id_each_team = os.getcwd()+'/Notebooks/raw_data/pokemon_id_each_team.csv'
        self.file_team_combat = os.getcwd()+'/Notebooks/raw_data/team_combat.csv'
        self.file_Processed = os.getcwd()+'/Notebooks/raw_data/ProcessedData.csv'

    def load_datasets(self):
        combats = pd.read_csv(self.file_combats)
        pokemon = pd.read_csv(self.file_pokemon)
        pokemon_id_each_team = pd.read_csv(self.file_pokemon_id_each_team)
        team_combat = pd.read_csv(self.file_team_combat)
        Processed = pd.read_csv(self.file_Processed)
        return combats, pokemon, pokemon_id_each_team, team_combat, Processed 

        


### Adding new columns to Pokemon DataFrame

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