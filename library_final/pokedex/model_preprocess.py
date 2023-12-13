import pandas as pd
from sklearn.model_selection import train_test_split



### 2. Load Datasets. No need for function since we already have RUI's df.
pokemon_data = pd.read_csv('/Users/luispoli/Documents/BSE/T1/Computing_DS/Practice/CDS_final_pokemon/raw_data/ProcessedData.csv')

columns_to_use = ['#', 'Name', 'Type 1', 'Type 2', 'HP', 'Attack','Defense', 'Sp. Atk', 'Sp. Def', 'Speed', 'Generation', 'Legendary','Victory_Rate', 'Total_Battles', 'Victory_Counts', 'Offensive_Power','Defensive_Power', 'Speed_to_Power_Ratio', 'Bug', 'Dark', 'Dragon','Electric', 'Fairy', 'Fighting', 'Fire', 'Flying', 'Ghost', 'Grass','Ground', 'Ice', 'Normal', 'Poison', 'Psychic', 'Rock', 'Steel','Water']
pokemon_data = pokemon_data[columns_to_use]


### 3. Divide Data into subsets
class TrainTestDivider:
    def __init__(self, df):
        self.df = df

    def train_test(self,columns_to_drop):
        # Assuming df_X and df_y are attributes you want to access outside the method
        self.df_X = self.df.drop(columns_to_drop, axis=1)
        self.df_y = self.df[column_to_predict]

        # 20% split into test data.
        X_train, X_test, y_train, y_test = train_test_split(self.df_X, self.df_y, test_size=0.2, random_state=42)
        return X_train, X_test, y_train, y_test, self.df_X, self.df_y
    

columns_to_drop = ['Victory_Rate', 'Total_Battles', 'Victory_Counts', 'Type 1', 'Type 2', 'Name']
column_to_predict = ['Victory_Rate']
pokemon_data_foo = TrainTestDivider(pokemon_data)
X_train, X_test, y_train, y_test, df_X, df_y = pokemon_data_foo.train_test(columns_to_drop)


print(X_train.head())