import os
import sys
import pandas as pd
from sklearn.model_selection import train_test_split

class diabetes_loader_jupyter():
    def __init__(self):
        self.file_path = os.getcwd()+'/sample_diabetes_mellitus_data.csv'
        self.df = pd.read_csv(self.file_path)

    def train_and_test_data(self):
        df_train, df_test = train_test_split(self.df, test_size=0.2, random_state=42)
        return df_train, df_test


class diabetes_loader_vscode():
    def __init__(self):
        self.file_path = os.getcwd()+'/Notebooks/sample_diabetes_mellitus_data.csv'
        self.df = pd.read_csv(self.file_path)

    def train_and_test_data(self):
        df_train, df_test = train_test_split(self.df, test_size=0.2, random_state=42)
        return df_train, df_test



#data_loader = DiabetesDataLoader()
# Call the train_and_test_data method to get the split datasets
#df_train, df_test = data_loader.train_and_test_data()


