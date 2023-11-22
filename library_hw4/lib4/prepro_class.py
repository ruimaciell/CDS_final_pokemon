import os
import pandas as pd
#df=pd.read_csv(os.getcwd()+'/Notebooks/sample_diabetes_mellitus_data.csv')

class DropMissingValuesGenderEthnicity:
    def __init__(self, df):
        self.df = df

    def process(self, columns=["age", "gender", "ethnicity"]):
        return self.df.dropna(subset=columns)

class FillMissingValuesWithMeanHeightWeight:
    def __init__(self, df):
        self.df = df

    def process(self, columns=["height", "weight"]):
        dz = self.df.copy()
        for column in columns:
            dz[column] = self.df[column].fillna(self.df[column].mean())
        return dz



# Create an instance and apply both transformations
#transformation_handler = FillMissingValuesWithMeanHeightWeight(df).process()
#final_df = DropMissingValuesGenderEthnicity(transformation_handler).process()
#print(final_df.isnull().sum())



