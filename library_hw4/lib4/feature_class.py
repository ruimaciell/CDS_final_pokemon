### Class that takes the data as features.
#Method that does one hot encoding for specific categorical value
#Method that inputs the average on 'weight' column
import pandas as pd

class Feature_prep:
    def __init__(self,df):
        self.df = df


class One_hot_enc(Feature_prep):

    def one_hot_enc(self,column_name: str):
        column_data_type = self.df[column_name].dtype
        print(column_data_type)
        if column_data_type != 'object':
            raise TypeError("Must ve a categorical variable")
        df2 = pd.get_dummies(self.df, columns=[column_name], prefix=column_name)
        df2 = df2.replace({True: 1, False: 0})
        return df2

class replace_booleans(Feature_prep):
    def input_mean_weight(self,column_name: str):
        df2 = self.df.copy()
        df2[column_name] = df2[column_name].replace('M',1).replace('F',0)
        return df2


