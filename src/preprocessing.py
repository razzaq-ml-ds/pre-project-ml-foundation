import pandas as pd
from sklearn.preprocessing import StandardScaler

class DataPreprocessor:
    def __init__(self,df):
        self.scaler = StandardScaler()
        self.columns_after_encoding = None
        
    
    def clean_data(self):
        df = df.copy()

        # droping columns that are not usefull as of now
        df = df.drop(columns=["PassengerId", "Name", "Ticket", "Cabin"])

        # filling missing age value with median
        df["Age"] = df["Age"].fillna(df["Age"].median())

        # filing embarked's value with mode(say most comman value)
        df["Embarked"] = df["Embarked"].fillna(df["Embarked"].mode()[0])

        # using one hot encoding to categorical values
        df = pd.get_dummies(df,columns=["Sex","Embarked"],drop_first=True)

        return df
    
    def fit_transform(self,df):

        df_cleaned = self.clean_data(df)

        self.columns_after_encoding = df_cleaned.columns.tolist()

        scaled_array = self.scaler.fit_transform(df_cleaned)

        return scaled_array

    
    def transform(self,df):

        if self.columns_after_encoding is None:
            raise RuntimeError("You must call fit_transform on training data before calling transform.")

        df_cleaned = self.clean_data(df)

        df_cleaned = df_cleaned.reindex(columns=self.columns_after_encoding, fill_value=0)

        scaled_array = self.scaler.transform(df_cleaned)

        return scaled_array