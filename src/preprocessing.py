import pandas as pd

class DataPreprocessor:
    def __init__(self,df):
        self.df = df
    
    def clean_data(self):
        df = self.df.copy()

        # droping columns that are not usefull as of now
        df = df.drop(columns=["PassengerId", "Name", "Ticket", "Cabin"])

        # filling missing age value with median
        df["Age"] = df["Age"].fillna(df["Age"].median())

        # filing embarked's value with mode(say most comman value)
        df["Embarked"] = df["Embarked"].fillna(df["Embarked"].mode()[0])

        # using one hot encoding to categorical values
        df = pd.get_dummies(df,columns=["Sex","Embarked"],drop_first=True)

        return df