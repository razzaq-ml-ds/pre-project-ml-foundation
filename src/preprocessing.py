import pandas as pd
from sklearn.preprocessing import StandardScaler



class DataPreprocessor:
    def __init__(self,config):
        self.config = config
        self.scaler = StandardScaler()
        self.columns_after_encoding = None
    def clean_data(self,df):
        df = df.copy()
        drop_col = self.config["data"]["drop_columns"]
        df = df.drop(columns=drop_col)

        numerical_col = self.config["features"]['numerical_columns']
        categorical_col = self.config["features"]["categorical_columns"]

        df["HasBalance"] = (df["Balance"] > 0).astype(int)
        df["IsAge50To60"] = ((df["Age"] > 50) & (df["Age"] <= 60)).astype(int)
        df["HasTwoProducts"] = (df["Num Of Products"] == 2).astype(int)
        df["HasThreePlusProducts"] = (df["Num Of Products"] >= 3).astype(int)
        # combining label to make one feature 
        df["InactivewithBalance"] = ((df["Is Active Member"] == 0 ) & (df["Balance"]>0)).astype(int)
        df["GermanyCustomer"] = (df["Geography"]=="Germany").astype(int)
        df["OneProductCustomer"] = (df["Num Of Products"] == 1).astype(int)
        df["BalanceSalaryRatio"] = df["Balance"] / (df["Estimated Salary"] + 1)
        df["TenureByAge"] = df["Tenure"] / (df["Age"] + 1)
        df["CreditScoreBucket"] = pd.cut(df["CreditScore"], bins=5, labels=False)
            
        # filling the missing values
        for col in numerical_col:
            if col in df.columns:
                df[col] = df[col].fillna(df[col].median())


        for col in categorical_col:
            if col in df.columns:
                df[col] = df[col].fillna(df[col].mode()[0])
        #one hot encoding
        df = pd.get_dummies(df,columns=categorical_col,drop_first=True)

        return df

    def fit_transform(self,df):
        df_cleaned = self.clean_data(df)
        self.columns_after_encoding = df_cleaned.columns.tolist()
        df_scaled = self.scaler.fit_transform(df_cleaned)
        return df_scaled
    

    def transform(self,df):
        df_cleaned = self.clean_data(df)
        # belwo line make new column if not in test learning from training data
        df_cleaned = df_cleaned.reindex(columns=self.columns_after_encoding, fill_value=0)
        df_scaled = self.scaler.transform(df_cleaned)
        return df_scaled