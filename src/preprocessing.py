import pandas as pd
from sklearn.preprocessing import StandardScaler

class DataPreprocessor:
    def __init__(self,config):
        self.config = config
        self.scaler = StandardScaler()
        self.columns_after_encoding = None

    
    def feature_engineering(self,df):

        df  = df.copy()

        if "SibSp" in df.columns and "Parch" in df.columns:
            df["FamilySize"] = df["SibSp"] + df["Parch"] + 1
            df["IsAlone"] = (df["FamilySize"]==1).astype(int)   

        return df     
    
    def clean_data(self,df):
        df = df.copy()

        columns_to_drop = self.config.get("columns_to_drop", [])

        columns_to_drop = [col for col in columns_to_drop if col in df.columns]
        df = df.drop(columns=columns_to_drop)

        fill_strategies = self.config.get("numerical_columns_fill_strategy", {})

        for column, strategy in fill_strategies.items():
            if column not in df.columns:
                continue  # Skip silently if column doesn't exist in this dataset

            if strategy == "median":
                df[column] = df[column].fillna(df[column].median())
            elif strategy == "mean":
                df[column] = df[column].fillna(df[column].mean())
            elif strategy == "zero":
                df[column] = df[column].fillna(0)
            else:
                raise ValueError(
                    f"Unknown fill strategy '{strategy}' for column '{column}'. "
                    f"Use 'median', 'mean', or 'zero'."
                )
            
        categorical_columns = self.config.get("categorical_columns", [])

        for column in categorical_columns:
            if column not in df.columns:
                continue
            df[column] = df[column].fillna(df[column].mode()[0])


        existing_categoricals = [col for col in categorical_columns if col in df.columns]
        if existing_categoricals:
            df = pd.get_dummies(df, columns=existing_categoricals, drop_first=True)

        return df


    def fit_transform(self,df):

        df = self.feature_engineering(df)
        df_cleaned = self.clean_data(df)

        self.columns_after_encoding = df_cleaned.columns.tolist()

        
        scaled_array = self.scaler.fit_transform(df_cleaned)
    
        return scaled_array
    

    def transform(self,df):

        if self.columns_after_encoding is None:
            raise RuntimeError(
                "fit_transform() must be called on training data before transform(). "
                "The scaler has not been fitted yet."
            )
        
        df = self.feature_engineering(df)
        df_cleaned = self.clean_data(df)

        df_cleaned = df_cleaned.reindex(columns=self.columns_after_encoding, fill_value=0)

        
        scaled_array = self.scaler.transform(df_cleaned)

        return scaled_array