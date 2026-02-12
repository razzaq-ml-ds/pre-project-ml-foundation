import csv
import pandas as pd 
class CustomerDataLoader:

    def __init__(self,file_path):
        self.file_path = file_path

    def load_data(self):
        try:
            df = pd.read_csv(self.file_path)
            return df
        except FileNotFoundError:
            raise FileNotFoundError("the specified file was not found!")

        
    def validate_columns(self,required_columns):
        df = self.load_data()
        

        for col in required_columns:
            if col not in df.columns:
                raise ValueError(f"missing required column:{col}")
            
        return True