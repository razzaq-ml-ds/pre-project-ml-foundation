import pandas as pd
from pathlib import Path

class DataLoader:
    def __init__(self,file_path):
        self.file_path = file_path
    def load_data(self):
        path = Path(self.file_path)

        if not path.exists():
            raise FileNotFoundError(f"file not found: {self.file_path}")
        
        return pd.read_csv(path)

        


