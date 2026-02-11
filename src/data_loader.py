import csv

class CustomerDataLoader:
    def __init__(self,file_path):
        self.file_path = file_path

    def load_data(self):
        try:
            data = []
            with open (self.file_path,mode="r") as file:
                reader = csv.reader(file)
                for row in reader:
                    data.append(row)
            return data
        except FileNotFoundError:
            raise FileNotFoundError("The specified file was not found.")
        
    def validate_columns(self,required_columns):
        data = self.load_data()
        header = data[0]

        for col in required_columns:
            if col not in header:
                raise ValueError(f"missing required column:{col}")
            
        return True