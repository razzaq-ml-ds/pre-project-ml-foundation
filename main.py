from src.data_loader import CustomerDataLoader
from src.config_loader import load_config
from src.logger import setup_logger
from src.preprocessing import DataPreprocessor
from sklearn.model_selection import train_test_split
import logging

def main():
    setup_logger()
    logging.info('Project started')

        
    config = load_config("config.json")

    loader = CustomerDataLoader(config["data_path"])
    df = loader.load_data()

    target_column = "Survived"
    X = df.drop(columns=[target_column])
    y = df[target_column]

    X_train,X_test,y_train,y_test = train_test_split(
        X,y,
        test_size=0.2,
        random_state=42
    )

    # preprocessing the train data
    preprocessing_train = DataPreprocessor(X_train)
    X_train_cleaned = preprocessing_train.clean_data()

    # preprocessing the test data
    preprocessing_test = DataPreprocessor(X_test)
    X_test_cleaned = preprocessing_test.clean_data()

    print("X_train shape:", X_train_cleaned.shape)
    print("X_test shape:", X_test_cleaned.shape)
    print("y_train shape:", y_train.shape)
    print("y_test shape:", y_test.shape)

if __name__ == "__main__":
    main()