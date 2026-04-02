from project_settings import PROJECT_CONFIG
import logging 
from src.data_loader import DataLoader
from src.logger import setup_logger
from src.preprocessing import DataPreprocessor
from sklearn.model_selection import train_test_split

def main():
    setup_logger()
    logging.info("project started!")

    config = PROJECT_CONFIG

    file_path = config["data"]["path"]

    loader = DataLoader(file_path)
    df = loader.load_data()

    X = df.drop(columns=[config["data"]["target_column"]])
    y = df[config["data"]["target_column"]]

    X_train, X_test, y_train, y_test =  train_test_split(
        X,
        y,
        test_size= config["training"]['test_size'],
        random_state=config["training"]['random_state'],
        stratify=y
    )

    preprocessor = DataPreprocessor(config)
    X_train_preprocessed = preprocessor.fit_transform(X_train)
    X_test_preprocessed = preprocessor.transform(X_test)

    print(f"training data shape: {X_train_preprocessed.shape}")
    print(f"testing data shape: {X_test_preprocessed.shape}")


if __name__ == "__main__":
    main()