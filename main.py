from src.data_loader import CustomerDataLoader
from src.config_loader import load_config
from src.logger import setup_logger
from src.preprocessing import DataPreprocessor
from sklearn.model_selection import train_test_split
from src.model_training import Modeltrainer
import pandas as pd
import logging

def main():
    setup_logger()
    logging.info('Project started')

        
    config = load_config("config.json")
    logging.info("config loaded successfully")


    loader = CustomerDataLoader(config["data_path"])
    df = loader.load_data()
    logging.info(f"Data loaded successfully. Shape: {df.shape}")

    target_column = config["target_column"]
    X = df.drop(columns=[target_column])
    y = df[target_column]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=config["test_size"],
        random_state=config["random_state"]
    )
    logging.info(f"Train size: {len(X_train)} | Test size: {len(X_test)}")

    preprocessor = DataPreprocessor(config)
    
    X_train_scaled = preprocessor.fit_transform(X_train)

    X_test_scaled = preprocessor.transform(X_test)

    logging.info("Preprocessing complete")

    trainer = Modeltrainer()
    trainer.train(X_train_scaled, y_train)
    

    accuracy, report, matrix = trainer.evaluate(X_test_scaled, y_test)
    

    print("\n" + "="*50)
    print(f"Model Accuracy: {accuracy:.4f}")
    print("\nClassification Report:\n", report)
    print("\nConfusion Matrix:\n", matrix)
    print("="*50)

    trainer.save_model("models/model.pkl")
    
    
if __name__ == "__main__":
    main()