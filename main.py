from src.data_loader import CustomerDataLoader
from src.config_loader import load_config
from src.logger import setup_logger
from src.preprocessing import DataPreprocessor
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from src.model_training import Modeltrainer
import pandas as pd
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

    # initialising scaler
    scaler = StandardScaler()#what actually is a scaler 

    # Fiting  only on training data
    X_train_scaled = scaler.fit_transform(X_train_cleaned)

    # Transform test data using same scaler
    X_test_scaled = scaler.transform(X_test_cleaned)
    
    trainer = Modeltrainer()
    trainer.train(X_train_scaled,y_train)#why scaled data is used btw

    accuracy, report, matrix = trainer.evaluate(X_test_scaled, y_test)#i not even 1 % understood this line of code like we assigned these three a value then we again did evalution what is happing 
    
    print("\nModel Accuracy:", accuracy)
    print("\nClassification Report:\n", report)
    print("\nConfusion Matrix:\n", matrix)
    
    # so my doubt is what is happing in the code like should i run it now or i should first run it with scalling code when run again with model training code 

if __name__ == "__main__":
    main()