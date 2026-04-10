from project_settings import PROJECT_CONFIG
import logging 
from src.data_loader import DataLoader
from src.logger import setup_logger
from src.preprocessing import DataPreprocessor
from sklearn.model_selection import train_test_split
from src.model_training import ModelTrainer
import joblib
from pathlib import Path

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
    
    thresholds = [i/100 for i in range(5, 60, 1)]


    model_trainer = ModelTrainer(config)
    results,best_result  = model_trainer.train_and_compare(
        X_train_preprocessed,
        y_train,
        X_test_preprocessed,
        y_test,
        thresholds,
    )

    confu_matrix = model_trainer.get_confusion_matrix(
        model_trainer.best_model,
        X_test_preprocessed,
        y_test,
        model_trainer.best_threshold
    )

    print(f"all model results: {results}")
    print(f"best tuned model result: {best_result}")


    logging.info(f"all models resuls: {results}")
    logging.info(f"the best tuned model result: {best_result}")



    Path("models").mkdir(exist_ok=True)

    model_trainer.save_model("models/model.pkl")
    model_trainer.save_experiment_results(
        "models/experiment_results.json",
        results,
        best_result,
        thresholds,
        confu_matrix,
        )

    joblib.dump(preprocessor, "models/preprocessor.pkl")


if __name__ == "__main__":
    main()