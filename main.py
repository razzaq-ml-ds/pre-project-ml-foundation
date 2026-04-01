from project_settings import PROJECT_CONFIG
import logging 
from src.data_loader import DataLoader
from src.logger import setup_logger

def main():
    setup_logger()
    logging.info("project started!")

    config = PROJECT_CONFIG

    file_path = config["data"]["path"]

    loader = DataLoader(file_path)
    df = loader.load_data()

    print(df.shape)

    logging.info(f"dataset loaded successfully with shape {df.shape}")


if __name__ == "__main__":
    main()