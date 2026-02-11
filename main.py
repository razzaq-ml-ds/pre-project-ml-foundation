from src.data_loader import CustomerDataLoader
from src.config_loader import load_config
from src.logger import setup_logger
import logging

def main():
    setup_logger()
    logging.info('Project started')

        
    config = load_config("config.json")

    loader = CustomerDataLoader(config["data_path"])
    required_columns = ["customer_id", "purchase_amount", "purchase_date"]

    loader.validate_columns(required_columns)

    logging.info("All required columns are present.")


if __name__ == "__main__":
    main()