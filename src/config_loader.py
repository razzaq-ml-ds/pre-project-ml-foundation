import json

def load_config(config_path):
    try:
        with open(config_path, "r") as file:
            config = json.load(file)
        return config
    except FileNotFoundError:
        raise FileNotFoundError("Config file not found")
