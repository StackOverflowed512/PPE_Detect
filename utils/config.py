import json

def load_config(config_path="config.json"):
    try:
        with open(config_path, 'r') as f:
            config = json.load(f)
        return config
    except FileNotFoundError:
        raise Exception(f"Configuration file '{config_path}' not found")
    except json.JSONDecodeError:
        raise Exception(f"Invalid JSON in configuration file '{config_path}'")