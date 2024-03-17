import json


def load_config(config_path: str) -> dict:
  with open(config_path, 'r') as f:
    config = json.load(f)
  return config


def save_config(config: dict, config_path: str) -> None:
  with open(config_path, 'w') as f:
    json.dump(config, f)
