import os
import yaml
import json
import logging
from box.exceptions import BoxValueError
from ensure import ensure_annotations
from box import ConfigBox
from pathlib import Path
from typing import Any

logging.basicConfig(level=logging.INFO, format='[%(asctime)s]: %(message)s:')

@ensure_annotations
def read_yaml(path_to_yaml: Path) -> ConfigBox:
    """Reads yaml file and returns a ConfigBox."""
    try:
        with open(path_to_yaml) as yaml_file:
            content = yaml.safe_load(yaml_file)
            logging.info(f"yaml file: {path_to_yaml} loaded successfully")
            return ConfigBox(content)
    except BoxValueError:
        raise ValueError("yaml file is empty")
    except Exception as e:
        raise e

@ensure_annotations
def create_directories(path_to_directories: list, verbose=True):
    """Creates a list of directories."""
    for path in path_to_directories:
        os.makedirs(path, exist_ok=True)
        if verbose:
            logging.info(f"Created directory at: {path}")

@ensure_annotations
def save_json(path: Path, data: dict):
    """Saves data to a json file (Used during Model Evaluation)."""
    with open(path, "w") as f:
        json.dump(data, f, indent=4)
    logging.info(f"json file saved at: {path}")