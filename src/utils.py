import yaml
import joblib
from datetime import datetime
from sklearn.preprocessing import OneHotEncoder
import joblib
import pandas as pd

CONFIG_DIR = "../config/config.yaml"


# load config
def load_config() -> dict: 
    try:
        with open(CONFIG_DIR, "r") as file:
            config = yaml.safe_load(file)
    except FileNotFoundError as fe:
        raise RuntimeError("Parameters file not found in path.")

    return config


# pickle load
def pickle_load(file_path: str):
    # Load and return pickle file
    return joblib.load(file_path)


# pickle dump
def pickle_dump(data, file_path: str) -> None:
    # Dump data into file
    joblib.dump(data, file_path)


print('utils good')