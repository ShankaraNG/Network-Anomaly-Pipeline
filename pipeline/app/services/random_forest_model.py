import joblib
import os
import pandas as pd
import numpy as np
from app.config_loader import load_config
from app.logger import get_logger


BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

log = get_logger("Random forest service")

def test_model():
    try:
        conf = load_config()
        model_name = conf['model']['name']
        model_path = conf['model']['save_path']
        log.info("Loading the model path")
        if model_name is None or model_path is None:
            raise ValueError("Model name or model path is not defined in the configuration.")
        log.info("Model path found")
        model_file = os.path.join(BASE_DIR, model_path, f"{model_name}.pkl")
        if not os.path.exists(model_file):
            raise FileNotFoundError("Model file not found.")
        log.info("Model exists in the path")
        log.info("Loading the model")
        pipeline = joblib.load(model_file)
        log.info("Model Loaded successfully")
        return pipeline
    except Exception as e:
        print(e)
        return None