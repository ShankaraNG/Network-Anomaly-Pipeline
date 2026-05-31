from ml_build.services.modelling import NetworkAnomalyNN
import torch
import torch.nn as nn
import torch.optim as optim
import os
import joblib
from app.config_loader import load_config
from app.logger import get_logger

BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

log = get_logger("Neural Network service")

def loadneuralnets():
    try:
        conf = load_config()
        log.info("Loading the neural network path")
        modelpath = conf['model']['save_path']
        modelname = conf['model']['namenn']
        if modelpath is None or modelname is None:
            raise ValueError("The model path and model name not found")
        log.info("Model path found")
        scalarmodelname = 'scaler.pkl'
        neuralnetmodelpath = os.path.join(BASE_DIR, modelpath, f"{modelname}.pth")
        scalarmodelpath = os.path.join(BASE_DIR, modelpath, scalarmodelname)
        if not os.path.exists(neuralnetmodelpath):
            raise ValueError(f"Required neural network model file missing at: {neuralnetmodelpath}")
        if not os.path.exists(scalarmodelpath):
            raise ValueError(f"Required scalar model file missing at: {scalarmodelpath}")
        log.info("All the scaler and Neural Network weights exists in the path")
        log.info("Loading the Scaler model")
        scalar = joblib.load(scalarmodelpath)
        log.info("Scaler Model Loaded successfully")
        log.info("Loading the Neural network Model")
        model = NetworkAnomalyNN()
        model.load_state_dict(torch.load(neuralnetmodelpath))
        log.info("Neural Network model loaded successfully")
        model.eval()
        log.info("Converting it to eval state")
        return scalar, model
    except Exception as e:
        log.error(e)
        return None, None
