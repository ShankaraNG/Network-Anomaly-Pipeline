from sklearn.preprocessing import StandardScaler
import torch
import torch.nn as nn
import torch.optim as optim
from ml_build.config_loader import load_config
from ml_build.logger import get_logger
import os
import joblib


BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

log = get_logger("Neural Network Preprocessing")

def standardizationfornetural(x_train, y_train, x_test, y_test):
    try:
        conf = load_config()
        modelpath = conf['model']['save_path']
        if modelpath is None:
            raise ValueError("model path configuration does not exists")
        modelname = 'scaler.pkl'
        modelpath = os.path.join(BASE_DIR, modelpath, modelname)
        if not os.path.exists(os.path.dirname(os.path.join(BASE_DIR, modelpath))):
            raise ValueError("The model path does not exists")
        log.info("Creating the Standard Scaler model")
        scaler = StandardScaler()
        log.info("Fitting and Transforming the data for the scaler")
        x_train_scaled = scaler.fit_transform(x_train)
        log.info("Saving the scaler model to the path")
        joblib.dump(scaler, modelpath)
        log.info("Scaler model has been successfully saved")
        log.info("Transforming the test data using the scaler")
        x_test_scaled = scaler.transform(x_test)
        log.info("Creating the tensors")
        X_train_tensor = torch.tensor(
            x_train_scaled,
            dtype=torch.float32
        )
        X_test_tensor = torch.tensor(
            x_test_scaled,
            dtype=torch.float32
        )
        y_train_tensor = torch.tensor(
            y_train.values,
            dtype=torch.float32
        ).view(-1, 1)
        y_test_tensor = torch.tensor(
            y_test.values,
            dtype=torch.float32
        ).view(-1, 1)
        log.info("Tensors has been created successfully")
        return X_train_tensor, y_train_tensor, X_test_tensor, y_test_tensor
    except Exception as e:
        log.error(f"Error during Neural Network Preprocessing: {e}")
        return None, None, None, None