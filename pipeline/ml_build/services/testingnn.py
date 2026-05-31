import torch
import torch.nn as nn
import torch.optim as optim
import joblib
import os
import pandas as pd
from ml_build.config_loader import load_config
from ml_build.services.modelling import NetworkAnomalyNN
from ml_build.logger import get_logger
from sklearn.metrics import accuracy_score


BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

log = get_logger("Testing the Neural Network")

def testneuralnets():
    try:
        conf = load_config()
        modelpath = conf['model']['save_path']
        modelname = conf['model']['namenn']
        if modelpath is None or modelname is None:
            raise ValueError("The model path and model name not found")
        scalarmodelname = 'scaler.pkl'
        neuralnetmodelpath = os.path.join(BASE_DIR, modelpath, f"{modelname}.pth")
        scalarmodelpath = os.path.join(BASE_DIR, modelpath, scalarmodelname)
        if not os.path.exists(neuralnetmodelpath):
            raise ValueError(f"Required neural network model file missing at: {neuralnetmodelpath}")
        if not os.path.exists(scalarmodelpath):
            raise ValueError(f"Required scalar model file missing at: {scalarmodelpath}")
        log.info("Loading the scaler model")
        scalar = joblib.load(scalarmodelpath)
        log.info("Scaler model loaded successfully")
        log.info("Loading the neural network model")
        model = NetworkAnomalyNN()
        log.info("Neural Network model loaded successfully")
        log.info("Loading the Nueral Network Weights")
        model.load_state_dict(torch.load(neuralnetmodelpath))
        log.info("Nueral Network Weights loaded successfully")
        log.info("Starting the Neural Network Model evaluation")
        model.eval()
        log.info("Changed to Evaluate Mode")
        log.info("Creating the sample test case")
        data_df = pd.DataFrame({

            'Inbound_Rate': [
                -0.56,
                -0.61,
                -0.69,
                -0.73,
                -0.77,
                -0.80,
                -0.82,
                -0.84,
                -0.85,
                -0.86
            ],

            'Outbound_Rate': [
                -0.70,
                -0.72,
                -0.76,
                -0.78,
                -0.79,
                -0.80,
                -0.81,
                -0.83,
                -0.84,
                -0.85
            ],

            'Inbound_Bandwidth_Utilization': [
                -0.55,
                -0.61,
                -0.69,
                -0.73,
                -0.77,
                -0.80,
                -0.82,
                -0.84,
                -0.85,
                -0.86
            ],

            'Outbound_Bandwidth_Utilization': [
                -0.70,
                -0.72,
                -0.76,
                -0.78,
                -0.79,
                -0.81,
                -0.82,
                -0.84,
                -0.85,
                -0.86
            ],

            'Expected_Label': [
                0,
                0,
                0,
                0,
                0,
                1,
                1,
                1,
                1,
                1
            ]
        })
        log.info("Sample test case created")
        x_data = data_df.drop('Expected_Label', axis=1)
        x_test = data_df['Expected_Label']
        log.info("Scaling the sample test case")
        scaled_data = scalar.transform(x_data)
        log.info("Sample test case has been scaled")
        log.info("Converting the data to torch tensor")
        data_tensor = torch.tensor(scaled_data, dtype=torch.float32)
        log.info("Converted successfully")
        log.info("Predicting the test samples")
        with torch.no_grad():
            probabilities = model(data_tensor)
            predictions = ( probabilities >= 0.5 ).float()
        log.info("Prediction completed")
        predictions = predictions.flatten()
        log.info("Calculating the accuracy score")
        accuracyscore = accuracy_score(x_test, predictions)
        artifacts_path = conf['artifacts']['save_path']
        accuracyscorepath = os.path.join(BASE_DIR, artifacts_path, 'neural_network_testing_accuracy.txt')
        with open(accuracyscorepath, 'w') as f:
            f.write(f"Accuracy Score: {accuracyscore:.4f}")
        log.info(f"Accuracy score has been calculated for the test sample {accuracyscore}")
        return "successfull"
    except Exception as e:
        log.error(f"Error during Neural Network model testing: {e}")
        return None
        


