from fastapi import APIRouter, HTTPException, Depends, Request
import pandas as pd
from app.logger import get_logger
from app.schemas import PredictionRequest
from app.services.random_forest_model import test_model
from app.services.neural_network import loadneuralnets
import torch

router = APIRouter()

log = get_logger("predictor routes")

randomforestmodel = test_model()

scaler, neuralnetworkmodel = loadneuralnets()

@router.get("/")
def root():
    try:
        log.info("Accessing the root directory")
        return {"message": "Welcome to the Network Anomaly Detection API"}
    except Exception as e:
        log.error(e)
        raise

@router.get("/health")
def root():
    try:
        log.info("Accessing the health directory")
        if randomforestmodel is None or scaler is None or neuralnetworkmodel is None:
            log.info("Health Status is down")
            result = "DOWN"
        else:
            log.info("Health Status is up")
            result = "UP"
        return {"Health Status": result}
    except Exception as e:
        log.error(e)
        raise


@router.post("/v1/predict")
def randomforestprediction(request:list[PredictionRequest]):
    try:
        log.info("Running the random forest prediction")
        data = []
        for r in request:
            data.append({
                'Inbound_Rate' : r.Inbound_Rate,
                'Outbound_Rate': r.Outbound_Rate,
                'Inbound_Bandwidth_Utilization': r.Inbound_Bandwidth_Utilization,
                'Outbound_Bandwidth_Utilization': r.Outbound_Bandwidth_Utilization
            })
        data_df = pd.DataFrame(data)
        predictions = randomforestmodel.predict(data_df)
        log.info("Prediction complete")

        result = []
        for p in predictions:
            result.append({
                "Prediction" : int(p),
                "message" : "Anomaly" if p == 1 else "Normal"
            })
        log.info("Returning the results")
        return {'results': result}
    except Exception as e:
        log.error(e)
        raise

@router.post("/v2/predict")
def randomforestprediction(request:list[PredictionRequest]):
    try:
        log.info("Running the Neural Network Prediction")
        data = []
        for r in request:
            data.append({
                'Inbound_Rate' : r.Inbound_Rate,
                'Outbound_Rate': r.Outbound_Rate,
                'Inbound_Bandwidth_Utilization': r.Inbound_Bandwidth_Utilization,
                'Outbound_Bandwidth_Utilization': r.Outbound_Bandwidth_Utilization
            })
        data_df = pd.DataFrame(data)
        log.info("Scaling the Data")
        scaled_data = scaler.transform(data_df)
        log.info("Converting it to Torch Tensor")
        data_tensor = torch.tensor(scaled_data, dtype=torch.float32)
        log.info("Starting the prediction")
        with torch.no_grad():
            probabilities = neuralnetworkmodel(data_tensor)
            predictions = ( probabilities >= 0.5 ).float().flatten()
        log.info("Prediction completed")
        result = []
        for p in predictions:
            prediction_value = int(
                p.item()
            )
            result.append({
                "Prediction" : int(prediction_value),
                "message" : "Anomaly" if int(prediction_value) == 1 else "Normal"
            })
        log.info("Returning the result")
        return {'results': result}
    except Exception as e:
        log.error(e)
        raise






