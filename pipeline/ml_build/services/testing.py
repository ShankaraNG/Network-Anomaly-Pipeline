import joblib
import os
import pandas as pd
import numpy as np
from ml_build.config_loader import load_config
from ml_build.logger import get_logger

log = get_logger("Testing")

BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

def test_model():
    log.info("Starting model testing")
    try:
        conf = load_config()
        # data_df = pd.DataFrame({'Inbound_Rate': [0.95, 1.08, 1.15, 1.22, 1.30, 1.55, 1.70, 0.70, 1.45, 1.62],
        #                                 'Outbound_Rate': [1.22, 1.34, 1.40, 1.38, 1.45, 1.90, 2.10, 2.00, 1.82, 2.05],
        #                                 'Inbound_Bandwidth_Utilization': [ 0.95, 1.08, 1.15, 1.21, 1.29, 1.56, 1.72, 0.72, 1.47, 1.63],
        #                                 'Outbound_Bandwidth_Utilization': [ 1.22, 1.35, 1.39, 1.37, 1.46, 1.91, 2.08, 2.05, 1.84, 2.07],
        #                                 'Expected_Label': [0, 0, 0, 0, 0, 1, 1, 1, 1, 1]
        #                                 })
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
        log.info(f"Test dataset created with {len(data_df)} samples")
        model_name = conf['model']['name']
        model_path = conf['model']['save_path']
        if model_name is None or model_path is None:
            raise ValueError("Model name or model path is not defined in the configuration.")
        model_file = os.path.join(BASE_DIR, model_path, f"{model_name}.pkl")
        log.info(f"Looking for model file at: {model_file}")
        if not os.path.exists(model_file):
            raise FileNotFoundError("Model file not found.")
        pipeline = joblib.load(model_file)
        log.info("Model loaded successfully")
        sampletest = data_df.drop('Expected_Label', axis=1)
        expected_labels = data_df['Expected_Label']
        predictions = pipeline.predict(sampletest)
        log.info("Predictions generated successfully")
        data_df['Expected_Label'] = expected_labels
        data_df['Predicted_Label'] = predictions
        artifacts_path = conf['artifacts']['save_path']
        if artifacts_path is None:
            raise ValueError("Artifacts path is not defined in the configuration.")
        testcasespath = os.path.join(BASE_DIR, artifacts_path, 'test_cases.csv')
        data_df['Correct_Prediction'] = (data_df['Expected_Label'] == data_df['Predicted_Label'])
        data_df.to_csv(testcasespath, index=False)
        log.info(f"Test cases saved to: {testcasespath}")
        testingaccuracypath = os.path.join(BASE_DIR, artifacts_path, 'testing_accuracy.txt')
        testing_accuracy = (data_df['Correct_Prediction'].sum() / len(data_df)) * 100
        with open(testingaccuracypath, 'w') as f:
            f.write(f"Testing Accuracy: {testing_accuracy:.2f}%")
        log.info(f"Testing accuracy: {testing_accuracy:.2f}% saved to: {testingaccuracypath}")
        return "successfull"
    except Exception as e:
        log.error(f"Error during model testing: {e}")
        return None