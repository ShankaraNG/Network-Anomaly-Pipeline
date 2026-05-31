import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from ml_build.services.validation import data_validation
from ml_build.services.augmentation import dataaugmentation
from ml_build.services.pipelinebuilder import building_pipeline
from ml_build.services.training import train_model
from ml_build.services.testing import test_model
from ml_build.services.preprocessing import standardizationfornetural
from ml_build.services.trainingnn import trainingneuralnets
from ml_build.services.testingnn import testneuralnets
from ml_build.config_loader import load_config
from ml_build.logger import get_logger
import os

base_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

log = get_logger("PipelineRunner")
def pipelinerunner():
    try:
        log.info("Starting the Pipeline")
        log.info("Loading the configurations")
        conf = load_config()
        data_path = conf['dataPath']['path']
        data_file = conf['dataPath']['file']
        fulldata_path = os.path.join(base_dir, data_path, data_file)
        if not os.path.exists(fulldata_path):
            raise FileNotFoundError("Data file not found")
        log.info("Fetched the Data path")
        log.info("Reading the Data file")
        data_df = pd.read_csv(fulldata_path)
        if data_df.empty:
            raise ValueError("Data file is empty.")
        columns_list = conf['features']
        if columns_list is None:
            raise ValueError("Columns list is not defined in the configuration")
        log.info("Data file has been read successfully")
        log.info("Performing data validations on the data file")
        cleaned_data_df = data_validation(data_df, conf['features'])
        if cleaned_data_df.empty or cleaned_data_df is None:
            raise ValueError("Data validation failed. Cleaned data is empty or None")
        log.info("Data Validation completed")
        log.info("Getting the prediction columns")
        columnsofprediction = conf['prediction']['column']
        if columnsofprediction is None:
            raise ValueError("Prediction column is not defined in the configuration")
        log.info("Prediction column fetched successfully")
        log.info("Getting the Augmentation factor and Noise factor")
        augmentation_factor = conf['augmentation']['factor']
        noise_factor = conf['augmentation']['noise_level']
        if augmentation_factor is None or noise_factor is None:
            raise ValueError("augmentation_factor or noise_factor is missing in the configuration file")
        log.info("Spliting the data set into train and test")
        x = cleaned_data_df.drop(columnsofprediction, axis=1)
        y = cleaned_data_df[columnsofprediction]
        x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.35, random_state=42, stratify=y)
        log.info("Performing Data augmentatino on the training data")
        x_train_final, y_train_final = dataaugmentation(x_train, y_train, augmentation_factor, noise_factor)
        log.info("Data Augmentation completed")
        log.info("Building the pipeline")
        pipeline = building_pipeline()
        if pipeline is None:
            raise ValueError("Pipeline building failed. Pipeline is None")
        log.info("Pipeline build has been completed")
        log.info("Training the model and the pipeline with the data")
        training_result = train_model(x_train_final, y_train_final, x_test, y_test, pipeline)
        if training_result is None or training_result != "successfull":
            raise ValueError("Model training failed")
        log.info("Model training has been completed")
        log.info("Testing the model")
        testing_result = test_model()
        if testing_result is None or testing_result != "successfull":
            raise ValueError("Model testing failed")
        log.info("Model Testing has been completed")
        log.info("Starting Neural Network Pipeline")
        log.info("Starting to standardize the data")
        X_train_tensor, y_train_tensor, X_test_tensor, y_test_tensor = standardizationfornetural(x_train_final, y_train_final, x_test, y_test)
        if X_train_tensor is None or y_train_tensor is None or X_test_tensor is None or y_test_tensor is None:
            raise ValueError("Standardization failed for Neural networks")
        log.info("Standardization of the data completed")
        log.info("Starting the train the neural network model")
        trainingnnresults = trainingneuralnets(X_train_tensor, y_train_tensor, X_test_tensor, y_test_tensor)
        if trainingnnresults is None or trainingnnresults != "successfull":
            raise ValueError("Neural Network training failed")
        log.info("Neural network Training has been completed")
        log.info("Starting to test the Neural Network Model")
        testing_neural_networks_result = testneuralnets()
        if testing_neural_networks_result is None or testing_neural_networks_result != "successfull":
            raise ValueError("Neural Network Testing failed")
        log.info("Neural Network Model testing has been completed")
        log.info("Neural Network Pipeline run Completed")
        log.info("Pipeline run has been completed")
        return "successfull"
    except Exception as e:
        log.error(f"Error during pipeline execution: {e}")
        return None
    

    