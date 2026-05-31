import pandas as pd
import numpy as np
import os
from ml_build.config_loader import load_config
from ml_build.logger import get_logger

log = get_logger("validaation")

BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

def data_validation(data_df, columns_list):
    try:
        log.info("Checking for the presence of the required columns in the dataset")
        if columns_list is None:
            raise ValueError("Columns list is not defined in the configuration.")
        if not all(col in data_df.columns for col in columns_list):
            missing_cols = [col for col in columns_list if col not in data_df.columns]
            raise ValueError(f"Missing columns in the dataset: {missing_cols}")
        log.info("All the columns are present in the dataset")
        log.info("Renaming the columns in the dataset")
        dictofcolumns = {
            columns_list[0]: 'Inbound_Rate',
            columns_list[1]: 'Outbound_Rate',
            columns_list[2]: 'Inbound_Bandwidth_Utilization',
            columns_list[3]: 'Outbound_Bandwidth_Utilization'
        }
        data_df.rename(columns=dictofcolumns, inplace=True)
        log.info("Columns have been renamed successfully")
        return data_df
    except Exception as e:
        log.error(f"Error during data validation: {e}")
        return None
