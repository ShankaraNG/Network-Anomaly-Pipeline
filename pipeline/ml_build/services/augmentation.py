import pandas as pd
import numpy as np
from ml_build.config_loader import load_config
from ml_build.logger import get_logger

log = get_logger("Data Augmentation")


def dataaugmentation(x_train, y_train, augmentation_factor, noise_factor):
    try:
        X_augmented_list = []
        y_augmented_list = []
        log.info("Starting the data augmentation process")

        for i in range(augmentation_factor):
            noise = np.random.normal(
                loc=0,
                scale=x_train.std() * noise_factor,
                size=x_train.shape
            )
            X_synthetic = x_train + noise
            X_augmented_list.append(X_synthetic)
            y_augmented_list.append(y_train.copy())
        
        log.info("Data augmentation process has beem completed")
        log.info("Merging the augmented data with the original training data")

        X_augmented = pd.concat(X_augmented_list, ignore_index=True)
        y_augmented = pd.concat(y_augmented_list, ignore_index=True)

        x_train_final = pd.concat([x_train, X_augmented],ignore_index=True)
        y_train_final = pd.concat([y_train, y_augmented],ignore_index=True)
        log.info("Merging completed successfully")
        return x_train_final, y_train_final
    except Exception as e:
        log.error(f"Error during data augmentation: {e}")
        return None, None


