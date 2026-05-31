import joblib
from sklearn.metrics import classification_report, recall_score, precision_score, f1_score, ConfusionMatrixDisplay, accuracy_score
import matplotlib.pyplot as plt
from ml_build.config_loader import load_config
from ml_build.logger import get_logger
import os

BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

log = get_logger("Training")

def train_model(X_train_final, y_train_final, X_test, y_test, pipeline):
    try:
        conf = load_config()
        log.info("Training the model")
        log.info("Fitting the data into the model")
        pipeline.fit(X_train_final, y_train_final)
        log.info("Getting the best model")
        bestmodel = pipeline.best_estimator_
        log.info("Testing the best model")
        y_pred = bestmodel.predict(X_test)
        log.info("Testing has been completed")
        log.info("Retrieveing the path to save the model")
        modelpath = conf['model']['save_path']
        modelname = conf['model']['name']
        if modelpath is None or modelname is None:
            raise ValueError("Model path or model name is not defined in the configuration.")
        modelfullpath = os.path.join(BASE_DIR, modelpath, f'{modelname}.pkl')
        if not os.path.exists(os.path.dirname(os.path.join(BASE_DIR, modelpath))):
            os.makedirs(os.path.join(BASE_DIR, modelpath))
        log.info("Model path exists and proceeding to save the model")
        joblib.dump(pipeline, modelfullpath)
        log.info("Model has been successfully saved")
        artifacts_path = conf['artifacts']['save_path']
        if artifacts_path is None:
            raise ValueError("Artifacts path is not defined in the configuration.")
        if not os.path.exists(os.path.join(BASE_DIR, artifacts_path)):
            os.makedirs(os.path.join(BASE_DIR, artifacts_path))
        accuracy_path = os.path.join(BASE_DIR, artifacts_path, 'accuracy.txt')
        log.info("Calculating the accuracy")
        accuracy = accuracy_score(y_test, y_pred)
        with open(accuracy_path, 'w') as f:
            f.write(f"Best Model Accuracy: {accuracy:.4f}")
        log.info(f"Accuracy has been saved to the file in the path {accuracy_path}")
        log.info("Generating the classification report")
        report = classification_report(y_test, y_pred)
        report_path = os.path.join(BASE_DIR, artifacts_path, 'classification_report.txt')
        with open(report_path, 'w') as f:
            f.write(report)
        log.info(f"Classification report has been saved in the path {report_path}")
        log.info("Calculating the recall score")
        recall = recall_score(y_test, y_pred)
        recall_path = os.path.join(BASE_DIR, artifacts_path, 'recall.txt')
        with open(recall_path, 'w') as f:
            f.write(f"Recall Score: {recall:.4f}")
        log.info(f"Recall score has been saved in the path {recall_path}")
        log.info("Calculating the precision score")
        precision = precision_score(y_test, y_pred)
        precision_path = os.path.join(BASE_DIR, artifacts_path, 'precision.txt')
        with open(precision_path, 'w') as f:
            f.write(f"Precision Score: {precision:.4f}")
        log.info(f"Precision score has been saved in the path {precision_path}")
        log.info("Calculating the F1 score")
        f1 = f1_score(y_test, y_pred)
        f1_path = os.path.join(BASE_DIR, artifacts_path, 'f1.txt')
        with open(f1_path, 'w') as f:
            f.write(f"F1 Score: {f1:.4f}")
        log.info(f"F1 score has been saved in the path {f1_path}")
        log.info("Generating confusion matrix")
        confusionmatrix_path = os.path.join(BASE_DIR, artifacts_path, 'confusion_matrix.png')
        fig, ax = plt.subplots(figsize=(8, 6))
        confusion_matrix_display = ConfusionMatrixDisplay.from_predictions(y_test, y_pred, cmap=plt.cm.Blues,ax=ax)
        plt.title("Random Forest Confusion Matrix")
        plt.savefig(confusionmatrix_path)
        plt.close()
        log.info(f"Confusion matrix has been saved in the path {confusionmatrix_path}")
        return "successfull"
    except Exception as e:
        log.error(f"Error during model training: {e}")
        return None