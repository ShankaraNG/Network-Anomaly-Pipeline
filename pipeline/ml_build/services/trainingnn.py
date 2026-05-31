from ml_build.services.modelling import NetworkAnomalyNN
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report,f1_score,recall_score,accuracy_score,precision_score,ConfusionMatrixDisplay
from ml_build.config_loader import load_config
from ml_build.logger import get_logger
import os

BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

log = get_logger("Training Neural Network")

def trainingneuralnets(X_train_tensor, y_train_tensor, X_test_tensor, y_test_tensor):
    try:
        conf = load_config()
        log.info("Loading the Learning Rate")
        learningrate = conf['nnconfig']['lr']
        if learningrate is None:
            raise ValueError("Learning Rate not found for the neural network in config")
        log.info("Loading the model")
        model = NetworkAnomalyNN()
        log.info("Model Loading has been completed")
        log.info("Creating the loss criteria")
        criterion = nn.BCELoss()
        log.info("Loss criteria has been created as BCELoss")
        log.info("Fitting the Optimizer and the learning rate")
        optimizer = optim.Adam(model.parameters(), lr=learningrate)
        log.info("Optimizer initialized")
        log.info("Loading the Epoch")
        epochs = conf['nnconfig']['epochs']
        if epochs is None:
            raise ValueError("Epochs is not present in the configuration")
        log.info(f"Epoch rate has been loaded as {epochs}")
        artifacts_path = conf['artifacts']['save_path']
        if artifacts_path is None:
            raise ValueError("The artifact path is not present in the configuration")
        loss_epoch_path = os.path.join(BASE_DIR, artifacts_path, "neural_network_loss_epoch.txt")
        if not os.path.exists(os.path.dirname(os.path.join(BASE_DIR, artifacts_path))):
            os.makedirs(os.path.join(BASE_DIR, modelpath))
        epochsarray = []
        lossarray = []
        log.info("Starting the Epochs Trainings")
        for epoch in range(epochs):
            outputs = model(X_train_tensor)
            loss = criterion(outputs, y_train_tensor)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            if (epoch + 1) % 10 == 0:
                with open(loss_epoch_path, 'a') as f:
                    f.write(f"Epoch [{epoch+1}/{epochs}] => Loss: {loss.item():.4f}\n")
            epochsarray.append(epoch + 1)
            lossarray.append(loss.item())
        log.info("Training has been completed")
        log.info("Saving the model")
        modelpath = conf['model']['save_path']
        modelname = conf['model']['namenn']
        model_save_path  = os.path.join(BASE_DIR, modelpath, f"{modelname}.pth")
        torch.save( model.state_dict(), model_save_path)
        log.info("Neural Network model has been saved successfully")
        log.info("Creating the Gradient Decent")
        window_size = 15
        gradient_path  = os.path.join(BASE_DIR, artifacts_path, "gradient_decent.png")
        smoothed_loss = np.convolve(lossarray, np.ones(window_size)/window_size, mode='valid')
        smoothed_epochs = epochsarray[window_size - 1:]
        plt.figure(figsize=(8, 6))
        plt.plot(smoothed_epochs, smoothed_loss, label='Smoothed Training Loss', linewidth=2)
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.title('Training Loss Over Epochs')
        plt.legend()
        plt.tight_layout()
        plt.savefig(gradient_path)
        plt.close()
        log.info("Gradient Decent has been saved")
        log.info("Testing the Neural Network Model")
        with torch.no_grad():
            predictions = model(X_test_tensor)
            predicted_classes = (predictions >= 0.5).float()
        log.info("Testing has been completed")
        log.info("Generating the Classification Report")
        classification_report_path = os.path.join(BASE_DIR, artifacts_path, 'Neuralnetwork_classification_report.txt')
        classificationreport = classification_report(y_test_tensor,predicted_classes)
        with open(classification_report_path, 'w') as f:
            f.write(classificationreport)
        log.info("Classification report for the neural network model has been generated successfully")
        log.info("Calculating the recall score")
        recallscore = recall_score(y_test_tensor, predicted_classes)
        recallscore_path = os.path.join(BASE_DIR, artifacts_path, 'Neuralnetwork_recall_score.txt')
        with open(recallscore_path, 'w') as f:
            f.write(f"Recall Score: {recallscore:.4f}")
        log.info(f"Recall Score has been calculated successfully {recallscore}")
        log.info("Calculating the accuracy score")
        accuracyscore = accuracy_score(y_test_tensor, predicted_classes)
        accuracyscore_path = os.path.join(BASE_DIR, artifacts_path, 'Neuralnetwork_accuracy_score.txt')
        with open(accuracyscore_path, 'w') as f:
            f.write(f"Accuracy Score: {accuracyscore:.4f}")
        log.info(f"Accuracy Score has been calculated successfully {accuracyscore}")
        log.info("Calculating the precision score")
        precisionscore = precision_score(y_test_tensor, predicted_classes)
        precisionscore_path = os.path.join(BASE_DIR, artifacts_path, 'Neuralnetwork_precision_score.txt')
        with open(precisionscore_path, 'w') as f:
            f.write(f"Precision Score: {precisionscore:.4f}")
        log.info(f"Precision Score has been calculated successfully {precisionscore}")
        log.info("Calculating the f1 score")
        f1score = f1_score(y_test_tensor, predicted_classes)
        f1score_path = os.path.join(BASE_DIR, artifacts_path, 'Neuralnetwork_f1_score.txt')
        with open(f1score_path, 'w') as f:
            f.write(f"F1 Score: {f1score:.4f}")
        log.info(f"F1 Score has been calculated successfully {f1score}")
        log.info("Generating the Confusion matrix for the Neural Network")
        confusionmatrix_path = os.path.join(BASE_DIR, artifacts_path, 'Neuralnetwork_confusion_matrix.png')
        fig, ax = plt.subplots(figsize=(8, 6))
        confusion_matrix_display = ConfusionMatrixDisplay.from_predictions(
            y_test_tensor,
            predicted_classes,
            cmap=plt.cm.Blues,
            ax=ax
        )
        plt.title("Neural Network Confusion Matrix")
        plt.savefig(confusionmatrix_path)
        plt.close()
        log.info("Confusion matrix Genrated and Saved successfully")
        return "successfull"
    except Exception as e:
        log.error(f"Training of the Neural Network failed with the error: {e}")
        return None

