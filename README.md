# Network Anomaly Detection API

A production-ready Machine Learning API for real-time network anomaly detection.
---
## Overview

This project provides a **dual-model inference API** for detecting network anomalies in real time. It exposes two prediction endpoints — one powered by a **Random Forest classifier** and another by a **PyTorch Neural Network** wrapped in a clean FastAPI application with Docker and Poetry support.

---

## Features

- Real-time and batch network anomaly detection
- Two independent prediction endpoints (Random Forest & Neural Network)
- FastAPI REST API with automatic Swagger/OpenAPI documentation
- ML build pipeline with preprocessing, training, validation, and testing
- Dockerized for consistent, portable deployment
- Dependency management via Poetry

---

# Live Deployment
The API is deployed and publicly accessible on Hugging Face Spaces:
EndpointURLBase URLhttps://shankarang-network-anomaly-pipeline.hf.space
Health CheckGET /health
Random Forest PredictPOST /v1/predict
Neural Network PredictPOST /v2/predict
Swagger Docs/docs

## Tech Stack

| Category | Tools |
|---|---|
| Language | Python 3.11 |
| API Framework | FastAPI, Uvicorn |
| ML / DL | Scikit-learn, PyTorch |
| Data Processing | Pandas |
| Packaging | Poetry |
| Containerization | Docker |

---

## Project Structure

```
pipeline/
├── app/                          # FastAPI application
│   ├── routes/
│   │   └── predictor.py          # API route definitions
│   ├── services/
│   │   ├── neural_network.py     # Neural network inference service
│   │   └── random_forest_model.py# Random forest inference service
│   ├── config_loader.py          # App configuration loader
│   ├── logger.py                 # Application logger
│   ├── main.py                   # FastAPI app entry point
│   └── schemas.py                # Pydantic request/response schemas
│
├── ml_build/                     # ML training pipeline
│   ├── services/
│   │   ├── augmentation.py       # Data augmentation
│   │   ├── modelling.py          # Model definitions
│   │   ├── pipelinebuilder.py    # Pipeline construction
│   │   ├── pipelinerunner.py     # Pipeline execution
│   │   ├── preprocessing.py      # Data preprocessing
│   │   ├── testing.py            # RF model testing
│   │   ├── testingnn.py          # Neural network testing
│   │   ├── training.py           # RF model training
│   │   ├── trainingnn.py         # Neural network training
│   │   └── validation.py         # Model validation
│   ├── config_loader.py
│   ├── logger.py
│   └── main.py                   # ML pipeline entry point
│
├── artifacts/                    # Model evaluation outputs
│   ├── accuracy.txt
│   ├── classification_report.txt
│   ├── confusion_matrix.png
│   ├── f1.txt
│   ├── gradient_decent.png
│   ├── neural_network_loss_epoch.txt
│   ├── neural_network_testing_accuracy.txt
│   ├── Neuralnetwork_accuracy_score.txt
│   ├── Neuralnetwork_classification_report.txt
│   ├── Neuralnetwork_confusion_matrix.png
│   ├── Neuralnetwork_f1_score.txt
│   ├── Neuralnetwork_precision_score.txt
│   ├── Neuralnetwork_recall_score.txt
│   ├── precision.txt
│   ├── recall.txt
│   ├── test_cases.csv
│   └── testing_accuracy.txt
│
├── config/
│   └── config.yaml               # Central configuration file
│
├── data/
│   └── networkanomalydataset.csv # Training dataset
│
├── logs/
│   ├── application.log
│   └── training.log
│
├── models/                       # Saved model files
│   ├── network_anomaly_model.pkl # Trained Random Forest model
│   ├── network_anomaly_nn_model.pth # Trained Neural Network model
│   └── scaler.pkl                # Fitted feature scaler
│
├── Dockerfile
├── pyproject.toml
├── poetry.lock
└── requirements.txt
```

---

## API Endpoints

### `GET /`

Health check / welcome endpoint.

**Response:**
```json
{
  "message": "Welcome to the Network Anomaly Detection API"
}
```

---

### `POST /v1/predict` — Random Forest

Runs inference using the **Random Forest Classifier**.

---

### `POST /v2/predict` — Neural Network

Runs inference using the **PyTorch Neural Network**.

---

### Request Body

Both endpoints accept a batch of network traffic records:

```json
[
  {
    "Inbound_Rate": -0.82,
    "Outbound_Rate": -0.83,
    "Inbound_Bandwidth_Utilization": -0.81,
    "Outbound_Bandwidth_Utilization": -0.84
  },
  {
    "Inbound_Rate": -0.56,
    "Outbound_Rate": -0.70,
    "Inbound_Bandwidth_Utilization": -0.55,
    "Outbound_Bandwidth_Utilization": -0.70
  }
]
```

### Response

```json
{
  "results": [
    {
      "Prediction": 1,
      "message": "Anomaly"
    },
    {
      "Prediction": 0,
      "message": "Normal"
    }
  ]
}
```

---

## Local Development

### Prerequisites

- Python 3.11+
- [Poetry](https://python-poetry.org/docs/#installation)
- Docker (optional, for containerised deployment)

### Install Dependencies

```bash
poetry install
```

### Run the ML Build Pipeline

```bash
poetry run python -m ml_build.main
```

This will preprocess the data, train both models, validate, and save artefacts to `artifacts/` and models to `models/`.

### Start the FastAPI Server

```bash
poetry run uvicorn app.main:app --reload
```

### Swagger Documentation

Once the server is running, visit:

```
http://127.0.0.1:8000/docs
```

---

## Docker Deployment

### Build the Image

```bash
docker build -t network-anomaly-api .
```

### Run the Container

```bash
docker run -p 8000:8000 network-anomaly-api
```

---

## Machine Learning Models

### Random Forest

- Ensemble learning with multiple decision trees
- Fast inference, high interpretability
- Serialised with `pickle` → `models/network_anomaly_model.pkl`

### Neural Network (PyTorch)

A fully connected feedforward network with the following architecture:

- Fully Connected Layers
- Batch Normalisation
- ReLU Activation
- Dropout Regularisation
- Sigmoid Output Layer

Serialised as `models/network_anomaly_nn_model.pth`.

---

## Future Improvements

- Kafka integration for real-time stream processing
- Kubernetes deployment manifests
- CI/CD pipeline (GitHub Actions)
- Prometheus / Grafana monitoring
- GPU inference support
- Authentication & authorisation (API keys / OAuth2)

---

## Author

**Shankar Narayana**

---

## License

This project is licensed under the [MIT License](LICENSE).
