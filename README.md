# 🚀 MLOps End-to-End Pipeline: Customer Churn Prediction

[![CI/CD Pipeline](https://github.com/GoparapukethaN/mlops-end-to-end-pipeline/actions/workflows/CI-CD.yaml/badge.svg)](https://github.com/GoparapukethaN/mlops-end-to-end-pipeline/actions/workflows/CI-CD.yaml)
[![Python 3.10](https://img.shields.io/badge/python-3.10-blue.svg)](https://www.python.org/downloads/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.100+-green.svg)](https://fastapi.tiangolo.com/)
[![Docker](https://img.shields.io/badge/Docker-Ready-blue.svg)](https://www.docker.com/)

A production-ready MLOps pipeline demonstrating end-to-end machine learning workflow including data ingestion, model training with experiment tracking, REST API deployment, containerization, and CI/CD automation.

## 📊 Project Overview

This project predicts customer churn for a telecommunications company using the IBM Telco Customer Churn dataset. It demonstrates industry best practices for deploying ML models in production.

### 🎯 Model Performance

| Metric | Score |
|--------|-------|
| Train Accuracy | 84.5% |
| Test Accuracy | 80.1% |
| AUC-ROC | 0.84 |
| Precision | 0.65 |
| Recall | 0.54 |
| F1-Score | 0.59 |

## 🏗️ Architecture
```
┌─────────────────┐     ┌─────────────────┐     ┌─────────────────┐
│   Data Source   │────▶│  Data Pipeline  │────▶│  Model Training │
│  (IBM Telco)    │     │  (ingestion.py) │     │   (train.py)    │
└─────────────────┘     └─────────────────┘     └────────┬────────┘
                                                         │
                                                         ▼
┌─────────────────┐     ┌─────────────────┐     ┌─────────────────┐
│   Prometheus    │◀────│    FastAPI      │◀────│    MLflow       │
│   Monitoring    │     │   (main.py)     │     │   Tracking      │
└─────────────────┘     └─────────────────┘     └─────────────────┘
                                │
                                ▼
                        ┌─────────────────┐
                        │  Docker/K8s     │
                        │  Deployment     │
                        └─────────────────┘
```

## ✨ Features

- **Data Pipeline**: Automated data ingestion and preprocessing
- **ML Training**: XGBoost classifier with hyperparameter tuning
- **Experiment Tracking**: MLflow for model versioning and metrics logging
- **REST API**: FastAPI with automatic Swagger documentation
- **Monitoring**: Prometheus metrics for predictions and latency
- **Containerization**: Docker-ready with multi-stage builds
- **Orchestration**: Kubernetes manifests for scalable deployment
- **CI/CD**: GitHub Actions for automated testing and building

## 📸 Screenshots

### CI/CD Pipeline - All Checks Passing ✅
![CI/CD Pipeline](<img width="1280" height="593" alt="Screenshot 2026-01-15 at 1 36 23 PM" src="https://github.com/user-attachments/assets/a759f4e7-5233-4b8d-9dd0-f35bf54f9e4d" />


### API Documentation (Swagger UI)
![Swagger UI](screenshots/swagger-docs.png)

### Prediction Response
![Prediction] <img width="1080" height="2724" alt="modified_image (10)" src="https://github.com/user-attachments/assets/800a1deb-4792-43d9-9d5f-5e23e914fdf7" />


## 🚀 Quick Start

### Prerequisites
- Python 3.10+
- Docker (optional)

### Local Development
```bash
# Clone the repository
git clone https://github.com/GoparapukethaN/mlops-end-to-end-pipeline.git
cd mlops-end-to-end-pipeline

# Install dependencies
pip install -r requirements.txt

# Run data ingestion
python -m src.data.ingestion

# Train the model
python -m src.models.train

# Start the API
python -m uvicorn src.api.main:app --host 0.0.0.0 --port 8000
```

### Docker Deployment
```bash
# Build the image
docker build -t churn-prediction:latest -f docker/Dockerfile .

# Run the container
docker run -p 8000:8000 churn-prediction:latest
```

## 📡 API Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/` | GET | Service status |
| `/health` | GET | Health check |
| `/predict` | POST | Get churn prediction |
| `/metrics` | GET | Prometheus metrics |
| `/docs` | GET | Swagger documentation |

### Sample Prediction Request
```bash
curl -X POST "http://localhost:8000/predict" \
  -H "Content-Type: application/json" \
  -d '{
    "tenure": 12,
    "MonthlyCharges": 70.5,
    "TotalCharges": 846.0,
    "Contract": "Month-to-month",
    "PaymentMethod": "Electronic check",
    "gender": "Female",
    "SeniorCitizen": 0,
    "Partner": "Yes",
    "Dependents": "No",
    "PhoneService": "Yes",
    "MultipleLines": "No",
    "InternetService": "Fiber optic",
    "OnlineSecurity": "No",
    "OnlineBackup": "No",
    "DeviceProtection": "No",
    "TechSupport": "No",
    "StreamingTV": "Yes",
    "StreamingMovies": "Yes",
    "PaperlessBilling": "Yes"
  }'
```

### Sample Response
```json
{
  "churn_probability": 0.72,
  "churn_prediction": "Yes",
  "risk_level": "High"
}
```

## 🛠️ Tech Stack

| Category | Technologies |
|----------|-------------|
| **ML Framework** | XGBoost, Scikit-learn |
| **Experiment Tracking** | MLflow |
| **API Framework** | FastAPI, Uvicorn |
| **Monitoring** | Prometheus |
| **Containerization** | Docker |
| **Orchestration** | Kubernetes |
| **CI/CD** | GitHub Actions |
| **Language** | Python 3.10 |

## 📁 Project Structure
```
mlops-end-to-end-pipeline/
├── .github/workflows/     # CI/CD pipelines
│   └── CI-CD.yaml
├── configs/               # Configuration files
│   └── config.yaml
├── data/
│   ├── raw/              # Raw downloaded data
│   └── processed/        # Train/test splits
├── docker/               # Docker configurations
│   └── Dockerfile
├── kubernetes/           # K8s manifests
│   ├── deployment.yaml
│   └── service.yaml
├── mlruns/               # MLflow experiment logs
├── models/               # Saved model artifacts
│   └── churn_model.joblib
├── src/
│   ├── api/              # FastAPI application
│   │   └── main.py
│   ├── data/             # Data processing
│   │   └── ingestion.py
│   └── models/           # Model training
│       └── train.py
├── tests/                # Unit tests
├── requirements.txt
└── README.md
```

## 📈 MLflow Experiment Tracking

The project uses MLflow to track:
- Model parameters (n_estimators, max_depth, learning_rate)
- Metrics (accuracy, precision, recall, F1, AUC)
- Model artifacts (saved model files)

## 🔄 CI/CD Pipeline

The GitHub Actions workflow includes:

1. **Test Job**
   - Code checkout
   - Python setup
   - Dependency installation
   - Linting with Flake8
   - Unit tests with Pytest

2. **Build Job**
   - Docker image build
   - Image verification

## 👤 Author

**Kethan Goparapuketha**
- GitHub: [@GoparapukethaN](https://github.com/GoparapukethaN)
- LinkedIn: [Connect with me](https://www.linkedin.com/in/kethan-goparapu/)

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---

⭐ **If you found this project helpful, please give it a star!**
