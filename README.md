# Customer Churn Prediction - MLOps Pipeline

End-to-end machine learning pipeline for predicting customer churn, built with production-grade MLOps practices. This project demonstrates the complete ML lifecycle from data ingestion to model deployment with monitoring.

## Overview

Built this project to showcase how I approach ML systems in production. The pipeline predicts customer churn for a telecom dataset using XGBoost, with the entire workflow containerized and deployed on Kubernetes.

**What's included:**
- Automated data validation and feature engineering
- - Model training with experiment tracking (MLflow)
  - - REST API for real-time predictions (FastAPI)
    - - Docker containerization with multi-stage builds
      - - Kubernetes deployment with auto-scaling
        - - CI/CD pipelines (GitHub Actions)
          - - Monitoring with Prometheus & Grafana
           
            - ## Architecture
           
            - ```
              ┌─────────────┐     ┌─────────────┐     ┌─────────────┐
              │   Raw Data  │────▶│  Feature    │────▶│   Model     │
              │  Ingestion  │     │  Pipeline   │     │  Training   │
              └─────────────┘     └─────────────┘     └──────┬──────┘
                                                             │
                                  ┌──────────────────────────┘
                                  ▼
              ┌─────────────┐     ┌─────────────┐     ┌─────────────┐
              │   MLflow    │◀────│   Model     │────▶│  FastAPI    │
              │  Registry   │     │  Artifacts  │     │   Service   │
              └─────────────┘     └─────────────┘     └──────┬──────┘
                                                             │
                                  ┌──────────────────────────┘
                                  ▼
              ┌─────────────────────────────────────────────────────┐
              │              Kubernetes Cluster                      │
              │  ┌─────────┐  ┌─────────┐  ┌─────────┐            │
              │  │   API   │  │Prometheus│  │ Grafana │            │
              │  │  Pods   │  │          │  │         │            │
              │  └─────────┘  └─────────┘  └─────────┘            │
              └─────────────────────────────────────────────────────┘
              ```

              ## Tech Stack

              | Component | Technology |
              |-----------|------------|
              | ML Framework | XGBoost, scikit-learn |
              | Experiment Tracking | MLflow |
              | API | FastAPI, Uvicorn |
              | Containerization | Docker |
              | Orchestration | Kubernetes, Helm |
              | CI/CD | GitHub Actions |
              | Monitoring | Prometheus, Grafana |
              | Data Validation | Great Expectations |

              ## Project Structure

              ```
              ├── src/
              │   ├── data/
              │   │   ├── ingestion.py
              │   │   ├── validation.py
              │   │   └── preprocessing.py
              │   ├── features/
              │   │   ├── build_features.py
              │   │   └── feature_store.py
              │   ├── models/
              │   │   ├── train.py
              │   │   ├── evaluate.py
              │   │   └── predict.py
              │   └── api/
              │       ├── main.py
              │       ├── routes.py
              │       └── schemas.py
              ├── kubernetes/
              │   ├── deployment.yaml
              │   ├── service.yaml
              │   ├── hpa.yaml
              │   └── configmap.yaml
              ├── docker/
              │   ├── Dockerfile
              │   └── docker-compose.yml
              ├── monitoring/
              │   ├── prometheus/
              │   └── grafana/
              ├── tests/
              ├── notebooks/
              ├── .github/workflows/
              ├── configs/
              ├── requirements.txt
              └── Makefile
              ```

              ## Getting Started

              ### Prerequisites

              - Python 3.10+
              - - Docker & Docker Compose
                - - kubectl (for K8s deployment)
                  - - Make
                   
                    - ### Local Setup
                   
                    - ```bash
                      # Clone the repo
                      git clone https://github.com/GoparapukethaN/mlops-end-to-end-pipeline.git
                      cd mlops-end-to-end-pipeline

                      # Create virtual environment
                      python -m venv venv
                      source venv/bin/activate

                      # Install dependencies
                      pip install -r requirements.txt

                      # Run data pipeline
                      python -m src.data.ingestion
                      python -m src.features.build_features

                      # Train model
                      python -m src.models.train

                      # Start API locally
                      uvicorn src.api.main:app --reload --port 8000
                      ```

                      ### Docker Deployment

                      ```bash
                      # Build and run all services
                      docker-compose -f docker/docker-compose.yml up -d

                      # Services available at:
                      # API: http://localhost:8000
                      # MLflow: http://localhost:5000
                      # Prometheus: http://localhost:9090
                      # Grafana: http://localhost:3000
                      ```

                      ### Kubernetes Deployment

                      ```bash
                      # Apply manifests
                      kubectl apply -f kubernetes/

                      # Check pods
                      kubectl get pods -n mlops

                      # Port forward to test
                      kubectl port-forward svc/churn-api 8000:80 -n mlops
                      ```

                      ## API Usage

                      ### Health Check
                      ```bash
                      curl http://localhost:8000/health
                      ```

                      ### Prediction
                      ```bash
                      curl -X POST http://localhost:8000/predict \
                        -H "Content-Type: application/json" \
                        -d '{
                          "tenure": 24,
                          "monthly_charges": 65.5,
                          "total_charges": 1572.0,
                          "contract": "Month-to-month",
                          "payment_method": "Electronic check"
                        }'
                      ```

                      ### Response
                      ```json
                      {
                        "prediction": "churn",
                        "probability": 0.73,
                        "model_version": "1.0.2"
                      }
                      ```

                      ## Model Performance

                      | Metric | Value |
                      |--------|-------|
                      | Accuracy | 0.81 |
                      | Precision | 0.79 |
                      | Recall | 0.76 |
                      | F1 Score | 0.77 |
                      | AUC-ROC | 0.84 |

                      ## Monitoring

                      Grafana dashboards track:
                      - API latency (p50, p95, p99)
                      - - Request throughput
                        - - Model prediction distribution
                          - - Data drift metrics
                            - - Resource utilization
                             
                              - ## CI/CD Pipeline
                             
                              - The GitHub Actions workflow handles:
                              - 1. Code linting (flake8, black)
                                2. 2. Unit tests (pytest)
                                   3. 3. Docker image build
                                      4. 4. Push to container registry
                                         5. 5. Deploy to K8s cluster
                                           
                                            6. ## Future Improvements
                                           
                                            7. - [ ] Add A/B testing capability
                                               - [ ] - [ ] Implement feature store with Feast
                                               - [ ] - [ ] Add model retraining triggers
                                               - [ ] - [ ] Set up Slack alerts for drift detection
                                              
                                               - [ ] ## License
                                              
                                               - [ ] MIT
