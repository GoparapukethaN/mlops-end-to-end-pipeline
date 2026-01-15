"""
Model training module with MLflow experiment tracking.
"""
import os
import logging
import joblib
from pathlib import Path
from typing import Dict, Any, Optional

import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
import xgboost as xgb
import mlflow
import mlflow.xgboost

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class ChurnModel:
      """XGBoost model for churn prediction with MLflow tracking."""

    def __init__(self, model_params: Optional[Dict[str, Any]] = None):
              self.model_params = model_params or {
                            'max_depth': 6,
                            'learning_rate': 0.1,
                            'n_estimators': 100,
                            'objective': 'binary:logistic',
                            'random_state': 42,
                            'n_jobs': -1
              }
              self.model = None
              self.label_encoders = {}
              self.scaler = StandardScaler()
              self.feature_names = None

    def _preprocess_features(self, df: pd.DataFrame, fit: bool = False) -> np.ndarray:
              """Encode categorical and scale numerical features."""
              df = df.copy()

        # Identify column types
              cat_cols = df.select_dtypes(include=['object']).columns.tolist()
              num_cols = df.select_dtypes(include=['int64', 'float64']).columns.tolist()

        # Encode categoricals
              for col in cat_cols:
                            if fit:
                                              le = LabelEncoder()
                                              df[col] = le.fit_transform(df[col].astype(str))
                                              self.label_encoders[col] = le
else:
                le = self.label_encoders.get(col)
                  if le:
                      df[col] = df[col].astype(str).map(
                                                lambda x: le.transform([x])[0] if x in le.classes_ else -1
                      )

                            # Scale numerical
                            if fit:
                                self.feature_names = df.columns.tolist()
                                          return self.scaler.fit_transform(df)
        return self.scaler.transform(df)

    def train(self, X_train: pd.DataFrame, y_train: pd.Series, 
                            X_val: pd.DataFrame, y_val: pd.Series) -> Dict[str, float]:
                                      """Train model and log to MLflow."""

        mlflow.set_experiment("churn-prediction")

        with mlflow.start_run():
                      # Log parameters
                      mlflow.log_params(self.model_params)

            # Preprocess
            X_train_processed = self._preprocess_features(X_train, fit=True)
            X_val_processed = self._preprocess_features(X_val, fit=False)

            # Train
            logger.info("Training XGBoost model...")
            self.model = xgb.XGBClassifier(**self.model_params)
            self.model.fit(
                              X_train_processed, y_train,
                              eval_set=[(X_val_processed, y_val)],
                              verbose=False
            )

            # Evaluate
            y_pred = self.model.predict(X_val_processed)
            y_proba = self.model.predict_proba(X_val_processed)[:, 1]

            metrics = {
                              'accuracy': accuracy_score(y_val, y_pred),
                              'precision': precision_score(y_val, y_pred),
                              'recall': recall_score(y_val, y_pred),
                              'f1': f1_score(y_val, y_pred),
                              'auc_roc': roc_auc_score(y_val, y_proba)
            }

            # Log metrics
            mlflow.log_metrics(metrics)

            # Log model
            mlflow.xgboost.log_model(self.model, "model")

            logger.info(f"Model trained. Metrics: {metrics}")
            return metrics

    def predict(self, X: pd.DataFrame) -> np.ndarray:
              """Make predictions."""
        X_processed = self._preprocess_features(X, fit=False)
        return self.model.predict(X_processed)

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
              """Get prediction probabilities."""
              X_processed = self._preprocess_features(X, fit=False)
              return self.model.predict_proba(X_processed)[:, 1]

    def save(self, path: str = "models/churn_model.joblib"):
              """Save model artifacts."""
              os.makedirs(os.path.dirname(path), exist_ok=True)
              artifacts = {
                  'model': self.model,
                  'label_encoders': self.label_encoders,
                  'scaler': self.scaler,
                  'feature_names': self.feature_names
              }
              joblib.dump(artifacts, path)
              logger.info(f"Model saved to {path}")

    def load(self, path: str = "models/churn_model.joblib"):
              """Load model artifacts."""
              artifacts = joblib.load(path)
              self.model = artifacts['model']
              self.label_encoders = artifacts['label_encoders']
              self.scaler = artifacts['scaler']
              self.feature_names = artifacts['feature_names']
              logger.info(f"Model loaded from {path}")


if __name__ == "__main__":
      # Load data
      X_train = pd.read_csv("data/processed/X_train.csv")
      X_test = pd.read_csv("data/processed/X_test.csv")
      y_train = pd.read_csv("data/processed/y_train.csv").squeeze()
      y_test = pd.read_csv("data/processed/y_test.csv").squeeze()

    # Train
      model = ChurnModel()
      metrics = model.train(X_train, y_train, X_test, y_test)

    # Save
      model.save()
      print(f"Training complete! Metrics: {metrics}")
