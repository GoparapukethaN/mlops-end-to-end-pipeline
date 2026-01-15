"""
Model training module with MLflow tracking.
"""
import os
import logging
import joblib
from pathlib import Path
from typing import Dict, Any, Tuple

import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
import mlflow
import mlflow.sklearn

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class ChurnModel:
    """XGBoost model for churn prediction with MLflow tracking."""

    def __init__(self, model_dir: str = "models"):
        self.model_dir = Path(model_dir)
        self.model_dir.mkdir(parents=True, exist_ok=True)
        self.model = None
        self.label_encoders: Dict[str, LabelEncoder] = {}
        self.feature_columns = []

    def prepare_features(self, df: pd.DataFrame, fit: bool = False) -> Tuple[np.ndarray, np.ndarray]:
        """Encode categorical features and prepare X, y."""
        df = df.copy()
        y = df.pop("Churn").values

        categorical_cols = df.select_dtypes(include=["object"]).columns.tolist()

        for col in categorical_cols:
            if fit:
                le = LabelEncoder()
                df[col] = le.fit_transform(df[col].astype(str))
                self.label_encoders[col] = le
            else:
                le = self.label_encoders.get(col)
                if le:
                    df[col] = df[col].astype(str).map(lambda x: le.transform([x])[0] if x in le.classes_ else -1)

        if fit:
            self.feature_columns = df.columns.tolist()

        return df.values, y

    def train(self, train_df: pd.DataFrame, params: Dict[str, Any] = None) -> Dict[str, float]:
        """Train the model with MLflow tracking."""
        params = params or {
            "n_estimators": 100,
            "max_depth": 5,
            "learning_rate": 0.1,
            "random_state": 42
        }

        mlflow.set_experiment("churn-prediction")

        with mlflow.start_run(run_name="xgboost-training"):
            mlflow.log_params(params)

            X_train, y_train = self.prepare_features(train_df, fit=True)

            self.model = XGBClassifier(**params, eval_metric="logloss")
            self.model.fit(X_train, y_train)

            y_pred = self.model.predict(X_train)
            y_proba = self.model.predict_proba(X_train)[:, 1]

            metrics = {
                "train_accuracy": accuracy_score(y_train, y_pred),
                "train_precision": precision_score(y_train, y_pred),
                "train_recall": recall_score(y_train, y_pred),
                "train_f1": f1_score(y_train, y_pred),
                "train_auc": roc_auc_score(y_train, y_proba)
            }

            mlflow.log_metrics(metrics)
            mlflow.sklearn.log_model(self.model, "model")

            logger.info(f"Training metrics: {metrics}")
            return metrics

    def evaluate(self, test_df: pd.DataFrame) -> Dict[str, float]:
        """Evaluate model on test data."""
        X_test, y_test = self.prepare_features(test_df, fit=False)

        y_pred = self.model.predict(X_test)
        y_proba = self.model.predict_proba(X_test)[:, 1]

        metrics = {
            "test_accuracy": accuracy_score(y_test, y_pred),
            "test_precision": precision_score(y_test, y_pred),
            "test_recall": recall_score(y_test, y_pred),
            "test_f1": f1_score(y_test, y_pred),
            "test_auc": roc_auc_score(y_test, y_proba)
        }

        with mlflow.start_run(run_name="xgboost-evaluation"):
            mlflow.log_metrics(metrics)

        logger.info(f"Test metrics: {metrics}")
        return metrics

    def save(self, filename: str = "churn_model.joblib"):
        """Save model and encoders."""
        model_path = self.model_dir / filename
        artifacts = {
            "model": self.model,
            "label_encoders": self.label_encoders,
            "feature_columns": self.feature_columns
        }
        joblib.dump(artifacts, model_path)
        logger.info(f"Model saved to {model_path}")

    def load(self, filename: str = "churn_model.joblib"):
        """Load model and encoders."""
        model_path = self.model_dir / filename
        artifacts = joblib.load(model_path)
        self.model = artifacts["model"]
        self.label_encoders = artifacts["label_encoders"]
        self.feature_columns = artifacts["feature_columns"]
        logger.info(f"Model loaded from {model_path}")


if __name__ == "__main__":
    train_df = pd.read_csv("data/processed/train.csv")
    test_df = pd.read_csv("data/processed/test.csv")

    model = ChurnModel()
    train_metrics = model.train(train_df)
    test_metrics = model.evaluate(test_df)
    model.save()

    print("\n=== Training Complete ===")
    print(f"Train Accuracy: {train_metrics['train_accuracy']:.4f}")
    print(f"Test Accuracy: {test_metrics['test_accuracy']:.4f}")
    print(f"Test AUC: {test_metrics['test_auc']:.4f}")
