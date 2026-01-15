"""
Data ingestion module for loading and initial processing of churn data.
"""
import os
import logging
from pathlib import Path
from typing import Optional, Tuple

import pandas as pd
from sklearn.model_selection import train_test_split

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class DataIngestion:
      """Handles data loading and train/test splitting."""

    def __init__(self, data_path: str = "data/raw/churn_data.csv"):
              self.data_path = Path(data_path)
              self.raw_data: Optional[pd.DataFrame] = None

    def load_data(self) -> pd.DataFrame:
              """Load raw data from CSV file."""
              logger.info(f"Loading data from {self.data_path}")

        if not self.data_path.exists():
                      # If no local file, fetch from UCI repository
                      url = "https://raw.githubusercontent.com/IBM/telco-customer-churn-on-icp4d/master/data/Telco-Customer-Churn.csv"
                      logger.info(f"Local file not found, fetching from {url}")
                      self.raw_data = pd.read_csv(url)
else:
              self.raw_data = pd.read_csv(self.data_path)

        logger.info(f"Loaded {len(self.raw_data)} records")
        return self.raw_data

    def clean_data(self, df: pd.DataFrame) -> pd.DataFrame:
              """Basic data cleaning."""
              logger.info("Cleaning data...")

        # Drop customer ID - not useful for prediction
              if 'customerID' in df.columns:
                            df = df.drop('customerID', axis=1)

              # Handle TotalCharges - some blank values
              df['TotalCharges'] = pd.to_numeric(df['TotalCharges'], errors='coerce')
              df['TotalCharges'] = df['TotalCharges'].fillna(df['TotalCharges'].median())

        # Convert target to binary
              df['Churn'] = df['Churn'].map({'Yes': 1, 'No': 0})

        logger.info(f"Data cleaned. Shape: {df.shape}")
        return df

    def split_data(
              self, 
              df: pd.DataFrame, 
              test_size: float = 0.2,
              random_state: int = 42
    ) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
              """Split data into train and test sets."""
              logger.info(f"Splitting data with test_size={test_size}")

        X = df.drop('Churn', axis=1)
        y = df['Churn']

        X_train, X_test, y_train, y_test = train_test_split(
                      X, y, test_size=test_size, random_state=random_state, stratify=y
        )

        logger.info(f"Train size: {len(X_train)}, Test size: {len(X_test)}")
        return X_train, X_test, y_train, y_test

    def run(self) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
              """Run full ingestion pipeline."""
              df = self.load_data()
              df = self.clean_data(df)
              return self.split_data(df)


if __name__ == "__main__":
      ingestion = DataIngestion()
      X_train, X_test, y_train, y_test = ingestion.run()

    # Save processed data
      os.makedirs("data/processed", exist_ok=True)
      X_train.to_csv("data/processed/X_train.csv", index=False)
      X_test.to_csv("data/processed/X_test.csv", index=False)
      y_train.to_csv("data/processed/y_train.csv", index=False)
      y_test.to_csv("data/processed/y_test.csv", index=False)

    logger.info("Data ingestion complete!")
