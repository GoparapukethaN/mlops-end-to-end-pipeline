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
        """Load data from CSV or download if not exists."""
        if not self.data_path.exists():
            logger.info("Downloading Telco Customer Churn dataset...")
            url = "https://raw.githubusercontent.com/IBM/telco-customer-churn-on-icp4d/master/data/Telco-Customer-Churn.csv"
            self.raw_data = pd.read_csv(url)
            self.data_path.parent.mkdir(parents=True, exist_ok=True)
            self.raw_data.to_csv(self.data_path, index=False)
        else:
            logger.info(f"Loading data from {self.data_path}")
            self.raw_data = pd.read_csv(self.data_path)
        logger.info(f"Loaded {len(self.raw_data)} records")
        return self.raw_data

    def clean_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """Basic data cleaning."""
        df = df.drop("customerID", axis=1, errors="ignore")
        df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")
        df["TotalCharges"] = df["TotalCharges"].fillna(df["TotalCharges"].median())
        df["Churn"] = (df["Churn"] == "Yes").astype(int)
        return df

    def split_data(self, df: pd.DataFrame, test_size: float = 0.2) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """Split into train and test sets."""
        train_df, test_df = train_test_split(df, test_size=test_size, stratify=df["Churn"], random_state=42)
        logger.info(f"Train: {len(train_df)}, Test: {len(test_df)}")
        return train_df, test_df

    def run(self) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """Run full ingestion pipeline."""
        df = self.load_data()
        df = self.clean_data(df)
        return self.split_data(df)


if __name__ == "__main__":
    ingestion = DataIngestion()
    train_df, test_df = ingestion.run()

    os.makedirs("data/processed", exist_ok=True)
    train_df.to_csv("data/processed/train.csv", index=False)
    test_df.to_csv("data/processed/test.csv", index=False)
    print("Data saved to data/processed/")