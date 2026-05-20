"""Tests for churn data cleaning and splitting."""

import pandas as pd

from src.data.ingestion import DataIngestion


def test_clean_data_converts_target_and_total_charges() -> None:
    df = pd.DataFrame(
        [
            {
                "customerID": "C-1",
                "TotalCharges": " ",
                "Churn": "Yes",
                "Contract": "Month-to-month",
            },
            {
                "customerID": "C-2",
                "TotalCharges": "120.5",
                "Churn": "No",
                "Contract": "One year",
            },
        ]
    )

    cleaned = DataIngestion().clean_data(df)

    assert "customerID" not in cleaned.columns
    assert cleaned["Churn"].tolist() == [1, 0]
    assert cleaned["TotalCharges"].isna().sum() == 0
    assert cleaned["TotalCharges"].tolist() == [120.5, 120.5]


def test_split_data_preserves_train_and_test_rows() -> None:
    df = pd.DataFrame(
        {
            "feature": list(range(20)),
            "Churn": [0, 1] * 10,
        }
    )

    train_df, test_df = DataIngestion().split_data(df, test_size=0.25)

    assert len(train_df) == 15
    assert len(test_df) == 5
    assert set(train_df.index).isdisjoint(set(test_df.index))
