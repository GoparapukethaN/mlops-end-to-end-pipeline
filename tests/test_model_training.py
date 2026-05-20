"""Tests for model feature preparation helpers."""

import pandas as pd

from src.models.train import ChurnModel


def sample_training_frame() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "gender": "Female",
                "Contract": "Month-to-month",
                "tenure": 2,
                "MonthlyCharges": 70.0,
                "TotalCharges": 140.0,
                "Churn": 1,
            },
            {
                "gender": "Male",
                "Contract": "One year",
                "tenure": 24,
                "MonthlyCharges": 40.0,
                "TotalCharges": 960.0,
                "Churn": 0,
            },
        ]
    )


def test_prepare_features_records_feature_order_and_encoders() -> None:
    model = ChurnModel(model_dir="/tmp/mlops-test-models")

    features, target = model.prepare_features(sample_training_frame(), fit=True)

    assert features.shape == (2, 5)
    assert target.tolist() == [1, 0]
    assert model.feature_columns == [
        "gender",
        "Contract",
        "tenure",
        "MonthlyCharges",
        "TotalCharges",
    ]
    assert sorted(model.label_encoders) == ["Contract", "gender"]


def test_prepare_features_maps_unknown_categories_to_sentinel() -> None:
    model = ChurnModel(model_dir="/tmp/mlops-test-models")
    model.prepare_features(sample_training_frame(), fit=True)
    test_df = pd.DataFrame(
        [
            {
                "gender": "Unknown",
                "Contract": "Two year",
                "tenure": 8,
                "MonthlyCharges": 65.0,
                "TotalCharges": 520.0,
                "Churn": 0,
            }
        ]
    )

    features, target = model.prepare_features(test_df, fit=False)

    assert target.tolist() == [0]
    assert features[0, 0] == -1
    assert features[0, 1] == -1
