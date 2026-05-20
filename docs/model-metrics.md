# Model Metrics

Last recomputed: 2026-05-20

Dataset:

- Source: IBM Telco Customer Churn CSV committed under `data/raw/churn_data.csv`
- Processed split: `data/processed/train.csv` and `data/processed/test.csv`
- Split behavior: stratified 80/20 split with `random_state=42`
- Model: XGBoost classifier in `src/models/train.py`

Command used:

```bash
MLFLOW_TRACKING_URI="file:///tmp/mlops-metrics-verify" .venv/bin/python - <<'PY'
import json
import pandas as pd
from src.models.train import ChurnModel

train_df = pd.read_csv("data/processed/train.csv")
test_df = pd.read_csv("data/processed/test.csv")
model = ChurnModel(model_dir="/tmp/mlops-model-verify")
train = model.train(train_df)
test = model.evaluate(test_df)
print(json.dumps({**train, **test}, sort_keys=True, indent=2))
PY
```

Result:

| Metric | Value |
| ------ | ----- |
| Train accuracy | 0.8450 |
| Train precision | 0.7541 |
| Train recall | 0.6174 |
| Train F1 | 0.6789 |
| Train AUC | 0.9141 |
| Test accuracy | 0.8006 |
| Test precision | 0.6566 |
| Test recall | 0.5214 |
| Test F1 | 0.5812 |
| Test AUC | 0.8393 |

These metrics are a portfolio baseline for the current dataset and model settings. They
are not a claim of production performance or business impact.
