# Verification

Last local verification: 2026-05-20

Command:

```bash
make verify
# or
make ci-local
```

Container smoke check:

```bash
make docker-check
```

Coverage:

- 15 pytest checks across API, request validation, data cleaning, and model feature prep
- API tests for health, readiness, prediction, validation failures, unavailable model behavior, and Prometheus metrics
- Flake8 linting with normal failure behavior
- Black format check
- isort import-order check
- Source and test compilation check
- Training module import check
- Prometheus config parse
- Optional Docker image build plus health and prediction smoke check

This repo uses local verification as the current proof path. Hosted workflow cleanup is
tracked separately; when Actions minutes are limited, local `make verify` /
`make ci-local` remains the source of truth. Docker is optional for the core local
verification path and stays in the separate `make docker-check` / `make verify-full`
path.

Latest Docker smoke result:

- Command: `make docker-check`
- Result: image build completed, `/health` returned `{"status":"healthy","model_loaded":true}`, and the sample prediction returned `{"churn_probability":0.4744,"churn_prediction":0,"risk_level":"Medium"}`.

Artifact boundary:

- The checked-in model artifact is used for local smoke tests and demos.
- `MODEL_PATH` should point only to trusted local artifacts. Joblib/pickle model files
  are not a safe format for untrusted uploads.
