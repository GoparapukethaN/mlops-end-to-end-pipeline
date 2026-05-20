# Case Study: Churn Model Lifecycle Pipeline

I built this project to practice the full path around a classic ML model: prepare data,
train a model, save a reproducible artifact, serve predictions through an API, expose
metrics, and keep the repo verifiable from a fresh clone.

The project is intentionally scoped as a portfolio system. It does not claim production
traffic or business impact. The value is in showing the engineering workflow I would
expect around a model service before it is handed to a platform or product team.

## Problem

A notebook can prove that a model works once. A model service needs more structure:

- a repeatable ingestion and train/test split
- a saved model artifact that the API can load consistently
- input validation before predictions reach the model
- health and readiness checks for deployment environments
- metrics that make the service observable after it starts serving requests
- local verification that exercises the code paths a reviewer can inspect

This repo turns a customer churn model into that kind of workflow.

## What I Built

The pipeline starts with the IBM Telco Customer Churn dataset. The ingestion module loads
the CSV, removes the customer identifier, normalizes `TotalCharges`, converts `Churn`
into a binary label, and writes stratified train/test splits with `random_state=42`.

Training uses an XGBoost classifier with a small fixed parameter set. The training code
logs parameters and metrics to MLflow, evaluates on the test split, and saves a local
`joblib` artifact containing the model, label encoders, and feature column order.

The serving layer is a FastAPI app with:

- `/health` for service status
- `/health/ready` for model-load readiness
- `/predict` for typed churn prediction requests
- `/metrics` for Prometheus-compatible counters and latency histograms
- `/docs` for the generated API contract

The repo also includes Docker, Docker Compose, Kubernetes, and Prometheus configuration so
the service shape is visible beyond the Python modules.

## Current Baseline

The current model baseline is documented in [model-metrics.md](model-metrics.md). The
latest recompute produced:

| Metric | Value |
| ------ | ----- |
| Test accuracy | 0.8006 |
| Test precision | 0.6566 |
| Test recall | 0.5214 |
| Test F1 | 0.5812 |
| Test AUC | 0.8393 |

I treat these numbers as a baseline for the checked-in dataset and model settings. They
are not presented as production performance.

## Verification Path

The local verification path is the main proof point for this repo:

```bash
make verify
```

That command runs pytest, flake8, Black check, isort check, source compilation, training
module import, and Prometheus config parsing without requiring Docker.

For the container path:

```bash
make docker-check
# or
make verify-full
```

The latest Docker smoke result is recorded in [verification.md](verification.md): the
image built, `/health` returned a loaded model, and the sample prediction returned a
medium-risk churn response.

## Design Choices

I kept a trained artifact in the repo so reviewers can run the API without retraining
first. That makes the service easier to smoke test, but it also means the artifact is a
trusted local file only. `joblib` and pickle-style model files should not be accepted from
untrusted users.

I used FastAPI request models for input validation because bad categorical values or
negative numeric fields should fail before reaching the model. Unknown categorical values
inside the model feature encoder map to `-1`, which is a practical fallback for this demo
but would need stronger feature validation in a production workflow.

I kept Prometheus instrumentation small: request count and prediction latency are enough
to show the observability path without pretending this is a mature monitoring platform.

## What I Would Improve Next

- Replace the local artifact handoff with a real model registry flow.
- Add data validation checks before training and serving.
- Track feature drift and prediction distribution changes over time.
- Add a retraining workflow with approval gates instead of automatic overwrite.
- Clean up the hosted GitHub Actions workflow once workflow permissions and Actions
  minutes are available again.
- Add a small dashboard for model health, recent predictions, latency, and drift signals.
