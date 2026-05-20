#!/usr/bin/env bash
set -euo pipefail

image_name="${IMAGE_NAME:-churn-prediction:local-verify}"
container_name="${CONTAINER_NAME:-churn-prediction-local-verify}"
host_port="${HOST_PORT:-8001}"

cleanup() {
  docker rm -f "$container_name" >/dev/null 2>&1 || true
}

trap cleanup EXIT

docker info >/dev/null
docker build -t "$image_name" -f docker/Dockerfile .

cleanup
docker run -d \
  --name "$container_name" \
  -p "${host_port}:8000" \
  "$image_name" >/dev/null

ready=0
for _ in 1 2 3 4 5 6 7 8 9 10 11 12; do
  if curl -fsS "http://localhost:${host_port}/health/ready" >/dev/null 2>&1; then
    ready=1
    break
  fi
  sleep 2
done

if [[ "$ready" != "1" ]]; then
  docker logs "$container_name"
  exit 1
fi

health_json="$(curl -fsS "http://localhost:${host_port}/health")"
sample_payload='{
  "tenure": 12,
  "MonthlyCharges": 70.5,
  "TotalCharges": 846.0,
  "Contract": "Month-to-month",
  "PaymentMethod": "Electronic check",
  "gender": "Female",
  "SeniorCitizen": 0,
  "Partner": "Yes",
  "Dependents": "No",
  "PhoneService": "Yes",
  "MultipleLines": "No",
  "InternetService": "Fiber optic",
  "OnlineSecurity": "No",
  "OnlineBackup": "No",
  "DeviceProtection": "No",
  "TechSupport": "No",
  "StreamingTV": "Yes",
  "StreamingMovies": "Yes",
  "PaperlessBilling": "Yes"
}'
prediction_json="$(
  curl -fsS -X POST "http://localhost:${host_port}/predict" \
    -H "Content-Type: application/json" \
    -d "$sample_payload"
)"

HEALTH_JSON="$health_json" PREDICTION_JSON="$prediction_json" python3 - <<'PY'
import json
import os

health = json.loads(os.environ["HEALTH_JSON"])
prediction = json.loads(os.environ["PREDICTION_JSON"])

assert health == {"status": "healthy", "model_loaded": True}
assert prediction == {
    "churn_probability": 0.4744,
    "churn_prediction": 0,
    "risk_level": "Medium",
}
print("docker image and prediction smoke check passed")
PY
