#!/usr/bin/env bash
set -euo pipefail

if [[ -x ".venv/bin/python" ]]; then
  python_cmd=".venv/bin/python"
else
  python_cmd="${PYTHON:-python3}"
fi

"$python_cmd" -m pytest
"$python_cmd" -m flake8 src tests --max-line-length=88
"$python_cmd" -m black --check src tests
"$python_cmd" -m isort --check-only src tests
"$python_cmd" -m compileall -q src tests
"$python_cmd" - <<'PY'
from src.models.train import ChurnModel

assert ChurnModel is not None
print("training module import ok")
PY
"$python_cmd" - <<'PY'
import yaml

yaml.safe_load(open("configs/prometheus.yml", encoding="utf-8"))
print("prometheus config ok")
PY
docker compose config --quiet

echo "local verification passed"
