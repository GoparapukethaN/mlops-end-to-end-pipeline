PYTHON ?= python3

.PHONY: install test lint format-check prometheus-check compose-check verify verify-full ci-local docker-check

install:
	$(PYTHON) -m pip install -r requirements.txt

test:
	$(PYTHON) -m pytest

lint:
	$(PYTHON) -m flake8 src tests --max-line-length=88

format-check:
	$(PYTHON) -m black --check src tests
	$(PYTHON) -m isort --check-only src tests

prometheus-check:
	$(PYTHON) - <<'PY'
	import yaml
	yaml.safe_load(open("configs/prometheus.yml", encoding="utf-8"))
	print("prometheus config ok")
	PY

compose-check:
	docker compose config --quiet

verify:
	./scripts/verify-local.sh

ci-local: verify

verify-full: verify compose-check docker-check

docker-check:
	bash scripts/verify-docker.sh
