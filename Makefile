PYTHON ?= python3

.PHONY: install test lint format-check prometheus-check compose-check verify

install:
	$(PYTHON) -m pip install -r requirements.txt

test:
	$(PYTHON) -m pytest

lint:
	$(PYTHON) -m flake8 src tests --ignore=E501,W292,W293,F401,E302,E303,E261,E262 --max-line-length=150

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
