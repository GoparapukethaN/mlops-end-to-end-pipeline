# Verification

Last local verification: 2026-05-20

Command:

```bash
make verify
```

Coverage:

- API tests for health, readiness, prediction, and Prometheus metrics
- Flake8 linting
- Black format check
- isort import-order check
- Training module import check
- Prometheus config parse
- Docker Compose config validation

This repo uses local verification as the current proof path. Hosted workflow cleanup is
a follow-up, so local `make verify` is the source of truth for now.
