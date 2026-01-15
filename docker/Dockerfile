# Multi-stage build for smaller image
FROM python:3.10-slim as builder

WORKDIR /app

# Install build dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
        && rm -rf /var/lib/apt/lists/*

        # Copy requirements first for caching
        COPY requirements.txt .
        RUN pip wheel --no-cache-dir --no-deps --wheel-dir /app/wheels -r requirements.txt


        # Production stage
        FROM python:3.10-slim

        WORKDIR /app

        # Create non-root user
        RUN useradd --create-home appuser

        # Copy wheels from builder
        COPY --from=builder /app/wheels /wheels
        RUN pip install --no-cache /wheels/*

        # Copy application code
        COPY src/ ./src/
        COPY models/ ./models/
        COPY configs/ ./configs/

        # Set ownership
        RUN chown -R appuser:appuser /app

        USER appuser

        # Environment variables
        ENV PYTHONPATH=/app
        ENV MODEL_PATH=/app/models/churn_model.joblib

        EXPOSE 8000

        # Health check
        HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
            CMD curl -f http://localhost:8000/health || exit 1

            CMD ["uvicorn", "src.api.main:app", "--host", "0.0.0.0", "--port", "8000"]
