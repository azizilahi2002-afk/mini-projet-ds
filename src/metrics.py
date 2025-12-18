from prometheus_client import Counter, Histogram, Gauge, generate_latest
from typing import Optional

# Définir les métriques
PREDICTION_COUNTER = Counter(
    "predictions_total",
    "Total number of predictions",
    ["status", "endpoint"]
)

PREDICTION_DURATION = Histogram(
    "prediction_duration_seconds",
    "Time spent processing predictions",
    ["endpoint"]
)

REQUESTS_IN_PROGRESS = Gauge(
    "requests_in_progress",
    "Number of requests currently being processed",
    ["endpoint"]
)

ERROR_COUNTER = Counter(
    "errors_total",
    "Total number of errors",
    ["type", "endpoint"]
)

# Fonction pour exporter les métriques
def export_metrics():
    """Exporter toutes les métriques au format Prometheus."""
    return generate_latest()

# Décorateur pour instrumenter les fonctions
def instrument_predict(endpoint_name: str = "predict"):
    """Décorateur pour instrumenter les fonctions de prédiction."""
    def decorator(func):
        def wrapper(*args, **kwargs):
            REQUESTS_IN_PROGRESS.labels(endpoint=endpoint_name).inc()
            with PREDICTION_DURATION.labels(endpoint=endpoint_name).time():
                try:
                    result = func(*args, **kwargs)
                    PREDICTION_COUNTER.labels(
                        status="success",
                        endpoint=endpoint_name
                    ).inc()
                    return result
                except Exception as e:
                    ERROR_COUNTER.labels(
                        type=type(e).__name__,
                        endpoint=endpoint_name
                    ).inc()
                    PREDICTION_COUNTER.labels(
                        status="error",
                        endpoint=endpoint_name
                    ).inc()
                    raise
                finally:
                    REQUESTS_IN_PROGRESS.labels(endpoint=endpoint_name).dec()
        return wrapper
    return decorator