"""Aither SDK - Python client for the Aither platform."""

from __future__ import annotations

from typing import Any

from aither.client import AitherClient

__version__ = "0.2.0"
__all__ = ["AitherClient", "init", "log_prediction", "log_label", "flush", "close"]

_client: AitherClient | None = None


def init(
    api_key: str | None = None,
    endpoint: str | None = None,
    flush_interval: float = 1.0,
    batch_size: int = 100,
) -> None:
    """Initialize the global Aither client.

    Args:
        api_key: API key for authentication. Falls back to AITHER_API_KEY env var.
        endpoint: API endpoint URL. Falls back to AITHER_ENDPOINT env var or default.
        flush_interval: How often to flush queued predictions (seconds).
        batch_size: Maximum predictions per batch request.
    """
    global _client
    _client = AitherClient(
        api_key=api_key,
        endpoint=endpoint,
        flush_interval=flush_interval,
        batch_size=batch_size,
    )


def _get_client() -> AitherClient:
    """Get or create the global client."""
    global _client
    if _client is None:
        _client = AitherClient()
    return _client


def log_prediction(
    model_name: str,
    features: dict[str, Any],
    prediction: Any,
    *,
    version: str | None = None,
    probabilities: list[float] | None = None,
    classes: list[str] | None = None,
    environment: str | None = None,
    request_id: str | None = None,
    user_id: str | None = None,
) -> str:
    """Log a model prediction using the global client (non-blocking).

    Predictions are queued and sent asynchronously using OTLP format.
    Returns a trace_id that can be used to correlate ground truth labels.

    Args:
        model_name: Identifier for the model (e.g., "fraud_detector").
        features: Input features used for the prediction.
        prediction: The prediction value.
        version: Model version (e.g., "1.2.3", git sha).
        probabilities: Class probabilities (for classification).
        classes: Class labels corresponding to probabilities.
        environment: Deployment environment (e.g., "production").
        request_id: Unique request identifier.
        user_id: User/customer identifier (anonymized).

    Returns:
        trace_id: Hex-encoded trace ID for label correlation.
    """
    return _get_client().log_prediction(
        model_name=model_name,
        features=features,
        prediction=prediction,
        version=version,
        probabilities=probabilities,
        classes=classes,
        environment=environment,
        request_id=request_id,
        user_id=user_id,
    )


def log_label(trace_id: str, label: Any) -> None:
    """Log ground truth label for a previous prediction (non-blocking).

    Use the trace_id returned from log_prediction() to correlate
    the ground truth with the original prediction.

    Args:
        trace_id: The trace_id returned from log_prediction().
        label: The actual outcome/ground truth value.
    """
    _get_client().log_label(trace_id=trace_id, label=label)


def flush() -> None:
    """Force immediate flush of queued predictions and labels (blocking).

    Useful for ensuring data is sent before shutdown or in tests.
    """
    _get_client().flush()


def close() -> None:
    """Close the global client and flush remaining data."""
    global _client
    if _client is not None:
        _client.close()
        _client = None
