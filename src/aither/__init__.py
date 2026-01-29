"""Aither SDK - Python client for the Aither platform."""

from __future__ import annotations

import functools
import threading
import warnings
from typing import Any, Callable, TypeVar

from aither.client import AitherClient, Stats, UsageInfo
from aither.management import APIKeysNamespace, OrgNamespace, UserNamespace
from aither.models import APIKey, APIKeyWithSecret, Organization, UsageStats, User

__version__ = "0.2.0"
__all__ = [
    # Client
    "AitherClient",
    "Stats",
    "UsageInfo",
    # Models
    "APIKey",
    "APIKeyWithSecret",
    "Organization",
    "UsageStats",
    "User",
    # Module-level functions
    "init",
    "configure",
    "log_prediction",
    "log_label",
    "flush",
    "close",
    "track",
    "last_trace_id",
    "stats",
    "usage_info",
    # Management namespaces
    "api_keys",
    "org",
    "user",
]

_client: AitherClient | None = None

# Thread-local storage for last trace_id from @track() decorator
_thread_local = threading.local()

F = TypeVar("F", bound=Callable[..., Any])


def init(
    api_key: str | None = None,
    endpoint: str | None = None,
    flush_interval: float = 1.0,
    batch_size: int = 100,
    max_queue_size: int = 10_000,
    on_error: str = "warn",
    on_drop: Callable[[Any], None] | None = None,
    on_send: Callable[[list[Any]], None] | None = None,
) -> None:
    """Initialize the global Aither client.

    Args:
        api_key: API key for authentication. Falls back to AITHER_API_KEY env var.
        endpoint: API endpoint URL. Falls back to AITHER_ENDPOINT env var or default.
        flush_interval: How often to flush queued predictions (seconds).
        batch_size: Maximum predictions per batch request.
        max_queue_size: Maximum items in queue before dropping oldest.
        on_error: Error handling mode - "warn", "silent", or "raise".
        on_drop: Callback when item dropped (queue full or retries exhausted).
        on_send: Callback on successful batch send.
    """
    global _client
    _client = AitherClient(
        api_key=api_key,
        endpoint=endpoint,
        flush_interval=flush_interval,
        batch_size=batch_size,
        max_queue_size=max_queue_size,
        on_error=on_error,
        on_drop=on_drop,
        on_send=on_send,
    )


def configure(
    api_key: str | None = None,
    endpoint: str | None = None,
    flush_interval: float = 1.0,
    batch_size: int = 100,
    max_queue_size: int = 10_000,
    on_error: str = "warn",
    on_drop: Callable[[Any], None] | None = None,
    on_send: Callable[[list[Any]], None] | None = None,
) -> None:
    """Configure the global Aither client.

    This is the recommended way to initialize the SDK. Alias for init().

    Args:
        api_key: API key for authentication. Falls back to AITHER_API_KEY env var.
        endpoint: API endpoint URL. Falls back to AITHER_ENDPOINT env var or default.
        flush_interval: How often to flush queued predictions (seconds).
        batch_size: Maximum predictions per batch request.
        max_queue_size: Maximum items in queue before dropping oldest.
        on_error: Error handling mode - "warn", "silent", or "raise".
        on_drop: Callback when item dropped (queue full or retries exhausted).
        on_send: Callback on successful batch send.

    Example:
        import aither

        # Option 1: Environment variable (recommended)
        # export AITHER_API_KEY="aith_..."
        aither.configure()

        # Option 2: Explicit API key
        aither.configure(api_key="aith_...")
    """
    init(
        api_key=api_key,
        endpoint=endpoint,
        flush_interval=flush_interval,
        batch_size=batch_size,
        max_queue_size=max_queue_size,
        on_error=on_error,
        on_drop=on_drop,
        on_send=on_send,
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


def flush(timeout: float | None = None) -> None:
    """Force immediate flush of queued predictions and labels (blocking).

    Args:
        timeout: Maximum time to wait for flush. Raises TimeoutError if exceeded.

    Raises:
        TimeoutError: If flush doesn't complete within timeout.
    """
    _get_client().flush(timeout=timeout)


def close() -> None:
    """Close the global client and flush remaining data."""
    global _client
    if _client is not None:
        _client.close()
        _client = None


def stats() -> Stats:
    """Get current SDK statistics.

    Returns:
        Stats object with queued, sent, dropped, retries, label_misses counts.
    """
    return _get_client().stats()


def usage_info() -> UsageInfo:
    """Get the most recent rate limit and usage information from the server.

    This information is updated after each successful API request.
    Until a request is made, all fields will be None.

    Returns:
        UsageInfo object with limit, remaining, grace_ends_at, and warning.

    Example:
        usage = aither.usage_info()
        if usage.remaining is not None and usage.remaining < 100:
            print(f"Low on API calls: {usage.remaining} remaining")
        if usage.warning:
            print(f"Warning: {usage.warning}")
    """
    return _get_client().usage_info()


def last_trace_id() -> str | None:
    """Get the trace_id from the most recent @track() call in this thread.

    Returns:
        The trace_id string, or None if no tracked call has been made in this thread.
    """
    return getattr(_thread_local, "last_trace_id", None)


def track(
    model_name: str,
    *,
    features_arg: str | None = None,
    prediction_key: str | None = None,
    version: str | None = None,
    environment: str | None = None,
) -> Callable[[F], F]:
    """Decorator to automatically track model predictions.

    By default, captures the first positional argument as features and the
    return value as prediction. Use features_arg and prediction_key for
    non-standard signatures.

    Args:
        model_name: Identifier for the model (e.g., "fraud_detector").
        features_arg: Name of the argument to use as features (default: first positional).
        prediction_key: Key to extract from return dict as prediction (default: entire return).
        version: Model version string.
        environment: Deployment environment.

    Returns:
        Decorated function that logs predictions automatically.

    Example:
        @aither.track("fraud_detector")
        def predict(features):
            return model.predict(features)

        result = predict({"amount": 150.0})
        trace_id = aither.last_trace_id()
    """

    def decorator(func: F) -> F:
        @functools.wraps(func)
        def wrapper(*args: Any, **kwargs: Any) -> Any:
            # Extract features from first positional arg or named arg
            if features_arg is not None:
                features = kwargs.get(features_arg)
                if features is None and args:
                    # Try to find it by parameter position
                    import inspect

                    sig = inspect.signature(func)
                    params = list(sig.parameters.keys())
                    if features_arg in params:
                        idx = params.index(features_arg)
                        if idx < len(args):
                            features = args[idx]
            else:
                # Default: first positional argument
                features = args[0] if args else {}

            # Ensure features is a dict
            if not isinstance(features, dict):
                features = {"input": features}

            # Call the actual function
            result = func(*args, **kwargs)

            # Extract prediction from result
            if prediction_key is not None and isinstance(result, dict):
                prediction = result.get(prediction_key, result)
            else:
                prediction = result

            # Log the prediction
            trace_id = _get_client().log_prediction(
                model_name=model_name,
                features=features,
                prediction=prediction,
                version=version,
                environment=environment,
            )

            # Store trace_id in thread-local storage
            _thread_local.last_trace_id = trace_id

            return result

        return wrapper  # type: ignore[return-value]

    return decorator


# Management namespaces - lazy proxy to global client
api_keys = APIKeysNamespace(_get_client)
org = OrgNamespace(_get_client)
user = UserNamespace(_get_client)
