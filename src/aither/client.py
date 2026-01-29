"""Aither client implementation using OTLP for model prediction logging."""

from __future__ import annotations

import atexit
import json
import os
import secrets
import threading
import time
from collections import deque
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Callable, Optional

import httpx

# SDK version - keep in sync with __init__.py
SDK_VERSION = "0.2.0"
SDK_LANGUAGE = "python"

DEFAULT_ENDPOINT = "https://aither.computer"
DEFAULT_FLUSH_INTERVAL = 1.0  # seconds
DEFAULT_BATCH_SIZE = 100
DEFAULT_MAX_QUEUE_SIZE = 10_000
MAX_RETRIES = 3
BACKOFF_SECONDS = [1.0, 2.0, 4.0]


@dataclass
class UsageInfo:
    """Rate limit and usage information from the server.

    Updated after each successful API request. Use client.usage_info()
    to get the most recent values.

    Attributes:
        limit: Maximum API calls allowed in the current period.
        remaining: API calls remaining in the current period.
        grace_ends_at: When grace period expires (if in grace period).
        warning: Warning message from server (e.g., grace period notice).
    """

    limit: Optional[int] = None
    remaining: Optional[int] = None
    grace_ends_at: Optional[datetime] = None
    warning: Optional[str] = None


@dataclass
class Stats:
    """SDK statistics for observability."""

    queued: int = 0
    sent: int = 0
    dropped: int = 0
    retries: int = 0
    label_misses: int = 0


@dataclass
class PredictionSpan:
    """Internal representation of a prediction span."""

    trace_id: bytes
    span_id: bytes
    model_name: str
    features: dict[str, Any]
    prediction: Any
    version: str | None = None
    probabilities: list[float] | None = None
    classes: list[str] | None = None
    environment: str | None = None
    request_id: str | None = None
    user_id: str | None = None
    start_time_ns: int = 0
    end_time_ns: int = 0


@dataclass
class LabelUpdate:
    """Internal representation of a label update."""

    trace_id: str
    label: Any


@dataclass
class BatchContext:
    """Context manager for high-throughput batch logging.

    Usage:
        with client.batch() as batch:
            for item in dataset:
                batch.log(model_name="classifier", features=item, prediction=output)
        # trace_ids available after context exit
        trace_ids = batch.trace_ids
    """

    _client: "AitherClient"
    trace_ids: list[str] = field(default_factory=list)

    def log(
        self,
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
        """Log a prediction within the batch.

        Returns trace_id and appends it to trace_ids list.
        """
        trace_id = self._client.log_prediction(
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
        self.trace_ids.append(trace_id)
        return trace_id

    def __enter__(self) -> "BatchContext":
        return self

    def __exit__(self, *args: Any) -> None:
        # Trigger flush on batch exit
        self._client._flush_event.set()


@dataclass
class TraceContext:
    """Context manager for dynamic model name tracing.

    Usage:
        with client.trace("model_" + version) as t:
            prediction = model.predict(features)
            t.log(features=features, prediction=prediction)
            # t.trace_id available for label correlation
    """

    _client: "AitherClient"
    model_name: str
    version: str | None = None
    environment: str | None = None
    trace_id: str | None = None

    def log(
        self,
        features: dict[str, Any],
        prediction: Any,
        *,
        probabilities: list[float] | None = None,
        classes: list[str] | None = None,
        request_id: str | None = None,
        user_id: str | None = None,
    ) -> str:
        """Log the prediction for this trace.

        Sets trace_id on the context for label correlation.
        """
        self.trace_id = self._client.log_prediction(
            model_name=self.model_name,
            features=features,
            prediction=prediction,
            version=self.version,
            probabilities=probabilities,
            classes=classes,
            environment=self.environment,
            request_id=request_id,
            user_id=user_id,
        )
        return self.trace_id

    def __enter__(self) -> "TraceContext":
        return self

    def __exit__(self, *args: Any) -> None:
        pass


class AitherClient:
    """Client for the Aither platform API.

    Logs ML model predictions using OTLP (OpenTelemetry Protocol) format.
    Predictions are sent as spans with ml.* attributes.
    """

    def __init__(
        self,
        api_key: str | None = None,
        endpoint: str | None = None,
        base_url: str | None = None,  # Alias for endpoint (spec compliance)
        timeout: float = 30.0,
        flush_interval: float = DEFAULT_FLUSH_INTERVAL,
        batch_size: int = DEFAULT_BATCH_SIZE,
        max_queue_size: int = DEFAULT_MAX_QUEUE_SIZE,
        enable_background: bool = True,
        on_error: str = "warn",  # "warn" | "silent" | "raise"
        on_drop: Callable[[Any], None] | None = None,
        on_send: Callable[[list[Any]], None] | None = None,
    ) -> None:
        """Initialize the Aither client.

        Args:
            api_key: API key for authentication. Falls back to AITHER_API_KEY env var.
            endpoint: API endpoint URL. Falls back to AITHER_ENDPOINT env var or default.
            timeout: Request timeout in seconds.
            flush_interval: How often to flush queued predictions (seconds).
            batch_size: Maximum predictions per batch request.
            max_queue_size: Maximum items in queue before dropping oldest.
            enable_background: If False, predictions are sent immediately (blocking).
            on_error: Error handling mode - "warn", "silent", or "raise".
            on_drop: Callback when item dropped (queue full or retries exhausted).
            on_send: Callback on successful batch send.
        """
        self.api_key = api_key or os.environ.get("AITHER_API_KEY")
        # base_url is an alias for endpoint (spec compliance)
        self.endpoint = (
            base_url
            or endpoint
            or os.environ.get("AITHER_BASE_URL")
            or os.environ.get("AITHER_ENDPOINT")
            or DEFAULT_ENDPOINT
        )
        self.timeout = timeout
        self.flush_interval = flush_interval
        self.batch_size = batch_size
        self.max_queue_size = max_queue_size
        self.enable_background = enable_background
        self.on_error = on_error
        self.on_drop = on_drop
        self.on_send = on_send

        # Thread-safe queue for predictions and labels
        self._prediction_queue: deque[PredictionSpan] = deque()
        self._label_queue: deque[LabelUpdate] = deque()
        self._queue_lock = threading.Lock()

        # Statistics
        self._stats = Stats()
        self._stats_lock = threading.Lock()

        # Track known trace_ids for label validation
        self._known_trace_ids: set[str] = set()
        self._trace_ids_lock = threading.Lock()

        # Rate limiting state
        self._retry_after_until: float = 0.0

        # Usage info from server (updated on each response)
        self._usage_info = UsageInfo()
        self._usage_info_lock = threading.Lock()

        # Grace period warning tracking (warn once, not every request)
        self._grace_warned = False
        self._grace_warned_lock = threading.Lock()

        # Drop warning throttle (warn once per minute)
        self._last_drop_warning: float = 0.0
        self._drop_warning_interval: float = 60.0

        # First success tracking
        self._first_success_printed = False

        # Background worker management
        self._worker_thread: threading.Thread | None = None
        self._stop_event = threading.Event()
        self._flush_event = threading.Event()  # Signal for immediate flush

        # Start background worker
        if self.enable_background:
            self._start_worker()
            atexit.register(self.close)

    def _build_headers(
        self, content_type: str = "application/x-protobuf"
    ) -> dict[str, str]:
        """Build request headers including SDK identification."""
        headers = {
            "Content-Type": content_type,
            "X-Aither-SDK-Version": SDK_VERSION,
            "X-Aither-SDK-Language": SDK_LANGUAGE,
        }
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"
        return headers

    def _generate_trace_id(self) -> bytes:
        """Generate a random 128-bit trace ID."""
        return secrets.token_bytes(16)

    def _generate_span_id(self) -> bytes:
        """Generate a random 64-bit span ID."""
        return secrets.token_bytes(8)

    def _start_worker(self) -> None:
        """Start the background worker thread."""
        self._worker_thread = threading.Thread(target=self._worker_loop, daemon=True)
        self._worker_thread.start()

    def _worker_loop(self) -> None:
        """Worker thread that periodically flushes the queues."""
        while not self._stop_event.is_set():
            try:
                # Check rate limiting
                if time.time() < self._retry_after_until:
                    self._stop_event.wait(timeout=0.1)
                    continue

                self._flush_predictions()
                self._flush_labels()
            except Exception as e:
                self._handle_error(f"Error flushing queue: {e}")

            # Wait for flush interval, stop event, or flush signal
            self._flush_event.wait(timeout=self.flush_interval)
            self._flush_event.clear()

    def _handle_error(self, message: str) -> None:
        """Handle errors based on on_error setting."""
        if self.on_error == "raise":
            raise RuntimeError(message)
        elif self.on_error == "warn":
            print(f"[aither] {message}")
        # "silent" - do nothing

    def _drop_item(self, item: Any, reason: str = "queue full") -> None:
        """Handle dropping an item from the queue."""
        with self._stats_lock:
            self._stats.dropped += 1

        # Throttled warning
        now = time.time()
        if now - self._last_drop_warning >= self._drop_warning_interval:
            self._last_drop_warning = now
            self._handle_error(
                f"Dropping items ({reason}). Queue size: {self.max_queue_size}"
            )

        # Call callback
        if self.on_drop:
            try:
                self.on_drop(item)
            except Exception:
                pass  # Don't let callback errors break the SDK

    def _parse_rate_limit_headers(self, response: httpx.Response) -> None:
        """Parse rate limit headers from response and update usage info.

        Headers parsed:
        - X-RateLimit-Limit: Maximum API calls in period
        - X-RateLimit-Remaining: Calls remaining in period
        - X-RateLimit-Grace-Ends: ISO 8601 timestamp when grace period ends
        - X-Aither-Warning: Warning message (e.g., grace period notice)
        """
        limit: Optional[int] = None
        remaining: Optional[int] = None
        grace_ends_at: Optional[datetime] = None
        warning: Optional[str] = None

        # Parse limit
        limit_header = response.headers.get("X-RateLimit-Limit")
        if limit_header:
            try:
                limit = int(limit_header)
            except ValueError:
                pass  # Ignore malformed header

        # Parse remaining
        remaining_header = response.headers.get("X-RateLimit-Remaining")
        if remaining_header:
            try:
                remaining = int(remaining_header)
            except ValueError:
                pass

        # Parse grace period end (ISO 8601)
        grace_header = response.headers.get("X-RateLimit-Grace-Ends")
        if grace_header:
            try:
                # Try parsing with timezone info
                grace_ends_at = datetime.fromisoformat(
                    grace_header.replace("Z", "+00:00")
                )
            except ValueError:
                pass

        # Parse warning
        warning = response.headers.get("X-Aither-Warning")

        # Update usage info atomically
        with self._usage_info_lock:
            self._usage_info = UsageInfo(
                limit=limit,
                remaining=remaining,
                grace_ends_at=grace_ends_at,
                warning=warning,
            )

        # Handle grace period warning (log once, not every request)
        if warning:
            with self._grace_warned_lock:
                if not self._grace_warned:
                    self._grace_warned = True
                    # Log the warning (only once)
                    if self.on_error != "silent":
                        print(f"[aither] Warning: {warning}")
        else:
            # Warning cleared, reset so we warn again if it comes back
            with self._grace_warned_lock:
                self._grace_warned = False

    def _build_otlp_request(self, spans: list[PredictionSpan]) -> bytes:
        """Build OTLP ExportTraceServiceRequest protobuf."""
        # Import here to avoid loading protobuf at module level
        from opentelemetry.proto.collector.trace.v1.trace_service_pb2 import (
            ExportTraceServiceRequest,
        )
        from opentelemetry.proto.common.v1.common_pb2 import AnyValue, KeyValue
        from opentelemetry.proto.trace.v1.trace_pb2 import (
            ResourceSpans,
            ScopeSpans,
            Span,
        )

        otlp_spans = []
        for ps in spans:
            attributes = [
                KeyValue(
                    key="ml.model.name", value=AnyValue(string_value=ps.model_name)
                ),
                KeyValue(
                    key="ml.features",
                    value=AnyValue(string_value=json.dumps(ps.features)),
                ),
                KeyValue(
                    key="ml.prediction",
                    value=AnyValue(string_value=json.dumps(ps.prediction)),
                ),
            ]

            if ps.version:
                attributes.append(
                    KeyValue(
                        key="ml.model.version", value=AnyValue(string_value=ps.version)
                    )
                )
            if ps.probabilities:
                attributes.append(
                    KeyValue(
                        key="ml.prediction.probabilities",
                        value=AnyValue(string_value=json.dumps(ps.probabilities)),
                    )
                )
            if ps.classes:
                attributes.append(
                    KeyValue(
                        key="ml.prediction.classes",
                        value=AnyValue(string_value=json.dumps(ps.classes)),
                    )
                )
            if ps.environment:
                attributes.append(
                    KeyValue(
                        key="ml.environment",
                        value=AnyValue(string_value=ps.environment),
                    )
                )
            if ps.request_id:
                attributes.append(
                    KeyValue(
                        key="ml.request_id", value=AnyValue(string_value=ps.request_id)
                    )
                )
            if ps.user_id:
                attributes.append(
                    KeyValue(key="ml.user_id", value=AnyValue(string_value=ps.user_id))
                )

            span = Span(
                trace_id=ps.trace_id,
                span_id=ps.span_id,
                name=ps.model_name,
                start_time_unix_nano=ps.start_time_ns,
                end_time_unix_nano=ps.end_time_ns,
                attributes=attributes,
            )
            otlp_spans.append(span)

        request = ExportTraceServiceRequest(
            resource_spans=[ResourceSpans(scope_spans=[ScopeSpans(spans=otlp_spans)])]
        )

        return request.SerializeToString()

    def _send_with_retry(self, payload: bytes, items: list[Any]) -> bool:
        """Send payload with retry and exponential backoff.

        Returns True if successful, False if all retries exhausted.
        """
        for attempt in range(MAX_RETRIES + 1):
            try:
                with httpx.Client(timeout=self.timeout) as client:
                    response = client.post(
                        f"{self.endpoint}/v1/traces",
                        content=payload,
                        headers=self._build_headers("application/x-protobuf"),
                    )

                    # Handle rate limiting
                    if response.status_code == 429:
                        # Parse rate limit headers even on 429
                        self._parse_rate_limit_headers(response)

                        retry_after = response.headers.get("Retry-After")
                        if retry_after:
                            try:
                                wait_time = float(retry_after)
                            except ValueError:
                                wait_time = 60.0  # Default if header is malformed
                        else:
                            wait_time = 60.0
                        self._retry_after_until = time.time() + wait_time
                        # Don't drop items, they'll be retried after the wait
                        return False

                    # Success
                    if response.status_code < 400:
                        # Parse rate limit headers from response
                        self._parse_rate_limit_headers(response)

                        # Print first success message
                        if not self._first_success_printed:
                            self._first_success_printed = True
                            print(
                                f"[aither] Prediction logged. View at {self.endpoint}/dashboard"
                            )

                        with self._stats_lock:
                            self._stats.sent += len(items)

                        # Call on_send callback
                        if self.on_send:
                            try:
                                self.on_send(items)
                            except Exception:
                                pass

                        return True

                    # Server error - retry with backoff
                    if response.status_code >= 500:
                        if attempt < MAX_RETRIES:
                            with self._stats_lock:
                                self._stats.retries += 1
                            time.sleep(BACKOFF_SECONDS[attempt])
                            continue
                        else:
                            # Exhausted retries
                            for item in items:
                                self._drop_item(item, "retries exhausted")
                            return False

                    # Client error (4xx) - don't retry, just drop
                    self._handle_error(f"Send failed: HTTP {response.status_code}")
                    for item in items:
                        self._drop_item(item, f"HTTP {response.status_code}")
                    return False

            except httpx.TimeoutException:
                if attempt < MAX_RETRIES:
                    with self._stats_lock:
                        self._stats.retries += 1
                    time.sleep(BACKOFF_SECONDS[attempt])
                    continue
                else:
                    for item in items:
                        self._drop_item(item, "timeout")
                    return False
            except httpx.HTTPStatusError as e:
                # Don't retry HTTP status errors (they were already handled above)
                self._handle_error(f"Send failed: {e}")
                for item in items:
                    self._drop_item(item, str(e))
                return False
            except Exception as e:
                # Network errors, connection errors - retry
                self._handle_error(f"Send failed: {e}")
                if attempt < MAX_RETRIES:
                    with self._stats_lock:
                        self._stats.retries += 1
                    time.sleep(BACKOFF_SECONDS[attempt])
                    continue
                else:
                    for item in items:
                        self._drop_item(item, str(e))
                    return False

        return False

    def _flush_predictions(self) -> None:
        """Flush predictions from queue to API."""
        spans_to_send: list[PredictionSpan] = []

        with self._queue_lock:
            while self._prediction_queue and len(spans_to_send) < self.batch_size:
                spans_to_send.append(self._prediction_queue.popleft())
            # Update queued count
            with self._stats_lock:
                self._stats.queued = len(self._prediction_queue) + len(
                    self._label_queue
                )

        if not spans_to_send:
            return

        # Build and send OTLP request
        payload = self._build_otlp_request(spans_to_send)
        self._send_with_retry(payload, spans_to_send)

    def _flush_labels(self) -> None:
        """Flush label updates from queue to API."""
        labels_to_send: list[LabelUpdate] = []

        with self._queue_lock:
            while self._label_queue and len(labels_to_send) < self.batch_size:
                labels_to_send.append(self._label_queue.popleft())
            # Update queued count
            with self._stats_lock:
                self._stats.queued = len(self._prediction_queue) + len(
                    self._label_queue
                )

        if not labels_to_send:
            return

        # Build OTLP request with label spans
        from opentelemetry.proto.collector.trace.v1.trace_service_pb2 import (
            ExportTraceServiceRequest,
        )
        from opentelemetry.proto.common.v1.common_pb2 import AnyValue, KeyValue
        from opentelemetry.proto.trace.v1.trace_pb2 import (
            ResourceSpans,
            ScopeSpans,
            Span,
        )

        otlp_spans = []
        now_ns = time.time_ns()

        for label in labels_to_send:
            # Decode trace_id from hex string
            trace_id = bytes.fromhex(label.trace_id)
            span_id = self._generate_span_id()

            attributes = [
                KeyValue(
                    key="ml.label", value=AnyValue(string_value=json.dumps(label.label))
                ),
            ]

            span = Span(
                trace_id=trace_id,
                span_id=span_id,
                name="label_update",
                start_time_unix_nano=now_ns,
                end_time_unix_nano=now_ns,
                attributes=attributes,
            )
            otlp_spans.append(span)

        request = ExportTraceServiceRequest(
            resource_spans=[ResourceSpans(scope_spans=[ScopeSpans(spans=otlp_spans)])]
        )

        payload = request.SerializeToString()
        self._send_with_retry(payload, labels_to_send)

    def log_prediction(
        self,
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
        """Log a model prediction (non-blocking).

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
        trace_id = self._generate_trace_id()
        span_id = self._generate_span_id()
        now_ns = time.time_ns()

        span = PredictionSpan(
            trace_id=trace_id,
            span_id=span_id,
            model_name=model_name,
            features=features,
            prediction=prediction,
            version=version,
            probabilities=probabilities,
            classes=classes,
            environment=environment,
            request_id=request_id,
            user_id=user_id,
            start_time_ns=now_ns,
            end_time_ns=now_ns,
        )

        trace_id_hex = trace_id.hex()

        # Track this trace_id for label validation
        with self._trace_ids_lock:
            self._known_trace_ids.add(trace_id_hex)
            # Limit memory usage - keep only recent trace_ids
            if len(self._known_trace_ids) > self.max_queue_size * 2:
                # Remove oldest (this is approximate, set doesn't maintain order)
                to_remove = len(self._known_trace_ids) - self.max_queue_size
                for _ in range(to_remove):
                    self._known_trace_ids.pop()

        if self.enable_background:
            with self._queue_lock:
                # Drop oldest if queue full
                if len(self._prediction_queue) >= self.max_queue_size:
                    dropped = self._prediction_queue.popleft()
                    self._drop_item(dropped, "queue full")

                self._prediction_queue.append(span)

                with self._stats_lock:
                    self._stats.queued = len(self._prediction_queue) + len(
                        self._label_queue
                    )
        else:
            # Immediate mode: block and send synchronously
            payload = self._build_otlp_request([span])
            self._send_with_retry(payload, [span])

        return trace_id_hex

    def log_label(self, trace_id: str, label: Any) -> None:
        """Log ground truth label for a previous prediction (non-blocking).

        Use the trace_id returned from log_prediction() to correlate
        the ground truth with the original prediction.

        Args:
            trace_id: The trace_id returned from log_prediction().
            label: The actual outcome/ground truth value.
        """
        # Validate trace_id format
        try:
            bytes.fromhex(trace_id)
        except ValueError:
            with self._stats_lock:
                self._stats.label_misses += 1
            self._handle_error(f"Invalid trace_id format: {trace_id}")
            return

        # Check if we've seen this trace_id (soft check - not exhaustive)
        with self._trace_ids_lock:
            if trace_id not in self._known_trace_ids:
                with self._stats_lock:
                    self._stats.label_misses += 1
                # Don't raise, just warn (per spec)
                if self.on_error == "warn":
                    print(
                        f"[aither] Warning: trace_id {trace_id[:8]}... not found in recent predictions"
                    )
                # Still try to send - backend might have it

        update = LabelUpdate(trace_id=trace_id, label=label)

        if self.enable_background:
            with self._queue_lock:
                # Drop oldest if queue full
                if len(self._label_queue) >= self.max_queue_size:
                    dropped = self._label_queue.popleft()
                    self._drop_item(dropped, "queue full")

                self._label_queue.append(update)

                with self._stats_lock:
                    self._stats.queued = len(self._prediction_queue) + len(
                        self._label_queue
                    )
        else:
            # Immediate mode: build and send synchronously
            from opentelemetry.proto.collector.trace.v1.trace_service_pb2 import (
                ExportTraceServiceRequest,
            )
            from opentelemetry.proto.common.v1.common_pb2 import AnyValue, KeyValue
            from opentelemetry.proto.trace.v1.trace_pb2 import (
                ResourceSpans,
                ScopeSpans,
                Span,
            )

            now_ns = time.time_ns()
            trace_id_bytes = bytes.fromhex(trace_id)
            span_id = self._generate_span_id()

            span = Span(
                trace_id=trace_id_bytes,
                span_id=span_id,
                name="label_update",
                start_time_unix_nano=now_ns,
                end_time_unix_nano=now_ns,
                attributes=[
                    KeyValue(
                        key="ml.label", value=AnyValue(string_value=json.dumps(label))
                    ),
                ],
            )

            request = ExportTraceServiceRequest(
                resource_spans=[ResourceSpans(scope_spans=[ScopeSpans(spans=[span])])]
            )

            payload = request.SerializeToString()
            self._send_with_retry(payload, [update])

    def flush(self, timeout: float | None = None) -> None:
        """Force immediate flush of queued predictions and labels (blocking).

        Args:
            timeout: Maximum time to wait for flush. Raises TimeoutError if exceeded.

        Raises:
            TimeoutError: If flush doesn't complete within timeout.
        """
        start_time = time.time()

        # Signal the background worker to flush immediately
        if self.enable_background:
            self._flush_event.set()

        # Keep flushing until queues are empty
        while True:
            with self._queue_lock:
                queue_empty = (
                    len(self._prediction_queue) == 0 and len(self._label_queue) == 0
                )

            if queue_empty:
                break

            if timeout is not None:
                elapsed = time.time() - start_time
                if elapsed >= timeout:
                    raise TimeoutError(
                        f"Flush did not complete within {timeout}s. "
                        f"Remaining: {len(self._prediction_queue)} predictions, "
                        f"{len(self._label_queue)} labels"
                    )

            # Do a flush iteration
            self._flush_predictions()
            self._flush_labels()

            # Brief sleep to avoid tight loop
            time.sleep(0.01)

    def stats(self) -> Stats:
        """Get current SDK statistics.

        Returns:
            Stats object with queued, sent, dropped, retries, label_misses counts.
        """
        with self._stats_lock:
            with self._queue_lock:
                self._stats.queued = len(self._prediction_queue) + len(
                    self._label_queue
                )
            return Stats(
                queued=self._stats.queued,
                sent=self._stats.sent,
                dropped=self._stats.dropped,
                retries=self._stats.retries,
                label_misses=self._stats.label_misses,
            )

    def usage_info(self) -> UsageInfo:
        """Get the most recent rate limit and usage information from the server.

        This information is updated after each successful API request.
        Until a request is made, all fields will be None.

        Returns:
            UsageInfo object with limit, remaining, grace_ends_at, and warning.

        Example:
            usage = client.usage_info()
            if usage.remaining is not None and usage.remaining < 100:
                print(f"Low on API calls: {usage.remaining} remaining")
            if usage.warning:
                print(f"Warning: {usage.warning}")
        """
        with self._usage_info_lock:
            return UsageInfo(
                limit=self._usage_info.limit,
                remaining=self._usage_info.remaining,
                grace_ends_at=self._usage_info.grace_ends_at,
                warning=self._usage_info.warning,
            )

    def health(self) -> bool:
        """Check if the API is healthy.

        Returns:
            True if the API is healthy.
        """
        with httpx.Client(timeout=self.timeout) as client:
            response = client.get(f"{self.endpoint}/health")
            return response.status_code == 200

    def batch(self) -> BatchContext:
        """Create a batch context for high-throughput logging.

        Usage:
            with client.batch() as batch:
                for item in dataset:
                    batch.log(model_name="classifier", features=item, prediction=output)
            # Access trace_ids after batch completes
            trace_ids = batch.trace_ids

        Returns:
            BatchContext that can be used as a context manager.
        """
        return BatchContext(_client=self)

    def trace(
        self,
        model_name: str,
        *,
        version: str | None = None,
        environment: str | None = None,
    ) -> TraceContext:
        """Create a trace context for dynamic model names.

        Usage:
            with client.trace("model_" + version) as t:
                prediction = model.predict(features)
                t.log(features=features, prediction=prediction)
                # t.trace_id available for label correlation

        Args:
            model_name: Identifier for the model.
            version: Model version string.
            environment: Deployment environment.

        Returns:
            TraceContext that can be used as a context manager.
        """
        return TraceContext(
            _client=self,
            model_name=model_name,
            version=version,
            environment=environment,
        )

    def close(self) -> None:
        """Close the client and flush remaining data."""
        if not self.enable_background:
            return

        # Signal worker to stop
        self._stop_event.set()
        self._flush_event.set()  # Wake up the worker

        # Flush any remaining data
        try:
            self.flush(timeout=5.0)
        except TimeoutError:
            pass  # Best effort

        # Wait for worker thread to finish
        if self._worker_thread and self._worker_thread.is_alive():
            self._worker_thread.join(timeout=5.0)

    def __enter__(self) -> AitherClient:
        return self

    def __exit__(self, *args: Any) -> None:
        self.close()
