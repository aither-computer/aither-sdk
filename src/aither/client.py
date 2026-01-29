"""Aither client implementation using OTLP for model prediction logging."""

from __future__ import annotations

import atexit
import json
import os
import secrets
import threading
import time
from collections import deque
from dataclasses import dataclass
from typing import Any

import httpx

DEFAULT_ENDPOINT = "https://aither.computer"
DEFAULT_FLUSH_INTERVAL = 1.0  # seconds
DEFAULT_BATCH_SIZE = 100


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


class AitherClient:
    """Client for the Aither platform API.

    Logs ML model predictions using OTLP (OpenTelemetry Protocol) format.
    Predictions are sent as spans with ml.* attributes.
    """

    def __init__(
        self,
        api_key: str | None = None,
        endpoint: str | None = None,
        timeout: float = 30.0,
        flush_interval: float = DEFAULT_FLUSH_INTERVAL,
        batch_size: int = DEFAULT_BATCH_SIZE,
        enable_background: bool = True,
    ) -> None:
        """Initialize the Aither client.

        Args:
            api_key: API key for authentication. Falls back to AITHER_API_KEY env var.
            endpoint: API endpoint URL. Falls back to AITHER_ENDPOINT env var or default.
            timeout: Request timeout in seconds.
            flush_interval: How often to flush queued predictions (seconds).
            batch_size: Maximum predictions per batch request.
            enable_background: If False, predictions are sent immediately (blocking).
        """
        self.api_key = api_key or os.environ.get("AITHER_API_KEY")
        self.endpoint = (
            endpoint or os.environ.get("AITHER_ENDPOINT") or DEFAULT_ENDPOINT
        )
        self.timeout = timeout
        self.flush_interval = flush_interval
        self.batch_size = batch_size
        self.enable_background = enable_background

        # Thread-safe queue for predictions and labels
        self._prediction_queue: deque[PredictionSpan] = deque()
        self._label_queue: deque[LabelUpdate] = deque()
        self._queue_lock = threading.Lock()

        # Background worker management
        self._worker_thread: threading.Thread | None = None
        self._stop_event = threading.Event()

        # Start background worker
        if self.enable_background:
            self._start_worker()
            atexit.register(self.close)

    def _build_headers(
        self, content_type: str = "application/x-protobuf"
    ) -> dict[str, str]:
        """Build request headers."""
        headers = {"Content-Type": content_type}
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
                self._flush_predictions()
                self._flush_labels()
            except Exception as e:
                # Log errors but keep the worker running
                print(f"Error flushing queue: {e}")

            # Wait for flush interval or stop event
            self._stop_event.wait(timeout=self.flush_interval)

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

    def _flush_predictions(self) -> None:
        """Flush predictions from queue to API."""
        spans_to_send: list[PredictionSpan] = []

        with self._queue_lock:
            while self._prediction_queue and len(spans_to_send) < self.batch_size:
                spans_to_send.append(self._prediction_queue.popleft())

        if not spans_to_send:
            return

        # Build and send OTLP request
        payload = self._build_otlp_request(spans_to_send)

        with httpx.Client(timeout=self.timeout) as client:
            response = client.post(
                f"{self.endpoint}/v1/traces",
                content=payload,
                headers=self._build_headers("application/x-protobuf"),
            )
            response.raise_for_status()

    def _flush_labels(self) -> None:
        """Flush label updates from queue to API."""
        labels_to_send: list[LabelUpdate] = []

        with self._queue_lock:
            while self._label_queue and len(labels_to_send) < self.batch_size:
                labels_to_send.append(self._label_queue.popleft())

        if not labels_to_send:
            return

        # Build OTLP request with label spans
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

        with httpx.Client(timeout=self.timeout) as client:
            response = client.post(
                f"{self.endpoint}/v1/traces",
                content=payload,
                headers=self._build_headers("application/x-protobuf"),
            )
            response.raise_for_status()

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

        if self.enable_background:
            with self._queue_lock:
                self._prediction_queue.append(span)
        else:
            # Immediate mode: block and send synchronously
            payload = self._build_otlp_request([span])
            with httpx.Client(timeout=self.timeout) as client:
                response = client.post(
                    f"{self.endpoint}/v1/traces",
                    content=payload,
                    headers=self._build_headers("application/x-protobuf"),
                )
                response.raise_for_status()

        return trace_id_hex

    def log_label(self, trace_id: str, label: Any) -> None:
        """Log ground truth label for a previous prediction (non-blocking).

        Use the trace_id returned from log_prediction() to correlate
        the ground truth with the original prediction.

        Args:
            trace_id: The trace_id returned from log_prediction().
            label: The actual outcome/ground truth value.
        """
        update = LabelUpdate(trace_id=trace_id, label=label)

        if self.enable_background:
            with self._queue_lock:
                self._label_queue.append(update)
        else:
            # Immediate mode: build and send synchronously
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

            with httpx.Client(timeout=self.timeout) as client:
                response = client.post(
                    f"{self.endpoint}/v1/traces",
                    content=payload,
                    headers=self._build_headers("application/x-protobuf"),
                )
                response.raise_for_status()

    def flush(self) -> None:
        """Force immediate flush of queued predictions and labels (blocking).

        Useful for ensuring data is sent before shutdown or in tests.
        """
        self._flush_predictions()
        self._flush_labels()

    def health(self) -> bool:
        """Check if the API is healthy.

        Returns:
            True if the API is healthy.
        """
        with httpx.Client(timeout=self.timeout) as client:
            response = client.get(f"{self.endpoint}/health")
            return response.status_code == 200

    def close(self) -> None:
        """Close the client and flush remaining data."""
        if not self.enable_background:
            return

        # Signal worker to stop
        self._stop_event.set()

        # Flush any remaining data
        try:
            self.flush()
        except Exception:
            pass  # Best effort

        # Wait for worker thread to finish
        if self._worker_thread and self._worker_thread.is_alive():
            self._worker_thread.join(timeout=5.0)

    def __enter__(self) -> AitherClient:
        return self

    def __exit__(self, *args: Any) -> None:
        self.close()
