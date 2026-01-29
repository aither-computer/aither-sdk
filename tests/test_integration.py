"""Integration tests for Aither SDK.

These tests verify that the SDK correctly:
1. Builds OTLP protobuf messages with ml.* attributes
2. Sends requests to the backend
3. Handles batching and background workers
4. Correlates labels with trace IDs
5. Parses rate limit headers from responses

Run with: pytest tests/test_integration.py -v
"""

import time
from datetime import datetime, timezone
import pytest
from unittest.mock import patch, MagicMock

import aither
from aither.client import (
    AitherClient,
    PredictionSpan,
    LabelUpdate,
    UsageInfo,
    SDK_VERSION,
    SDK_LANGUAGE,
)


class TestOtlpMessageBuilding:
    """Test that SDK builds correct OTLP messages."""

    def test_log_prediction_returns_trace_id(self):
        """log_prediction should return a valid hex trace ID."""
        client = AitherClient(
            api_key="test_key",
            endpoint="http://localhost:8080",
            enable_background=False,
        )

        # Mock the HTTP client to prevent actual requests
        with patch("httpx.Client") as mock_client:
            mock_response = MagicMock()
            mock_response.status_code = 200
            mock_response.headers = {}  # Must be a dict, not MagicMock
            mock_response.raise_for_status = MagicMock()
            mock_client.return_value.__enter__.return_value.post.return_value = (
                mock_response
            )

            trace_id = client.log_prediction(
                model_name="test_model",
                features={"x": 1, "y": 2},
                prediction=0.95,
            )

            # Verify trace_id is a valid 32-character hex string (128 bits)
            assert len(trace_id) == 32
            assert all(c in "0123456789abcdef" for c in trace_id)

    def test_prediction_span_has_required_fields(self):
        """PredictionSpan should have all required ml.* fields."""
        import secrets

        span = PredictionSpan(
            trace_id=secrets.token_bytes(16),
            span_id=secrets.token_bytes(8),
            model_name="fraud_detector",
            features={"amount": 150.0, "country": "US"},
            prediction=0.87,
            version="1.2.3",
            environment="production",
        )

        assert span.model_name == "fraud_detector"
        assert span.features == {"amount": 150.0, "country": "US"}
        assert span.prediction == 0.87
        assert span.version == "1.2.3"
        assert span.environment == "production"

    def test_build_otlp_request_creates_valid_protobuf(self):
        """_build_otlp_request should create valid OTLP protobuf."""
        client = AitherClient(
            api_key="test_key",
            enable_background=False,
        )

        import secrets

        spans = [
            PredictionSpan(
                trace_id=secrets.token_bytes(16),
                span_id=secrets.token_bytes(8),
                model_name="test_model",
                features={"x": 1},
                prediction=0.5,
                start_time_ns=time.time_ns(),
                end_time_ns=time.time_ns(),
            )
        ]

        payload = client._build_otlp_request(spans)

        # Verify it's valid protobuf that can be decoded
        from opentelemetry.proto.collector.trace.v1.trace_service_pb2 import (
            ExportTraceServiceRequest,
        )

        request = ExportTraceServiceRequest()
        request.ParseFromString(payload)

        assert len(request.resource_spans) == 1
        assert len(request.resource_spans[0].scope_spans) == 1
        assert len(request.resource_spans[0].scope_spans[0].spans) == 1

        span = request.resource_spans[0].scope_spans[0].spans[0]
        assert span.name == "test_model"

        # Check ml.* attributes are present
        attr_keys = [attr.key for attr in span.attributes]
        assert "ml.model.name" in attr_keys
        assert "ml.features" in attr_keys
        assert "ml.prediction" in attr_keys


class TestLabelCorrelation:
    """Test trace ID correlation for ground truth labels."""

    def test_log_label_uses_trace_id(self):
        """log_label should use the trace_id from log_prediction."""
        client = AitherClient(
            api_key="test_key",
            endpoint="http://localhost:8080",
            enable_background=False,
        )

        with patch("httpx.Client") as mock_client:
            mock_response = MagicMock()
            mock_response.status_code = 200
            mock_response.headers = {}  # Must be a dict, not MagicMock
            mock_response.raise_for_status = MagicMock()
            mock_client.return_value.__enter__.return_value.post.return_value = (
                mock_response
            )

            # Get trace_id from prediction
            trace_id = client.log_prediction(
                model_name="test_model",
                features={"x": 1},
                prediction=0.5,
            )

            # Log label with same trace_id
            client.log_label(trace_id=trace_id, label=1)

            # Verify both calls were made
            assert mock_client.return_value.__enter__.return_value.post.call_count == 2

    def test_label_update_has_correct_trace_id(self):
        """LabelUpdate should preserve the trace_id."""
        trace_id = "abcdef1234567890abcdef1234567890"
        label = LabelUpdate(trace_id=trace_id, label="positive")

        assert label.trace_id == trace_id
        assert label.label == "positive"


class TestBackgroundWorker:
    """Test background worker and batching."""

    def test_predictions_are_queued(self):
        """Predictions should be queued in background mode."""
        client = AitherClient(
            api_key="test_key",
            enable_background=True,
            flush_interval=10.0,  # Don't auto-flush during test
        )

        try:
            # Log predictions without making HTTP requests
            for i in range(5):
                client.log_prediction(
                    model_name="test_model",
                    features={"i": i},
                    prediction=i * 0.1,
                )

            # Check queue has predictions
            assert len(client._prediction_queue) == 5
        finally:
            client._stop_event.set()  # Stop worker
            client.close()

    def test_flush_clears_queue(self):
        """flush() should send all queued predictions."""
        client = AitherClient(
            api_key="test_key",
            enable_background=True,
            flush_interval=10.0,
        )

        try:
            # Queue predictions
            for i in range(3):
                client.log_prediction(
                    model_name="test_model",
                    features={"i": i},
                    prediction=i * 0.1,
                )

            assert len(client._prediction_queue) == 3

            # Mock flush to prevent HTTP errors
            with patch.object(client, "_flush_predictions") as mock_flush:
                mock_flush.side_effect = lambda: client._prediction_queue.clear()
                client.flush()

            assert len(client._prediction_queue) == 0
        finally:
            client._stop_event.set()
            client.close()


class TestModuleLevelAPI:
    """Test the module-level convenience API."""

    def test_init_creates_global_client(self):
        """aither.init() should create a global client."""
        aither.init(api_key="test_key", endpoint="http://test")

        # Access internal client
        client = aither._get_client()
        assert client is not None
        assert client.api_key == "test_key"
        assert client.endpoint == "http://test"

        aither.close()

    def test_log_prediction_uses_global_client(self):
        """aither.log_prediction() should use the global client."""
        aither.init(
            api_key="test_key",
            endpoint="http://localhost:8080",
        )

        try:
            with patch.object(aither._get_client(), "_prediction_queue") as mock_queue:
                mock_queue.append = MagicMock()

                trace_id = aither.log_prediction(
                    model_name="test",
                    features={"x": 1},
                    prediction=0.5,
                )

                assert trace_id is not None
                assert len(trace_id) == 32
        finally:
            aither.close()


class TestRateLimitHeaders:
    """Test rate limit header parsing and usage info."""

    def test_usage_info_default_values(self):
        """UsageInfo should have None for all fields by default."""
        usage = UsageInfo()
        assert usage.limit is None
        assert usage.remaining is None
        assert usage.grace_ends_at is None
        assert usage.warning is None

    def test_parse_rate_limit_headers(self):
        """Client should parse rate limit headers from responses."""
        client = AitherClient(
            api_key="test_key",
            endpoint="http://localhost:8080",
            enable_background=False,
        )

        with patch("httpx.Client") as mock_client:
            mock_response = MagicMock()
            mock_response.status_code = 200
            mock_response.headers = {
                "X-RateLimit-Limit": "10000",
                "X-RateLimit-Remaining": "9500",
            }
            mock_client.return_value.__enter__.return_value.post.return_value = (
                mock_response
            )

            client.log_prediction(
                model_name="test_model",
                features={"x": 1},
                prediction=0.5,
            )

            usage = client.usage_info()
            assert usage.limit == 10000
            assert usage.remaining == 9500
            assert usage.grace_ends_at is None
            assert usage.warning is None

    def test_parse_grace_period_header(self):
        """Client should parse grace period end timestamp."""
        client = AitherClient(
            api_key="test_key",
            endpoint="http://localhost:8080",
            enable_background=False,
        )

        with patch("httpx.Client") as mock_client:
            mock_response = MagicMock()
            mock_response.status_code = 200
            mock_response.headers = {
                "X-RateLimit-Limit": "10000",
                "X-RateLimit-Remaining": "0",
                "X-RateLimit-Grace-Ends": "2026-01-31T12:00:00Z",
                "X-Aither-Warning": "Grace period active, upgrade to continue",
            }
            mock_client.return_value.__enter__.return_value.post.return_value = (
                mock_response
            )

            client.log_prediction(
                model_name="test_model",
                features={"x": 1},
                prediction=0.5,
            )

            usage = client.usage_info()
            assert usage.limit == 10000
            assert usage.remaining == 0
            assert usage.grace_ends_at == datetime(
                2026, 1, 31, 12, 0, 0, tzinfo=timezone.utc
            )
            assert usage.warning == "Grace period active, upgrade to continue"

    def test_grace_warning_logged_once(self, capsys):
        """Grace period warning should only be logged once."""
        client = AitherClient(
            api_key="test_key",
            endpoint="http://localhost:8080",
            enable_background=False,
            on_error="warn",  # Enable warning output
        )

        with patch("httpx.Client") as mock_client:
            mock_response = MagicMock()
            mock_response.status_code = 200
            mock_response.headers = {
                "X-Aither-Warning": "Grace period started",
            }
            mock_client.return_value.__enter__.return_value.post.return_value = (
                mock_response
            )

            # First request - should log warning
            client.log_prediction(
                model_name="test_model",
                features={"x": 1},
                prediction=0.5,
            )

            # Second request - should NOT log warning again
            client.log_prediction(
                model_name="test_model",
                features={"x": 2},
                prediction=0.6,
            )

            captured = capsys.readouterr()
            # Should only see the warning once (plus the first success message)
            warning_count = captured.out.count("Grace period started")
            assert warning_count == 1

    def test_warning_resets_when_cleared(self, capsys):
        """Warning should reset when server clears it, allowing future warns."""
        client = AitherClient(
            api_key="test_key",
            endpoint="http://localhost:8080",
            enable_background=False,
            on_error="warn",
        )

        with patch("httpx.Client") as mock_client:
            # First: response with warning
            mock_response_warning = MagicMock()
            mock_response_warning.status_code = 200
            mock_response_warning.headers = {
                "X-Aither-Warning": "Grace period started",
            }

            # Second: response without warning (user upgraded)
            mock_response_clear = MagicMock()
            mock_response_clear.status_code = 200
            mock_response_clear.headers = {}

            # Third: response with warning again
            mock_response_warning2 = MagicMock()
            mock_response_warning2.status_code = 200
            mock_response_warning2.headers = {
                "X-Aither-Warning": "Grace period started again",
            }

            mock_client.return_value.__enter__.return_value.post.side_effect = [
                mock_response_warning,
                mock_response_clear,
                mock_response_warning2,
            ]

            # Request 1: logs warning
            client.log_prediction(model_name="test", features={"x": 1}, prediction=0.5)
            # Request 2: no warning (cleared)
            client.log_prediction(model_name="test", features={"x": 2}, prediction=0.6)
            # Request 3: should log warning again
            client.log_prediction(model_name="test", features={"x": 3}, prediction=0.7)

            captured = capsys.readouterr()
            # Should see two warnings (one for each time it was set)
            assert "Grace period started" in captured.out
            assert "Grace period started again" in captured.out

    def test_sdk_version_headers_sent(self):
        """SDK should send version and language headers."""
        client = AitherClient(
            api_key="test_key",
            endpoint="http://localhost:8080",
            enable_background=False,
        )

        with patch("httpx.Client") as mock_client:
            mock_response = MagicMock()
            mock_response.status_code = 200
            mock_response.headers = {}
            mock_client.return_value.__enter__.return_value.post.return_value = (
                mock_response
            )

            client.log_prediction(
                model_name="test_model",
                features={"x": 1},
                prediction=0.5,
            )

            # Check the headers that were sent
            call_args = mock_client.return_value.__enter__.return_value.post.call_args
            headers = call_args.kwargs.get("headers", {})

            assert headers.get("X-Aither-SDK-Version") == SDK_VERSION
            assert headers.get("X-Aither-SDK-Language") == SDK_LANGUAGE

    def test_module_level_usage_info(self):
        """aither.usage_info() should work with global client."""
        aither.init(
            api_key="test_key",
            endpoint="http://localhost:8080",
        )

        try:
            with patch("httpx.Client") as mock_client:
                mock_response = MagicMock()
                mock_response.status_code = 200
                mock_response.headers = {
                    "X-RateLimit-Limit": "5000",
                    "X-RateLimit-Remaining": "4999",
                }
                mock_client.return_value.__enter__.return_value.post.return_value = (
                    mock_response
                )

                # Make a request to trigger header parsing
                # Use the client directly since module-level might not have flushed
                client = aither._get_client()
                client._send_with_retry(b"test", [])

                usage = aither.usage_info()
                assert usage.limit == 5000
                assert usage.remaining == 4999
        finally:
            aither.close()

    def test_parse_headers_on_429(self):
        """Rate limit headers should be parsed even on 429 responses."""
        client = AitherClient(
            api_key="test_key",
            endpoint="http://localhost:8080",
            enable_background=False,
        )

        with patch("httpx.Client") as mock_client:
            mock_response = MagicMock()
            mock_response.status_code = 429
            mock_response.headers = {
                "X-RateLimit-Limit": "10000",
                "X-RateLimit-Remaining": "0",
                "Retry-After": "60",
                "X-Aither-Warning": "Rate limit exceeded",
            }
            mock_client.return_value.__enter__.return_value.post.return_value = (
                mock_response
            )

            client.log_prediction(
                model_name="test_model",
                features={"x": 1},
                prediction=0.5,
            )

            usage = client.usage_info()
            assert usage.limit == 10000
            assert usage.remaining == 0
            assert usage.warning == "Rate limit exceeded"


class TestBatchContext:
    """Test batch context manager."""

    def test_batch_collects_trace_ids(self):
        """Batch context should collect trace_ids from all logged predictions."""
        client = AitherClient(
            api_key="test_key",
            enable_background=True,
            flush_interval=10.0,
        )

        try:
            with client.batch() as batch:
                batch.log(model_name="m1", features={"x": 1}, prediction=0.1)
                batch.log(model_name="m2", features={"x": 2}, prediction=0.2)
                batch.log(model_name="m3", features={"x": 3}, prediction=0.3)

            assert len(batch.trace_ids) == 3
            for trace_id in batch.trace_ids:
                assert len(trace_id) == 32
                assert all(c in "0123456789abcdef" for c in trace_id)
        finally:
            client._stop_event.set()
            client.close()

    def test_module_level_batch(self):
        """aither.batch() should work with global client."""
        aither.init(api_key="test_key", endpoint="http://test")

        try:
            with aither.batch() as batch:
                batch.log(model_name="test", features={"x": 1}, prediction=0.5)

            assert len(batch.trace_ids) == 1
        finally:
            aither.close()


class TestTraceContext:
    """Test trace context manager."""

    def test_trace_captures_trace_id(self):
        """Trace context should expose trace_id after logging."""
        client = AitherClient(
            api_key="test_key",
            enable_background=True,
            flush_interval=10.0,
        )

        try:
            with client.trace("my_model", version="1.0") as t:
                t.log(features={"x": 1}, prediction=0.5)

            assert t.trace_id is not None
            assert len(t.trace_id) == 32
        finally:
            client._stop_event.set()
            client.close()

    def test_module_level_trace(self):
        """aither.trace() should work with global client."""
        aither.init(api_key="test_key", endpoint="http://test")

        try:
            with aither.trace("dynamic_model") as t:
                t.log(features={"x": 1}, prediction=0.5)

            assert t.trace_id is not None
        finally:
            aither.close()


class TestQueueBehavior:
    """Test queue full, drop oldest, and stats tracking."""

    def test_queue_drops_oldest_when_full(self):
        """When queue is full, oldest items should be dropped."""
        dropped_items = []

        client = AitherClient(
            api_key="test_key",
            enable_background=True,
            flush_interval=10.0,
            max_queue_size=3,
            on_drop=lambda item: dropped_items.append(item),
        )

        try:
            # Fill queue beyond max_queue_size
            for i in range(5):
                client.log_prediction(
                    model_name="test",
                    features={"i": i},
                    prediction=i * 0.1,
                )

            # Queue should be at max size
            assert len(client._prediction_queue) == 3
            # Should have dropped 2 items (oldest)
            assert len(dropped_items) == 2
            # Stats should reflect drops
            stats = client.stats()
            assert stats.dropped == 2
        finally:
            client._stop_event.set()
            client.close()

    def test_on_drop_callback_called(self):
        """on_drop callback should be called when items are dropped."""
        dropped = []
        client = AitherClient(
            api_key="test_key",
            enable_background=True,
            flush_interval=10.0,
            max_queue_size=1,
            on_drop=lambda item: dropped.append(item),
        )

        try:
            client.log_prediction(model_name="t", features={}, prediction=1)
            client.log_prediction(model_name="t", features={}, prediction=2)

            assert len(dropped) == 1
        finally:
            client._stop_event.set()
            client.close()


class TestRetryBehavior:
    """Test retry on 5xx and backoff timing."""

    def test_retry_on_5xx_increments_stats(self):
        """5xx responses should trigger retries with stats tracking."""
        client = AitherClient(
            api_key="test_key",
            endpoint="http://localhost:8080",
            enable_background=False,
        )

        with patch("httpx.Client") as mock_client:
            mock_response = MagicMock()
            mock_response.status_code = 500
            mock_response.headers = {}
            mock_client.return_value.__enter__.return_value.post.return_value = (
                mock_response
            )

            # Patch time.sleep to avoid waiting
            with patch("time.sleep"):
                client.log_prediction(
                    model_name="test",
                    features={"x": 1},
                    prediction=0.5,
                )

            # Should have retried 3 times
            stats = client.stats()
            assert stats.retries == 3
            # Should have dropped the item after exhausting retries
            assert stats.dropped == 1

    def test_max_retries_exhausted_calls_on_drop(self):
        """After max retries, on_drop should be called for the batch."""
        dropped = []
        client = AitherClient(
            api_key="test_key",
            endpoint="http://localhost:8080",
            enable_background=False,
            on_drop=lambda item: dropped.append(item),
        )

        with patch("httpx.Client") as mock_client:
            mock_response = MagicMock()
            mock_response.status_code = 500
            mock_response.headers = {}
            mock_client.return_value.__enter__.return_value.post.return_value = (
                mock_response
            )

            with patch("time.sleep"):
                client.log_prediction(
                    model_name="test",
                    features={"x": 1},
                    prediction=0.5,
                )

            assert len(dropped) == 1


class TestRateLimitBehavior:
    """Test 429 handling with Retry-After."""

    def test_429_respects_retry_after(self):
        """429 response should set retry_after_until."""
        client = AitherClient(
            api_key="test_key",
            endpoint="http://localhost:8080",
            enable_background=False,
        )

        with patch("httpx.Client") as mock_client:
            mock_response = MagicMock()
            mock_response.status_code = 429
            mock_response.headers = {"Retry-After": "30"}
            mock_client.return_value.__enter__.return_value.post.return_value = (
                mock_response
            )

            before = time.time()
            client.log_prediction(
                model_name="test",
                features={"x": 1},
                prediction=0.5,
            )

            # retry_after_until should be set to ~30 seconds from now
            assert client._retry_after_until > before + 25
            assert client._retry_after_until < before + 35


class TestLabelMisses:
    """Test log_label with invalid trace_id."""

    def test_log_label_bad_trace_id_increments_misses(self):
        """log_label with unknown trace_id should increment label_misses."""
        client = AitherClient(
            api_key="test_key",
            enable_background=True,
            flush_interval=10.0,
            on_error="silent",  # Don't print warnings
        )

        try:
            # Log label with trace_id we never created
            client.log_label(trace_id="00000000000000000000000000000000", label=1)

            stats = client.stats()
            assert stats.label_misses == 1
        finally:
            client._stop_event.set()
            client.close()

    def test_log_label_invalid_format_increments_misses(self):
        """log_label with invalid trace_id format should increment label_misses."""
        client = AitherClient(
            api_key="test_key",
            enable_background=True,
            flush_interval=10.0,
            on_error="silent",
        )

        try:
            # Invalid hex format
            client.log_label(trace_id="not-valid-hex", label=1)

            stats = client.stats()
            assert stats.label_misses == 1
        finally:
            client._stop_event.set()
            client.close()


class TestConcurrency:
    """Test thread-safety of log_prediction and log_label."""

    def test_concurrent_log_prediction(self):
        """Multiple threads calling log_prediction should not cause race conditions."""
        import threading

        client = AitherClient(
            api_key="test_key",
            enable_background=True,
            flush_interval=10.0,
            max_queue_size=1000,
        )

        results = []
        errors = []

        def log_many(thread_id: int, count: int):
            try:
                for i in range(count):
                    trace_id = client.log_prediction(
                        model_name=f"model_{thread_id}",
                        features={"thread": thread_id, "i": i},
                        prediction=i * 0.1,
                    )
                    results.append(trace_id)
            except Exception as e:
                errors.append(e)

        try:
            threads = [
                threading.Thread(target=log_many, args=(t, 50)) for t in range(10)
            ]

            for t in threads:
                t.start()
            for t in threads:
                t.join()

            # All predictions should be logged without errors
            assert len(errors) == 0
            assert len(results) == 500
            # All trace_ids should be unique
            assert len(set(results)) == 500
        finally:
            client._stop_event.set()
            client.close()


class TestFlushTimeout:
    """Test flush(timeout) behavior."""

    def test_flush_timeout_raises(self):
        """flush() with short timeout should raise TimeoutError if not complete."""
        client = AitherClient(
            api_key="test_key",
            endpoint="http://localhost:8080",
            enable_background=False,
            batch_size=10,  # Small batch size so we need multiple requests
        )

        # Queue many items - needs 10 requests with batch_size=10
        for i in range(100):
            client._prediction_queue.append(
                PredictionSpan(
                    trace_id=client._generate_trace_id(),
                    span_id=client._generate_span_id(),
                    model_name="test",
                    features={"i": i},
                    prediction=i,
                    start_time_ns=time.time_ns(),
                    end_time_ns=time.time_ns(),
                )
            )

        with patch("httpx.Client") as mock_client:
            # Simulate slow network by having post sleep
            def slow_post(*args, **kwargs):
                time.sleep(0.05)  # 50ms per request, 10 requests = 500ms total
                mock_response = MagicMock()
                mock_response.status_code = 200
                mock_response.headers = {}
                return mock_response

            mock_client.return_value.__enter__.return_value.post = slow_post

            # Should timeout (needs 500ms, timeout is 100ms)
            with pytest.raises(TimeoutError):
                client.flush(timeout=0.1)


class TestBaseUrlAlias:
    """Test base_url parameter as alias for endpoint."""

    def test_base_url_parameter(self):
        """base_url should work as alias for endpoint."""
        client = AitherClient(
            api_key="test_key",
            base_url="https://custom.example.com",
            enable_background=False,
        )

        assert client.endpoint == "https://custom.example.com"

    def test_base_url_takes_precedence(self):
        """base_url should take precedence over endpoint."""
        client = AitherClient(
            api_key="test_key",
            base_url="https://base-url.example.com",
            endpoint="https://endpoint.example.com",
            enable_background=False,
        )

        assert client.endpoint == "https://base-url.example.com"

    def test_module_level_base_url(self):
        """aither.init(base_url=...) should work."""
        aither.init(api_key="test_key", base_url="https://example.com")

        try:
            client = aither._get_client()
            assert client.endpoint == "https://example.com"
        finally:
            aither.close()


class TestEndToEndWithMockServer:
    """End-to-end tests with mocked HTTP responses.

    These tests verify the full flow from SDK to (mocked) backend.
    For real integration tests, run against a live server.
    """

    @pytest.mark.skip(reason="Requires live backend server")
    def test_full_prediction_flow(self):
        """Test complete prediction + label flow against live server."""
        aither.init(
            api_key="aith_test_key_with_ingest_traces_scope",
            endpoint="http://localhost:8080",
        )

        # Log prediction
        trace_id = aither.log_prediction(
            model_name="integration_test_model",
            features={"test": True, "timestamp": time.time()},
            prediction=0.42,
            environment="test",
        )

        # Log label
        aither.log_label(trace_id=trace_id, label="actual_value")

        # Flush and verify no errors
        aither.flush()
        aither.close()

        # If we get here without exceptions, the test passed
        # In a real integration test, we'd query the DB to verify


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
