"""Integration tests for Aither SDK.

These tests verify that the SDK correctly:
1. Builds OTLP protobuf messages with ml.* attributes
2. Sends requests to the backend
3. Handles batching and background workers
4. Correlates labels with trace IDs

Run with: pytest tests/test_integration.py -v
"""

import time
import pytest
from unittest.mock import patch, MagicMock

import aither
from aither.client import AitherClient, PredictionSpan, LabelUpdate


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
