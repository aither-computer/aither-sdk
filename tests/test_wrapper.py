"""Tests for aither.wrap() and TrackedModel.

Tests model wrapping for different frameworks:
- sklearn (and sklearn-compatible)
- pytorch
- tensorflow
- tinygrad (mocked)

Tests feature extraction for different data types:
- numpy arrays
- pandas DataFrames
- polars DataFrames
- torch Tensors
- dicts
"""

import pytest
from unittest.mock import MagicMock, patch
import numpy as np

import aither
from aither.wrapper import wrap, detect_framework, detect_methods
from aither.extractors import (
    extract_features,
    extract_numpy,
    extract_pandas_dataframe,
    extract_torch_tensor,
    extract_dict,
    extract_list,
    ExtractionConfig,
)


# =============================================================================
# Mock Models for Testing
# =============================================================================


class MockSklearnModel:
    """Mock sklearn-style model."""

    __module__ = "sklearn.ensemble"

    def predict(self, X):
        return np.array([1, 0, 1])

    def predict_proba(self, X):
        return np.array([[0.2, 0.8], [0.9, 0.1], [0.3, 0.7]])

    @property
    def classes_(self):
        return np.array([0, 1])


class MockPytorchModel:
    """Mock pytorch-style model."""

    __module__ = "torch.nn.modules.container"

    def __call__(self, x):
        # Simulate returning a tensor-like result
        return {"logits": [0.1, 0.9]}

    def forward(self, x):
        return self(x)


class MockTensorflowModel:
    """Mock tensorflow-style model."""

    __module__ = "tensorflow.python.keras.engine.training"

    def predict(self, X):
        return np.array([[0.8], [0.2], [0.6]])

    def __call__(self, X):
        # TF models have both - __call__ returns Tensor
        return {"output": [0.8, 0.2, 0.6]}


class MockTinygradModel:
    """Mock tinygrad-style model."""

    __module__ = "tinygrad.nn"

    def __call__(self, x):
        return [0.5, 0.5]


class MockTransformersModel:
    """Mock huggingface transformers model."""

    __module__ = "transformers.models.bert.modeling_bert"

    def __call__(self, input_ids=None, attention_mask=None):
        return {"logits": [[0.1, 0.9]]}

    def generate(self, input_ids=None, max_length=50):
        return [[101, 102, 103]]


class MockCallableOnly:
    """Model with only __call__, no predict."""

    __module__ = "custom_module"

    def __call__(self, x):
        return x * 2


class MockNeitherMethod:
    """Model with no predict or __call__."""

    __module__ = "custom_module"

    def run(self, x):
        return x


# =============================================================================
# Framework Detection Tests
# =============================================================================


class TestFrameworkDetection:
    """Test detect_framework() for different model types."""

    def test_sklearn_detection(self):
        model = MockSklearnModel()
        assert detect_framework(model) == "sklearn"

    def test_pytorch_detection(self):
        model = MockPytorchModel()
        assert detect_framework(model) == "pytorch"

    def test_tensorflow_detection(self):
        model = MockTensorflowModel()
        assert detect_framework(model) == "tensorflow"

    def test_tinygrad_detection(self):
        model = MockTinygradModel()
        assert detect_framework(model) == "tinygrad"

    def test_transformers_detection(self):
        model = MockTransformersModel()
        assert detect_framework(model) == "transformers"

    def test_unknown_framework(self):
        model = MockNeitherMethod()
        assert detect_framework(model) == "unknown"


class TestMethodDetection:
    """Test detect_methods() for different model types."""

    def test_sklearn_methods(self):
        model = MockSklearnModel()
        methods = detect_methods(model, "sklearn")
        assert "predict" in methods
        assert "predict_proba" in methods

    def test_pytorch_methods(self):
        model = MockPytorchModel()
        methods = detect_methods(model, "pytorch")
        assert "__call__" in methods
        assert "forward" in methods

    def test_tensorflow_methods(self):
        model = MockTensorflowModel()
        methods = detect_methods(model, "tensorflow")
        assert "predict" in methods
        assert "__call__" in methods

    def test_transformers_methods(self):
        model = MockTransformersModel()
        methods = detect_methods(model, "transformers")
        assert "__call__" in methods
        assert "generate" in methods


# =============================================================================
# TrackedModel Tests
# =============================================================================


class TestTrackedModel:
    """Test TrackedModel wrapper."""

    def test_wrap_sklearn_model(self):
        """wrap() should work with sklearn models."""
        model = MockSklearnModel()
        tracked = wrap(model, name="test_model")

        assert tracked._name == "test_model"
        assert tracked.framework == "sklearn"
        assert "predict" in tracked.tracked_methods

    def test_wrap_pytorch_model(self):
        """wrap() should work with pytorch models."""
        model = MockPytorchModel()
        tracked = wrap(model, name="test_model")

        assert tracked.framework == "pytorch"
        assert "__call__" in tracked.tracked_methods

    def test_wrap_with_version(self):
        """wrap() should accept version parameter."""
        model = MockSklearnModel()
        tracked = wrap(model, name="test", version="1.2.3")

        assert tracked._version == "1.2.3"

    def test_wrap_with_environment(self):
        """wrap() should accept environment parameter."""
        model = MockSklearnModel()
        tracked = wrap(model, name="test", environment="production")

        assert tracked._environment == "production"

    def test_wrap_with_custom_methods(self):
        """wrap() should accept custom methods list."""
        model = MockSklearnModel()
        tracked = wrap(model, name="test", methods=["predict"])

        assert tracked.tracked_methods == ["predict"]

    def test_wrap_fails_without_methods(self):
        """wrap() should raise error if no methods detected."""
        model = MockNeitherMethod()

        with pytest.raises(ValueError, match="Could not detect"):
            wrap(model, name="test")

    def test_predict_logs_automatically(self):
        """Calling predict should log via aither.log_prediction()."""
        model = MockSklearnModel()
        tracked = wrap(model, name="test_model")

        with patch("aither.log_prediction") as mock_log:
            mock_log.return_value = "abc123"

            X = np.array([[1, 2], [3, 4], [5, 6]])
            result = tracked.predict(X)

            # Should return original result
            assert np.array_equal(result, np.array([1, 0, 1]))

            # Should have logged
            mock_log.assert_called_once()
            call_kwargs = mock_log.call_args.kwargs
            assert call_kwargs["model_name"] == "test_model"
            assert "features" in call_kwargs
            assert "prediction" in call_kwargs

    def test_call_logs_automatically(self):
        """Calling __call__ should log for pytorch-style models."""
        model = MockPytorchModel()
        tracked = wrap(model, name="test_model")

        with patch("aither.log_prediction") as mock_log:
            mock_log.return_value = "abc123"

            result = tracked({"x": 1})

            assert result == {"logits": [0.1, 0.9]}
            mock_log.assert_called_once()

    def test_last_trace_id_set(self):
        """last_trace_id should be set after prediction."""
        model = MockSklearnModel()
        tracked = wrap(model, name="test_model")

        with patch("aither.log_prediction") as mock_log:
            mock_log.return_value = "trace123"

            tracked.predict(np.array([[1, 2]]))

            assert tracked.last_trace_id == "trace123"

    def test_trace_ids_accumulated(self):
        """trace_ids should accumulate across multiple predictions."""
        model = MockSklearnModel()
        tracked = wrap(model, name="test_model")

        with patch("aither.log_prediction") as mock_log:
            mock_log.side_effect = ["trace1", "trace2", "trace3"]

            tracked.predict(np.array([[1]]))
            tracked.predict(np.array([[2]]))
            tracked.predict(np.array([[3]]))

            assert tracked.trace_ids == ["trace1", "trace2", "trace3"]
            assert tracked.last_trace_id == "trace3"

    def test_passthrough_attributes(self):
        """TrackedModel should pass through model attributes."""
        model = MockSklearnModel()
        tracked = wrap(model, name="test_model")

        # Should be able to access model attributes
        assert np.array_equal(tracked.classes_, np.array([0, 1]))

    def test_model_property(self):
        """model property should return underlying model."""
        model = MockSklearnModel()
        tracked = wrap(model, name="test_model")

        assert tracked.model is model

    def test_repr(self):
        """TrackedModel should have useful repr."""
        model = MockSklearnModel()
        tracked = wrap(model, name="fraud_detector")

        r = repr(tracked)
        assert "TrackedModel" in r
        assert "fraud_detector" in r
        assert "sklearn" in r


class TestTrackedModelWithCustomFeaturesFn:
    """Test TrackedModel with custom features_fn."""

    def test_custom_features_fn(self):
        """Custom features_fn should be used for extraction."""
        model = MockSklearnModel()

        def custom_fn(X):
            return {"custom": "features", "shape": list(X.shape)}

        tracked = wrap(model, name="test", features_fn=custom_fn)

        with patch("aither.log_prediction") as mock_log:
            mock_log.return_value = "abc123"

            X = np.array([[1, 2], [3, 4]])
            tracked.predict(X)

            call_kwargs = mock_log.call_args.kwargs
            assert call_kwargs["features"] == {"custom": "features", "shape": [2, 2]}


# =============================================================================
# Feature Extraction Tests
# =============================================================================


class TestNumpyExtraction:
    """Test feature extraction from numpy arrays."""

    def test_basic_extraction(self):
        """Should extract sample and metadata from numpy array."""
        X = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
        config = ExtractionConfig(sample_rows=2)

        result = extract_numpy(X, config)

        assert "sample" in result
        assert "_meta" in result
        assert result["_meta"]["type"] == "numpy.ndarray"
        assert result["_meta"]["shape"] == [3, 3]
        assert result["_meta"]["dtype"] == "int64"
        assert len(result["sample"]) == 2  # Only 2 rows sampled

    def test_truncation_flag(self):
        """Should set truncated=True when data is truncated."""
        X = np.array([[1, 2], [3, 4], [5, 6], [7, 8], [9, 10]])
        config = ExtractionConfig(sample_rows=2)

        result = extract_numpy(X, config)

        assert result["_meta"]["truncated"] is True

    def test_column_sampling(self):
        """Should sample columns for wide arrays."""
        X = np.array([[1] * 20])  # 1 row, 20 columns
        config = ExtractionConfig(sample_columns=5)

        result = extract_numpy(X, config)

        assert len(result["sample"][0]) == 5


class TestPandasExtraction:
    """Test feature extraction from pandas DataFrames."""

    def test_basic_extraction(self):
        """Should extract sample and metadata from DataFrame."""
        pd = pytest.importorskip("pandas")

        df = pd.DataFrame({"a": [1, 2, 3], "b": [4, 5, 6], "c": [7, 8, 9]})
        config = ExtractionConfig(sample_rows=2)

        result = extract_pandas_dataframe(df, config)

        assert result["_meta"]["type"] == "pandas.DataFrame"
        assert result["_meta"]["shape"] == [3, 3]
        assert result["_meta"]["columns"] == ["a", "b", "c"]
        assert "dtypes" in result["_meta"]
        assert len(result["sample"]) == 2

    def test_column_sampling(self):
        """Should sample columns for wide DataFrames."""
        pd = pytest.importorskip("pandas")

        df = pd.DataFrame({f"col_{i}": [i] for i in range(20)})
        config = ExtractionConfig(sample_columns=5)

        result = extract_pandas_dataframe(df, config)

        assert len(result["sample"][0]) == 5


class TestTorchExtraction:
    """Test feature extraction from torch Tensors."""

    def test_basic_extraction(self):
        """Should extract sample and metadata from Tensor."""
        torch = pytest.importorskip("torch")

        X = torch.tensor([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]])
        config = ExtractionConfig(sample_rows=2)

        result = extract_torch_tensor(X, config)

        assert result["_meta"]["type"] == "torch.Tensor"
        assert result["_meta"]["shape"] == [3, 2]
        assert "dtype" in result["_meta"]
        assert "device" in result["_meta"]
        assert len(result["sample"]) == 2


class TestDictExtraction:
    """Test feature extraction from dicts."""

    def test_basic_extraction(self):
        """Should extract nested dict contents."""
        X = {"input_ids": [1, 2, 3], "attention_mask": [1, 1, 1]}
        config = ExtractionConfig()

        result = extract_dict(X, config)

        assert result["_meta"]["type"] == "dict"
        assert result["_meta"]["keys"] == ["input_ids", "attention_mask"]
        assert "sample" in result

    def test_nested_tensors(self):
        """Should handle nested tensors in dict."""
        torch = pytest.importorskip("torch")

        X = {"input_ids": torch.tensor([1, 2, 3]), "values": [0.1, 0.2]}
        config = ExtractionConfig()

        result = extract_dict(X, config)

        assert "input_ids" in result["sample"]
        assert result["sample"]["input_ids"]["_meta"]["type"] == "torch.Tensor"


class TestListExtraction:
    """Test feature extraction from lists."""

    def test_basic_extraction(self):
        """Should extract sample from list."""
        X = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
        config = ExtractionConfig(sample_rows=3)

        result = extract_list(X, config)

        assert result["_meta"]["type"] == "list"
        assert result["_meta"]["length"] == 10
        assert len(result["sample"]) == 3
        assert result["_meta"]["truncated"] is True


class TestExtractFeatures:
    """Test the unified extract_features() function."""

    def test_numpy(self):
        """Should handle numpy arrays."""
        X = np.array([1, 2, 3])
        result = extract_features(X)
        assert result["_meta"]["type"] == "numpy.ndarray"

    def test_dict(self):
        """Should handle dicts."""
        X = {"key": "value"}
        result = extract_features(X)
        assert result["_meta"]["type"] == "dict"

    def test_list(self):
        """Should handle lists."""
        X = [1, 2, 3]
        result = extract_features(X)
        assert result["_meta"]["type"] == "list"

    def test_primitive(self):
        """Should handle primitives."""
        result = extract_features(42)
        assert result["_meta"]["type"] == "int"
        assert result["value"] == 42

    def test_unknown_type(self):
        """Should handle unknown types gracefully."""

        class CustomClass:
            pass

        result = extract_features(CustomClass())
        assert "_meta" in result
        assert "type" in result["_meta"]


# =============================================================================
# Integration Tests
# =============================================================================


class TestModuleLevelWrap:
    """Test aither.wrap() at module level."""

    def test_wrap_exported(self):
        """wrap should be exported from aither module."""
        assert hasattr(aither, "wrap")
        assert callable(aither.wrap)

    def test_tracked_model_exported(self):
        """TrackedModel should be exported from aither module."""
        assert hasattr(aither, "TrackedModel")

    def test_full_flow(self):
        """Test complete wrap -> predict -> log flow."""
        model = MockSklearnModel()

        # Initialize aither
        aither.init(api_key="test_key", endpoint="http://test")

        try:
            tracked = aither.wrap(model, name="test_model")

            with patch.object(aither._get_client(), "_prediction_queue") as mock_queue:
                mock_queue.append = MagicMock()

                X = np.array([[1, 2], [3, 4]])
                result = tracked.predict(X)

                assert result is not None
                assert tracked.last_trace_id is not None
        finally:
            aither.close()


# =============================================================================
# Edge Cases
# =============================================================================


class TestEdgeCases:
    """Test edge cases and error handling."""

    def test_empty_input(self):
        """Should handle empty arrays."""
        X = np.array([])
        result = extract_features(X)
        assert result["_meta"]["shape"] == [0]

    def test_very_large_input(self):
        """Should truncate very large inputs."""
        X = np.random.randn(10000, 100)
        config = ExtractionConfig(sample_rows=5, sample_columns=10, max_bytes=1000)

        result = extract_numpy(X, config)

        assert result["_meta"]["truncated"] is True
        assert len(result["sample"]) <= 5

    def test_model_without_predict(self):
        """Should fall back to __call__ if no predict."""
        model = MockCallableOnly()
        tracked = wrap(model, name="test", methods=["__call__"])

        with patch("aither.log_prediction") as mock_log:
            mock_log.return_value = "abc"

            result = tracked(5)

            assert result == 10  # 5 * 2
            mock_log.assert_called_once()


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
