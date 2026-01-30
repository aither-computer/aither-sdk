"""Model wrapper for automatic prediction logging.

Wraps ML models from various frameworks (sklearn, pytorch, tensorflow, tinygrad)
and automatically logs predictions via aither.log_prediction().
"""

from __future__ import annotations

from typing import Any, Callable, Optional

from aither.extractors import (
    DEFAULT_MAX_BYTES,
    DEFAULT_SAMPLE_COLUMNS,
    DEFAULT_SAMPLE_ROWS,
    ExtractionConfig,
    extract_features,
    extract_from_args_kwargs,
)

# Framework detection strings
SKLEARN_MODULES = ("sklearn", "xgboost", "lightgbm", "catboost")
PYTORCH_MODULES = ("torch",)
PYTORCH_LIGHTNING_MODULES = ("pytorch_lightning", "lightning")
TENSORFLOW_MODULES = ("tensorflow", "keras")
TINYGRAD_MODULES = ("tinygrad",)
JAX_MODULES = ("jax", "flax")
TRANSFORMERS_MODULES = ("transformers",)

# Methods to look for, in order of preference
PREDICT_METHODS = [
    "predict",
    "predict_proba",
    "predict_step",
    "__call__",
    "forward",
    "generate",
    "encode",
]


def detect_framework(model: Any) -> str:
    """Detect which ML framework a model belongs to.

    Args:
        model: The model object

    Returns:
        Framework name: 'sklearn', 'pytorch', 'tensorflow', 'tinygrad', 'jax', 'transformers', or 'unknown'
    """
    module = type(model).__module__

    if any(module.startswith(m) for m in SKLEARN_MODULES):
        return "sklearn"

    if any(module.startswith(m) for m in PYTORCH_LIGHTNING_MODULES):
        return "pytorch_lightning"

    if any(module.startswith(m) for m in PYTORCH_MODULES):
        return "pytorch"

    if any(module.startswith(m) for m in TENSORFLOW_MODULES):
        return "tensorflow"

    if any(module.startswith(m) for m in TINYGRAD_MODULES):
        return "tinygrad"

    if any(module.startswith(m) for m in JAX_MODULES):
        return "jax"

    if any(module.startswith(m) for m in TRANSFORMERS_MODULES):
        return "transformers"

    return "unknown"


def detect_methods(model: Any, framework: str) -> list[str]:
    """Detect which prediction methods a model has.

    Args:
        model: The model object
        framework: Detected framework name

    Returns:
        List of method names to wrap
    """
    methods = []

    # Check for each method in preference order
    for method in PREDICT_METHODS:
        if hasattr(model, method) and callable(getattr(model, method)):
            # For __call__, check it's not just object's default
            if method == "__call__":
                # Most ML models override __call__ to do inference
                if framework in (
                    "pytorch",
                    "pytorch_lightning",
                    "tensorflow",
                    "tinygrad",
                    "transformers",
                ):
                    methods.append(method)
            else:
                methods.append(method)

    return methods


class TrackedModel:
    """Wraps an ML model to automatically log predictions.

    Supports sklearn, pytorch, tensorflow, tinygrad, and other frameworks.
    Intercepts predict-like methods and logs via aither.log_prediction().

    Example:
        import aither
        from sklearn.ensemble import RandomForestClassifier

        model = RandomForestClassifier().fit(X_train, y_train)
        tracked = aither.wrap(model, name="fraud_detector")

        result = tracked.predict(X_test)  # Automatically logged
        trace_id = tracked.last_trace_id

    Attributes:
        last_trace_id: The trace_id from the most recent prediction call.
            Use this to correlate ground truth labels.
    """

    def __init__(
        self,
        model: Any,
        name: str,
        *,
        version: Optional[str] = None,
        environment: Optional[str] = None,
        sample_rows: int = DEFAULT_SAMPLE_ROWS,
        sample_columns: int = DEFAULT_SAMPLE_COLUMNS,
        max_bytes: int = DEFAULT_MAX_BYTES,
        methods: Optional[list[str]] = None,
        features_fn: Optional[Callable[..., dict]] = None,
        track_all_methods: bool = True,
    ):
        """Initialize a tracked model wrapper.

        Args:
            model: The underlying ML model to wrap.
            name: Model name for logging (e.g., "fraud_detector").
            version: Model version string (e.g., "1.2.3", git sha).
            environment: Deployment environment (e.g., "production", "staging").
            sample_rows: Max rows to include in feature sample (default: 5).
            sample_columns: Max columns to include in feature sample (default: 10).
            max_bytes: Max bytes for serialized features (default: 5000).
            methods: Specific methods to wrap. If None, auto-detects.
            features_fn: Custom function to extract features from args/kwargs.
                Signature: features_fn(*args, **kwargs) -> dict
            track_all_methods: If True, track predict, predict_proba, etc.
                If False, only track the primary method.
        """
        self._model = model
        self._name = name
        self._version = version
        self._environment = environment
        self._features_fn = features_fn
        self._track_all_methods = track_all_methods

        # Extraction config
        self._extraction_config = ExtractionConfig(
            sample_rows=sample_rows,
            sample_columns=sample_columns,
            max_bytes=max_bytes,
        )

        # Detect framework
        self._framework = detect_framework(model)

        # Detect or use provided methods
        if methods is not None:
            self._methods = methods
        else:
            detected = detect_methods(model, self._framework)
            if track_all_methods:
                self._methods = detected
            else:
                # Just the primary method
                self._methods = detected[:1] if detected else []

        if not self._methods:
            raise ValueError(
                f"Could not detect any prediction methods on model. "
                f"Model type: {type(model).__name__}, Framework: {self._framework}. "
                f"Either add a .predict() method or specify methods=['your_method']."
            )

        # Last trace_id from prediction
        self.last_trace_id: Optional[str] = None

        # Trace IDs from all predictions (for batch scenarios)
        self.trace_ids: list[str] = []

    def _extract_features(self, args: tuple, kwargs: dict) -> dict:
        """Extract features from method arguments."""
        if self._features_fn is not None:
            return self._features_fn(*args, **kwargs)
        return extract_from_args_kwargs(args, kwargs, self._extraction_config)

    def _extract_prediction(self, result: Any) -> dict:
        """Extract prediction output."""
        return extract_features(result, self._extraction_config)

    def _log_prediction(self, features: dict, prediction: dict, method: str) -> str:
        """Log prediction via aither.log_prediction()."""
        # Import here to avoid circular imports
        import aither

        trace_id = aither.log_prediction(
            model_name=self._name,
            features=features,
            prediction=prediction,
            version=self._version,
            environment=self._environment,
        )

        self.last_trace_id = trace_id
        self.trace_ids.append(trace_id)

        return trace_id

    def _make_tracked_method(self, method_name: str) -> Callable:
        """Create a tracked version of a method."""
        original_method = getattr(self._model, method_name)

        def tracked_method(*args, **kwargs):
            # Call original method
            result = original_method(*args, **kwargs)

            # Extract and log
            features = self._extract_features(args, kwargs)
            prediction = self._extract_prediction(result)
            self._log_prediction(features, prediction, method_name)

            return result

        # Preserve docstring and name
        tracked_method.__doc__ = original_method.__doc__
        tracked_method.__name__ = method_name

        return tracked_method

    def predict(self, *args, **kwargs) -> Any:
        """Call predict on the underlying model and log.

        This method always exists on TrackedModel, even if the underlying
        model uses __call__ or another method. It routes to the appropriate
        underlying method.
        """
        # If model has predict, use it
        if hasattr(self._model, "predict"):
            return self._make_tracked_method("predict")(*args, **kwargs)

        # Otherwise use __call__
        if callable(self._model):
            return self._make_tracked_method("__call__")(*args, **kwargs)

        raise AttributeError(
            f"Model {type(self._model).__name__} has no predict() or __call__() method"
        )

    def predict_proba(self, *args, **kwargs) -> Any:
        """Call predict_proba on the underlying model and log."""
        if not hasattr(self._model, "predict_proba"):
            raise AttributeError(
                f"Model {type(self._model).__name__} has no predict_proba() method"
            )
        return self._make_tracked_method("predict_proba")(*args, **kwargs)

    def predict_step(self, *args, **kwargs) -> Any:
        """Call predict_step on the underlying model and log (PyTorch Lightning)."""
        if not hasattr(self._model, "predict_step"):
            raise AttributeError(
                f"Model {type(self._model).__name__} has no predict_step() method"
            )
        return self._make_tracked_method("predict_step")(*args, **kwargs)

    def generate(self, *args, **kwargs) -> Any:
        """Call generate on the underlying model and log (transformers)."""
        if not hasattr(self._model, "generate"):
            raise AttributeError(
                f"Model {type(self._model).__name__} has no generate() method"
            )
        return self._make_tracked_method("generate")(*args, **kwargs)

    def encode(self, *args, **kwargs) -> Any:
        """Call encode on the underlying model and log (sentence-transformers)."""
        if not hasattr(self._model, "encode"):
            raise AttributeError(
                f"Model {type(self._model).__name__} has no encode() method"
            )
        return self._make_tracked_method("encode")(*args, **kwargs)

    def __call__(self, *args, **kwargs) -> Any:
        """Call the model directly and log.

        For pytorch, tensorflow, tinygrad models that use __call__ for inference.
        Also works as unified interface - tracked(X) always works.
        """
        # If model is callable, use __call__
        if callable(self._model):
            return self._make_tracked_method("__call__")(*args, **kwargs)

        # Otherwise fall back to predict
        return self.predict(*args, **kwargs)

    def __getattr__(self, name: str) -> Any:
        """Pass through attribute access to the underlying model.

        This allows accessing model attributes like .classes_, .feature_importances_,
        .coef_, etc. directly on the wrapped model.
        """
        # Avoid infinite recursion for our own attributes
        if name.startswith("_"):
            raise AttributeError(
                f"'{type(self).__name__}' object has no attribute '{name}'"
            )
        return getattr(self._model, name)

    def __repr__(self) -> str:
        return (
            f"TrackedModel(name={self._name!r}, "
            f"framework={self._framework!r}, "
            f"methods={self._methods!r}, "
            f"model={type(self._model).__name__})"
        )

    @property
    def model(self) -> Any:
        """Access the underlying model directly."""
        return self._model

    @property
    def framework(self) -> str:
        """The detected framework of the underlying model."""
        return self._framework

    @property
    def tracked_methods(self) -> list[str]:
        """List of methods being tracked on this model."""
        return self._methods.copy()


def wrap(
    model: Any,
    name: str,
    *,
    version: Optional[str] = None,
    environment: Optional[str] = None,
    sample_rows: int = DEFAULT_SAMPLE_ROWS,
    sample_columns: int = DEFAULT_SAMPLE_COLUMNS,
    max_bytes: int = DEFAULT_MAX_BYTES,
    methods: Optional[list[str]] = None,
    features_fn: Optional[Callable[..., dict]] = None,
) -> TrackedModel:
    """Wrap an ML model for automatic prediction logging.

    This is the recommended way to use aither with existing models.
    The wrapped model behaves exactly like the original, but automatically
    logs predictions to aither.

    Supports:
        - sklearn (and sklearn-compatible: xgboost, lightgbm, catboost)
        - pytorch
        - pytorch lightning
        - tensorflow/keras
        - tinygrad
        - huggingface transformers

    Example:
        import aither
        from sklearn.ensemble import RandomForestClassifier

        # Train your model
        model = RandomForestClassifier().fit(X_train, y_train)

        # Wrap it
        tracked = aither.wrap(model, name="fraud_detector", version="1.0.0")

        # Use normally - predictions are logged automatically
        predictions = tracked.predict(X_test)

        # Get trace_id for label correlation
        trace_id = tracked.last_trace_id
        aither.log_label(trace_id, actual_labels)

    Args:
        model: The ML model to wrap.
        name: Model name for logging (e.g., "fraud_detector").
        version: Model version string (e.g., "1.2.3").
        environment: Deployment environment (e.g., "production").
        sample_rows: Max rows to sample from features (default: 5).
        sample_columns: Max columns to sample from features (default: 10).
        max_bytes: Max serialized size for features (default: 5000 bytes).
        methods: Specific methods to track. Auto-detects if None.
        features_fn: Custom function to extract features.
            Signature: features_fn(*args, **kwargs) -> dict

    Returns:
        TrackedModel that wraps the original model.
    """
    return TrackedModel(
        model=model,
        name=name,
        version=version,
        environment=environment,
        sample_rows=sample_rows,
        sample_columns=sample_columns,
        max_bytes=max_bytes,
        methods=methods,
        features_fn=features_fn,
    )
