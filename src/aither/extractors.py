"""Feature extraction for different data types.

Extracts sample data + metadata from various ML data formats:
- numpy arrays
- pandas DataFrames
- polars DataFrames
- torch Tensors
- tensorflow Tensors
- jax arrays
- dicts (common in transformers)
- lists
"""

from __future__ import annotations

import json
import sys
from dataclasses import dataclass
from typing import Any, Optional

# Default limits
DEFAULT_SAMPLE_ROWS = 5
DEFAULT_SAMPLE_COLUMNS = 10
DEFAULT_MAX_BYTES = 5000


@dataclass
class ExtractionConfig:
    """Configuration for feature extraction."""

    sample_rows: int = DEFAULT_SAMPLE_ROWS
    sample_columns: int = DEFAULT_SAMPLE_COLUMNS
    max_bytes: int = DEFAULT_MAX_BYTES


def _is_numpy_array(x: Any) -> bool:
    """Check if x is a numpy array without importing numpy."""
    return type(x).__module__ == "numpy" and type(x).__name__ == "ndarray"


def _is_pandas_dataframe(x: Any) -> bool:
    """Check if x is a pandas DataFrame without importing pandas."""
    return type(x).__module__.startswith("pandas") and type(x).__name__ == "DataFrame"


def _is_pandas_series(x: Any) -> bool:
    """Check if x is a pandas Series without importing pandas."""
    return type(x).__module__.startswith("pandas") and type(x).__name__ == "Series"


def _is_polars_dataframe(x: Any) -> bool:
    """Check if x is a polars DataFrame without importing polars."""
    return type(x).__module__.startswith("polars") and type(x).__name__ == "DataFrame"


def _is_polars_series(x: Any) -> bool:
    """Check if x is a polars Series without importing polars."""
    return type(x).__module__.startswith("polars") and type(x).__name__ == "Series"


def _is_torch_tensor(x: Any) -> bool:
    """Check if x is a torch Tensor without importing torch."""
    return type(x).__module__ == "torch" and type(x).__name__ == "Tensor"


def _is_tf_tensor(x: Any) -> bool:
    """Check if x is a TensorFlow tensor without importing tensorflow."""
    module = type(x).__module__
    return module.startswith("tensorflow") and "Tensor" in type(x).__name__


def _is_jax_array(x: Any) -> bool:
    """Check if x is a JAX array without importing jax."""
    module = type(x).__module__
    return module.startswith("jax") or module.startswith("jaxlib")


def _estimate_size(obj: Any) -> int:
    """Estimate serialized size of an object."""
    try:
        return len(json.dumps(obj))
    except (TypeError, ValueError):
        return sys.getsizeof(obj)


def _truncate_to_size(data: Any, max_bytes: int) -> tuple[Any, bool]:
    """Truncate data to fit within max_bytes. Returns (data, was_truncated)."""
    if _estimate_size(data) <= max_bytes:
        return data, False

    if isinstance(data, list):
        # Binary search for how many items fit
        low, high = 0, len(data)
        while low < high:
            mid = (low + high + 1) // 2
            if _estimate_size(data[:mid]) <= max_bytes:
                low = mid
            else:
                high = mid - 1
        return data[:low], low < len(data)

    if isinstance(data, dict):
        # Keep as many keys as fit
        result = {}
        for k, v in data.items():
            result[k] = v
            if _estimate_size(result) > max_bytes:
                del result[k]
                break
        return result, len(result) < len(data)

    return data, False


def extract_numpy(x: Any, config: ExtractionConfig) -> dict[str, Any]:
    """Extract sample + metadata from numpy array."""
    shape = list(x.shape)
    dtype = str(x.dtype)

    # Sample rows
    sample = x[: config.sample_rows]

    # Sample columns if 2D+
    if len(shape) > 1 and shape[1] > config.sample_columns:
        sample = sample[:, : config.sample_columns]

    # Convert to list
    try:
        sample_list = sample.tolist()
    except (TypeError, ValueError):
        sample_list = []

    sample_list, truncated_by_size = _truncate_to_size(sample_list, config.max_bytes)

    return {
        "sample": sample_list,
        "_meta": {
            "type": "numpy.ndarray",
            "shape": shape,
            "dtype": dtype,
            "sample_rows": min(config.sample_rows, shape[0]) if shape else 0,
            "truncated": shape[0] > config.sample_rows or truncated_by_size,
        },
    }


def extract_pandas_dataframe(x: Any, config: ExtractionConfig) -> dict[str, Any]:
    """Extract sample + metadata from pandas DataFrame."""
    shape = list(x.shape)
    columns = list(x.columns)
    dtypes = {str(k): str(v) for k, v in x.dtypes.items()}

    # Sample rows and columns
    sample_df = x.head(config.sample_rows)
    if len(columns) > config.sample_columns:
        sample_df = sample_df.iloc[:, : config.sample_columns]

    try:
        sample_list = sample_df.to_dict(orient="records")
    except (TypeError, ValueError):
        sample_list = []

    sample_list, truncated_by_size = _truncate_to_size(sample_list, config.max_bytes)

    return {
        "sample": sample_list,
        "_meta": {
            "type": "pandas.DataFrame",
            "shape": shape,
            "columns": columns,
            "dtypes": dtypes,
            "sample_rows": min(config.sample_rows, shape[0]),
            "sample_columns": min(config.sample_columns, len(columns)),
            "truncated": shape[0] > config.sample_rows
            or len(columns) > config.sample_columns
            or truncated_by_size,
        },
    }


def extract_pandas_series(x: Any, config: ExtractionConfig) -> dict[str, Any]:
    """Extract sample + metadata from pandas Series."""
    length = len(x)
    dtype = str(x.dtype)
    name = x.name

    sample = x.head(config.sample_rows)
    try:
        sample_list = sample.tolist()
    except (TypeError, ValueError):
        sample_list = []

    sample_list, truncated_by_size = _truncate_to_size(sample_list, config.max_bytes)

    return {
        "sample": sample_list,
        "_meta": {
            "type": "pandas.Series",
            "length": length,
            "dtype": dtype,
            "name": str(name) if name is not None else None,
            "sample_size": min(config.sample_rows, length),
            "truncated": length > config.sample_rows or truncated_by_size,
        },
    }


def extract_polars_dataframe(x: Any, config: ExtractionConfig) -> dict[str, Any]:
    """Extract sample + metadata from polars DataFrame."""
    shape = x.shape
    columns = x.columns
    dtypes = {col: str(x[col].dtype) for col in columns}

    # Sample rows and columns
    sample_df = x.head(config.sample_rows)
    if len(columns) > config.sample_columns:
        sample_df = sample_df.select(columns[: config.sample_columns])

    try:
        sample_list = sample_df.to_dicts()
    except (TypeError, ValueError):
        sample_list = []

    sample_list, truncated_by_size = _truncate_to_size(sample_list, config.max_bytes)

    return {
        "sample": sample_list,
        "_meta": {
            "type": "polars.DataFrame",
            "shape": list(shape),
            "columns": columns,
            "dtypes": dtypes,
            "sample_rows": min(config.sample_rows, shape[0]),
            "sample_columns": min(config.sample_columns, len(columns)),
            "truncated": shape[0] > config.sample_rows
            or len(columns) > config.sample_columns
            or truncated_by_size,
        },
    }


def extract_polars_series(x: Any, config: ExtractionConfig) -> dict[str, Any]:
    """Extract sample + metadata from polars Series."""
    length = len(x)
    dtype = str(x.dtype)
    name = x.name

    sample = x.head(config.sample_rows)
    try:
        sample_list = sample.to_list()
    except (TypeError, ValueError):
        sample_list = []

    sample_list, truncated_by_size = _truncate_to_size(sample_list, config.max_bytes)

    return {
        "sample": sample_list,
        "_meta": {
            "type": "polars.Series",
            "length": length,
            "dtype": dtype,
            "name": name,
            "sample_size": min(config.sample_rows, length),
            "truncated": length > config.sample_rows or truncated_by_size,
        },
    }


def extract_torch_tensor(x: Any, config: ExtractionConfig) -> dict[str, Any]:
    """Extract sample + metadata from torch Tensor."""
    shape = list(x.shape)
    dtype = str(x.dtype)
    device = str(x.device)

    # Sample rows
    sample = x[: config.sample_rows]

    # Sample columns if 2D+
    if len(shape) > 1 and shape[1] > config.sample_columns:
        sample = sample[:, : config.sample_columns]

    # Move to CPU and convert
    try:
        sample = sample.detach().cpu()
        sample_list = sample.tolist()
    except (TypeError, ValueError, RuntimeError):
        sample_list = []

    sample_list, truncated_by_size = _truncate_to_size(sample_list, config.max_bytes)

    return {
        "sample": sample_list,
        "_meta": {
            "type": "torch.Tensor",
            "shape": shape,
            "dtype": dtype,
            "device": device,
            "numel": x.numel() if hasattr(x, "numel") else None,
            "sample_rows": min(config.sample_rows, shape[0]) if shape else 0,
            "truncated": (shape[0] > config.sample_rows if shape else False)
            or truncated_by_size,
        },
    }


def extract_tf_tensor(x: Any, config: ExtractionConfig) -> dict[str, Any]:
    """Extract sample + metadata from TensorFlow tensor."""
    shape = list(x.shape) if x.shape else []
    dtype = str(x.dtype)

    # Sample rows
    sample = x[: config.sample_rows]

    # Sample columns if 2D+
    if len(shape) > 1 and shape[1] > config.sample_columns:
        sample = sample[:, : config.sample_columns]

    # Convert to numpy then list
    try:
        sample_list = sample.numpy().tolist()
    except (TypeError, ValueError, AttributeError):
        sample_list = []

    sample_list, truncated_by_size = _truncate_to_size(sample_list, config.max_bytes)

    return {
        "sample": sample_list,
        "_meta": {
            "type": "tensorflow.Tensor",
            "shape": shape,
            "dtype": dtype,
            "sample_rows": min(config.sample_rows, shape[0]) if shape else 0,
            "truncated": (shape[0] > config.sample_rows if shape else False)
            or truncated_by_size,
        },
    }


def extract_jax_array(x: Any, config: ExtractionConfig) -> dict[str, Any]:
    """Extract sample + metadata from JAX array."""
    shape = list(x.shape)
    dtype = str(x.dtype)

    # Sample rows
    sample = x[: config.sample_rows]

    # Sample columns if 2D+
    if len(shape) > 1 and shape[1] > config.sample_columns:
        sample = sample[:, : config.sample_columns]

    # Convert via numpy
    try:
        import numpy as np

        sample_list = np.array(sample).tolist()
    except (TypeError, ValueError, ImportError):
        sample_list = []

    sample_list, truncated_by_size = _truncate_to_size(sample_list, config.max_bytes)

    return {
        "sample": sample_list,
        "_meta": {
            "type": "jax.Array",
            "shape": shape,
            "dtype": dtype,
            "sample_rows": min(config.sample_rows, shape[0]) if shape else 0,
            "truncated": (shape[0] > config.sample_rows if shape else False)
            or truncated_by_size,
        },
    }


def extract_dict(x: dict, config: ExtractionConfig) -> dict[str, Any]:
    """Extract sample + metadata from dict (recursive for nested tensors)."""
    result = {}
    for k, v in x.items():
        result[k] = extract_features(v, config)

    result, truncated = _truncate_to_size(result, config.max_bytes)

    return {
        "sample": result,
        "_meta": {
            "type": "dict",
            "keys": list(x.keys()),
            "truncated": truncated,
        },
    }


def extract_list(x: list, config: ExtractionConfig) -> dict[str, Any]:
    """Extract sample + metadata from list."""
    length = len(x)
    sample = x[: config.sample_rows]

    # If list contains complex objects, try to extract them
    if sample and not isinstance(sample[0], (int, float, str, bool, type(None))):
        sample = [extract_features(item, config) for item in sample]

    sample, truncated_by_size = _truncate_to_size(sample, config.max_bytes)

    return {
        "sample": sample,
        "_meta": {
            "type": "list",
            "length": length,
            "sample_size": min(config.sample_rows, length),
            "truncated": length > config.sample_rows or truncated_by_size,
        },
    }


def extract_primitive(x: Any) -> dict[str, Any]:
    """Extract primitive values (int, float, str, bool, None)."""
    return {
        "value": x,
        "_meta": {
            "type": type(x).__name__,
        },
    }


def extract_features(
    x: Any,
    config: Optional[ExtractionConfig] = None,
) -> dict[str, Any]:
    """Extract sample + metadata from any supported input type.

    Args:
        x: Input data (numpy, pandas, torch, tf, dict, list, or primitive)
        config: Extraction configuration (sample sizes, max bytes)

    Returns:
        Dict with 'sample' (actual data) and '_meta' (metadata)
    """
    if config is None:
        config = ExtractionConfig()

    # Check types in order of likelihood
    if _is_numpy_array(x):
        return extract_numpy(x, config)

    if _is_pandas_dataframe(x):
        return extract_pandas_dataframe(x, config)

    if _is_pandas_series(x):
        return extract_pandas_series(x, config)

    if _is_polars_dataframe(x):
        return extract_polars_dataframe(x, config)

    if _is_polars_series(x):
        return extract_polars_series(x, config)

    if _is_torch_tensor(x):
        return extract_torch_tensor(x, config)

    if _is_tf_tensor(x):
        return extract_tf_tensor(x, config)

    if _is_jax_array(x):
        return extract_jax_array(x, config)

    if isinstance(x, dict):
        return extract_dict(x, config)

    if isinstance(x, list):
        return extract_list(x, config)

    if isinstance(x, (int, float, str, bool, type(None))):
        return extract_primitive(x)

    # Unknown type - try to get basic info
    return {
        "sample": str(x)[:100],
        "_meta": {
            "type": f"{type(x).__module__}.{type(x).__name__}",
            "repr_truncated": len(str(x)) > 100,
        },
    }


def extract_from_args_kwargs(
    args: tuple,
    kwargs: dict,
    config: Optional[ExtractionConfig] = None,
) -> dict[str, Any]:
    """Extract features from function args and kwargs.

    For model.predict(X) calls, X is typically the first positional arg.
    For transformers, inputs are often kwargs (input_ids=..., attention_mask=...).

    Args:
        args: Positional arguments
        kwargs: Keyword arguments
        config: Extraction configuration

    Returns:
        Extracted features dict
    """
    if config is None:
        config = ExtractionConfig()

    # If there are kwargs that look like features, prefer those
    feature_kwargs = {
        k: v
        for k, v in kwargs.items()
        if k
        not in (
            "self",
            "training",
            "return_dict",
            "output_attentions",
            "output_hidden_states",
        )
    }

    if feature_kwargs:
        # Transformers-style: kwargs are the features
        return extract_features(feature_kwargs, config)

    # Standard sklearn/pytorch style: first positional arg is features
    if args:
        return extract_features(args[0], config)

    # No features found
    return {
        "sample": None,
        "_meta": {"type": "none", "note": "no features detected"},
    }
