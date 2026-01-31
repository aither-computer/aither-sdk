# aither

Python SDK for the [aither](https://aither.computer) platform - contextual intelligence and model observability.

## Features

- **Zero-code logging**: Wrap any model and predictions are logged automatically
- **Framework support**: sklearn, pytorch, tensorflow, tinygrad, transformers
- **Smart sampling**: Captures sample + metadata from features (never blocks inference)
- **Label correlation**: Track ground truth with trace ID correlation
- **Non-blocking**: Background worker handles all API communication

## Installation

```bash
pip install aither
```

## Quick Start

```python
import aither
from sklearn.ensemble import RandomForestClassifier

# Initialize
aither.init()  # Uses AITHER_API_KEY env var

# Train your model
model = RandomForestClassifier().fit(X_train, y_train)

# Wrap it - predictions are now logged automatically
tracked = aither.wrap(model, name="fraud_detector")

# Use normally
predictions = tracked.predict(X_test)

# Get trace_id for label correlation
trace_id = tracked.last_trace_id
```

## Supported Frameworks

| Framework | Example |
|-----------|---------|
| **sklearn** | `aither.wrap(RandomForestClassifier(), name="clf")` |
| **pytorch** | `aither.wrap(MyNet(), name="net")` |
| **tensorflow** | `aither.wrap(keras_model, name="tf_model")` |
| **tinygrad** | `aither.wrap(tinygrad_model, name="tiny")` |
| **transformers** | `aither.wrap(pipeline("sentiment"), name="sentiment")` |

Also works with sklearn-compatible libraries: xgboost, lightgbm, catboost.

## Configuration

### Environment Variables

```bash
export AITHER_API_KEY="aith_your_api_key"
export AITHER_BASE_URL="https://aither.computer"  # optional
```

### Explicit Initialization

```python
aither.init(
    api_key="aith_your_api_key",
    base_url="https://aither.computer",
    flush_interval=1.0,
    batch_size=100,
)
```

## API Reference

### `aither.wrap(model, name, **options)` - Recommended

Wrap a model for automatic prediction logging.

```python
tracked = aither.wrap(
    model,                          # Any ML model
    name="fraud_detector",          # Required: model identifier
    version="1.2.3",                # Optional: model version
    environment="production",       # Optional: deployment env
    sample_rows=5,                  # Max rows to sample (default: 5)
    sample_columns=10,              # Max columns to sample (default: 10)
    features_fn=custom_extractor,   # Custom feature extraction
)

# Use the wrapped model normally
result = tracked.predict(X)
result = tracked(X)                 # Also works
probs = tracked.predict_proba(X)    # Also tracked

# Access trace_id for label correlation
trace_id = tracked.last_trace_id
all_trace_ids = tracked.trace_ids   # All predictions

# Access underlying model
tracked.model                       # Original model
tracked.classes_                    # Passthrough to model attributes
```

### `aither.log_prediction(...)` - Manual Control

For custom pipelines or when you need full control:

```python
trace_id = aither.log_prediction(
    model_name="my_pipeline",
    features={"amount": 150.0, "country": "US"},
    prediction=0.87,
    version="1.2.3",
    environment="production",
)
```

### `aither.log_label(trace_id, label)`

Log ground truth for a prediction:

```python
aither.log_label(trace_id=trace_id, label=1)
```

### `@aither.track(name)` - Decorator

For functions instead of model objects:

```python
@aither.track("my_function")
def predict(features):
    return model.predict(features)

result = predict(X)
trace_id = aither.last_trace_id()
```

## Feature Extraction

Wrapped models automatically extract features with smart sampling:

```python
# Input: pandas DataFrame with 50,000 rows, 100 columns
# What gets logged:
{
    "sample": [
        {"col1": 1.5, "col2": "A", ...},  # 5 rows
        ...
    ],
    "_meta": {
        "type": "pandas.DataFrame",
        "shape": [50000, 100],
        "columns": ["col1", "col2", ...],
        "dtypes": {"col1": "float64", ...},
        "truncated": True
    }
}
```

**Supported types:**
- numpy arrays
- pandas DataFrames/Series
- polars DataFrames/Series
- torch Tensors
- tensorflow Tensors
- dicts (common in transformers)
- lists

**Custom extraction:**

```python
def my_extractor(X):
    return {"shape": X.shape, "mean": X.mean()}

tracked = aither.wrap(model, name="m", features_fn=my_extractor)
```

## Usage Patterns

### Basic Model Wrapping

```python
import aither
from sklearn.ensemble import RandomForestClassifier

aither.init()

model = RandomForestClassifier().fit(X_train, y_train)
tracked = aither.wrap(model, name="churn_predictor", version="1.0")

predictions = tracked.predict(X_test)
```

### PyTorch Model

```python
import aither
import torch

aither.init()

class MyNet(torch.nn.Module):
    def forward(self, x):
        return self.layers(x)

model = MyNet()
tracked = aither.wrap(model, name="my_net")

# Both work:
output = tracked(input_tensor)
output = tracked.predict(input_tensor)
```

### Ground Truth Correlation

```python
# At prediction time
predictions = tracked.predict(X)
trace_ids = tracked.trace_ids  # List of all trace IDs

# Store trace_ids with your predictions
save_to_db(prediction_ids, trace_ids)

# Later, when ground truth is known
for pred_id, trace_id in load_from_db():
    actual = get_actual_outcome(pred_id)
    aither.log_label(trace_id, actual)
```

### FastAPI Integration

```python
import aither
from fastapi import FastAPI

aither.init()
model = aither.wrap(load_model(), name="api_model")
app = FastAPI()

@app.post("/predict")
async def predict(data: dict):
    prediction = model.predict(data)
    return {
        "prediction": prediction,
        "trace_id": model.last_trace_id
    }

@app.post("/label")
async def label(trace_id: str, actual: int):
    aither.log_label(trace_id, actual)
    return {"status": "ok"}

@app.on_event("shutdown")
async def shutdown():
    aither.close()
```

## Management API

The SDK provides namespaces for managing your organization, API keys, and user account.

```python
import aither

aither.init()

# Organization info
org = aither.org.get()
print(org.name, org.plan)

# Usage stats for current billing period
usage = aither.org.usage()
print(f"API calls: {usage.api_calls}")

# Current user
me = aither.user.me()
print(me.email)

# API key management (requires admin scope)
keys = aither.api_keys.list()
new_key = aither.api_keys.create(name="Production", scopes=["read", "write"])
aither.api_keys.revoke(key_id="...")
```

### Why `.get()` instead of direct attributes?

You might wonder why `aither.org.get().name` instead of just `aither.org.name`. This is intentional:

1. **Network calls are explicit** - `.get()` makes it clear you're making an HTTP request. Hidden network calls on attribute access would be surprising and expensive.

2. **Caching semantics are clear** - The returned `Organization` is a point-in-time snapshot. You control when to refresh by calling `.get()` again.

3. **Error handling is predictable** - Exceptions from HTTP failures occur at the `.get()` call site, not on attribute access.

```python
# Recommended: fetch once, use the snapshot
org = aither.org.get()
print(f"{org.name} on {org.plan} plan")

# Compare states over time
org_before = aither.org.get()
# ... make changes ...
org_after = aither.org.get()
if org_before.plan != org_after.plan:
    print("Plan changed!")
```

## Data Format

The SDK uses OTLP (OpenTelemetry Protocol) to send predictions as spans with `ml.*` attributes:

| Attribute | Description |
|-----------|-------------|
| `ml.model.name` | Model identifier |
| `ml.model.version` | Model version |
| `ml.features` | JSON-encoded feature sample + metadata |
| `ml.prediction` | JSON-encoded prediction |
| `ml.label` | Ground truth value |
| `ml.environment` | Deployment environment |

## License

MIT
