# aither

Python SDK for the [Aither](https://aither.computer) platform - contextual intelligence and model observability.

## Features

- **OTLP Native**: Uses OpenTelemetry Protocol for efficient, standardized data transfer
- **Non-blocking**: Predictions are logged asynchronously without blocking your application
- **Automatic batching**: Multiple predictions are sent in batches for efficiency
- **Label correlation**: Track ground truth labels with trace ID correlation
- **Zero latency impact**: Background worker handles all API communication

## Installation

```bash
pip install aither
```

## Quick Start

```python
import aither

# Initialize with your API key
aither.init(api_key="aith_your_api_key")

# Log a prediction - returns trace_id for label correlation
trace_id = aither.log_prediction(
    model_name="fraud_detector",
    features={"amount": 150.0, "country": "US"},
    prediction=0.87,
)

# Later, when you know the ground truth:
aither.log_label(trace_id=trace_id, label=1)  # Was actually fraud
```

## Configuration

### Environment Variables

```bash
export AITHER_API_KEY="aith_your_api_key"
export AITHER_ENDPOINT="https://aither.computer"  # optional
```

### Explicit Initialization

```python
import aither

aither.init(
    api_key="aith_your_api_key",
    endpoint="https://aither.computer",
    flush_interval=1.0,  # seconds between flushes
    batch_size=100,      # max predictions per batch
)
```

## API Reference

### `aither.init(api_key=None, endpoint=None, flush_interval=1.0, batch_size=100)`

Initialize the global client.

- `api_key`: Your Aither API key (or set `AITHER_API_KEY` env var)
- `endpoint`: API endpoint URL (default: `https://aither.computer`)
- `flush_interval`: How often to flush queued predictions in seconds
- `batch_size`: Maximum predictions per batch request

### `aither.log_prediction(...) -> str`

Log a model prediction (non-blocking). Returns a trace_id for label correlation.

```python
trace_id = aither.log_prediction(
    model_name="fraud_detector",           # Required: model identifier
    features={"amount": 150.0},            # Required: input features
    prediction=0.87,                       # Required: prediction value
    # Optional parameters:
    version="1.2.3",                       # Model version
    probabilities=[0.13, 0.87],            # Class probabilities
    classes=["legit", "fraud"],            # Class labels
    environment="production",              # Deployment environment
    request_id="req-abc123",               # Request identifier
    user_id="user-anonymized-hash",        # User identifier (anonymized)
)
```

**Returns**: `trace_id` (hex string) for correlating ground truth labels.

### `aither.log_label(trace_id, label)`

Log ground truth for a previous prediction (non-blocking).

```python
aither.log_label(
    trace_id=trace_id,  # From log_prediction()
    label=1,            # Actual outcome
)
```

### `aither.flush()`

Force immediate flush of all queued data (blocking).

```python
aither.log_prediction(model_name="my-model", features={}, prediction=0.5)
aither.flush()  # Wait for all data to be sent
```

### `aither.close()`

Close the global client and flush remaining data.

```python
aither.close()  # Flush and shutdown background worker
```

## Usage Patterns

### Basic Prediction Logging

```python
import aither

aither.init(api_key="aith_...")

# Log prediction, get trace_id
trace_id = aither.log_prediction(
    model_name="churn_predictor",
    features={"tenure": 24, "monthly_charges": 65.5},
    prediction=0.73,
)

# Store trace_id with your prediction for later label correlation
save_to_database(prediction_id, trace_id)
```

### Ground Truth Correlation

```python
# Later, when ground truth is known:
trace_id = get_trace_id_from_database(prediction_id)
aither.log_label(trace_id=trace_id, label="churned")
```

### Classification with Probabilities

```python
trace_id = aither.log_prediction(
    model_name="sentiment_classifier",
    features={"text": "Great product!"},
    prediction="positive",
    probabilities=[0.05, 0.15, 0.80],
    classes=["negative", "neutral", "positive"],
    version="2.1.0",
    environment="production",
)
```

### FastAPI Integration

```python
import aither
from fastapi import FastAPI

aither.init(api_key="aith_...")
app = FastAPI()

@app.post("/predict")
async def predict(data: dict):
    prediction = model.predict(data)

    # Non-blocking - returns instantly
    trace_id = aither.log_prediction(
        model_name="my-model",
        features=data,
        prediction=prediction,
    )

    return {"prediction": prediction, "trace_id": trace_id}

@app.post("/label")
async def label(trace_id: str, actual: int):
    aither.log_label(trace_id=trace_id, label=actual)
    return {"status": "ok"}

@app.on_event("shutdown")
async def shutdown():
    aither.close()  # Flush remaining data
```

### Using the Client Directly

```python
from aither import AitherClient

# With background worker (default)
client = AitherClient(
    api_key="aith_...",
    flush_interval=1.0,
    batch_size=100,
)

trace_id = client.log_prediction(
    model_name="my-model",
    features={"x": 1},
    prediction=0.5,
)
client.log_label(trace_id=trace_id, label=1)
client.close()

# Immediate mode (blocking, no background worker)
with AitherClient(api_key="aith_...", enable_background=False) as client:
    client.log_prediction(...)  # Sends immediately
```

## Data Format

The SDK uses OTLP (OpenTelemetry Protocol) to send predictions as spans with `ml.*` attributes:

| Attribute | Description |
|-----------|-------------|
| `ml.model.name` | Model identifier |
| `ml.model.version` | Model version |
| `ml.features` | JSON-encoded input features |
| `ml.prediction` | JSON-encoded prediction value |
| `ml.prediction.probabilities` | Class probabilities |
| `ml.prediction.classes` | Class labels |
| `ml.label` | Ground truth value |
| `ml.environment` | Deployment environment |
| `ml.request_id` | Request identifier |
| `ml.user_id` | User identifier |

## License

MIT
