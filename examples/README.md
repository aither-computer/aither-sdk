# Aither SDK Examples

This directory contains examples demonstrating how to use the Aither SDK in different scenarios.

## Examples

### 1. `quickstart.py` - Minimal Getting Started

The simplest possible example - just initialize and log a prediction.

```bash
python quickstart.py
```

**Use this when**: You want to see the absolute minimum code needed to get started.

### 2. `basic_usage.py` - Basic Features

Demonstrates the core features including:
- Initializing the client
- Logging multiple predictions with features and metadata
- Manual flushing
- Non-blocking behavior

```bash
python basic_usage.py
```

**Use this when**: You want to understand the main SDK features in a simple script.

### 3. `fastapi_example.py` - FastAPI Integration

Complete FastAPI application showing:
- SDK initialization at startup
- Non-blocking prediction logging in API endpoints
- Graceful shutdown with flush
- Request/response models

```bash
# Install dependencies
pip install fastapi uvicorn

# Run the server
uvicorn fastapi_example:app --reload

# Test with curl
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{"text": "This product is amazing!", "user_id": "user_123"}'
```

**Use this when**: You're building a web API and want to see how to integrate Aither without blocking requests.

### 4. `test_async.py` - Performance Testing

Test script that demonstrates and measures:
- Non-blocking mode performance (default)
- Blocking mode comparison
- Batch flushing behavior
- Timing measurements

```bash
python test_async.py
```

**Use this when**: You want to verify the async behavior and measure performance characteristics.

## Key Concepts

### Non-Blocking by Default

All examples use non-blocking mode by default. When you call `log_prediction()`, it:
1. Adds the prediction to an in-memory queue
2. Returns immediately (microseconds)
3. Background worker sends it asynchronously
4. Predictions are batched automatically

### Manual Flushing

Use `aither.flush()` when you need to ensure predictions are sent immediately:
- Before program exit (automatic via `atexit`)
- At the end of batch processing
- In tests
- When you need synchronous behavior

### Configuration

Customize behavior via `aither.init()`:

```python
aither.init(
    api_key="ak_your_key",
    flush_interval=1.0,  # Flush every N seconds
    batch_size=100,      # Max predictions per batch
)
```

## Next Steps

1. Start with `quickstart.py` to see the basics
2. Review `basic_usage.py` for core features
3. Use `fastapi_example.py` as a template for web APIs
4. Run `test_async.py` to understand performance

For full API documentation, see the main [README.md](../README.md).
