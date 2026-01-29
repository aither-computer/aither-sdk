"""FastAPI integration example for Aither SDK.

This demonstrates how to use the SDK in a FastAPI application.
Predictions are logged without blocking request handling.

Run with: uvicorn fastapi_example:app --reload

Requirements:
    pip install fastapi uvicorn
"""

from typing import Any
import aither
from fastapi import FastAPI
from pydantic import BaseModel

# Initialize Aither SDK once at startup
aither.init(
    api_key="aith_your_api_key_here",
    flush_interval=1.0,  # Flush every 1 second
    batch_size=100,  # Up to 100 predictions per batch
)

app = FastAPI(title="ML Prediction API")


class PredictionRequest(BaseModel):
    """Request model for predictions."""

    text: str
    user_id: str | None = None


class PredictionResponse(BaseModel):
    """Response model for predictions."""

    sentiment: str
    confidence: float
    trace_id: str


class LabelRequest(BaseModel):
    """Request model for ground truth labels."""

    trace_id: str
    actual_sentiment: str


# Simulate a trained model (replace with your actual model)
def predict_sentiment(text: str) -> tuple[str, float]:
    """Mock sentiment prediction."""
    # In real use: return model.predict(text)
    sentiment = "positive" if len(text) > 50 else "negative"
    confidence = 0.75 + (len(text) % 20) * 0.01
    return sentiment, confidence


@app.post("/predict", response_model=PredictionResponse)
async def predict(request: PredictionRequest) -> dict[str, Any]:
    """Make a prediction and log it to Aither.

    The prediction is returned to the user immediately.
    Logging happens asynchronously in the background without blocking.
    """
    # Make prediction
    sentiment, confidence = predict_sentiment(request.text)

    # Log to Aither (non-blocking - returns trace_id for label correlation)
    trace_id = aither.log_prediction(
        model_name="sentiment-api-v1",
        features={
            "text_length": len(request.text),
            "word_count": len(request.text.split()),
        },
        prediction=sentiment,
        # Optional: add probability and class information
        probabilities=[1 - confidence, confidence]
        if sentiment == "positive"
        else [confidence, 1 - confidence],
        classes=["negative", "positive"],
        environment="production",
        user_id=request.user_id,
    )

    # Return prediction immediately with trace_id
    return {
        "sentiment": sentiment,
        "confidence": confidence,
        "trace_id": trace_id,
    }


@app.post("/label")
async def label(request: LabelRequest) -> dict[str, str]:
    """Submit ground truth label for a previous prediction.

    Use the trace_id from the prediction response to correlate
    the ground truth with the original prediction.
    """
    aither.log_label(trace_id=request.trace_id, label=request.actual_sentiment)
    return {"status": "ok"}


@app.get("/health")
async def health() -> dict[str, str]:
    """Health check endpoint."""
    return {"status": "healthy"}


@app.on_event("shutdown")
async def shutdown_event():
    """Flush any remaining predictions on shutdown."""
    print("Flushing predictions before shutdown...")
    aither.flush()
    aither.close()
    print("Aither SDK closed.")


# Example usage:
#
# 1. Start the server:
#    uvicorn fastapi_example:app --reload
#
# 2. Make a prediction:
#    curl -X POST http://localhost:8000/predict \
#      -H "Content-Type: application/json" \
#      -d '{"text": "This product is amazing!", "user_id": "user_123"}'
#
# 3. Response includes trace_id for ground truth correlation:
#    {"sentiment": "negative", "confidence": 0.84, "trace_id": "abc123..."}
#
# 4. Later, submit ground truth:
#    curl -X POST http://localhost:8000/label \
#      -H "Content-Type: application/json" \
#      -d '{"trace_id": "abc123...", "actual_sentiment": "positive"}'
