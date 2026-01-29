"""Quickstart example - minimal code to get started with Aither SDK."""

import aither

# Set your API key (or use AITHER_API_KEY environment variable)
aither.init(api_key="aith_your_api_key_here")

# Log a simple prediction - returns trace_id for label correlation
trace_id = aither.log_prediction(
    model_name="my-first-model",
    features={"input": "test"},
    prediction="success",
)

print(f"Prediction logged! trace_id: {trace_id}")
print("It will be sent to Aither in the background.")
print("The function returned immediately without waiting for the HTTP request.")

# Later, when you know the ground truth:
aither.log_label(trace_id=trace_id, label="actual_success")
print("Ground truth label logged for correlation!")

# Optional: wait for it to be sent
aither.flush()
print("Flushed - all data has been sent!")
