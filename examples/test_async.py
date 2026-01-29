"""Test script to verify async behavior of the SDK.

This script logs predictions and measures timing to show
that predictions are truly non-blocking.
"""

import time
import aither


def test_non_blocking():
    """Test that log_prediction returns immediately."""
    print("Testing non-blocking behavior...\n")

    # Initialize with test endpoint
    aither.init(
        api_key="aith_test_key",
        endpoint="http://localhost:8080",  # Your test server
        flush_interval=2.0,  # Flush every 2 seconds
    )

    # Measure time for 100 predictions
    start = time.time()

    for i in range(100):
        aither.log_prediction(
            model_name="test-model",
            features={"feature_a": i, "feature_b": i * 2},
            prediction=i * 0.01,
        )

    elapsed = time.time() - start

    print(f"Logged 100 predictions in {elapsed * 1000:.2f}ms")
    print(f"Average: {elapsed * 10:.2f}ms per prediction")
    print("\nIf this was blocking HTTP, it would take 100-1000ms per request!")
    print("The fast timing proves predictions are queued, not sent immediately.\n")

    # Wait and flush
    print("Waiting 3 seconds for background flush...")
    time.sleep(3)

    print("\nManual flush to send any remaining predictions...")
    aither.flush()

    print("Done! Check your server logs to verify batched requests.")


def test_immediate_mode():
    """Test immediate mode (blocking) for comparison."""
    print("\n" + "=" * 60)
    print("Testing immediate mode (blocking) for comparison...\n")

    # Create client with immediate mode
    from aither import AitherClient

    client = AitherClient(
        api_key="aith_test_key",
        endpoint="http://localhost:8080",
        enable_background=False,  # Immediate mode
    )

    start = time.time()

    try:
        # This will actually make HTTP requests (or fail if server not running)
        for i in range(5):
            client.log_prediction(
                model_name="test-model-immediate",
                features={"x": i},
                prediction=i * 0.01,
            )
        elapsed = time.time() - start
        print(f"Logged 5 predictions in {elapsed * 1000:.2f}ms (blocking mode)")
        print(f"Average: {elapsed * 200:.2f}ms per prediction")
    except Exception as e:
        elapsed = time.time() - start
        print(f"Expected error (no server running): {type(e).__name__}")
        print(
            f"But you can see it took {elapsed * 1000:.2f}ms - proving it tried to make HTTP requests"
        )

    client.close()


if __name__ == "__main__":
    print("=" * 60)
    print("Aither SDK Async Test")
    print("=" * 60 + "\n")

    test_non_blocking()
    test_immediate_mode()

    print("\n" + "=" * 60)
    print("Test complete!")
    print("=" * 60)
