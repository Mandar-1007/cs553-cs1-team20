from inference_local import run_inference_local

def test_local_sentiment_pipeline_returns_label_and_score():
    out = run_inference_local("I really enjoyed this product.")
    assert isinstance(out, (list, tuple)) and len(out) >= 1

    first = out[0]
    assert isinstance(first, dict)
    assert "label" in first and "score" in first
    assert isinstance(first["label"], str)
    assert 0.0 <= float(first["score"]) <= 1.0


def test_negative_sentiment_prediction_contains_negative_label():
    out = run_inference_local("This was a terrible experience.")
    labels = [result["label"].upper() for result in out]
    assert any("NEG" in label for label in labels), f"Labels were: {labels}"
