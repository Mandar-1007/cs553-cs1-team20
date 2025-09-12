from transformers import pipeline

# Use the same model ID as API for a fair comparison
_classifier = pipeline(
    "sentiment-analysis",
    model="distilbert/distilbert-base-uncased-finetuned-sst-2-english",
)

def run_inference_local(text: str):
    try:
        return _classifier(text)  # list of {label, score}
    except Exception as e:
        return {"error": str(e)}