import os
from huggingface_hub import InferenceClient

# Full repo ID (owner/name) — keep consistent with local for apples-to-apples comparison
MODEL_ID = os.getenv(
    "HF_API_MODEL_ID",
    "distilbert/distilbert-base-uncased-finetuned-sst-2-english",
)

# Space secret name
RUNTIME_TOKEN = os.getenv("runtime_inference")

_client = InferenceClient(model=MODEL_ID, token=RUNTIME_TOKEN)

def run_inference_api(text: str):
    try:
        if not RUNTIME_TOKEN:
            return {"error": "Missing Space secret 'runtime_inference'."}
        return _client.text_classification(text)
    except Exception as e:
        return {"error": str(e)}