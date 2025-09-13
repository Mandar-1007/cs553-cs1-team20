import gradio as gr
import time
import os
import csv
import datetime

from inference_api import run_inference_api
from inference_local import run_inference_local

LOG_PATH = os.path.join("logs", "latency.csv")

def log_latency(backend: str, text: str, latency_ms: float):
    """Append a row to logs/latency.csv (auto-creates folder/file)."""
    os.makedirs(os.path.dirname(LOG_PATH), exist_ok=True)
    header = ["timestamp_utc", "backend", "text_len", "latency_ms"]
    row = [datetime.datetime.utcnow().isoformat(), backend, len(text or ""), f"{latency_ms:.2f}"]
    write_header = not os.path.exists(LOG_PATH)
    with open(LOG_PATH, "a", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        if write_header:
            w.writerow(header)
        w.writerow(row)

def predict(text, backend):
    text = (text or "").strip()
    if not text:
        return {"error": "Please enter some text."}, ""

    start = time.perf_counter()
    try:
        if backend == "API (InferenceClient)":
            result = run_inference_api(text)
        else:
            result = run_inference_local(text)
    except Exception as e:
        result = {"error": str(e)}
    latency_ms = (time.perf_counter() - start) * 1000

    # record latency for comparison
    try:
        log_latency(backend, text, latency_ms)
    except Exception:
        # logging failures should never break the UI
        pass

    return result, f"{latency_ms:.2f} ms"


with gr.Blocks(theme=gr.themes.Base(), fill_height=True) as demo:
    gr.Markdown(
        """
        # Case Study 1 — Sentiment Analysis
        This application demonstrates two approaches to running a machine learning model:
        1) Using a remote API (Hugging Face Inference Client), and
        2) Running the model locally within this Space using a Transformers pipeline.

        Enter text, choose the backend, and compare performance and results.
        """
    )

    with gr.Row():
        with gr.Column(scale=1):
            text_input = gr.Textbox(
                label="Input Text",
                placeholder="Type a sentence or review...",
                lines=3
            )
            backend = gr.Radio(
                ["API (InferenceClient)", "Local (Transformers pipeline)"],
                value="API (InferenceClient)",
                label="Select Backend"
            )
            gr.Examples(
                examples=[
                    ["I really enjoyed this product.", "API (InferenceClient)"],
                    ["The service was disappointing.", "Local (Transformers pipeline)"],
                ],
                inputs=[text_input, backend],
                label="Example Inputs"
            )
            with gr.Row():
                submit_btn = gr.Button("Analyze Sentiment")
                clear_btn = gr.Button("Clear")

        with gr.Column(scale=1):
            output = gr.JSON(label="Prediction Result")
            latency = gr.Textbox(label="Latency", interactive=False)

    submit_btn.click(fn=predict, inputs=[text_input, backend], outputs=[output, latency])

    clear_btn.click(
        fn=lambda: ("", "API (InferenceClient)", "", None),
        outputs=[text_input, backend, latency, output],
    )

if __name__ == "__main__":
    demo.launch()