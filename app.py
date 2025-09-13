import gradio as gr
import time
import os
import csv
import datetime
from typing import List, Dict, Any

from inference_api import run_inference_api
from inference_local import run_inference_local

HEADER = ["timestamp_utc", "backend", "text_len", "latency_ms"]

def make_row(backend: str, text: str, latency_ms: float) -> Dict[str, Any]:
    return {
        "timestamp_utc": datetime.datetime.utcnow().isoformat(),
        "backend": backend,
        "text_len": len(text or ""),
        "latency_ms": float(f"{latency_ms:.2f}"),
    }

def download_csv(rows: List[Dict[str, Any]]) -> str:
    """Write rows to a temp CSV and return its path for download."""
    tmp_path = "/tmp/latency.csv" if os.name != "nt" else os.path.join(os.getcwd(), "latency.csv")
    os.makedirs(os.path.dirname(tmp_path), exist_ok=True)
    with open(tmp_path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=HEADER)
        w.writeheader()
        for r in rows or []:
            w.writerow(r)
    return tmp_path

def predict_and_log(text, backend, rows_state: List[Dict[str, Any]]):
    text = (text or "").strip()
    if not text:
        return {"error": "Please enter some text."}, "", rows_state, rows_state

    start = time.perf_counter()
    try:
        if backend == "API (InferenceClient)":
            result = run_inference_api(text)
        else:
            result = run_inference_local(text)
    except Exception as e:
        result = {"error": str(e)}
    latency_ms = (time.perf_counter() - start) * 1000

    new_row = make_row(backend, text, latency_ms)
    rows_state = (rows_state or []) + [new_row]

    return result, f"{latency_ms:.2f} ms", rows_state, rows_state


with gr.Blocks(theme=gr.themes.Base(), fill_height=True) as demo:
    rows_state = gr.State([])

    gr.Mardown = gr.Markdown( 
        """
        # Case Study 1 — Sentiment Analysis
        Compare two backends:
        1) Remote API (Hugging Face Inference Client)
        2) Local Transformers pipeline

        Enter text, choose backend, and compare latency.
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
                    ["This was a terrible experience.", "Local (Transformers pipeline)"],
                    ["The restaurant exceeded my expectations with excellent food, great service, and a cozy atmosphere.", "API (InferenceClient)"],
                    ["The movie was unnecessarily long, the plot was confusing, and the characters were poorly written.", "Local (Transformers pipeline)"],
                ],
                inputs=[text_input, backend],
                label="Example Inputs"
            )
            with gr.Row():
                submit_btn = gr.Button("Analyze Sentiment")
                clear_btn = gr.Button("Clear")

        with gr.Column(scale=1):
            output = gr.JSON(label="Prediction Result")
            latency = gr.Textbox(label="Latency (ms)", interactive=False)
            table = gr.Dataframe(
                headers=HEADER,
                label="Latency log (session)",
                interactive=False,
                wrap=True,
                value=[],
            )
            download_btn = gr.DownloadButton(label="Download CSV")

    submit_btn.click(
        fn=predict_and_log,
        inputs=[text_input, backend, rows_state],
        outputs=[output, latency, table, rows_state],
    )

    def clear_all():
        return "", "API (InferenceClient)", "", [], []

    clear_btn.click(
        fn=clear_all,
        outputs=[text_input, backend, latency, table, rows_state],
    )

    download_btn.click(
        fn=download_csv,
        inputs=[rows_state],
        outputs=download_btn,
    )

if __name__ == "__main__":
    demo.launch()