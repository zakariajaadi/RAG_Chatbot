"""
Gradio frontend — chat interface for the RAG.

Run:
    python frontend.py

Requires the backend to be running on http://localhost:8000
"""

import json
import gradio as gr
import requests


BACKEND_URL = "http://localhost:8000/rag/stream"


def predict(message: str, history: list):
    """Stream the RAG response token by token."""
    with requests.post(
        BACKEND_URL,
        json={"input": message},
        stream=True,
        headers={"Accept": "text/event-stream"},
    ) as response:
        response.raise_for_status()

        partial = ""
        for line in response.iter_lines():
            if not line:
                continue

            decoded = line.decode("utf-8")

            if not decoded.startswith("data: "):
                continue

            payload = decoded[len("data: "):]

            if payload.strip() == "[DONE]":
                break

            try:
                data = json.loads(payload)
            except json.JSONDecodeError:
                continue

            # LangServe format: {"ops": [{"op": "add", "path": "/streamed_output/-", "value": "..."}]}
            if isinstance(data, dict):
                for op in data.get("ops", []):
                    if op.get("op") == "add" and op.get("path") == "/streamed_output/-":
                        value = op.get("value", "")
                        if isinstance(value, str):
                            partial += value
                            yield partial
            # Fallback: data is directly the token string
            elif isinstance(data, str):
                partial += data
                yield partial


with gr.Blocks(title="RAG Assistant") as demo:
    gr.Markdown("## 💬 RAG Assistant")
    gr.Markdown("Posez vos questions sur les documents indexés.")

    gr.ChatInterface(
        fn=predict,
        examples=[
            "Quand on va en prison dans le jeu.",
            "Quend on fait faillite dans le jeu ?",
        ],
        cache_examples=False,
    )


if __name__ == "__main__":
    demo.launch(
        server_name="0.0.0.0",
        server_port=7860,
        share=False,
        theme=gr.themes.Soft(),
    )