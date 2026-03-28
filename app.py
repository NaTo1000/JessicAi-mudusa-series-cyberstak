"""
app.py – Hugging Face Spaces (Gradio) demo for JessicaAi Mudusa.

Deploy this file at the root of a Hugging Face Space to get an interactive
chat UI powered by the Mudusa model.

Environment variables
---------------------
HF_MODEL_ID : str
    Hub model id to load (default: ``NaTo1000/jessicai-mudusa``).
HF_DEVICE : str
    Compute device: ``cuda``, ``mps``, or ``cpu`` (auto-detected).
"""

from __future__ import annotations

import os
from typing import Iterator

import gradio as gr
import torch

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
MODEL_ID = os.getenv("HF_MODEL_ID", "NaTo1000/jessicai-mudusa")
DEVICE = os.getenv("HF_DEVICE", None)

DEFAULT_SYSTEM_PROMPT = (
    "You are JessicaAi Mudusa, an advanced quantum-inspired AI assistant "
    "developed by NaTo1000. You are highly cooperative, creative, and "
    "deeply knowledgeable across science, technology, and the arts. "
    "You answer thoughtfully and concisely."
)

# ---------------------------------------------------------------------------
# Lazy model loading (first request triggers load)
# ---------------------------------------------------------------------------
_pipeline = None


def get_pipeline():
    global _pipeline
    if _pipeline is None:
        from jessicai_mudusa import MudusaPipeline
        _pipeline = MudusaPipeline.from_pretrained(
            MODEL_ID,
            device=DEVICE,
            torch_dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32,
        )
    return _pipeline


# ---------------------------------------------------------------------------
# Gradio chat function
# ---------------------------------------------------------------------------

def chat(
    message: str,
    history: list[list[str]],
    system_prompt: str,
    max_new_tokens: int,
    temperature: float,
    top_p: float,
    top_k: int,
    repetition_penalty: float,
) -> Iterator[str]:
    """Stream response tokens one-by-one into the Gradio chat component."""
    pipe = get_pipeline()

    # Build messages list from Gradio history format
    messages: list[dict[str, str]] = []
    for user_msg, assistant_msg in history:
        messages.append({"role": "user", "content": user_msg})
        if assistant_msg:
            messages.append({"role": "assistant", "content": assistant_msg})
    messages.append({"role": "user", "content": message})

    prompt = pipe.tokenizer.apply_chat_template(
        messages,
        system_prompt=system_prompt or DEFAULT_SYSTEM_PROMPT,
        add_generation_prompt=True,
    )

    partial = ""
    for token in pipe.stream(
        prompt,
        max_new_tokens=max_new_tokens,
        temperature=temperature,
        top_p=top_p,
        top_k=top_k,
        repetition_penalty=repetition_penalty,
    ):
        partial += token
        yield partial


# ---------------------------------------------------------------------------
# Gradio UI
# ---------------------------------------------------------------------------

with gr.Blocks(
    title="JessicaAi Mudusa",
    theme=gr.themes.Soft(primary_hue="violet", secondary_hue="indigo"),
    css=".gradio-container { max-width: 860px; margin: auto; }",
) as demo:
    gr.HTML(
        """
        <div style="text-align:center; padding: 20px 0 10px;">
            <h1 style="font-size:2.2rem; font-weight:700;
                       background: linear-gradient(135deg,#7c3aed,#4f46e5);
                       -webkit-background-clip:text; -webkit-text-fill-color:transparent;">
                ✨ JessicaAi Mudusa
            </h1>
            <p style="color:#6b7280; font-size:0.95rem;">
                Quantum-Inspired Neural Mesh Transformer &nbsp;|&nbsp; NaTo1000
            </p>
        </div>
        """
    )

    with gr.Row():
        with gr.Column(scale=1):
            gr.Markdown("### ⚙️ Settings")

            system_prompt_box = gr.Textbox(
                label="System Prompt",
                value=DEFAULT_SYSTEM_PROMPT,
                lines=4,
                placeholder="Customise the assistant's persona …",
            )
            max_new_tokens_slider = gr.Slider(
                label="Max New Tokens",
                minimum=64,
                maximum=2048,
                step=64,
                value=512,
            )
            temperature_slider = gr.Slider(
                label="Temperature",
                minimum=0.1,
                maximum=2.0,
                step=0.05,
                value=0.7,
            )
            top_p_slider = gr.Slider(
                label="Top-P",
                minimum=0.1,
                maximum=1.0,
                step=0.05,
                value=0.9,
            )
            top_k_slider = gr.Slider(
                label="Top-K",
                minimum=0,
                maximum=200,
                step=5,
                value=50,
            )
            rep_penalty_slider = gr.Slider(
                label="Repetition Penalty",
                minimum=1.0,
                maximum=2.0,
                step=0.05,
                value=1.1,
            )

        with gr.Column(scale=3):
            chatbot = gr.Chatbot(
                label="Mudusa Chat",
                height=520,
                bubble_full_width=False,
            )
            with gr.Row():
                msg_box = gr.Textbox(
                    label="Your message",
                    placeholder="Ask me anything …",
                    scale=5,
                    show_label=False,
                    container=False,
                )
                send_btn = gr.Button("Send ➤", scale=1, variant="primary")

            clear_btn = gr.Button("🗑 Clear conversation", variant="secondary")

    # Wire up events
    def user_input(message, history):
        return "", history + [[message, None]]

    def bot_response(history, system_prompt, max_new_tokens, temperature, top_p, top_k, rep_penalty):
        history[-1][1] = ""
        for token in chat(
            history[-1][0],
            history[:-1],
            system_prompt,
            max_new_tokens,
            temperature,
            top_p,
            top_k,
            rep_penalty,
        ):
            history[-1][1] = token
            yield history

    msg_box.submit(
        user_input, [msg_box, chatbot], [msg_box, chatbot], queue=False
    ).then(
        bot_response,
        [chatbot, system_prompt_box, max_new_tokens_slider, temperature_slider,
         top_p_slider, top_k_slider, rep_penalty_slider],
        chatbot,
    )

    send_btn.click(
        user_input, [msg_box, chatbot], [msg_box, chatbot], queue=False
    ).then(
        bot_response,
        [chatbot, system_prompt_box, max_new_tokens_slider, temperature_slider,
         top_p_slider, top_k_slider, rep_penalty_slider],
        chatbot,
    )

    clear_btn.click(lambda: [], outputs=[chatbot])

    gr.Markdown(
        """
        ---
        **Model architecture highlights:**
        - 🔮 Quantum-Phase Multi-Head Attention with learnable phase angles
        - 🧠 Triple Neural Mesh (Cortical · Limbic · Quantum channels)
        - 💡 Synaptic Plasticity Memory for context retention
        - ⚡ Grouped-Query Attention + RoPE for efficient long contexts
        """
    )

demo.queue()

if __name__ == "__main__":
    demo.launch(share=False)
