"""
serve.py – FastAPI REST server for JessicaAi Mudusa.

Provides an OpenAI-compatible ``/v1/chat/completions`` endpoint.

Usage::

    python inference/serve.py --model NaTo1000/jessicai-mudusa --port 8000

    # Test with curl:
    curl -X POST http://localhost:8000/v1/chat/completions \\
        -H "Content-Type: application/json" \\
        -d '{
            "messages": [{"role": "user", "content": "Hello!"}],
            "max_tokens": 256
        }'
"""

from __future__ import annotations

import argparse
import logging
import os
import sys
import time
import uuid
from typing import Iterator, Optional

import torch
import uvicorn
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
from jessicai_mudusa import MudusaPipeline

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(
    title="JessicaAi Mudusa API",
    description=(
        "REST API for the JessicaAi Mudusa quantum-inspired language model. "
        "Compatible with the OpenAI Chat Completions specification."
    ),
    version="1.0.0",
)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global pipeline (initialised at startup)
_pipeline: Optional[MudusaPipeline] = None


# ---------------------------------------------------------------------------
# Pydantic schemas
# ---------------------------------------------------------------------------


class ChatMessage(BaseModel):
    role: str = Field(..., description="One of 'system', 'user', 'assistant'.")
    content: str = Field(..., description="Message content.")


class ChatCompletionRequest(BaseModel):
    messages: list[ChatMessage] = Field(..., description="Conversation history.")
    max_tokens: int = Field(512, ge=1, le=4096, description="Max tokens to generate.")
    temperature: float = Field(0.7, ge=0.0, le=2.0, description="Sampling temperature.")
    top_p: float = Field(0.9, ge=0.0, le=1.0, description="Nucleus sampling probability.")
    top_k: int = Field(50, ge=0, description="Top-k filtering.")
    repetition_penalty: float = Field(1.1, ge=1.0, description="Repetition penalty.")
    stream: bool = Field(False, description="Whether to stream the response.")
    system_prompt: Optional[str] = Field(None, description="Override the system prompt.")


class ChatCompletionResponse(BaseModel):
    id: str
    object: str = "chat.completion"
    created: int
    model: str = "jessicai-mudusa"
    choices: list[dict]
    usage: dict


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------


@app.get("/health")
def health_check():
    return {"status": "ok", "model_loaded": _pipeline is not None}


@app.get("/v1/models")
def list_models():
    return {
        "object": "list",
        "data": [
            {
                "id": "jessicai-mudusa",
                "object": "model",
                "created": 1700000000,
                "owned_by": "NaTo1000",
            }
        ],
    }


@app.post("/v1/chat/completions")
def chat_completions(request: ChatCompletionRequest):
    if _pipeline is None:
        raise HTTPException(status_code=503, detail="Model not loaded yet.")

    messages = [m.model_dump() for m in request.messages]
    gen_kwargs = dict(
        max_new_tokens=request.max_tokens,
        temperature=request.temperature,
        top_p=request.top_p,
        top_k=request.top_k,
        repetition_penalty=request.repetition_penalty,
    )

    if request.stream:
        prompt = _pipeline.tokenizer.apply_chat_template(
            messages,
            system_prompt=request.system_prompt or _pipeline.system_prompt,
            add_generation_prompt=True,
        )

        def token_stream() -> Iterator[str]:
            for token in _pipeline.stream(prompt, **gen_kwargs):
                chunk = (
                    '{"id":"'
                    + str(uuid.uuid4())
                    + '","object":"chat.completion.chunk",'
                    '"choices":[{"delta":{"content":'
                    + repr(token)
                    + "}]}\n"
                )
                yield f"data: {chunk}\n"
            yield "data: [DONE]\n"

        return StreamingResponse(token_stream(), media_type="text/event-stream")

    response_text = _pipeline.chat(messages, system_prompt=request.system_prompt, **gen_kwargs)

    return ChatCompletionResponse(
        id=str(uuid.uuid4()),
        created=int(time.time()),
        choices=[
            {
                "index": 0,
                "message": {"role": "assistant", "content": response_text},
                "finish_reason": "stop",
            }
        ],
        usage={
            "prompt_tokens": -1,
            "completion_tokens": -1,
            "total_tokens": -1,
        },
    )


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Serve JessicaAi Mudusa via FastAPI.")
    parser.add_argument("--model", type=str, default="NaTo1000/jessicai-mudusa")
    parser.add_argument("--tokenizer", type=str, default=None)
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--host", type=str, default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--workers", type=int, default=1)
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    global _pipeline
    logger.info("Loading model from '%s' …", args.model)
    _pipeline = MudusaPipeline.from_pretrained(
        args.model,
        tokenizer_name_or_path=args.tokenizer,
        device=args.device,
    )
    logger.info("Model loaded.  Starting server on %s:%d …", args.host, args.port)
    uvicorn.run(app, host=args.host, port=args.port, workers=args.workers)


if __name__ == "__main__":
    main()
