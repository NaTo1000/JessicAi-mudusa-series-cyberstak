# Deployment Guide – JessicaAi Mudusa

---

## Option 1 – Hugging Face Spaces (Recommended)

The easiest way to share Mudusa with the world.

### Steps

1. Create a new Space at [huggingface.co/new-space](https://huggingface.co/new-space).
2. Choose **Gradio** SDK.
3. Clone the Space repo and copy in the files:

```bash
git clone https://huggingface.co/spaces/NaTo1000/jessicai-mudusa-demo
cp app.py requirements.txt jessicai_mudusa/ -r ./jessicai-mudusa-demo/
cd jessicai-mudusa-demo
git add . && git commit -m "Deploy Mudusa" && git push
```

4. Set the `HF_MODEL_ID` environment variable in the Space settings to
   `NaTo1000/jessicai-mudusa`.

---

## Option 2 – FastAPI Server (Self-hosted)

### Start the server

```bash
python inference/serve.py \
    --model NaTo1000/jessicai-mudusa \
    --host 0.0.0.0 \
    --port 8000
```

### Test with cURL

```bash
curl -X POST http://localhost:8000/v1/chat/completions \
     -H "Content-Type: application/json" \
     -d '{
       "messages": [{"role": "user", "content": "Hello, Mudusa!"}],
       "max_tokens": 256,
       "temperature": 0.7
     }'
```

### Streaming response

```bash
curl -X POST http://localhost:8000/v1/chat/completions \
     -H "Content-Type: application/json" \
     -d '{"messages": [{"role":"user","content":"Tell me a story."}], "stream": true}'
```

---

## Option 3 – Docker

```dockerfile
FROM pytorch/pytorch:2.2.0-cuda12.1-cudnn8-runtime

WORKDIR /app
COPY . .

RUN pip install --no-cache-dir -r requirements.txt

EXPOSE 8000
CMD ["python", "inference/serve.py", "--model", "NaTo1000/jessicai-mudusa"]
```

```bash
docker build -t jessicai-mudusa .
docker run --gpus all -p 8000:8000 jessicai-mudusa
```

---

## Option 4 – Python Library

Install and use directly in your own application:

```bash
pip install jessicai-mudusa
```

```python
from jessicai_mudusa import MudusaPipeline

pipe = MudusaPipeline.from_pretrained("NaTo1000/jessicai-mudusa")

# Single prompt
print(pipe("What is quantum consciousness?"))

# Multi-turn chat
messages = [{"role": "user", "content": "Hi! Who are you?"}]
print(pipe.chat(messages))

# Streaming
for token in pipe.stream("Write a haiku about the cosmos."):
    print(token, end="", flush=True)
print()
```

---

## Performance Tuning

### Half-precision inference

```python
import torch
from jessicai_mudusa import MudusaForCausalLM

model = MudusaForCausalLM.from_pretrained(
    "NaTo1000/jessicai-mudusa",
    torch_dtype=torch.bfloat16,
    device_map="auto",
)
```

### torch.compile (PyTorch 2.x)

```python
model = torch.compile(model, mode="reduce-overhead")
```

### Batch inference

```python
prompts = ["Prompt 1", "Prompt 2", "Prompt 3"]
inputs = tokenizer(prompts, padding=True, return_tensors="pt").to(model.device)
outputs = model.generate(**inputs, max_new_tokens=128)
```

---

## API Reference

### `GET /health`
Returns server status.

### `GET /v1/models`
Lists available model IDs.

### `POST /v1/chat/completions`

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `messages` | `list` | — | Chat history |
| `max_tokens` | `int` | 512 | Max tokens to generate |
| `temperature` | `float` | 0.7 | Sampling temperature |
| `top_p` | `float` | 0.9 | Nucleus probability |
| `top_k` | `int` | 50 | Top-k filter |
| `repetition_penalty` | `float` | 1.1 | Repetition penalty |
| `stream` | `bool` | false | Enable SSE streaming |
| `system_prompt` | `str` | null | Override system prompt |
