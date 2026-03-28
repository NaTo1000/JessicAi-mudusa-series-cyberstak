"""
run_inference.py – Command-line inference tool for JessicaAi Mudusa.

Usage::

    # Interactive REPL
    python inference/run_inference.py --model NaTo1000/jessicai-mudusa

    # Single prompt
    python inference/run_inference.py \\
        --model NaTo1000/jessicai-mudusa \\
        --prompt "Explain quantum consciousness in simple terms."

    # Streaming output
    python inference/run_inference.py \\
        --model NaTo1000/jessicai-mudusa \\
        --prompt "Write a short poem about the cosmos." \\
        --stream
"""

from __future__ import annotations

import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
from jessicai_mudusa import MudusaPipeline


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run inference with the JessicaAi Mudusa model."
    )
    parser.add_argument(
        "--model",
        type=str,
        default="NaTo1000/jessicai-mudusa",
        help="Hub repo id or local path to the Mudusa model.",
    )
    parser.add_argument(
        "--tokenizer",
        type=str,
        default=None,
        help="Tokenizer repo id or path (defaults to --model).",
    )
    parser.add_argument(
        "--prompt",
        type=str,
        default=None,
        help="Single prompt to run.  If omitted, starts an interactive REPL.",
    )
    parser.add_argument(
        "--max_new_tokens",
        type=int,
        default=512,
        help="Maximum tokens to generate.",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.7,
        help="Sampling temperature.",
    )
    parser.add_argument(
        "--top_p",
        type=float,
        default=0.9,
        help="Nucleus sampling probability.",
    )
    parser.add_argument(
        "--stream",
        action="store_true",
        help="Stream output tokens as they are generated.",
    )
    parser.add_argument(
        "--device",
        type=str,
        default=None,
        help="Device to run on (cuda / cpu / mps).",
    )
    return parser.parse_args()


def interactive_repl(pipe: MudusaPipeline, stream: bool, **gen_kwargs) -> None:
    """Run a simple multi-turn interactive REPL."""
    print("\n✨  JessicaAi Mudusa – Interactive Mode  ✨")
    print("Type your message and press Enter.  Type 'exit' or Ctrl-C to quit.\n")
    history: list[dict] = []

    while True:
        try:
            user_input = input("You: ").strip()
        except (KeyboardInterrupt, EOFError):
            print("\nGoodbye!")
            break

        if not user_input:
            continue
        if user_input.lower() in {"exit", "quit", "bye"}:
            print("Goodbye!")
            break

        history.append({"role": "user", "content": user_input})

        print("Mudusa: ", end="", flush=True)
        if stream:
            prompt = pipe.tokenizer.apply_chat_template(history, add_generation_prompt=True)
            response_parts: list[str] = []
            for token in pipe.stream(prompt, **gen_kwargs):
                print(token, end="", flush=True)
                response_parts.append(token)
            response = "".join(response_parts)
            print()
        else:
            response = pipe.chat(history, **gen_kwargs)
            print(response)

        history.append({"role": "assistant", "content": response})


def main() -> None:
    args = parse_args()

    print(f"Loading model from '{args.model}' …")
    pipe = MudusaPipeline.from_pretrained(
        args.model,
        tokenizer_name_or_path=args.tokenizer,
        device=args.device,
    )

    gen_kwargs = {
        "max_new_tokens": args.max_new_tokens,
        "temperature": args.temperature,
        "top_p": args.top_p,
    }

    if args.prompt:
        if args.stream:
            print("Mudusa: ", end="", flush=True)
            for token in pipe.stream(args.prompt, **gen_kwargs):
                print(token, end="", flush=True)
            print()
        else:
            response = pipe(args.prompt, **gen_kwargs)
            print(f"Mudusa: {response}")
    else:
        interactive_repl(pipe, stream=args.stream, **gen_kwargs)


if __name__ == "__main__":
    main()
