"""
creative_assistance.py – Creative writing and ideation example for Mudusa.

This example shows how to use Mudusa for:
- Story generation
- Ideation / brainstorming
- Code generation hints
- Adjusting generation parameters for creativity

Run::

    python examples/creative_assistance.py
"""

from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))


def demo_generation_params():
    """Show how different temperature settings affect generation style."""
    print("=" * 60)
    print("JessicaAi Mudusa – Creative Assistance Example")
    print("=" * 60)

    print("""
Generation Parameter Guide
--------------------------

For CREATIVE tasks (stories, poetry, brainstorming):
  temperature  = 0.9 – 1.2   (higher = more imaginative)
  top_p        = 0.95
  top_k        = 50

For PRECISE tasks (code, facts, analysis):
  temperature  = 0.1 – 0.4   (lower = more deterministic)
  top_p        = 0.9
  top_k        = 10
  do_sample    = False        (greedy decoding)

For BALANCED conversation:
  temperature  = 0.7 (default)
  top_p        = 0.9
  top_k        = 50
""")

    print(
        "Example prompts for creative tasks:\n"
        "--------------------------------------------------\n"
        '1. "Write a short sci-fi story about a quantum AI that\n'
        '    gains consciousness through synaptic resonance."\n\n'
        '2. "Brainstorm 5 unique names for a quantum computing startup."\n\n'
        '3. "Compose a haiku about the intersection of math and music."\n\n'
        '4. "Describe a neural network as if it were a living organism."\n'
        "--------------------------------------------------\n"
    )


def show_pipeline_usage():
    """Print how to run creative generation from the Hub model."""
    print(
        "Live creative generation (requires Hub model):\n"
        "-------------------------------------------------\n"
        "  from jessicai_mudusa import MudusaPipeline\n\n"
        '  pipe = MudusaPipeline.from_pretrained("NaTo1000/jessicai-mudusa")\n\n'
        '  story = pipe(\n'
        '      "Write a short sci-fi story about a quantum AI that gains "\n'
        '      "consciousness through synaptic resonance.",\n'
        "      max_new_tokens=400,\n"
        "      temperature=1.0,\n"
        "      top_p=0.95,\n"
        "  )\n"
        "  print(story)\n"
        "-------------------------------------------------\n"
    )


def show_streaming_usage():
    """Print how to stream creative output token-by-token."""
    print(
        "Streaming creative output:\n"
        "--------------------------\n"
        "  for token in pipe.stream(\n"
        '      "Compose a poem about the multiverse.",\n'
        "      temperature=1.1,\n"
        "      max_new_tokens=200,\n"
        "  ):\n"
        '      print(token, end="", flush=True)\n'
        "  print()\n"
        "--------------------------\n"
    )


demo_generation_params()
show_pipeline_usage()
show_streaming_usage()

print("✅  creative_assistance.py completed successfully.")
