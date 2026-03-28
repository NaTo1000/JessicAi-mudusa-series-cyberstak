"""
Command-line interface for the JessicAi HuggingFace Fork.

Usage
-----
::

    # Enroll voice-print (run once)
    python -m huggingface_fork.cli enroll-voice

    # Unlock and load a model
    python -m huggingface_fork.cli load-model bert-base-uncased \\
        --signed-token @/path/to/token.asc \\
        --voice-file /path/to/sample.wav

    # Unlock and load a dataset
    python -m huggingface_fork.cli load-dataset squad \\
        --split train \\
        --signed-token @/path/to/token.asc \\
        --voice-file /path/to/sample.wav

    # Check auth status
    python -m huggingface_fork.cli auth-status
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

logging.basicConfig(
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    level=logging.INFO,
    stream=sys.stderr,
)


def _read_token(value: str) -> str:
    """Support ``@path`` syntax for reading a signed token from a file."""
    if value.startswith("@"):
        path = Path(value[1:])
        if not path.exists():
            print(f"ERROR: Token file not found: {path}", file=sys.stderr)
            sys.exit(1)
        return path.read_text()
    return value


def cmd_enroll_voice(args: argparse.Namespace) -> int:
    from huggingface_fork.auth.voice_auth import VoiceAuthenticator

    va = VoiceAuthenticator()
    audio_files = [Path(f) for f in args.files] if args.files else None
    try:
        va.enroll(audio_files)
        print("Voice-print enrolled successfully.")
        return 0
    except Exception as exc:
        print(f"Enrollment failed: {exc}", file=sys.stderr)
        return 1


def cmd_load_model(args: argparse.Namespace) -> int:
    from huggingface_fork.models.loader import ModelLoader

    loader = ModelLoader(session_id=args.session_id)
    if args.signed_token:
        token_str = _read_token(args.signed_token)
        voice_file = Path(args.voice_file) if args.voice_file else None
        ok = loader.unlock(token_str, audio_file=voice_file)
        if not ok:
            print("Authentication failed. Running in restricted mode.", file=sys.stderr)
    try:
        model = loader.load(args.model_id, task=args.task)
        print(f"Model loaded: {model.__class__.__name__}")
        return 0
    except Exception as exc:
        print(f"Failed to load model: {exc}", file=sys.stderr)
        return 1


def cmd_load_dataset(args: argparse.Namespace) -> int:
    from huggingface_fork.datasets.loader import DatasetLoader

    loader = DatasetLoader(session_id=args.session_id)
    if args.signed_token:
        token_str = _read_token(args.signed_token)
        voice_file = Path(args.voice_file) if args.voice_file else None
        ok = loader.unlock(token_str, audio_file=voice_file)
        if not ok:
            print("Authentication failed. Running in restricted mode.", file=sys.stderr)
    try:
        ds = loader.load(args.dataset_id, subset=args.subset, split=args.split)
        print(f"Dataset loaded: {ds}")
        return 0
    except Exception as exc:
        print(f"Failed to load dataset: {exc}", file=sys.stderr)
        return 1


def cmd_auth_status(args: argparse.Namespace) -> int:
    from huggingface_fork.auth.voice_auth import VoiceAuthenticator
    from huggingface_fork.config import load_config

    cfg = load_config()
    va = VoiceAuthenticator(cfg.auth)
    print(f"Authorised user  : {cfg.auth.authorized_user}")
    print(f"GPG fingerprint  : {cfg.auth.gpg_fingerprint or '(not configured)'}")
    print(f"Voice enrolled   : {va.is_enrolled()}")
    print(f"HF Hub token set : {bool(cfg.huggingface.hub_token)}")
    print(f"Cache directory  : {cfg.huggingface.cache_dir}")
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="jessicai-hf",
        description="JessicAi HuggingFace Fork — autonomous AI with GPG+voice auth",
    )
    parser.add_argument(
        "--session-id",
        default="cli",
        help="Session identifier used in audit logs (default: cli)",
    )

    sub = parser.add_subparsers(dest="command", required=True)

    # enroll-voice
    p_enroll = sub.add_parser("enroll-voice", help="Enroll a voice-print")
    p_enroll.add_argument(
        "files",
        nargs="*",
        help="WAV files to use for enrollment (omit to use microphone)",
    )
    p_enroll.set_defaults(func=cmd_enroll_voice)

    # load-model
    p_model = sub.add_parser("load-model", help="Load a HuggingFace model")
    p_model.add_argument("model_id", help="HuggingFace model identifier")
    p_model.add_argument("--task", help="Pipeline task (e.g. text-classification)")
    p_model.add_argument("--signed-token", help="GPG-signed auth token or @path/to/file")
    p_model.add_argument("--voice-file", help="WAV voice sample for voice auth")
    p_model.set_defaults(func=cmd_load_model)

    # load-dataset
    p_ds = sub.add_parser("load-dataset", help="Load a HuggingFace dataset")
    p_ds.add_argument("dataset_id", help="HuggingFace dataset identifier")
    p_ds.add_argument("--subset", help="Dataset configuration/subset name")
    p_ds.add_argument("--split", help="Dataset split (train/validation/test)")
    p_ds.add_argument("--signed-token", help="GPG-signed auth token or @path/to/file")
    p_ds.add_argument("--voice-file", help="WAV voice sample for voice auth")
    p_ds.set_defaults(func=cmd_load_dataset)

    # auth-status
    p_status = sub.add_parser("auth-status", help="Display authentication configuration status")
    p_status.set_defaults(func=cmd_auth_status)

    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
