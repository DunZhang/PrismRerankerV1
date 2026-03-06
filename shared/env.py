"""Shared helpers for loading optional environment variables."""

from __future__ import annotations

from pathlib import Path

DEFAULT_PROJECT_ENV_FILE = Path(__file__).resolve().parents[1] / ".env"


def load_optional_dotenv(
    env_file: Path | None = None,
    default_env_file: Path | None = DEFAULT_PROJECT_ENV_FILE,
) -> None:
    """Load a ``.env`` file when one is available and python-dotenv is installed."""
    path = env_file or default_env_file
    if path is None or not path.exists():
        return

    try:
        from dotenv import load_dotenv
    except ImportError:
        return

    load_dotenv(path)
