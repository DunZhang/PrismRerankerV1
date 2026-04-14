"""Runtime configuration helpers for reranker evaluation."""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

from shared.env import DEFAULT_PROJECT_ENV_FILE, load_optional_dotenv

PACKAGE_DIR = Path(__file__).parent
DEFAULT_OUTPUT = PACKAGE_DIR / "evaluation_results.xlsx"
DEFAULT_CACHE_DIR = PACKAGE_DIR / "cache"
DEFAULT_ENV_FILE = DEFAULT_PROJECT_ENV_FILE


def load_env(env_file: Path | None = None) -> None:
    """Load environment variables from a .env file if python-dotenv is installed."""
    load_optional_dotenv(env_file=env_file, default_env_file=DEFAULT_ENV_FILE)


def resolve_data_dir(data_dir: Path | None) -> Path:
    """Resolve the benchmark data directory from CLI input or environment."""
    if data_dir is not None:
        return data_dir

    env_dir = os.environ.get("POSIR_DATA_DIR")
    if env_dir:
        return Path(env_dir)

    raise ValueError(
        "--data_dir not specified and POSIR_DATA_DIR environment variable is not set."
    )


def make_run_tag(num_neg: int, seed: int) -> str:
    """Build the cache subdirectory name for a specific evaluation setup."""
    return f"neg{num_neg}_seed{seed}"


@dataclass(slots=True, frozen=True)
class EvaluationConfig:
    """Resolved runtime configuration for one evaluation run."""

    model_name: str
    model_path: Path | None
    num_neg: int
    data_dir: Path
    output_path: Path
    cache_dir: Path
    seed: int
    max_queries: int | None = None

    @property
    def run_tag(self) -> str:
        return make_run_tag(self.num_neg, self.seed)

    def validate(self) -> None:
        """Raise ValueError when the configuration is inconsistent."""
        if self.num_neg < 0:
            raise ValueError("--num_neg must be >= 0.")
        if self.max_queries is not None and self.max_queries <= 0:
            raise ValueError("--max_queries must be > 0 when provided.")
        if not self.data_dir.exists():
            raise ValueError(f"data_dir not found: {self.data_dir}")
        if not self.data_dir.is_dir():
            raise ValueError(f"data_dir is not a directory: {self.data_dir}")
