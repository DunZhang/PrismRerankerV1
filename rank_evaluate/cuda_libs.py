"""Pre-load CUDA shared libraries bundled with PyTorch's nvidia packages.

On systems where libcudart.so.12 etc. are not in the system library path
(e.g. WSL2 without a system-level CUDA toolkit), llama-cpp-python's pre-built
CUDA wheels fail to import because the dynamic linker cannot find the libraries.

This module explicitly loads every .so file from the nvidia packages installed
in the current venv, making them available to any subsequent ctypes.CDLL call.

Must be imported BEFORE `import llama_cpp`.
"""

import ctypes
import sys
from pathlib import Path


def preload_cuda_libs() -> None:
    """Load CUDA .so files from the venv's nvidia packages into the process."""
    nvidia_dir = Path(
        sys.prefix,
        f"lib/python{sys.version_info.major}.{sys.version_info.minor}/site-packages/nvidia",
    )
    if not nvidia_dir.exists():
        return  # system CUDA or no nvidia packages installed

    # Collect all .so files (only real files, not symlinks)
    so_files = sorted(
        so
        for lib_dir in nvidia_dir.glob("*/lib")
        for so in lib_dir.glob("*.so.*")
        if so.is_file() and not so.is_symlink()
    )

    # Two-pass load: some libs depend on others; second pass catches stragglers
    for _ in range(2):
        for so in so_files:
            try:
                ctypes.CDLL(str(so))
            except OSError:
                pass
