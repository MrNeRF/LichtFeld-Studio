# SPDX-FileCopyrightText: 2025 LichtFeld Studio Authors
# SPDX-License-Identifier: GPL-3.0-or-later
"""Plugin utilities for resource management."""

import logging

_log = logging.getLogger(__name__)


def get_gpu_memory() -> int:
    """Get current GPU memory usage in bytes."""
    try:
        import torch
        if torch.cuda.is_available():
            return torch.cuda.memory_allocated()
    except ImportError:
        pass
    return 0


def log_gpu_memory(label: str = ""):
    """Log current GPU memory for debugging."""
    mb = get_gpu_memory() / (1024 * 1024)
    suffix = f" ({label})" if label else ""
    _log.info(f"GPU Memory{suffix}: {mb:.1f} MB")


def cleanup_torch_model(model):
    """Mark model for cleanup.

    Note: We intentionally don't do aggressive GPU cleanup here.
    PyTorch and LichtFeld share the same CUDA context, and calling
    gc.collect() or torch.cuda operations can corrupt shared state.

    Just set references to None and let Python's GC handle it naturally.

    Args:
        model: A PyTorch model or None (will be set to None by caller)
    """
    # Intentionally minimal - aggressive cleanup corrupts shared CUDA context
    pass


def refresh_scene_sidecars(scene=None) -> int:
    """Discover post-import dataset masks for the active scene."""
    if scene is None:
        import lichtfeld as lf
        scene = lf.get_scene()
    return scene.refresh_sidecars() if scene else 0
