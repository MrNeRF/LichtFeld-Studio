"""Neural network inference"""

import os

from numpy.typing import NDArray

import lichtfeld


def softmax(input: lichtfeld.Tensor, mask: lichtfeld.Tensor | None = None) -> lichtfeld.Tensor:
    """
    Softmax over the last dimension with an optional broadcastable additive mask
    """

def silu(input: lichtfeld.Tensor) -> lichtfeld.Tensor:
    """SiLU activation: x * sigmoid(x)"""

def rms_norm(input: lichtfeld.Tensor, weight: lichtfeld.Tensor, eps: float = 9.999999974752427e-07) -> lichtfeld.Tensor:
    """RMS normalization over the last dimension followed by channel weights"""

def residual_scale(x: lichtfeld.Tensor, hidden: lichtfeld.Tensor, gamma: lichtfeld.Tensor) -> lichtfeld.Tensor:
    """
    Compute x + hidden * gamma with gamma broadcast over the last dimension
    """

def window_partition(input: lichtfeld.Tensor, window_size: int) -> lichtfeld.Tensor:
    """Partition [B,H,N,d] into [B*n_windows,H,window_size,d], zero-padding N"""

def window_unpartition(windows: lichtfeld.Tensor, window_size: int, original_n: int) -> lichtfeld.Tensor:
    """Restore [B,H,N,d] from 1D windows and crop padding to original_n"""

class Sam2:
    """SAM 2.1 image predictor"""

    def __init__(self, weights: str | os.PathLike | None = None) -> None:
        """
        Create a SAM 2.1 predictor. weights=None resolves the default cached .lfw via ensure_sam2_weights (downloads on first use).
        """

    def set_image(self, image: NDArray) -> None:
        """
        Set the image for prompting. numpy HWC uint8 or float32 RGB in [0, 1], any size.
        """

    def predict(self, points: object | None = None, labels: object | None = None, box: object | None = None, multimask: bool = True) -> tuple:
        """
        Predict masks from point and/or box prompts. Returns (masks [N,H,W] float32 logits, scores [N] float32).
        """

class RomaV1Image:
    """An image prepared for RoMa v1 matching."""

class RomaV1:
    """RoMa v1 dense feature matcher (MIT, on an Apache-2.0 DINOv2 backbone)"""

    def __init__(self, weights: str | os.PathLike | None = None, resolution: int = 448) -> None:
        """
        Create a RoMa v1 matcher. weights=None resolves the default cached .lfw via ensure_romav1_weights (downloads on first use). resolution must be one the weight file baked a position embedding for.
        """

    def prepare(self, image: lichtfeld.Tensor) -> RomaV1Image:
        """
        Prepare one image for matching. A lichtfeld Tensor: [1,3,H,W] or [3,H,W] float, or [H,W,3] uint8/float. A camera's load_image() already has that shape.
        """

    def match_gpu(self, a: RomaV1Image, b: RomaV1Image) -> tuple:
        """Match two prepared images, returning GPU lichtfeld tensors."""

    def match_grid(self, a: RomaV1Image, b: RomaV1Image) -> tuple:
        """
        Match two prepared images and prepend the reference pixel grid. Returns GPU lichtfeld tensors (warp [R,R,4] of (x_a, y_a, x_b, y_b), certainty [R,R]).
        """

    def close(self) -> None:
        """
        Release the weights and their device memory. The next call reloads them.
        """

    @property
    def weights_bytes(self) -> int:
        """Resident weight bytes, or zero before loading and after close."""

    @property
    def resolution(self) -> int:
        """Working resolution of the matcher (square, in pixels)."""

    @property
    def is_loaded(self) -> bool:
        """Whether the weights are currently resident on the device."""

def romav1_weights_path() -> str | None:
    """
    Path to the cached RoMa v1 weight file, or None if it has not been downloaded yet. Building a matcher downloads it.
    """
