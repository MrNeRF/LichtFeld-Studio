"""Neural network inference"""

import os

from numpy.typing import NDArray


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
