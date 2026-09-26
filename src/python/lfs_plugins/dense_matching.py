# SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
# SPDX-License-Identifier: GPL-3.0-or-later
"""Dense image matching for densification, without PyTorch and without numpy.

Wraps the native RoMa v1 matcher in the shape the densification pipeline wants:
one reference image matched against a list of neighbours, returning a per-pixel
warp and a certainty map. RoMa v1 is MIT and its DINOv2 backbone is Apache-2.0,
so the whole model ships with LichtFeld Studio.

Everything here stays on the GPU in lichtfeld tensors. A dataset camera's
``load_image()`` is a valid input as it comes, and the results are tensors the
rest of the pipeline can keep using on the device. Nothing in this module needs
a third-party array library, because LichtFeld Studio does not ship one.

The warp is ``[H, W, 4]``: the first two channels are the reference pixel grid
and the last two are the matched coordinates in the neighbour, both in
normalized ``[-1, 1]`` coordinates, so a row reads ``(x_a, y_a, x_b, y_b)``.
That is the layout the triangulation step consumes, and the last two channels
double as a sampling grid for warping a neighbour's mask.
"""

from __future__ import annotations

from typing import List, Optional, Sequence, Tuple

import lichtfeld as lf

__all__ = ["DenseMatcher", "weights_path", "has_weights"]


def weights_path() -> Optional[str]:
    """Path to the cached weight file, or None if it has not been downloaded.

    Building a matcher downloads it, so this is only for telling the user what
    is about to happen.
    """
    return lf.nn.romav1_weights_path()


def has_weights() -> bool:
    """Whether a matcher can be built without downloading first."""
    return weights_path() is not None


class DenseMatcher:
    """RoMa v1 dense matcher.

    ``prepare`` does the per-image work (the projected feature pyramid) once, so
    matching one reference against many neighbours pays for the backbone a
    single time.

    ``setting`` picks the working resolution: ``fast`` (448) or ``base`` (560).
    Both are multiples of the patch size times the coarsest stride, which is
    what the baked position embedding requires.
    """

    #: Working resolution for each setting.
    SETTINGS = {"fast": 448, "base": 560}

    def __init__(self, weights=None, setting: str = "fast",
                 resolution: int | None = None) -> None:
        if resolution is None:
            if setting not in self.SETTINGS:
                raise ValueError(
                    f"unknown setting {setting!r}; expected one of {sorted(self.SETTINGS)}")
            resolution = self.SETTINGS[setting]
        self._model = lf.nn.RomaV1(weights, resolution)
        self._resolution = int(self._model.resolution)

    @property
    def resolution(self) -> int:
        return self._resolution

    @property
    def is_loaded(self) -> bool:
        """Whether the weights are currently resident on the GPU."""
        return bool(self._model.is_loaded)

    def prepare(self, image):
        """Prepare one image; the result is reusable across matches.

        ``image`` is a lichtfeld tensor, ``[1, 3, H, W]`` or ``[3, H, W]``
        float, or ``[H, W, 3]`` uint8 or float. A camera's ``load_image()``
        already has that shape.
        """
        return self._model.prepare(image)

    def match_grids(self, image_a, image_b) -> Tuple["lf.Tensor", "lf.Tensor"]:
        """Match one pair. Returns (warp [H, W, 4], certainty [H, W])."""
        return self.match_grids_batch(image_a, [image_b])[0]

    def match_grids_batch(self, image_a,
                          images_b: Sequence) -> List[Tuple["lf.Tensor", "lf.Tensor"]]:
        """Match one reference against several neighbours.

        The reference is prepared once. Each result is ``(warp, certainty)`` as
        GPU tensors, with the reference grid already in the warp's first two
        channels.
        """
        if not images_b:
            return []
        prepared_a = self.prepare(image_a)
        results: List[Tuple["lf.Tensor", "lf.Tensor"]] = []
        for image_b in images_b:
            prepared_b = self.prepare(image_b)
            results.append(self._model.match_grid(prepared_a, prepared_b))
        return results

    def match_pairs(self, prepared_a, prepared_b) -> Tuple["lf.Tensor", "lf.Tensor"]:
        """Match two already-prepared images.

        Returns (warp [H, W, 2], certainty [H, W]) as GPU tensors, with no
        reference grid attached.
        """
        return self._model.match_gpu(prepared_a, prepared_b)

    def close(self) -> None:
        """Release the weights and their device memory.

        The matcher stays usable: the next call reloads. Densification holds one
        matcher for a whole run and frees it before training resumes.
        """
        self._model.close()

    def __enter__(self) -> "DenseMatcher":
        return self

    def __exit__(self, *_exc) -> None:
        self.close()
