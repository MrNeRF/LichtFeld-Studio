# SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
# SPDX-License-Identifier: GPL-3.0-or-later
"""Smoke test for lichtfeld.nn.Sam2 image masking."""

import os

import pytest


@pytest.mark.gpu
def test_sam2_predict_shapes(lf, numpy):
    weights = os.environ.get("LFS_SAM2_WEIGHTS")
    if not weights:
        pytest.skip("set LFS_SAM2_WEIGHTS to run SAM2 binding smoke test")

    h, w = 32, 48
    img = numpy.zeros((h, w, 3), dtype=numpy.uint8)
    img[8:24, 12:36] = 255

    model = lf.nn.Sam2(weights=weights)
    model.set_image(img)
    masks, scores = model.predict(points=[[24.0, 16.0]], labels=[1], multimask=True)

    assert masks.dtype == numpy.float32
    assert scores.dtype == numpy.float32
    assert masks.shape == (3, h, w)
    assert scores.shape == (3,)
    assert numpy.isfinite(scores).all()
    assert float(scores.min()) >= 0.0
    assert float(scores.max()) <= 1.0
