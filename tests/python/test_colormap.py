# SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
# SPDX-License-Identifier: GPL-3.0-or-later
"""The colormap helper accepts one value per item and returns RGB rows."""

import pytest


def test_colormap_rejects_rank_two_values(lf):
    values = lf.Tensor.zeros((2, 2), device="cpu")

    with pytest.raises(ValueError, match="rank 1"):
        lf.colormap(values)


@pytest.mark.parametrize("device", ["cpu", pytest.param("cuda", marks=pytest.mark.gpu)])
def test_colormap_preserves_documented_vector_behavior(lf, numpy, gpu_available, device):
    if device == "cuda" and not gpu_available:
        pytest.skip("GPU not available")

    values = lf.Tensor.from_numpy(numpy.array([0.0, 0.5, 1.0], dtype=numpy.float32))
    if device == "cuda":
        values = values.cuda()

    result = lf.colormap(values)

    assert result.shape == (3, 3)
    assert result.device == device
    numpy.testing.assert_allclose(
        result.cpu().numpy(),
        numpy.array([[0.0, 0.0, 1.0], [0.5, 1.0, 0.0], [0.5, 0.0, 0.0]], dtype=numpy.float32),
    )
