# SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
# SPDX-License-Identifier: GPL-3.0-or-later
"""Tensor factories reject negative dimensions before allocation."""

import pytest


@pytest.mark.parametrize("device", ["cpu", pytest.param("gpu", marks=pytest.mark.gpu)])
@pytest.mark.parametrize("shape", [(-1,), (2, -1)])
def test_tensor_factories_reject_negative_dimensions(lf, device, shape):
    with pytest.raises(ValueError, match="negative dimension"):
        lf.Tensor.zeros(shape, device=device, dtype="uint8")
