# SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
# SPDX-License-Identifier: GPL-3.0-or-later
"""Inverse trig operations retain their mathematical real-domain behavior."""

import pytest


@pytest.mark.parametrize("device", ["cpu", "cuda"])
@pytest.mark.parametrize("operation", ["asin", "acos"])
def test_inverse_trig_domain_matches_numpy(lf, numpy, device, operation):
    values = numpy.array([-numpy.inf, -2, -1.01, -1, -.5, -0., 0., .5, 1, 1.01, 2, numpy.inf, numpy.nan], dtype=numpy.float32)
    tensor = getattr(lf.Tensor.from_numpy(values), device)()
    with numpy.errstate(invalid="ignore"):
        expected = getattr(numpy, "arc" + operation[1:])(values)
    result = getattr(tensor, operation)().numpy()
    numpy.testing.assert_array_equal(numpy.isnan(result), numpy.isnan(expected))
    numpy.testing.assert_allclose(result, expected, rtol=1e-6, atol=1e-6, equal_nan=True)


@pytest.mark.parametrize("device", ["cpu", "cuda"])
def test_inverse_trig_just_outside_endpoints(lf, numpy, device):
    values = numpy.array([numpy.nextafter(numpy.float32(-1), numpy.float32(-2)), numpy.nextafter(numpy.float32(1), numpy.float32(2))])
    tensor = getattr(lf.Tensor.from_numpy(values), device)()
    assert numpy.isnan(tensor.asin().numpy()).all()
    assert numpy.isnan(tensor.acos().numpy()).all()
