# SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
# SPDX-License-Identifier: GPL-3.0-or-later

"""Check the actual mask's distribution and both required blue-noise spectra."""

from pathlib import Path

import numpy as np
import pytest


@pytest.fixture(scope="module")
def volume():
    root = Path(__file__).resolve().parents[2]
    data = root / "src/rendering/rasterizer/vulkan/assets/stbn_128x128x64.bin"
    return np.frombuffer(data.read_bytes(), dtype=np.uint8).reshape(64, 128, 128)


def test_scalar_stbn_has_uniform_marginal(volume):
    np.testing.assert_array_equal(np.bincount(volume.ravel()), np.full(256, 4096))


@pytest.mark.parametrize("threshold", [26, 64, 128, 192, 230])
def test_thresholded_masks_are_blue_in_space_and_time(volume, threshold):
    # A stack of independent 2D masks fails the temporal test. Ordinary 3D
    # blue noise fails the slice test. Test binary visibility, not just values.
    visible = (volume < threshold).astype(np.float64)
    spectrum_xy = np.abs(np.fft.fft2(visible, axes=(-2, -1))) ** 2
    frequencies = np.fft.fftfreq(128) * 128
    radius = np.hypot(frequencies[:, None], frequencies[None, :])
    low_xy = spectrum_xy[:, (radius > 0) & (radius < 8)].mean()
    high_xy = spectrum_xy[:, radius > 32].mean()
    assert low_xy / high_xy < 0.12

    spectrum_z = np.abs(np.fft.rfft(visible, axis=0)) ** 2
    assert spectrum_z[1:5].mean() / spectrum_z[16:].mean() < 0.5
