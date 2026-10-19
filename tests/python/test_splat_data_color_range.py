# SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
# SPDX-License-Identifier: GPL-3.0-or-later
"""SplatData RGB colors obey the documented finite [0, 1] range."""

import pytest
import struct


def _load_splat(lf, tmp_path):
    properties = [
        "x",
        "y",
        "z",
        "f_dc_0",
        "f_dc_1",
        "f_dc_2",
        "opacity",
        "scale_0",
        "scale_1",
        "scale_2",
        "rot_0",
        "rot_1",
        "rot_2",
        "rot_3",
    ]
    rows = [
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0],
        [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0],
    ]
    header = [
        "ply",
        "format binary_little_endian 1.0",
        f"element vertex {len(rows)}",
        *(f"property float {name}" for name in properties),
        "end_header",
    ]
    path = tmp_path / "splat.ply"
    path.write_bytes(
        "\n".join(header).encode("ascii")
        + b"\n"
        + b"".join(struct.pack("<" + "f" * len(properties), *row) for row in rows)
    )
    result = lf.io.load(str(path))
    assert result.splat_data is not None
    return result.splat_data


@pytest.mark.gpu
@pytest.mark.parametrize("device", ["cpu", pytest.param("cuda", marks=pytest.mark.gpu)])
@pytest.mark.parametrize(
    "colors",
    [
        [[2.0, -1.0, 0.5], [0.0, 0.0, 0.0]],
        [[float("nan"), 0.0, 0.0], [0.0, 0.0, 0.0]],
        [[float("inf"), 0.0, 0.0], [0.0, 0.0, 0.0]],
    ],
    ids=["outside-range", "nan", "infinity"],
)
def test_set_colors_rgb_rejects_invalid_values(lf, numpy, gpu_available, tmp_path, colors, device):
    if not gpu_available:
        pytest.skip("GPU not available")

    splat = _load_splat(lf, tmp_path)
    values = lf.Tensor.from_numpy(numpy.asarray(colors, dtype=numpy.float32))
    if device == "cuda":
        values = values.cuda()

    with pytest.raises(ValueError, match=r"finite and within \[0, 1\]"):
        splat.set_colors_rgb(values)


@pytest.mark.gpu
@pytest.mark.parametrize("device", ["cpu", pytest.param("cuda", marks=pytest.mark.gpu)])
def test_set_colors_rgb_preserves_valid_values(lf, numpy, gpu_available, tmp_path, device):
    if not gpu_available:
        pytest.skip("GPU not available")

    splat = _load_splat(lf, tmp_path)
    expected = numpy.asarray([[0.0, 0.5, 1.0], [1.0, 0.25, 0.0]], dtype=numpy.float32)
    values = lf.Tensor.from_numpy(expected)
    if device == "cuda":
        values = values.cuda()
    splat.set_colors_rgb(values)

    numpy.testing.assert_allclose(splat.get_colors_rgb().cpu().numpy(), expected, atol=1e-6)
