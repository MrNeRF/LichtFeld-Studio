# SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
# SPDX-License-Identifier: GPL-3.0-or-later
"""Splat simplification rejects non-finite ratios at its Python entry point."""

import struct

import pytest


@pytest.mark.gpu
def test_simplify_splat_data_with_history_rejects_nan_ratio(lf, tmp_path):
    properties = [
        "x", "y", "z", "f_dc_0", "f_dc_1", "f_dc_2", "opacity",
        "scale_0", "scale_1", "scale_2", "rot_0", "rot_1", "rot_2", "rot_3",
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
    source = lf.io.load(str(path)).splat_data
    assert source is not None

    with pytest.raises(ValueError, match="ratio must be finite"):
        lf.simplify_splat_data_with_history(source, ratio=float("nan"), opacity_prune_threshold=0.0)

    with pytest.raises(ValueError, match="ratio must be finite"):
        lf.simplify_splats("missing", ratio=float("nan"))
