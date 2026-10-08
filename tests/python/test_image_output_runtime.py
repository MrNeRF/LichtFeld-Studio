# SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
# SPDX-License-Identifier: GPL-3.0-or-later
"""Real Studio image-output binding contracts; reuse the independent EXR reader."""

import importlib.util
import json
from pathlib import Path
import struct

import pytest


@pytest.fixture(scope="module")
def exr_reader():
    path = Path(__file__).parents[1] / "media" / "test_float_exr.py"
    spec = importlib.util.spec_from_file_location("image_output_exr_reference", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.uncompressed_exr


@pytest.mark.parametrize("dtype", ["float16", "float32"])
@pytest.mark.parametrize("precision", ["Half", "Float"])
@pytest.mark.parametrize("channels", [3, 4])
@pytest.mark.parametrize("value", [-0.25, 0.1, 2.0])
def test_save_exr_image_preserves_declared_float_samples(lf, tmp_path, exr_reader, dtype, precision, channels, value):
    image = lf.Tensor.full([2, 3, 3], value, device="cpu", dtype=dtype)
    if channels == 4:
        alpha = lf.Tensor.full([2, 3, 1], 0.5, device="cpu", dtype=dtype)
        image = lf.Tensor.cat([image, alpha], dim=2)
    color = lf.media.FrameColor()
    color.transfer = lf.media.ColorTransfer.Linear
    color.primaries = lf.media.ColorPrimaries.Bt709
    color.alpha = lf.media.AlphaMode.Straight if channels == 4 else lf.media.AlphaMode.NoAlpha
    options = lf.media.ExrOutputOptions()
    options.precision = getattr(lf.media.ExrPrecision, precision)
    options.compression = lf.media.ExrCompression.Uncompressed
    path = tmp_path / "linear 日本語.exr"
    lf.io.save_exr_image(str(path), image, color, options, include_provenance=False)
    width, height, encoded_channels, attrs, pixels = exr_reader(path)
    assert (width, height) == (3, 2)
    assert {kind for _, kind in encoded_channels} == {1 if precision == "Half" else 2}
    source_code = "e" if dtype == "float16" else "f"
    expected = struct.unpack("<" + source_code, struct.pack("<" + source_code, value))[0]
    if channels == 4:
        expected *= 0.5
        assert pixels["A"] == [0.5] * 6
    if precision == "Half":
        expected = struct.unpack("<e", struct.pack("<e", expected))[0]
    for channel in "RGB":
        assert pixels[channel] == [expected] * 6
    assert attrs["lfsOrigin"][1] == b"external"
    assert "lfsSourceTimestamp" not in attrs
    assert "lfsOutputTimestamp" not in attrs
    assert isinstance(json.loads(attrs["lfsProvenance"][1]), dict)


def test_save_exr_image_rejects_display_color_and_byte_input_without_replacing_target(lf, tmp_path):
    path = tmp_path / "keep.exr"
    path.write_bytes(b"keep")
    options = lf.media.ExrOutputOptions()
    options.overwrite = True
    color = lf.media.FrameColor()
    color.transfer = lf.media.ColorTransfer.Srgb
    color.primaries = lf.media.ColorPrimaries.Bt709
    for dtype in ("float32", "uint8"):
        image = lf.Tensor.full([2, 3, 3], 1, device="cpu", dtype=dtype)
        with pytest.raises(Exception) as error:
            lf.io.save_exr_image(str(path), image, color, options)
        assert "linear" in str(error.value) if dtype == "float32" else "Float16/Float32" in str(error.value)
        assert path.read_bytes() == b"keep"


def test_save_exr_image_preserves_strided_hwc_order_and_explicit_provenance(lf, tmp_path, exr_reader):
    image = lf.Tensor.arange(0, 18, device="cpu", dtype="float32").reshape([3, 2, 3]).transpose(0, 1)
    assert not image.is_contiguous
    color = lf.media.FrameColor()
    color.transfer = lf.media.ColorTransfer.Linear
    color.primaries = lf.media.ColorPrimaries.Bt709
    color.alpha = lf.media.AlphaMode.NoAlpha
    options = lf.media.ExrOutputOptions()
    options.precision = lf.media.ExrPrecision.Float
    options.compression = lf.media.ExrCompression.Uncompressed
    options.provenance = '{"caller":"preserve"}'
    path = tmp_path / "strided.exr"
    lf.io.save_exr_image(str(path), image, color, options)
    width, height, _, attrs, pixels = exr_reader(path)
    assert (width, height) == (3, 2)
    for channel, offset in zip("RGB", range(3)):
        assert pixels[channel] == [float(offset + 3 * (x * 2 + y)) for y in range(2) for x in range(3)]
    assert json.loads(attrs["lfsProvenance"][1]) == {"caller": "preserve"}
