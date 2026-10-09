# SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
# SPDX-License-Identifier: GPL-3.0-or-later
"""Real Studio image-output binding contracts; reuse the independent EXR reader."""

import importlib.util
import json
from pathlib import Path
import struct
import os
import subprocess
import sys

import pytest


@pytest.mark.parametrize("channels,extension", [(c, e) for c in (1, 2, 3, 4)
                                             for e in ("png", "jpg", "tiff") if not (c == 2 and e == "jpg")])
def test_byte_export_independent_decoder(lf, tmp_path, channels, extension):
    path = tmp_path / f"export 日本語-{channels}.{extension}"
    lf.io.save_image(str(path), lf.Tensor.full([channels, 8, 19], 0.5, device="cpu"), False)
    if extension == "tiff":
        data = path.read_bytes()
        order = "<" if data[:2] == b"II" else ">"
        position = struct.unpack_from(order + "I", data, 4)[0]
        count = struct.unpack_from(order + "H", data, position)[0]
        entries = [struct.unpack_from(order + "HHII", data, position + 2 + i * 12) for i in range(count)]
        tags = {entry[0]: entry for entry in entries}
        assert tags[278][3] > 0
        assert tags[262][3] == (1 if channels < 3 else 2)
        if channels in (2, 4):
            assert tags[338][3] == 2
    if extension == "jpg" and sys.platform == "darwin":
        converted = path.with_suffix(".imageio.png")
        subprocess.run(["/usr/bin/sips", "-s", "format", "png", str(path), "--out", str(converted)],
                       check=True, capture_output=True)
        path = converted
    pixel_format = "rgba" if channels in (2, 4) and extension != "jpg" else "rgb24"
    raw = subprocess.check_output([os.environ.get("LFS_MEDIA_TEST_FFMPEG", "ffmpeg"), "-v", "error",
                                   "-i", str(path), "-f", "rawvideo", "-pix_fmt", pixel_format, "-"])
    assert len(raw) == 19 * 8 * (4 if pixel_format == "rgba" else 3)
    assert set(raw) == {128}


def test_float_render_requires_active_viewer(lf):
    with pytest.raises(RuntimeError, match="No active viewer"):
        lf.render_linear_image(32, 32)


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
