# SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
# SPDX-License-Identifier: GPL-3.0-or-later
"""Production Media Ingest bindings: GIL, ownership, file parity and callbacks."""
import argparse
import concurrent.futures
import errno
import hashlib
import importlib.util
import json
import math
import os
import shutil
from pathlib import Path
import sys
import subprocess
import struct
import tempfile
import threading
import unicodedata
import unittest

ROOT = Path(__file__).resolve().parents[2]
_exif_spec = importlib.util.spec_from_file_location("source_exif_fixture", ROOT / "tests/media/source_exif_fixture.py")
source_exif_fixture = importlib.util.module_from_spec(_exif_spec)
_exif_spec.loader.exec_module(source_exif_fixture)
_jpeg_spec = importlib.util.spec_from_file_location("jpeg_metadata_fixture", ROOT / "tests/media/jpeg_metadata_fixture.py")
jpeg_metadata_fixture = importlib.util.module_from_spec(_jpeg_spec)
_jpeg_spec.loader.exec_module(jpeg_metadata_fixture)
_png_spec = importlib.util.spec_from_file_location("png_metadata_fixture", ROOT / "tests/media/png_metadata_fixture.py")
png_metadata_fixture = importlib.util.module_from_spec(_png_spec)
_png_spec.loader.exec_module(png_metadata_fixture)
spec = importlib.util.spec_from_file_location("media_fixtures", ROOT / "scripts/prepare_media_fixtures.py")
fixtures = importlib.util.module_from_spec(spec)
spec.loader.exec_module(fixtures)
media = None
MEDIA_ERROR = None
METADATA_CLI = None


class SourceMetadataBindings(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(prefix="lfs-python-metadata-")
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)

    def test_catalog_and_summary_match_cli_and_legacy_exif(self):
        cli = METADATA_CLI
        if cli is None or not cli.is_file():
            self.fail("--cli must identify the production metadata CLI")
        for carrier in ("jpeg", "png", "tiff", "limited_tiff"):
            with self.subTest(carrier=carrier):
                path = self.root / ("metadata é 日本語." + ("tiff" if carrier == "limited_tiff" else carrier))
                if carrier == "limited_tiff":
                    source_exif_fixture.write_tiff(path, xmp_packets=[b"x" * (1024 * 1024 + 1)])
                else:
                    source_exif_fixture.write_camera_metadata(path, carrier=carrier)
                original, stamp = path.read_bytes(), path.stat().st_mtime_ns
                encoded = media.MediaIngest.source_metadata_json(path)
                result = json.loads(encoded)
                reference = subprocess.run([str(cli), "source-metadata", str(path)],
                                           check=True, capture_output=True, timeout=30)
                self.assertEqual(result, json.loads(reference.stdout))
                self.assertEqual(result["schema_version"], 1)
                self.assertTrue(result["success"])
                self.assertEqual(result["metadata"]["image_exif"],
                                 json.loads(media.MediaIngest.source_exif_json(path)))
                if carrier == "limited_tiff":
                    self.assertEqual(result["metadata"]["image_xmp"]["status"], "limited")
                    self.assertTrue(result["summary"]["limited"])
                else:
                    facts = {f["semantic"]: f for f in result["summary"]["facts"]}
                    self.assertEqual(facts["camera.make"]["value"], "Test Camera")
                self.assertEqual((path.read_bytes(), path.stat().st_mtime_ns), (original, stamp))
                path.unlink()
                self.assertEqual(json.loads(encoded), result)

    def test_remaining_carriers_use_the_same_catalog_contract(self):
        for carrier in ("jxl", "avif", "heic"):
            with self.subTest(carrier=carrier):
                path = self.root / ("metadata." + carrier)
                if carrier == "jxl":
                    shutil.copyfile(ROOT / "tests/data/jxl/rgb8.jxl", path)
                else:
                    source_exif_fixture.write_heif_metadata(path, carrier=carrier)
                original, stamp = path.read_bytes(), path.stat().st_mtime_ns
                result = json.loads(media.MediaIngest.source_metadata_json(path))
                reference = subprocess.run([str(METADATA_CLI), "source-metadata", str(path)],
                                           check=True, capture_output=True, timeout=30)
                self.assertEqual(result, json.loads(reference.stdout))
                self.assertEqual((path.read_bytes(), path.stat().st_mtime_ns), (original, stamp))

    def test_concurrent_calls_keep_results_independent(self):
        valid = self.root / "valid.tiff"
        invalid = self.root / "invalid.tiff"
        source_exif_fixture.write_tiff(valid)
        invalid.write_bytes(b"II*\0")
        expected = media.MediaIngest.source_metadata_json(valid)
        def read(index):
            return index, json.loads(media.MediaIngest.source_metadata_json(valid if index % 2 else invalid))
        with concurrent.futures.ThreadPoolExecutor(max_workers=4) as pool:
            for index, result in pool.map(read, range(24)):
                if index % 2:
                    self.assertEqual(result, json.loads(expected))
                else:
                    self.assertEqual(result["metadata"]["image_exif"]["status"], "malformed")

    @unittest.skipIf(os.name == "nt", "Native non-UTF-8 filename bytes are POSIX-specific")
    def test_native_filename_bytes_and_pathlike(self):
        native = os.fsencode(self.root) + b"/metadata-\xff.jpg"
        path = Path(os.fsdecode(native))
        try:
            source_exif_fixture.write_jpeg(path)
        except OSError as error:
            if error.errno == errno.EILSEQ:
                self.skipTest("The filesystem rejects non-UTF-8 filename bytes")
            raise
        class NativePath:
            def __fspath__(self):
                return os.fsdecode(native)
        result = json.loads(media.MediaIngest.source_metadata_json(NativePath()))
        self.assertEqual(result["metadata"]["source_locator_bytes"], native.hex())
        self.assertTrue(result["summary"]["facts"])


class MediaBindings(unittest.TestCase):
    def test_tiff_decoded_tags_and_original_source_ranges(self):
        path = self.root / "TIFF metadata é 日本語.bin"
        for big in (False, True):
            for little in (False, True):
                with self.subTest(big=big, little=little):
                    payload = source_exif_fixture.write_tiff(path, big=big, little=little, pages=2)
                    value = json.loads(media.MediaIngest.source_exif_json(path))
                    source_exif_fixture.check_tiff(self, value, path, payload, big=big, little=little, pages=2)
                    maker = source_exif_fixture.write_tiff_graph(path, big=big, little=little)
                    value = json.loads(media.MediaIngest.source_exif_json(path))
                    source_exif_fixture.check_tiff_graph(self, value, path, maker)

    def test_tiff_review_budgets_cycles_and_baseline_fields(self):
        path = self.root / "review-budgets.tif"
        for big in (False, True):
            for little in (False, True):
                with self.subTest(big=big, little=little):
                    blobs = [(700, b"x" * (600 * 1024)), (34377, b"p" * (500 * 1024))]
                    source_exif_fixture.write_tiff_graph(path, big=big, little=little, blobs=blobs, alpha=True)
                    value = json.loads(media.MediaIngest.source_exif_json(path))
                    self.assertEqual(value["status"], "collected", value.get("diagnostic"))
                    self.assertIn("exif", {node["kind"] for node in value["tiff"]["directories"]})
                    tags = {tag["id"]: tag for tag in value["tiff"]["directories"][0]["tags"]}
                    for id_, blob in blobs:
                        self.assertEqual(bytes.fromhex(tags[id_]["bytes"]), blob)
                    for id_, expected in ((254, [1]), (266, [1]), (338, [2])):
                        self.assertEqual(tags[id_]["unsigned_values"], expected)
                    source_exif_fixture.write_tiff_graph(path, big=big, little=little, subimages=70, blobs=[(65000, b"retained-root")])
                    value = json.loads(media.MediaIngest.source_exif_json(path))
                    self.assertEqual(value["status"], "limited")
                    tags = {tag["id"]: tag for tag in value["tiff"]["directories"][0]["tags"]}
                    self.assertEqual(tags[274]["unsigned_values"], [6])
                    self.assertEqual(bytes.fromhex(tags[65000]["bytes"]), b"retained-root")
                    for cycle in (True, "exif", "self-subimage"):
                        source_exif_fixture.write_tiff_graph(path, big=big, little=little, cycle=cycle)
                        value = json.loads(media.MediaIngest.source_exif_json(path))
                        self.assertEqual(value["status"], "malformed", value.get("diagnostic"))
                    source_exif_fixture.write_tiff_graph(path, big=big, little=little, subimages=2)
                    value = json.loads(media.MediaIngest.source_exif_json(path))
                    self.assertEqual(value["status"], "unsupported", "A shared directory alias is not an ancestor cycle")
        source_exif_fixture.write_tiff_graph(path, blobs=[(700, b"x" * 1153433)])
        value = json.loads(media.MediaIngest.source_exif_json(path))
        self.assertEqual(value["status"], "collected", value.get("diagnostic"))
        self.assertIn("exif", {node["kind"] for node in value["tiff"]["directories"]})
        source_exif_fixture.write_tiff_graph(path, blobs=[(37724, bytes(20 * 1024 * 1024))])
        value = json.loads(media.MediaIngest.source_exif_json(path))
        self.assertEqual(value["status"], "limited", value.get("diagnostic"))

    def test_tiff_concurrent_handles_keep_diagnostics_and_storage_separate(self):
        valid = self.root / "concurrent-valid.tif"
        invalid = self.root / "concurrent-invalid.tif"
        source_exif_fixture.write_tiff_graph(valid, big=True)
        invalid.write_bytes(b"II*\0")
        expected = json.loads(media.MediaIngest.source_exif_json(valid))
        self.assertEqual(expected["status"], "collected")
        def read(index):
            return index, json.loads(media.MediaIngest.source_exif_json(valid if index % 2 else invalid))
        with concurrent.futures.ThreadPoolExecutor(max_workers=4) as pool:
            for index, value in pool.map(read, range(32)):
                if index % 2:
                    self.assertEqual(value, expected)
                else:
                    self.assertEqual(value["status"], "malformed")
                    self.assertTrue(value["diagnostic"])

    def test_tiff_fixed_ascii_count_is_not_a_decoded_buffer_extent(self):
        path = self.root / "short-datetime.tif"
        source_exif_fixture.write_tiff(path, short_datetime=True)
        result = json.loads(media.MediaIngest.source_exif_json(path))
        self.assertIn(result["status"], ("collected", "malformed"))
        tags = {tag["id"]: tag for tag in result["tiff"]["directories"][0]["tags"]}
        if 306 in tags:
            decoded = bytes.fromhex(tags[306]["bytes"])
            self.assertEqual(tags[306]["count"], len(decoded))
            self.assertEqual(decoded[-1:], b"\0")
            self.assertLess(len(decoded), 20)

    def test_tiff_metadata_skips_large_pixels_and_owns_results(self):
        path = self.root / "large TIFF metadata.tif"
        payload = source_exif_fixture.write_tiff(path, pixel_offset=16 * 1024 * 1024)
        value = json.loads(media.MediaIngest.source_exif_json(path))
        source_exif_fixture.check_tiff(self, value, path, payload)
        self.assertLess(sum(len(bytes.fromhex(span["bytes"])) for span in value["tiff"]["source_ranges"]), 4096)
        self.assertTrue(all(span["offset"] + len(bytes.fromhex(span["bytes"])) < 16 * 1024 * 1024
                            for span in value["tiff"]["source_ranges"]))
        path.unlink()
        self.assertTrue(value["tiff"]["directories"])

    def test_png_idat_segmentation_and_ancillary_crc_recovery(self):
        path = self.root / "review PNG.png"
        for options, count, status in (({"idat_chunks": 5000}, 1, "collected"),
                                        ({"idat_chunks": 5000, "include_exif": False}, 0, "absent"),
                                        ({"damaged_ancillary": True}, 1, "malformed")):
            with self.subTest(options=options):
                payload = source_exif_fixture.write_png(path, **options)
                result = json.loads(media.MediaIngest.source_exif_json(path))
                source_exif_fixture.check_png(self, result, payload, count=count, status=status)

    def test_png_source_exif_owned_original_and_typed_values(self):
        path = self.root / "PNG EXIF é 日本語.png"
        payload = source_exif_fixture.write_png(path)
        result = media.MediaIngest.source_exif_json(path)
        path.unlink()
        source_exif_fixture.check_png(self, json.loads(result), payload)

    def test_source_exif_diagnostics_and_large_header_prefix(self):
        path = self.root / "invalid-diagnostic.avif"
        source_exif_fixture.write_invalid_heif(path)
        result = json.loads(media.MediaIngest.source_exif_json(path))
        self.assertEqual(result["status"], "malformed")
        self.assertIn("\ufffd", result["diagnostic"])
        path = self.root / "large-header.jpg"
        for long_header in (False, True):
            with self.subTest(long_header=long_header):
                payload = source_exif_fixture.write_header_failure(path, long_header)
                result = json.loads(media.MediaIngest.source_exif_json(path))
                self.assertEqual(result["status"], "limited" if long_header else "malformed")
                self.assertEqual(len(result["blocks"]), 2)
                for block in result["blocks"]:
                    self.assertEqual(bytes.fromhex(block["bytes"]), payload)
                    self.assertEqual(block["interpretation"]["status"], "parsed")

    def test_source_exif_json_unicode_originals_and_ownership(self):
        path = self.root / "original EXIF é 日本語.jpg"
        payload = source_exif_fixture.write_jpeg(path)
        value = media.MediaIngest.source_exif_json(path)
        path.unlink()
        source_exif_fixture.check(self, json.loads(value), payload)
        self.assertEqual(json.loads(media.MediaIngest.source_exif_json(path))["status"], "io_error")

    def test_jxl_public_writer_rejects_incompatible_color_without_publication(self):
        path = self.root / "jxl-color-policy.jxl"
        color = media.FrameColor()
        color.transfer = media.ColorTransfer.Srgb
        color.primaries = media.ColorPrimaries.Bt709
        frame = media.FrameSurface.from_bytes(1, 1, media.FramePixelFormat.RGB8,
                                              bytes((10, 20, 30)), color=color)
        options = media.JxlOutputOptions()
        media.ImageOutput.write_jxl(path, frame, options)
        original = path.read_bytes()
        options.overwrite = True
        cases = (
            (media.FramePixelFormat.RGB8, media.ColorTransfer.Linear, media.ColorPrimaries.Bt709, media.AlphaMode.NoAlpha, media.ColorUnits.Relative),
            (media.FramePixelFormat.RGB8, media.ColorTransfer.Srgb, media.ColorPrimaries.Bt2020, media.AlphaMode.NoAlpha, media.ColorUnits.Relative),
            (media.FramePixelFormat.RGBFloat32, media.ColorTransfer.Unspecified, media.ColorPrimaries.Bt709, media.AlphaMode.NoAlpha, media.ColorUnits.Relative),
            (media.FramePixelFormat.RGBFloat32, media.ColorTransfer.Linear, media.ColorPrimaries.Unspecified, media.AlphaMode.NoAlpha, media.ColorUnits.Relative),
            (media.FramePixelFormat.RGBFloat32, media.ColorTransfer.Linear, media.ColorPrimaries.Bt709, media.AlphaMode.NoAlpha, media.ColorUnits.Nits),
            (media.FramePixelFormat.RGBFloat32, media.ColorTransfer.Linear, media.ColorPrimaries.Bt709, media.AlphaMode.NoAlpha, media.ColorUnits.SceneRelative),
            (media.FramePixelFormat.RGBFloat32, media.ColorTransfer.Linear, media.ColorPrimaries.Bt709, media.AlphaMode.Straight, media.ColorUnits.Relative),
            (media.FramePixelFormat.RGBAFloat32, media.ColorTransfer.Linear, media.ColorPrimaries.Bt709, media.AlphaMode.NoAlpha, media.ColorUnits.Relative),
        )
        for fmt, transfer, primaries, alpha, units in cases:
            with self.subTest(format=fmt, transfer=transfer, primaries=primaries, alpha=alpha, units=units):
                color.transfer, color.primaries, color.alpha, color.units = transfer, primaries, alpha, units
                pixels = bytes((10, 20, 30)) if fmt == media.FramePixelFormat.RGB8 else struct.pack("=4f", .1, .2, .3, 1.) if fmt == media.FramePixelFormat.RGBAFloat32 else struct.pack("=3f", .1, .2, .3)
                invalid = media.FrameSurface.from_bytes(1, 1, fmt, pixels, color=color)
                for target in (path, self.root / "jxl-invalid-new.jxl"):
                    with self.assertRaises(MEDIA_ERROR):
                        media.ImageOutput.write_jxl(target, invalid, options)
                self.assertEqual(path.read_bytes(), original)
                self.assertFalse((self.root / "jxl-invalid-new.jxl").exists())
        media.ImageOutput.write_jxl(path, frame, options)
        self.assertEqual(path.read_bytes(), original)

    def test_jxl_uint16_options_publication_and_verified_resume(self):
        color = media.FrameColor()
        color.transfer = media.ColorTransfer.Linear
        color.primaries = media.ColorPrimaries.Bt2020
        color.alpha = media.AlphaMode.Straight
        payload = struct.pack("=8f", -.25, .0031308, 1.25, .5, .1, .25, .5, 0.)
        frame = media.FrameSurface.from_bytes(2, 1, media.FramePixelFormat.RGBAFloat32, payload, color=color)
        options = media.JxlOutputOptions()
        options.precision = media.JxlPrecision.UInt16
        options.effort = 1
        options.overwrite = True
        path = self.root / "uint16 日本語.jxl"
        for transfer in (media.ColorTransfer.Linear, media.ColorTransfer.Srgb):
            options.transfer = transfer
            media.ImageOutput.write_jxl(path, frame, options)
            previous = path.read_bytes()
            with self.assertRaises(MEDIA_ERROR):
                media.ImageOutput.write_jxl(path, frame, options, cancelled=lambda: True)
            self.assertEqual(path.read_bytes(), previous)
            def failed():
                raise ValueError("UInt16 cancellation callback")
            with self.assertRaisesRegex(ValueError, "UInt16 cancellation callback"):
                media.ImageOutput.write_jxl(path, frame, options, cancelled=failed)
            self.assertEqual(path.read_bytes(), previous)
            for value in (float("nan"), float("inf"), -float("inf")):
                invalid = media.FrameSurface.from_bytes(2, 1, media.FramePixelFormat.RGBAFloat32,
                                struct.pack("=8f", value, .1, .2, .5, .1, .2, .3, 1.), color=color)
                with self.assertRaises(MEDIA_ERROR):
                    media.ImageOutput.write_jxl(path, invalid, options)
                self.assertEqual(path.read_bytes(), previous)
        options.transfer = media.ColorTransfer.Bt709
        with self.assertRaises(MEDIA_ERROR):
            media.ImageOutput.write_jxl(path, frame, options)
        self.assertEqual(path.read_bytes(), previous)
        options.transfer = media.ColorTransfer.Srgb
        options.precision = media.JxlPrecision.Float32
        options.effort = 4
        with self.assertRaises(MEDIA_ERROR):
            media.ImageOutput.write_jxl(path, frame, options)
        self.assertEqual(path.read_bytes(), previous)
        request = self.request()
        request.output_format = media.FramePixelFormat.RGBFloat32
        request.input_color.transfer = media.ColorTransfer.Linear
        request.input_color.primaries = media.ColorPrimaries.Bt709
        for transfer, name in ((media.ColorTransfer.Linear, "linear"), (media.ColorTransfer.Srgb, "srgb")):
            files = self.resume_files(self.root / ("jxl-u16-" + name), media.FrameFileFormat.JXL)
            files.files.jxl.precision = media.JxlPrecision.UInt16
            files.files.jxl.transfer = transfer
            files.files.jxl.effort = 1
            self.interrupt_resume(request, files)
            self.assertEqual(media.MediaIngest.extract_files(request, files).frames_accepted, 4)
            images = sorted(files.files.output_directory.glob("*.jxl"))
            original = {p: (p.read_bytes(), p.stat().st_mtime_ns) for p in images}
            metadata = json.loads((files.files.output_directory / "extraction_metadata.json").read_text())
            self.assertEqual(metadata["output"]["bit_depth"], 16)
            self.assertEqual(metadata["output"]["transfer"], name)
            self.assertEqual(metadata["output"]["compression"], "lossless-integer-codes")
            self.assertEqual(metadata["output"]["quantization"], "clamp-round-half-up-uint16")
            files.files.jxl.overwrite = True
            media.MediaIngest.extract_files(request, files)
            self.assertEqual({p: (p.read_bytes(), p.stat().st_mtime_ns) for p in images}, original)
            files.files.jxl.transfer = media.ColorTransfer.Srgb if name == "linear" else media.ColorTransfer.Linear
            with self.assertRaises(MEDIA_ERROR):
                media.MediaIngest.extract_files(request, files)
            self.assertEqual({p: (p.read_bytes(), p.stat().st_mtime_ns) for p in images}, original)
            files.files.jxl.transfer = media.ColorTransfer.Linear
            files.files.jxl.precision = media.JxlPrecision.Float32
            files.files.jxl.effort = 4
            with self.assertRaises(MEDIA_ERROR):
                media.MediaIngest.extract_files(request, files)
            self.assertEqual({p: (p.read_bytes(), p.stat().st_mtime_ns) for p in images}, original)

    def test_jxl_lossless_float_effort_is_rejected_before_publication(self):
        color = media.FrameColor()
        color.transfer = media.ColorTransfer.Linear
        color.primaries = media.ColorPrimaries.Bt709
        frame = media.FrameSurface.from_bytes(1, 1, media.FramePixelFormat.RGBFloat32,
                                              struct.pack("=3f", -0., -.25, 1.25), color=color)
        options = media.JxlOutputOptions()
        path = self.root / "jxl-effort-existing.jxl"
        media.ImageOutput.write_jxl(path, frame, options)
        original = path.read_bytes()
        options.overwrite = True
        for effort in (1, 2, 3):
            options.effort = effort
            for target in (path, self.root / "jxl-effort-new.jxl"):
                with self.assertRaisesRegex(MEDIA_ERROR, "lossless Float32 requires effort"):
                    media.ImageOutput.write_jxl(target, frame, options)
            self.assertEqual(path.read_bytes(), original)
            self.assertFalse((self.root / "jxl-effort-new.jxl").exists())
            request = self.request()
            request.output_format = media.FramePixelFormat.RGBFloat32
            request.input_color.transfer = media.ColorTransfer.Linear
            request.input_color.primaries = media.ColorPrimaries.Bt709
            files = media.FileExtraction()
            files.files.format = media.FrameFileFormat.JXL
            files.files.output_directory = self.root / f"jxl-effort-preflight-{effort}"
            files.files.jxl.effort = effort
            with self.assertRaisesRegex(MEDIA_ERROR, "lossless Float32 requires effort"):
                media.MediaIngest.extract_files(request, files)
            self.assertFalse(files.files.output_directory.exists())
        options.lossless = False
        options.effort = 1
        media.ImageOutput.write_jxl(self.root / "jxl-lossy-effort1.jxl", frame, options)

    def test_avif_public_writer_rejects_incompatible_color_without_publication(self):
        path = self.root / "avif-color-policy.avif"
        color = media.FrameColor()
        color.transfer = media.ColorTransfer.Srgb
        color.primaries = media.ColorPrimaries.Bt709
        frame = media.FrameSurface.from_bytes(1, 1, media.FramePixelFormat.RGB8,
                                              bytes((10, 20, 30)), color=color)
        options = media.AvifOutputOptions()
        media.ImageOutput.write_avif(path, frame, options)
        original = path.read_bytes()
        options.overwrite = True
        cases = (
            (media.FramePixelFormat.RGB8, media.ColorTransfer.Linear, media.ColorPrimaries.Bt709, media.AlphaMode.NoAlpha, media.ColorUnits.Relative),
            (media.FramePixelFormat.RGB8, media.ColorTransfer.Srgb, media.ColorPrimaries.Bt2020, media.AlphaMode.NoAlpha, media.ColorUnits.Relative),
            (media.FramePixelFormat.RGBFloat32, media.ColorTransfer.Unspecified, media.ColorPrimaries.Bt709, media.AlphaMode.NoAlpha, media.ColorUnits.Relative),
            (media.FramePixelFormat.RGBFloat32, media.ColorTransfer.Linear, media.ColorPrimaries.Unspecified, media.AlphaMode.NoAlpha, media.ColorUnits.Relative),
            (media.FramePixelFormat.RGBFloat32, media.ColorTransfer.Linear, media.ColorPrimaries.Bt709, media.AlphaMode.NoAlpha, media.ColorUnits.Nits),
            (media.FramePixelFormat.RGBFloat32, media.ColorTransfer.Linear, media.ColorPrimaries.Bt709, media.AlphaMode.NoAlpha, media.ColorUnits.SceneRelative),
            (media.FramePixelFormat.RGBFloat32, media.ColorTransfer.Linear, media.ColorPrimaries.Bt709, media.AlphaMode.Straight, media.ColorUnits.Relative),
            (media.FramePixelFormat.RGBAFloat32, media.ColorTransfer.Linear, media.ColorPrimaries.Bt709, media.AlphaMode.NoAlpha, media.ColorUnits.Relative),
        )
        for fmt, transfer, primaries, alpha, units in cases:
            with self.subTest(format=fmt, transfer=transfer, primaries=primaries, alpha=alpha, units=units):
                color.transfer, color.primaries, color.alpha, color.units = transfer, primaries, alpha, units
                pixels = bytes((10, 20, 30)) if fmt == media.FramePixelFormat.RGB8 else struct.pack("=4f", .1, .2, .3, 1.) if fmt == media.FramePixelFormat.RGBAFloat32 else struct.pack("=3f", .1, .2, .3)
                invalid = media.FrameSurface.from_bytes(1, 1, fmt, pixels, color=color)
                for target in (path, self.root / "avif-invalid-new.avif"):
                    with self.assertRaises(MEDIA_ERROR):
                        media.ImageOutput.write_avif(target, invalid, options)
                self.assertEqual(path.read_bytes(), original)
                self.assertFalse((self.root / "avif-invalid-new.avif").exists())
        media.ImageOutput.write_avif(path, frame, options)
        self.assertEqual(path.read_bytes(), original)

    def test_avif_bt2020_premultiplied_alpha_and_nonfinite_rejection(self):
        color = media.FrameColor()
        color.transfer = media.ColorTransfer.Linear
        color.primaries = media.ColorPrimaries.Bt2020
        color.alpha = media.AlphaMode.Straight
        options = media.AvifOutputOptions()
        options.bit_depth = 12
        straight = media.FrameSurface.from_bytes(2, 1, media.FramePixelFormat.RGBAFloat32,
                    struct.pack("=8f", .25, .5, .75, .5, 0., 0., 0., 0.), color=color)
        path = self.root / "avif-bt2020-straight.avif"
        media.ImageOutput.write_avif(path, straight, options)
        original = path.read_bytes()
        color.alpha = media.AlphaMode.Premultiplied
        premult = media.FrameSurface.from_bytes(2, 1, media.FramePixelFormat.RGBAFloat32,
                    struct.pack("=8f", .125, .25, .375, .5, 9., 3., 1., 0.), color=color)
        other = self.root / "avif-bt2020-premult.avif"
        media.ImageOutput.write_avif(other, premult, options)
        self.assertEqual(other.read_bytes(), original)
        info = json.loads(subprocess.check_output([
            os.environ.get("LFS_MEDIA_TEST_FFPROBE", "ffprobe"), "-v", "error",
            "-show_streams", "-of", "json", str(path)], timeout=30))
        primary = info["streams"][0]
        self.assertEqual(primary["color_primaries"], "bt2020")
        self.assertEqual(primary["color_transfer"], "iec61966-2-1")
        options.overwrite = True
        color.alpha = media.AlphaMode.Straight
        for values in ((float("nan"), .2, .3, 1.), (.1, float("inf"), .3, 1.),
                       (.1, .2, float("-inf"), 1.), (.1, .2, .3, float("nan")),
                       (.1, .2, .3, -.1), (.1, .2, .3, 1.1)):
            invalid = media.FrameSurface.from_bytes(1, 1, media.FramePixelFormat.RGBAFloat32,
                                                   struct.pack("=4f", *values), color=color)
            for target in (path, self.root / "avif-invalid-samples.avif"):
                with self.assertRaises(MEDIA_ERROR):
                    media.ImageOutput.write_avif(target, invalid, options)
            self.assertEqual(path.read_bytes(), original)
            self.assertFalse((self.root / "avif-invalid-samples.avif").exists())
        media.ImageOutput.write_avif(path, straight, options)
        self.assertEqual(path.read_bytes(), original)

    def test_avif_resume_rejects_corruption_and_inconsistent_checkpoint(self):
        for damage in ("truncate", "mutate", "checksum", "record"):
            with self.subTest(damage=damage):
                request = self.request()
                request.output_format = media.FramePixelFormat.RGBFloat32
                request.input_color.transfer = media.ColorTransfer.Linear
                request.input_color.primaries = media.ColorPrimaries.Bt709
                files = self.resume_files(self.root / ("avif-corrupt-" + damage), media.FrameFileFormat.AVIF)
                self.interrupt_resume(request, files)
                checkpoint = files.files.output_directory / ".media-ingest-checkpoint"
                images = sorted(files.files.output_directory.glob("*.avif"))
                self.assertTrue(images)
                original = images[0].read_bytes()
                record = checkpoint / "frame-0.json"
                record_before = record.read_bytes()
                if damage == "truncate":
                    images[0].write_bytes(original[:16])
                elif damage == "mutate":
                    changed = bytearray(original)
                    changed[-1] ^= 1
                    images[0].write_bytes(changed)
                elif damage == "checksum":
                    value = json.loads(record_before)
                    value["file_sha256"] = "0" * 64
                    record.write_text(json.dumps(value))
                else:
                    record.write_bytes(b'{"partial":')
                before = {p: (p.read_bytes(), p.stat().st_mtime_ns) for p in images}
                damaged_record = record.read_bytes()
                with self.assertRaises(MEDIA_ERROR):
                    media.MediaIngest.extract_files(request, files)
                self.assertEqual({p: (p.read_bytes(), p.stat().st_mtime_ns) for p in images}, before)
                self.assertEqual(record.read_bytes(), damaged_record)
                self.assertFalse((files.files.output_directory / "extraction_metadata.json").exists())
                images[0].write_bytes(original)
                record.write_bytes(record_before)
                self.assertEqual(media.MediaIngest.extract_files(request, files).frames_accepted, 4)
                self.assertEqual(images[0].read_bytes(), original)

    def test_avif_float_quantization_matches_independent_decoder(self):
        color = media.FrameColor()
        color.transfer = media.ColorTransfer.Linear
        color.primaries = media.ColorPrimaries.Bt709
        samples = (-.5, .003, 2., .25, .5, .75)
        frame = media.FrameSurface.from_bytes(2, 1, media.FramePixelFormat.RGBFloat32,
                                              struct.pack("=6f", *samples), color=color)
        options = media.AvifOutputOptions()
        def code(value, maximum, alpha=False):
            value = min(1., max(0., value))
            if not alpha:
                value = 12.92 * value if value <= .0031308 else 1.055 * value ** (1./2.4) - .055
            return math.floor(value * maximum + .5)
        for bits in (8, 10, 12):
            with self.subTest(bits=bits):
                options.bit_depth = bits
                path = self.root / f"avif-quantization-{bits}.avif"
                media.ImageOutput.write_avif(path, frame, options)
                fmt = "gbrp" if bits == 8 else f"gbrp{bits}le"
                raw = subprocess.check_output([os.environ.get("LFS_MEDIA_TEST_FFMPEG", "ffmpeg"), "-v", "error", "-i", str(path), "-f", "rawvideo", "-pix_fmt", fmt, "-"], timeout=30)
                actual = tuple(raw) if bits == 8 else struct.unpack("<6H", raw)
                maximum = (1 << bits)-1
                expected = tuple(code(samples[pixel*3+channel], maximum)
                                 for channel in (1, 2, 0) for pixel in range(2))
                self.assertEqual(actual, expected)
        options.overwrite = True
        before = path.read_bytes()
        for units in (media.ColorUnits.Nits, media.ColorUnits.SceneRelative):
            color.units = units
            invalid = media.FrameSurface.from_bytes(2, 1, media.FramePixelFormat.RGBFloat32,
                                                    struct.pack("=6f", *samples), color=color)
            with self.assertRaises(MEDIA_ERROR):
                media.ImageOutput.write_avif(path, invalid, options)
            self.assertEqual(path.read_bytes(), before)

    def test_avif_public_writer_and_lossless_lossy_sequences(self):
        color = media.FrameColor()
        color.transfer = media.ColorTransfer.Linear
        color.primaries = media.ColorPrimaries.Bt709
        samples = (-.5, 0., 2., .25, .5, .75)
        frame = media.FrameSurface.from_bytes(2, 1, media.FramePixelFormat.RGBFloat32,
                                              struct.pack("=6f", *samples), color=color)
        options = media.AvifOutputOptions()
        path = self.root / "AVIF export é 日本語.avif"
        media.ImageOutput.write_avif(path, frame, options)
        original = path.read_bytes()
        with self.assertRaises(MEDIA_ERROR):
            media.ImageOutput.write_avif(path, frame, options)
        options.overwrite = True
        with self.assertRaises(MEDIA_ERROR):
            media.ImageOutput.write_avif(path, frame, options, cancelled=lambda: True)
        self.assertEqual(path.read_bytes(), original)
        def failed():
            raise ValueError("AVIF cancellation sentinel")
        with self.assertRaisesRegex(ValueError, "AVIF cancellation sentinel"):
            media.ImageOutput.write_avif(path, frame, options, cancelled=failed)
        self.assertEqual(path.read_bytes(), original)
        self.assertTrue(media.MediaIngest.capabilities().avif)
        request = self.request()
        request.output_format = media.FramePixelFormat.RGBFloat32
        request.input_color.transfer = media.ColorTransfer.Linear
        request.input_color.primaries = media.ColorPrimaries.Bt709
        for lossless in (True, False):
            output = media.FileExtraction()
            output.files.output_directory = self.root / ("avif-lossless" if lossless else "avif-lossy")
            output.files.format = media.FrameFileFormat.AVIF
            output.files.avif.lossless = lossless
            output.resume = True
            report = media.MediaIngest.extract_files(request, output)
            paths = sorted(output.files.output_directory.glob("*.avif"))
            self.assertEqual(len(paths), report.frames_accepted)
            self.assertEqual(report.frames_accepted, 4)
            original = {p.name:p.read_bytes() for p in paths}
            mtimes = {p.name:p.stat().st_mtime_ns for p in paths}
            output.files.avif.overwrite = True
            media.MediaIngest.extract_files(request, output)
            self.assertEqual({p.name:p.read_bytes() for p in paths}, original)
            self.assertEqual({p.name:p.stat().st_mtime_ns for p in paths}, mtimes)
            if lossless:
                output.files.avif.quality = 80
                media.MediaIngest.extract_files(request, output)
                self.assertEqual({p.name:p.stat().st_mtime_ns for p in paths}, mtimes)
            paths[0].unlink()
            media.MediaIngest.extract_files(request, output)
            self.assertEqual({p.name:p.read_bytes() for p in paths}, original)
            output.files.avif.speed -= 1
            with self.assertRaises(MEDIA_ERROR):
                media.MediaIngest.extract_files(request, output)

    def test_jxl_public_writer_and_lossless_lossy_sequences(self):
        color = media.FrameColor()
        color.transfer = media.ColorTransfer.Linear
        color.primaries = media.ColorPrimaries.Bt709
        samples = (-.5, 0., 2., .25, .5, .75)
        frame = media.FrameSurface.from_bytes(2, 1, media.FramePixelFormat.RGBFloat32,
                                              struct.pack("=6f", *samples), color=color)
        options = media.JxlOutputOptions()
        path = self.root / "JXL export é 日本語.jxl"
        media.ImageOutput.write_jxl(path, frame, options)
        original = path.read_bytes()
        with self.assertRaises(MEDIA_ERROR):
            media.ImageOutput.write_jxl(path, frame, options)
        options.overwrite = True
        with self.assertRaises(MEDIA_ERROR):
            media.ImageOutput.write_jxl(path, frame, options, cancelled=lambda: True)
        self.assertEqual(path.read_bytes(), original)
        def failed():
            raise ValueError("JXL cancellation sentinel")
        with self.assertRaisesRegex(ValueError, "JXL cancellation sentinel"):
            media.ImageOutput.write_jxl(path, frame, options, cancelled=failed)
        self.assertEqual(path.read_bytes(), original)
        self.assertTrue(media.MediaIngest.capabilities().jxl)
        request = self.request()
        request.output_format = media.FramePixelFormat.RGBFloat32
        request.input_color.transfer = media.ColorTransfer.Linear
        request.input_color.primaries = media.ColorPrimaries.Bt709
        for lossless in (True, False):
            output = media.FileExtraction()
            output.files.output_directory = self.root / ("jxl-lossless" if lossless else "jxl-lossy")
            output.files.format = media.FrameFileFormat.JXL
            output.files.jxl.lossless = lossless
            output.resume = True
            report = media.MediaIngest.extract_files(request, output)
            paths = sorted(output.files.output_directory.glob("*.jxl"))
            self.assertEqual(len(paths), report.frames_accepted)
            self.assertEqual(report.frames_accepted, 4)
            original = {p.name:p.read_bytes() for p in paths}
            mtimes = {p.name:p.stat().st_mtime_ns for p in paths}
            output.files.jxl.overwrite = True
            media.MediaIngest.extract_files(request, output)
            self.assertEqual({p.name:p.read_bytes() for p in paths}, original)
            self.assertEqual({p.name:p.stat().st_mtime_ns for p in paths}, mtimes)
            if lossless:
                output.files.jxl.distance = 2.
                media.MediaIngest.extract_files(request, output)
                self.assertEqual({p.name:p.stat().st_mtime_ns for p in paths}, mtimes)
            paths[0].unlink()
            media.MediaIngest.extract_files(request, output)
            self.assertEqual({p.name:p.read_bytes() for p in paths}, original)
            output.files.jxl.effort += 1
            with self.assertRaises(MEDIA_ERROR):
                media.MediaIngest.extract_files(request, output)

    def test_tiff16_public_output_precision_alpha_and_failures(self):
        ffmpeg = os.environ.get("LFS_MEDIA_TEST_FFMPEG", "ffmpeg")
        color = media.FrameColor()
        color.transfer = media.ColorTransfer.Linear
        color.primaries = media.ColorPrimaries.Bt709
        options = media.Tiff16OutputOptions()
        samples = (0., .125, .5, 1., .9, .0001)
        frame = media.FrameSurface.from_bytes(2, 1, media.FramePixelFormat.RGBFloat32,
                                              struct.pack("=6f", *samples), color=color)
        path = self.root / "tiff16-precision.tif"
        for transfer in (media.ColorTransfer.Linear, media.ColorTransfer.Srgb, media.ColorTransfer.Bt709):
            options.transfer = transfer
            options.overwrite = True
            media.ImageOutput.write_tiff16(path, frame, options)
            archive = json.loads(media.MediaIngest.source_exif_json(path))
            self.assertEqual(archive["status"], "collected", archive)
            self.assertEqual(archive["tiff"]["directories"][0]["kind"], "image")
            self.assertTrue(any(tag["id"] == 270 and b"tiff16-sdr-v1" in bytes.fromhex(tag["bytes"])
                                for tag in archive["tiff"]["directories"][0]["tags"]))
            tags = {tag["id"]: tag for tag in archive["tiff"]["directories"][0]["tags"]}
            for id_, expected in ((318, (.3127, .3290)), (319, (.64, .33, .30, .60, .15, .06))):
                self.assertEqual(tags[id_]["type"], 5)
                self.assertEqual(tags[id_]["count"], len(expected))
                self.assertEqual(len(tags[id_]["floating_values"]), len(expected))
                for actual_value, expected_value in zip(tags[id_]["floating_values"], expected):
                    self.assertAlmostEqual(actual_value, expected_value, delta=1e-6)
            result = subprocess.run([ffmpeg, "-v", "error", "-i", str(path), "-f", "rawvideo", "-pix_fmt", "rgb48le", "-"],
                                    check=True, capture_output=True, timeout=30)
            actual = struct.unpack("<6H", result.stdout)
            def encode(v):
                if transfer == media.ColorTransfer.Srgb:
                    v = 12.92 * v if v <= .0031308 else 1.055 * v ** (1 / 2.4) - .055
                elif transfer == media.ColorTransfer.Bt709:
                    v = 4.5 * v if v < .018 else 1.099 * v ** .45 - .099
                return math.floor(max(0., min(1., v)) * 65535 + .5)
            self.assertLessEqual(max(abs(a - encode(v)) for a, v in zip(actual, samples)), 1)
            self.assertTrue(any(v % 257 for v in actual), "genuine precision beyond RGB8")
        before = path.read_bytes()
        options.overwrite = False
        with self.assertRaises(MEDIA_ERROR):
            media.ImageOutput.write_tiff16(path, frame, options)
        options.overwrite = True
        with self.assertRaises(MEDIA_ERROR):
            media.ImageOutput.write_tiff16(path, frame, options, cancelled=lambda: True)
        self.assertEqual(path.read_bytes(), before)
        for units in (media.ColorUnits.Nits, media.ColorUnits.SceneRelative):
            color.units = units
            invalid = media.FrameSurface.from_bytes(2, 1, media.FramePixelFormat.RGBFloat32,
                                                    struct.pack("=6f", *samples), color=color)
            with self.assertRaises(MEDIA_ERROR):
                media.ImageOutput.write_tiff16(path, invalid, options)
            self.assertEqual(path.read_bytes(), before)
        color.units = media.ColorUnits.Relative
        options.transfer = media.ColorTransfer.Linear
        for alpha in (media.AlphaMode.Straight, media.AlphaMode.Premultiplied):
            color.alpha = alpha
            rgba = (.25, .125, .5, .5) if alpha == media.AlphaMode.Premultiplied else (.5, .25, 1., .5)
            rgba_frame = media.FrameSurface.from_bytes(1, 1, media.FramePixelFormat.RGBAFloat32,
                                                       struct.pack("=4f", *rgba), color=color)
            media.ImageOutput.write_tiff16(path, rgba_frame, options)
            result = subprocess.run([ffmpeg, "-v", "error", "-i", str(path), "-f", "rawvideo", "-pix_fmt", "rgba64le", "-"],
                                    check=True, capture_output=True, timeout=30)
            self.assertEqual(struct.unpack("<4H", result.stdout), (32768, 16384, 65535, 32768))
        zero_alpha = media.FrameSurface.from_bytes(1, 1, media.FramePixelFormat.RGBAFloat32,
                                                   struct.pack("=4f", 1., .5, .25, 0.), color=color)
        media.ImageOutput.write_tiff16(path, zero_alpha, options)
        raw = subprocess.check_output([ffmpeg, "-v", "error", "-i", str(path), "-f", "rawvideo", "-pix_fmt", "rgba64le", "-"], timeout=30)
        self.assertEqual(struct.unpack("<4H", raw), (0, 0, 0, 0))
        before = path.read_bytes()
        for rgba in ((float("nan"), 0., 0., 1.), (0., float("inf"), 0., 1.), (0., 0., 0., -.1), (0., 0., 0., 1.1)):
            invalid = media.FrameSurface.from_bytes(1, 1, media.FramePixelFormat.RGBAFloat32,
                                                    struct.pack("=4f", *rgba), color=color)
            with self.assertRaises(MEDIA_ERROR):
                media.ImageOutput.write_tiff16(path, invalid, options)
            self.assertEqual(path.read_bytes(), before)

    def test_tiff16_sequence_precision_metadata_and_resume(self):
        ffmpeg = os.environ.get("LFS_MEDIA_TEST_FFMPEG", "ffmpeg")
        source = self.root / "tiff16-source.mkv"
        expression = ("nullsrc=s=32x24:r=4:d=1,format=gbrp10le,"
                      "geq=r='mod(X*17+Y*3+N*31,1024)':g='mod(X*7+Y*11+N*29,1024)':b='mod(X*13+Y*5+N*23,1024)',"
                      "setparams=color_primaries=bt709:color_trc=linear:colorspace=gbr:range=full")
        subprocess.run([ffmpeg, "-v", "error", "-f", "lavfi", "-i", expression, "-c:v", "ffv1", "-level", "3",
                        "-pix_fmt", "gbrp10le", "-color_trc", "linear", "-color_primaries", "bt709",
                        "-colorspace", "rgb", "-color_range", "pc", str(source)], check=True, capture_output=True, timeout=60)
        request = media.IngestRequest()
        request.input = source
        request.allow_hardware_decode = False
        request.output_format = media.FramePixelFormat.RGBFloat32
        request.input_color.transfer = media.ColorTransfer.Linear
        request.input_color.primaries = media.ColorPrimaries.Bt709
        request.selection.fps = 4
        output = media.FileExtraction()
        output.files.output_directory = self.root / "tiff16-sequence"
        output.files.format = media.FrameFileFormat.TIFF16
        output.files.tiff16.transfer = media.ColorTransfer.Linear
        output.resume = True
        output.write_metadata = True
        report = media.MediaIngest.extract_files(request, output)
        self.assertEqual(report.frames_accepted, 4)
        paths = sorted(output.files.output_directory.glob("*.tif"))
        self.assertEqual(len(paths), 4)
        values = subprocess.run([ffmpeg, "-v", "error", "-i", str(paths[0]), "-f", "rawvideo", "-pix_fmt", "rgb48le", "-"],
                                check=True, capture_output=True, timeout=30).stdout
        actual = struct.unpack("<" + "H" * (len(values) // 2), values)
        expected = [math.floor((v / 1023) * 65535 + .5) for y in range(24) for x in range(32)
                    for v in ((x * 17 + y * 3) % 1024, (x * 7 + y * 11) % 1024, (x * 13 + y * 5) % 1024)]
        self.assertLessEqual(max(abs(a - b) for a, b in zip(actual, expected)), 1)
        metadata = json.loads((output.files.output_directory / "extraction_metadata.json").read_text())
        self.assertEqual(metadata["output"]["bit_depth"], 16)
        self.assertEqual(metadata["output"]["source_component_depth"], 10)
        self.assertEqual(metadata["output"]["pixel_format"], "rgb_uint16")
        self.assertEqual(metadata["output"]["transfer"], "linear")
        hashes = [hashlib.sha256(path.read_bytes()).hexdigest() for path in paths]
        mtimes = [path.stat().st_mtime_ns for path in paths]
        metadata_path = output.files.output_directory / "extraction_metadata.json"
        self.assertEqual(media.MediaIngest.extract_files(request, output).frames_accepted, 4)
        self.assertEqual(json.loads(metadata_path.read_bytes())["output"], metadata["output"])
        output.files.tiff16.overwrite = True
        self.assertEqual(media.MediaIngest.extract_files(request, output).frames_accepted, 4)
        self.assertEqual([hashlib.sha256(path.read_bytes()).hexdigest() for path in paths], hashes)
        self.assertEqual([path.stat().st_mtime_ns for path in paths], mtimes)
        output.files.tiff16.transfer = media.ColorTransfer.Srgb
        with self.assertRaises(MEDIA_ERROR):
            media.MediaIngest.extract_files(request, output)
        output.files.tiff16.transfer = media.ColorTransfer.Linear
        identity_path = output.files.output_directory / ".media-ingest-checkpoint" / "identity.json"
        identity = json.loads(identity_path.read_text())
        self.assertEqual(identity["tiff16"][0], "tiff16-sdr-v1")
        # Older or incompatible encoding identities must not reuse verified files.
        identity["tiff16"][0] = "tiff16-sdr-v0"
        identity_path.write_text(json.dumps(identity))
        with self.assertRaises(MEDIA_ERROR):
            media.MediaIngest.extract_files(request, output)
        self.assertEqual([hashlib.sha256(path.read_bytes()).hexdigest() for path in paths], hashes)
        self.assertEqual([path.stat().st_mtime_ns for path in paths], mtimes)

    @classmethod
    def setUpClass(cls):
        cls.temp = tempfile.TemporaryDirectory(prefix="lfs-python-media-")
        cls.addClassCleanup(cls.cleanup_temp)
        cls.root = Path(cls.temp.name)
        cls.corpus = cls.root / "é 日本語"
        fixtures.prepare(cls.corpus, os.environ.get("LFS_MEDIA_TEST_FFMPEG", "ffmpeg"),
                         os.environ.get("LFS_MEDIA_TEST_FFPROBE", "ffprobe"))

    @classmethod
    def cleanup_temp(cls):
        try:
            cls.temp.cleanup()
        except OSError as error:
            if sys.platform != "darwin" or error.errno != errno.ENOTEMPTY:
                raise
            # macOS/exFAT can enumerate Unicode fixture names in NFD while
            # lookup requires the NFC spelling used to create them. Restrict
            # this fallback to normalized children of our own temporary root.
            for entry in cls.root.iterdir():
                normalized = entry.with_name(unicodedata.normalize("NFC", entry.name))
                if normalized == entry or not normalized.exists():
                    continue
                if normalized.is_symlink() or not normalized.is_dir():
                    normalized.unlink()
                else:
                    shutil.rmtree(normalized)
            cls.temp.cleanup()

    def request(self):
        request = media.IngestRequest()
        request.input = self.corpus / "cfr-asymmetric.nut"
        request.selection.mode = media.SelectionMode.Interval
        request.selection.interval = 1
        request.end_seconds = 0.35
        request.allow_hardware_decode = False
        return request

    def test_hdr_units_survive_owned_surfaces_and_verified_file_resume(self):
        for hlg,profile,transfer,units in [(False,media.FloatColorProfile.PqNits,"smpte2084",media.ColorUnits.Nits),
                                           (True,media.FloatColorProfile.HlgScene,"arib-std-b67",media.ColorUnits.SceneRelative)]:
            source=self.root/("hlg-bindings.mkv" if hlg else "pq-bindings.mkv")
            subprocess.run([os.environ.get("LFS_MEDIA_TEST_FFMPEG","ffmpeg"),"-v","error",
                            "-f","lavfi","-i","testsrc2=size=64x32:rate=10:duration=1",
                            "-vf",f"format=yuv444p10le,setparams=color_primaries=bt2020:color_trc={transfer}:colorspace=bt2020nc:range=limited",
                            "-c:v","ffv1","-level","3",str(source)],check=True,capture_output=True)
            request=self.request()
            request.input=source
            request.output_format=media.FramePixelFormat.RGBFloat32
            request.float_color_profile=profile
            request.end_seconds=.5
            report,frames=media.MediaIngest.extract(request)
            self.assertGreater(report.frames_accepted,0)
            self.assertEqual(frames[0].layout.color.units,units)
            self.assertEqual(frames[0].layout.color.transfer,media.ColorTransfer.Linear)
            self.assertEqual(frames[0].layout.color.primaries,media.ColorPrimaries.Bt2020)
            snapshot=media.FrameSurface.from_bytes(frames[0].layout.width,frames[0].layout.height,
                        frames[0].layout.format,frames[0].pixels,color=frames[0].layout.color,info=frames[0].info)
            self.assertEqual(snapshot.layout.color.units,units)
            self.assertEqual(snapshot.pixels,frames[0].pixels)
            files=media.FileExtraction()
            files.files.format=media.FrameFileFormat.EXR
            files.files.output_directory=self.root/("hlg-resume" if hlg else "pq-resume")
            files.files.exr.precision=media.ExrPrecision.Float
            files.files.exr.overwrite=True
            files.resume=True
            files.write_metadata=True
            def stop(progress):
                if progress.processed>=2:
                    raise ValueError("interrupt PQ checkpoint")
            with self.assertRaisesRegex(ValueError,"interrupt PQ checkpoint"):
                media.MediaIngest.extract_files(request,files,progress=stop)
            media.MediaIngest.extract_files(request,files)
            def proof():
                return {path.name:(path.read_bytes(),path.stat().st_mtime_ns) for path in files.files.output_directory.glob("*.exr")}
            before=proof()
            media.MediaIngest.extract_files(request,files)
            self.assertEqual(proof(),before)
            missing=next(iter(before))
            (files.files.output_directory/missing).unlink()
            media.MediaIngest.extract_files(request,files)
            self.assertEqual((files.files.output_directory/missing).read_bytes(),before[missing][0])
            manifest=json.loads((files.files.output_directory/"extraction_metadata.json").read_text())
            self.assertEqual(manifest["output"]["units"],"scene-relative" if hlg else "cd/m2")
            request.float_color_profile=media.FloatColorProfile.Sdr
            with self.assertRaises(MEDIA_ERROR):
                media.MediaIngest.extract_files(request,files)
            self.assertEqual({name:data for name,(data,_) in proof().items()},
                             {name:data for name,(data,_) in before.items()})

    def test_probe_optional_inventory_unicode(self):
        self.assertTrue(media.MediaIngest.codec_build_info().ffmpeg_license)
        source = media.MediaIngest.probe(self.request().input)
        self.assertEqual(source.streams[0].width, 64)
        self.assertIsNotNone(source.streams[0].time_base)
        self.assertFalse(media.MediaIngest.probe(self.request().input, headers_only=True).stream_info_probed)
        with self.assertRaises(MEDIA_ERROR):
            media.MediaIngest.probe(self.root / "missing.nut")

    def test_nested_options_owned_frames_and_file_report(self):
        request = self.request()
        request.geometry.clockwise_rotation = 90
        progress = []
        report, frames = media.MediaIngest.extract(request, progress=lambda value: progress.append(value.processed))
        self.assertEqual(report.frames_accepted, len(frames))
        self.assertTrue(progress)
        before = frames[0].pixels
        self.assertEqual(len(before), frames[0].layout.row_stride * frames[0].layout.height)
        files = media.FileExtraction()
        files.files.output_directory = self.root / "output é 日本語"
        files.write_metadata = True
        file_report = media.MediaIngest.extract_files(request, files)
        self.assertEqual(file_report.frames_accepted, len(frames))
        self.assertEqual(len(list(files.files.output_directory.glob("frame_*.png"))), len(frames))
        del request, report, file_report
        self.assertEqual(before, frames[0].pixels)

    def metadata_request(self, multitrack=False):
        request = self.request()
        path = self.root / (self._testMethodName + ".mkv")
        command = [os.environ.get("LFS_MEDIA_TEST_FFMPEG", "ffmpeg"), "-v", "error", "-y", "-i", str(request.input)]
        if multitrack:
            command += ["-f", "lavfi", "-i", "anullsrc=r=48000:cl=mono,aformat=sample_fmts=s16"]
        command += ["-map", "0:v:0", "-c:v", "copy"]
        if multitrack:
            command += ["-map", "1:a:0", "-c:a", "pcm_s16le", "-metadata:s:a:0", "title=AUDIO_METADATA_SENTINEL", "-shortest"]
        command += ["-metadata", "comment=STATIC_METADATA_SENTINEL", "-metadata", "location=+01.2500+002.5000/",
                    "-write_crc32", "0", str(path)]
        subprocess.run(command, check=True, capture_output=True, timeout=30)
        request.input = path
        return request

    def test_source_metadata_manifest_survives_empty_selection_and_resume(self):
        request = self.metadata_request()
        request.sharpness.enabled = True
        request.sharpness.threshold = 1e30
        for resume in (False, True):
            for preserve in (False, True):
                with self.subTest(resume=resume, preserve=preserve):
                    files = media.FileExtraction()
                    files.files.output_directory = self.root / f"metadata-empty-{resume}-{preserve}"
                    files.write_metadata = True
                    files.resume = resume
                    files.preserve_source_metadata = preserve
                    report = media.MediaIngest.extract_files(request, files)
                    self.assertEqual(report.frames_accepted, 0)
                    self.assertGreater(report.discarded, 0)
                    path = files.files.output_directory / "extraction_metadata.json"
                    self.assertEqual(path.is_file(), preserve)
                    if not preserve:
                        continue
                    manifest = json.loads(path.read_text(encoding="utf-8"))
                    self.assertEqual(manifest["frames"], [])
                    self.assertTrue(manifest["source_metadata"]["success"])
                    previous = manifest["source_metadata"]
                    self.assertEqual(media.MediaIngest.extract_files(request, files).frames_accepted, 0)
                    repeated = json.loads(path.read_text(encoding="utf-8"))
                    self.assertEqual(repeated["frames"], [])
                    self.assertEqual(repeated["source_metadata"], previous)
        # An empty extraction must replace an older nonempty frame manifest.
        files = media.FileExtraction()
        files.preserve_source_metadata = True
        files.files.output_directory = self.root / "metadata-empty-replaces-old"
        request.sharpness.enabled = False
        self.assertGreater(media.MediaIngest.extract_files(request, files).frames_accepted, 0)
        path = files.files.output_directory / "extraction_metadata.json"
        previous = json.loads(path.read_text(encoding="utf-8"))
        request.sharpness.enabled = True
        before = path.read_bytes()
        cancelled = [False]
        with self.assertRaises(MEDIA_ERROR):
            media.MediaIngest.extract_files(request, files, cancelled=lambda: cancelled[0],
                                            progress=lambda value: cancelled.__setitem__(0, value.processed == value.estimated))
        self.assertEqual(path.read_bytes(), before)
        self.assertEqual(media.MediaIngest.extract_files(request, files).frames_accepted, 0)
        current = json.loads(path.read_text(encoding="utf-8"))
        self.assertTrue(previous["frames"])
        self.assertEqual(current["frames"], [])
        self.assertEqual(current["source_metadata"], previous["source_metadata"])
        files.files.output_directory = self.root / "metadata-empty-collision"
        files.files.output_directory.mkdir()
        collision = files.files.output_directory / "extraction_metadata.json"
        collision.mkdir()
        with self.assertRaises(MEDIA_ERROR):
            media.MediaIngest.extract_files(request, files)
        self.assertTrue(collision.is_dir())

    def test_source_metadata_oversized_rotation_stays_bounded_through_extraction(self):
        value = "9" * (9 * 1024 * 1024)
        request = self.request()
        source = self.root / "oversized-rotation.mkv"
        subprocess.run([os.environ.get("LFS_MEDIA_TEST_FFMPEG", "ffmpeg"), "-v", "error", "-y",
                        "-i", str(request.input), "-map", "0:v:0", "-c:v", "copy",
                        "-metadata:s:v:0", "ROTATX=ROTATION_PLACEHOLDER", "-live", "1",
                        "-write_crc32", "0", str(source)], check=True, capture_output=True, timeout=30)
        # Resize only the metadata EBML elements. Streaming output has no
        # finite Segment size or cue offsets to invalidate when tags grow.
        def rewrite(elements):
            result = bytearray()
            offset = 0
            while offset < len(elements):
                def width(first, maximum):
                    for count in range(1, maximum + 1):
                        if first & (1 << (8 - count)):
                            return count
                    raise AssertionError("invalid EBML fixture integer")
                id_size = width(elements[offset], 4)
                identifier = elements[offset:offset + id_size]
                size_offset = offset + id_size
                size_width = width(elements[size_offset], 8)
                size_bytes = elements[size_offset:size_offset + size_width]
                size = int.from_bytes(size_bytes, "big") & ((1 << (7 * size_width)) - 1)
                unknown = size == (1 << (7 * size_width)) - 1
                begin = size_offset + size_width
                end = len(elements) if unknown else begin + size
                self.assertLessEqual(end, len(elements))
                payload = elements[begin:end]
                if identifier == b"\x11\x4d\x9b\x74":
                    offset = end
                    continue  # Optional SeekHead offsets precede the resized tags.
                if identifier in (b"\x18\x53\x80\x67", b"\x12\x54\xc3\x67", b"\x73\x73", b"\x67\xc8"):
                    payload = rewrite(payload)
                elif identifier == b"\x44\x87" and payload == b"ROTATION_PLACEHOLDER":
                    payload = value.encode("utf-8")
                elif identifier == b"\x45\xa3" and payload == b"ROTATX":
                    payload = b"ROTATE"
                if not unknown:
                    count = next(count for count in range(1, 9) if len(payload) < (1 << (7 * count)) - 1)
                    size_bytes = (len(payload) | (1 << (7 * count))).to_bytes(count, "big")
                result += identifier + size_bytes + payload
                offset = end
            return bytes(result)
        original = source.read_bytes()
        self.assertIn(b"ROTATION_PLACEHOLDER", original)
        source.write_bytes(rewrite(original))
        request.input = source
        files = media.FileExtraction()
        files.files.output_directory = self.root / "rotation-output"
        files.preserve_source_metadata = True
        self.assertEqual(media.MediaIngest.extract_files(request, files).frames_accepted, 4)
        self.assertEqual(len(list(files.files.output_directory.glob("frame_*.png"))), 4)
        path = files.files.output_directory / "extraction_metadata.json"
        self.assertLess(path.stat().st_size, 128 * 1024)
        snapshot = json.loads(path.read_text(encoding="utf-8"))["source_metadata"]
        orientation = snapshot["metadata"]["description"]["streams"][0]["orientation"]
        self.assertTrue(orientation["rotate_tag_limited"])
        self.assertIsNone(orientation["rotate_tag"])
        self.assertIsNone(orientation["clockwise_degrees"])
        self.assertTrue(snapshot["summary"]["limited"])
        observed = snapshot["metadata"]["source_metadata"]
        self.assertTrue(observed["limited"])
        probed = media.MediaIngest.probe(source)
        self.assertTrue(probed.streams[0].orientation.rotate_tag_limited)

    def test_source_metadata_is_opt_in_and_shared_by_output_frames(self):
        request = self.metadata_request()
        for preserve in (False, True):
            files = media.FileExtraction()
            files.files.output_directory = self.root / ("preserved" if preserve else "omitted")
            files.write_metadata = True
            files.preserve_source_metadata = preserve
            media.MediaIngest.extract_files(request, files)
            manifest = json.loads((files.files.output_directory / "extraction_metadata.json").read_text(encoding="utf-8"))
            self.assertEqual("source_metadata" in manifest, preserve)
            self.assertEqual("source_metadata_policy" in manifest, preserve)
            if preserve:
                self.assertEqual(manifest["source_metadata_policy"], "preserve_static_container_v1")
                snapshot = manifest["source_metadata"]
                description = snapshot["metadata"]["description"]
                self.assertEqual(description["selected_video_stream"], manifest["input"]["video"]["stream_index"])
                native = str(request.input).encode("utf-16-le") if os.name == "nt" else os.fsencode(request.input)
                self.assertEqual(snapshot["metadata"]["source_locator_bytes"], native.hex())
                self.assertEqual((snapshot["schema_version"], snapshot["success"]), (1, True))
                observed = snapshot["metadata"]["source_metadata"]
                tags = {bytes.fromhex(tag["key_bytes"]).lower(): bytes.fromhex(tag["value_bytes"]) for tag in observed["tags"]}
                self.assertEqual(tags[b"comment"], b"STATIC_METADATA_SENTINEL")
                self.assertEqual(tags[b"location"], b"+01.2500+002.5000/")
                self.assertFalse(snapshot["summary"]["limited"])
                self.assertGreater(len(manifest["frames"]), 1)
                self.assertTrue(all("source_metadata" not in frame and "source_pts" in frame for frame in manifest["frames"]))
        omitted, preserved = [self.root / name for name in ("omitted", "preserved")]
        self.assertEqual({p.name: p.read_bytes() for p in omitted.glob("frame_*.png")},
                         {p.name: p.read_bytes() for p in preserved.glob("frame_*.png")})

    def test_source_metadata_keeps_other_stream_tags(self):
        request = self.metadata_request(multitrack=True)
        files = media.FileExtraction()
        files.files.output_directory = self.root / "metadata-multitrack"
        files.preserve_source_metadata = True
        media.MediaIngest.extract_files(request, files)
        manifest = json.loads((files.files.output_directory / "extraction_metadata.json").read_text(encoding="utf-8"))
        catalog = manifest["source_metadata"]["metadata"]
        audio, = [stream for stream in catalog["description"]["streams"] if stream["kind"] == "audio"]
        observed, = [stream for stream in catalog["source_metadata"]["streams"] if stream["stream_index"] == audio["index"]]
        self.assertTrue(any(bytes.fromhex(tag["value_bytes"]) == b"AUDIO_METADATA_SENTINEL" for tag in observed["tags"]))

    def test_source_metadata_preservation_all_image_formats(self):
        request = self.metadata_request()
        request.geometry.clockwise_rotation = 90
        request.geometry.mode = media.ResizeMode.Scale
        request.geometry.scale = 0.5
        for name in ("PNG", "JPEG", "EXR", "TIFF16", "JXL", "AVIF"):
            with self.subTest(format=name):
                files = media.FileExtraction()
                files.files.output_directory = self.root / ("metadata-" + name)
                files.files.format = getattr(media.FrameFileFormat, name)
                files.preserve_source_metadata = True
                request.output_format = media.FramePixelFormat.RGB8 if name in ("PNG", "JPEG") else media.FramePixelFormat.RGBFloat32
                request.input_color.transfer = media.ColorTransfer.Unspecified if name in ("PNG", "JPEG") else media.ColorTransfer.Linear
                request.input_color.primaries = media.ColorPrimaries.Unspecified if name in ("PNG", "JPEG") else media.ColorPrimaries.Bt709
                report = media.MediaIngest.extract_files(request, files)
                manifest = json.loads((files.files.output_directory / "extraction_metadata.json").read_text(encoding="utf-8"))
                self.assertEqual(len(manifest["frames"]), report.frames_accepted)
                self.assertEqual(manifest["output"]["size"], [16, 32])
                dimensions = {fact["semantic"]: fact["value"] for fact in manifest["source_metadata"]["summary"]["facts"]}
                self.assertEqual((dimensions["image.width"], dimensions["image.height"]), (64, 32))

    def test_jpeg_embedded_source_metadata_roundtrip_and_unchanged_scan(self):
        request = self.metadata_request()
        request.geometry.mode = media.ResizeMode.Scale
        request.geometry.scale = 0.5
        request.geometry.clockwise_rotation = 90
        plain = media.FileExtraction()
        plain.files.format = media.FrameFileFormat.JPEG
        plain.files.output_directory = self.root / "jpeg-original"
        media.MediaIngest.extract_files(request, plain)
        output = media.FileExtraction()
        output.files.format = media.FrameFileFormat.JPEG
        output.files.embed_source_metadata = True
        output.files.output_directory = self.root / "jpeg-embedded é 日本語"
        report = media.MediaIngest.extract_files(request, output)
        manifest = json.loads((output.files.output_directory / "extraction_metadata.json").read_text(encoding="utf-8"))
        self.assertEqual(manifest["embedded_metadata_policy"], "jpeg-static-source-xmp-v1")
        self.assertEqual(len(manifest["frames"]), report.frames_accepted)
        for image in output.files.output_directory.glob("*.jpg"):
            original, orientation, dimensions, facts, catalog, chunks = jpeg_metadata_fixture.read(image.read_bytes())
            self.assertEqual(original, (plain.files.output_directory / image.name).read_bytes())
            self.assertEqual((orientation, dimensions), (1, (16, 32)))
            self.assertEqual((facts["PixelWidth"], facts["PixelHeight"]), ("16", "32"))
            self.assertEqual(catalog, manifest["source_metadata"])
            self.assertEqual(catalog["metadata"]["scope"], "container_stream")
            self.assertIn("SourceTimestampTicks", facts)
            self.assertEqual(facts["AppliedClockwiseRotation"], "90")
            self.assertEqual(chunks, 0)
        self.assertGreater(len(list(output.files.output_directory.glob("*.jpg"))), 0)

    def test_jpeg_embedded_metadata_resume_policy_and_replay(self):
        self._check_embedded_metadata_resume(media.FrameFileFormat.JPEG, ".jpg")

    def test_png_embedded_metadata_serial_parallel_and_unchanged_pixels(self):
        for large in (False, True):
            with self.subTest(parallel=large):
                request = self.metadata_request()
                request.end_seconds = 0.01
                request.geometry.mode = media.ResizeMode.Custom if large else media.ResizeMode.Scale
                if large:
                    request.geometry.width, request.geometry.height = 1025, 3073
                    request.geometry.clockwise_rotation = 0
                else:
                    request.geometry.scale = 0.5
                    request.geometry.clockwise_rotation = 90
                plain = media.FileExtraction()
                plain.files.output_directory = self.root / ("png-plain-" + str(large))
                media.MediaIngest.extract_files(request, plain)
                output = media.FileExtraction()
                output.files.output_directory = self.root / ("png-metadata-" + str(large))
                output.files.embed_source_metadata = True
                result = media.MediaIngest.extract_files(request, output)
                self.assertGreater(result.frames_accepted, 0)
                manifest = json.loads((output.files.output_directory / "extraction_metadata.json").read_text(encoding="utf-8"))
                self.assertEqual(manifest["embedded_metadata_policy"], "png-static-source-xmp-v1")
                for image in output.files.output_directory.glob("*.png"):
                    data = image.read_bytes()
                    original, dimensions, facts, catalog = png_metadata_fixture.read(data)
                    self.assertEqual(original, (plain.files.output_directory / image.name).read_bytes())
                    self.assertEqual(dimensions, (1025, 3073) if large else (16, 32))
                    self.assertEqual((facts["PixelWidth"], facts["PixelHeight"]), tuple(map(str, dimensions)))
                    self.assertEqual(facts["MetadataPolicy"], "png-static-source-xmp-v1")
                    self.assertEqual(facts["TimestampOrigin"], "best_effort")
                    self.assertEqual(catalog, manifest["source_metadata"])
                    decoded = subprocess.check_output([os.environ.get("LFS_MEDIA_TEST_FFMPEG", "ffmpeg"),
                                                       "-v", "error", "-i", str(image), "-f", "rawvideo", "-pix_fmt", "rgb24", "-"], timeout=30)
                    self.assertEqual(len(decoded), dimensions[0] * dimensions[1] * 3)
                    corrupted = bytearray(data)
                    at = data.index(b"iTXt") + 4 + 22
                    corrupted[at] ^= 1
                    with self.assertRaises(AssertionError):
                        png_metadata_fixture.read(corrupted)
                    with self.assertRaises(AssertionError):
                        png_metadata_fixture.read(data[:-12])

    def test_png_embedded_metadata_resume_policy_and_replay(self):
        self._check_embedded_metadata_resume(media.FrameFileFormat.PNG, ".png")

    def test_tiff_embedded_metadata_preserves_color_strips_and_pixels(self):
        import xml.etree.ElementTree as ET
        request = self.metadata_request()
        request.output_format = media.FramePixelFormat.RGBFloat32
        request.input_color.transfer = media.ColorTransfer.Linear
        request.input_color.primaries = media.ColorPrimaries.Bt709
        request.end_seconds = 0.01
        request.geometry.clockwise_rotation = 90
        plain = media.FileExtraction()
        plain.files.format = media.FrameFileFormat.TIFF16
        plain.files.output_directory = self.root / "plain-tiff-metadata"
        self.assertGreater(media.MediaIngest.extract_files(request, plain).frames_accepted, 0)
        output = media.FileExtraction()
        output.files.format = media.FrameFileFormat.TIFF16
        output.files.embed_source_metadata = True
        output.files.output_directory = self.root / "embedded TIFF é 日本語"
        self.assertGreater(media.MediaIngest.extract_files(request, output).frames_accepted, 0)
        manifest = json.loads((output.files.output_directory / "extraction_metadata.json").read_text(encoding="utf-8"))
        self.assertEqual(manifest["embedded_metadata_policy"], "tiff16-static-source-xmp-v1")
        paths = list(output.files.output_directory.glob("*.tif"))
        self.assertTrue(paths)
        for image in paths:
            tags, strips, packet = source_exif_fixture.read_export_tiff(image.read_bytes())
            plain_path = plain.files.output_directory / image.name
            plain_tags, plain_strips, absent = source_exif_fixture.read_export_tiff(plain_path.read_bytes())
            self.assertIsNone(absent)
            self.assertEqual(tags, plain_tags)
            self.assertEqual(strips, plain_strips)
            self.assertTrue(packet.endswith(b'<?xpacket end="r"?>'))
            self.assertNotIn(b"\0", packet)
            tree = ET.fromstring(packet)
            facts = {node.tag.rsplit("}", 1)[-1]: node.text for node in tree.iter()
                     if node.tag.startswith("{https://lichtfeld.io/ns/media/1.0/}")}
            self.assertEqual(facts["MetadataPolicy"], "tiff16-static-source-xmp-v1")
            self.assertEqual((facts["PixelWidth"], facts["PixelHeight"]), ("32", "64"))
            self.assertEqual(facts["AppliedClockwiseRotation"], "90")
            self.assertEqual(facts["TimestampOrigin"], "best_effort")
            self.assertEqual(json.loads(facts["SourceMetadata"]), manifest["source_metadata"])
            observed = json.loads(media.MediaIngest.source_metadata_json(image))
            self.assertTrue(observed["success"])
            blocks = observed["metadata"]["image_xmp"]["blocks"]
            self.assertEqual([bytes.fromhex(block["bytes"]) for block in blocks], [packet])
            pixels = []
            for path in (plain_path, image):
                pixels.append(subprocess.check_output([os.environ.get("LFS_MEDIA_TEST_FFMPEG", "ffmpeg"),
                              "-v", "error", "-i", str(path), "-f", "rawvideo", "-pix_fmt", "rgb48le", "-"], timeout=30))
            self.assertEqual(len(pixels[0]), 32 * 64 * 6)
            self.assertEqual(pixels[0], pixels[1])
            with self.assertRaises(AssertionError):
                source_exif_fixture.read_export_tiff(image.read_bytes()[:12])

    def test_tiff_embedded_metadata_resume_policy_and_replay(self):
        self._check_embedded_metadata_resume(media.FrameFileFormat.TIFF16, ".tif")

    def test_exr_embedded_metadata_preserves_attributes_chunks_and_pixels(self):
        import xml.etree.ElementTree as ET
        request = self.metadata_request()
        request.output_format = media.FramePixelFormat.RGBFloat32
        request.input_color.transfer = media.ColorTransfer.Linear
        request.input_color.primaries = media.ColorPrimaries.Bt709
        request.geometry.clockwise_rotation = 90
        request.end_seconds = 0.01
        for precision in (media.ExrPrecision.Half, media.ExrPrecision.Float):
            for compression in (media.ExrCompression.Uncompressed, media.ExrCompression.ZIP):
                with self.subTest(precision=precision, compression=compression):
                    name = f"exr-{precision}-{compression}"
                    plain = media.FileExtraction()
                    plain.files.format = media.FrameFileFormat.EXR
                    plain.files.exr.precision, plain.files.exr.compression = precision, compression
                    plain.files.output_directory = self.root / (name + "-plain")
                    self.assertGreater(media.MediaIngest.extract_files(request, plain).frames_accepted, 0)
                    output = media.FileExtraction()
                    output.files.format = media.FrameFileFormat.EXR
                    output.files.exr.precision, output.files.exr.compression = precision, compression
                    output.files.embed_source_metadata = True
                    output.files.output_directory = self.root / (name + " é 日本語")
                    self.assertGreater(media.MediaIngest.extract_files(request, output).frames_accepted, 0)
                    manifest = json.loads((output.files.output_directory / "extraction_metadata.json").read_text(encoding="utf-8"))
                    self.assertEqual(manifest["embedded_metadata_policy"], "exr-static-source-xmp-v1")
                    paths = list(output.files.output_directory.glob("*.exr"))
                    self.assertTrue(paths)
                    for image in paths:
                        attrs, chunks, packet = source_exif_fixture.read_export_exr(image.read_bytes())
                        original = plain.files.output_directory / image.name
                        before, before_chunks, absent = source_exif_fixture.read_export_exr(original.read_bytes())
                        self.assertIsNone(absent)
                        self.assertEqual(attrs, before)
                        self.assertEqual(chunks, before_chunks)
                        self.assertTrue(packet.endswith(b'<?xpacket end="r"?>'))
                        self.assertNotIn(b"\0", packet)
                        facts = {node.tag.rsplit("}", 1)[-1]: node.text for node in ET.fromstring(packet).iter()
                                 if node.tag.startswith("{https://lichtfeld.io/ns/media/1.0/}")}
                        self.assertEqual(facts["MetadataPolicy"], "exr-static-source-xmp-v1")
                        self.assertEqual((facts["PixelWidth"], facts["PixelHeight"]), ("32", "64"))
                        self.assertEqual(facts["AppliedClockwiseRotation"], "90")
                        self.assertEqual(facts["TimestampOrigin"], "best_effort")
                        self.assertEqual(json.loads(facts["SourceMetadata"]), manifest["source_metadata"])
                        observed = json.loads(media.MediaIngest.source_metadata_json(image))
                        self.assertTrue(observed["success"])
                        blocks = observed["metadata"]["image_xmp"]["blocks"]
                        self.assertEqual([block["envelope"] for block in blocks], ["exr_string"])
                        self.assertEqual([bytes.fromhex(block["bytes"]) for block in blocks], [packet])
                        decoded = []
                        for path in (original, image):
                            decoded.append(subprocess.check_output([os.environ.get("LFS_MEDIA_TEST_FFMPEG", "ffmpeg"),
                                           "-v", "error", "-i", str(path), "-f", "rawvideo", "-pix_fmt", "gbrpf32le", "-"], timeout=30))
                        self.assertEqual(len(decoded[0]), 32 * 64 * 12)
                        self.assertEqual(decoded[0], decoded[1])
                        with self.assertRaises(AssertionError):
                            source_exif_fixture.read_export_exr(image.read_bytes()[:12])

    def test_exr_embedded_metadata_resume_policy_and_replay(self):
        self._check_embedded_metadata_resume(media.FrameFileFormat.EXR, ".exr")

    def _check_embedded_metadata_resume(self, file_format, extension):
        request = self.metadata_request()
        if file_format in (media.FrameFileFormat.TIFF16, media.FrameFileFormat.EXR):
            request.output_format = media.FramePixelFormat.RGBFloat32
            request.input_color.transfer = media.ColorTransfer.Linear
            request.input_color.primaries = media.ColorPrimaries.Bt709
        for enabled in (False, True):
            output = self.resume_files(self.root / f"metadata-resume-{extension[1:]}-{enabled}")
            output.files.format = file_format
            output.files.embed_source_metadata = enabled
            media.MediaIngest.extract_files(request, output)
            files = sorted(output.files.output_directory.glob("*" + extension))
            self.assertTrue(files)
            before = [(p.read_bytes(), p.stat().st_mtime_ns) for p in files]
            output.files.embed_source_metadata = not enabled
            with self.assertRaises(MEDIA_ERROR):
                media.MediaIngest.extract_files(request, output)
            self.assertEqual(before, [(p.read_bytes(), p.stat().st_mtime_ns) for p in files])
            output.files.embed_source_metadata = enabled
            media.MediaIngest.extract_files(request, output)
            self.assertEqual(before, [(p.read_bytes(), p.stat().st_mtime_ns) for p in files])
            record_path = output.files.output_directory / ".media-ingest-checkpoint/frame-0.json"
            original_record = record_path.read_bytes()
            record = json.loads(original_record)
            if not enabled:
                self.assertNotIn("timestamp_origin", record)
            if enabled:
                self.assertEqual(record["timestamp_origin"], 1)
                record["timestamp_origin"] = 2
                record_path.write_text(json.dumps(record), encoding="utf-8")
                with self.assertRaises(MEDIA_ERROR):
                    media.MediaIngest.extract_files(request, output)
                self.assertEqual(before, [(p.read_bytes(), p.stat().st_mtime_ns) for p in files])
                record_path.write_bytes(original_record)
                media.MediaIngest.extract_files(request, output)
                self.assertEqual(before, [(p.read_bytes(), p.stat().st_mtime_ns) for p in files])
                original_source = request.input
                alias = self.root / "same-bytes-different-source-path.mkv"
                shutil.copyfile(original_source, alias)
                request.input = alias
                with self.assertRaises(MEDIA_ERROR):
                    media.MediaIngest.extract_files(request, output)
                self.assertEqual(before, [(p.read_bytes(), p.stat().st_mtime_ns) for p in files])
                request.input = original_source
                media.MediaIngest.extract_files(request, output)

    def test_jpeg_extended_xmp_preserves_raw_source_bytes(self):
        request = self.request()
        source = self.root / "large-source-tags.mkv"
        # A normal command-size tag whose JSON and XML representations require
        # more than one APP1 segment; no ffmetadata demuxer dependency.
        value = 'A<&"' * 4000
        subprocess.run([os.environ.get("LFS_MEDIA_TEST_FFMPEG", "ffmpeg"), "-v", "error", "-y",
                        "-i", str(request.input), "-map", "0:v:0", "-c:v", "copy",
                        "-metadata", "comment=" + value, "-write_crc32", "0", str(source)],
                       check=True, capture_output=True, timeout=30)
        request.input = source
        output = media.FileExtraction()
        output.files.format = media.FrameFileFormat.JPEG
        output.files.embed_source_metadata = True
        output.files.output_directory = self.root / "extended-jpeg"
        self.assertGreater(media.MediaIngest.extract_files(request, output).frames_accepted, 0)
        manifest = json.loads((output.files.output_directory / "extraction_metadata.json").read_text(encoding="utf-8"))
        for image in output.files.output_directory.glob("*.jpg"):
            _, _, _, _, catalog, chunks = jpeg_metadata_fixture.read(image.read_bytes())
            self.assertGreater(chunks, 1)
            # Decode the whole pixel stream, not only its metadata/header.
            decoded = subprocess.check_output([os.environ.get("LFS_MEDIA_TEST_FFMPEG", "ffmpeg"),
                                               "-v", "error", "-i", str(image), "-f", "rawvideo",
                                               "-pix_fmt", "rgb24", "-"], timeout=30)
            self.assertEqual(len(decoded), 64 * 32 * 3)
            data = image.read_bytes()
            offset = 2
            extended_offsets = []
            while data[offset:offset + 2] != b"\xff\xda":
                length = int.from_bytes(data[offset + 2:offset + 4], "big")
                if data[offset + 4:].startswith(b"http://ns.adobe.com/xmp/extension/\0"):
                    extended_offsets.append(offset + 4)
                offset += 2 + length
            with self.assertRaises(AssertionError):
                jpeg_metadata_fixture.read(data[:offset])
            damaged = bytearray(data)
            at = extended_offsets[0] + 67
            total = int.from_bytes(damaged[at:at + 4], "big")
            damaged[at:at + 4] = (total + 1).to_bytes(4, "big")
            with self.assertRaises(AssertionError):
                jpeg_metadata_fixture.read(damaged)
            self.assertEqual(catalog, manifest["source_metadata"])
            tags = catalog["metadata"]["source_metadata"]["tags"]
            observed = {bytes.fromhex(t["key_bytes"]).lower(): bytes.fromhex(t["value_bytes"]) for t in tags}
            self.assertEqual(observed[b"comment"], value.encode("utf-8"))

    def test_embedding_rejects_other_formats_before_output(self):
        output = media.FileExtraction()
        output.files.embed_source_metadata = True
        output.files.format = media.FrameFileFormat.JXL
        output.files.output_directory = self.root / "invalid-embedded-jxl"
        with self.assertRaises(MEDIA_ERROR):
            media.MediaIngest.extract_files(self.request(), output)
        self.assertFalse(output.files.output_directory.exists())

    def test_source_metadata_policy_is_bound_to_resume_identity(self):
        request = self.metadata_request()
        for preserve in (False, True):
            with self.subTest(preserve=preserve):
                files = self.resume_files(self.root / ("metadata-resume-" + str(preserve)))
                files.preserve_source_metadata = preserve
                self.interrupt_resume(request, files)
                before = {p.name: (p.read_bytes(), p.stat().st_mtime_ns) for p in files.files.output_directory.glob("frame_*.png")}
                files.preserve_source_metadata = not preserve
                with self.assertRaises(MEDIA_ERROR):
                    media.MediaIngest.extract_files(request, files)
                self.assertEqual(before, {p.name: (p.read_bytes(), p.stat().st_mtime_ns) for p in files.files.output_directory.glob("frame_*.png")})
                files.preserve_source_metadata = preserve
                media.MediaIngest.extract_files(request, files)
                manifest = json.loads((files.files.output_directory / "extraction_metadata.json").read_text(encoding="utf-8"))
                self.assertEqual("source_metadata" in manifest, preserve)

    def test_non_utf8_source_tag_bytes_survive_manifest_transport(self):
        request = self.metadata_request()
        data = request.input.read_bytes()
        self.assertIn(b"STATIC_METADATA_SENTINEL", data)
        request.input.write_bytes(data.replace(b"STATIC_METADATA_SENTINEL", b"\xffTATIC_METADATA_SENTINEL"))
        files = media.FileExtraction()
        files.files.output_directory = self.root / "metadata-bytes"
        files.preserve_source_metadata = True
        media.MediaIngest.extract_files(request, files)
        manifest = json.loads((files.files.output_directory / "extraction_metadata.json").read_text(encoding="utf-8"))
        tags = manifest["source_metadata"]["metadata"]["source_metadata"]["tags"]
        comment, = [tag for tag in tags if bytes.fromhex(tag["key_bytes"]).lower() == b"comment"]
        self.assertEqual(bytes.fromhex(comment["value_bytes"]), b"\xffTATIC_METADATA_SENTINEL")
        self.assertIn("\ufffd", comment["value"])

    def test_budget_and_cancellation_failures(self):
        with self.assertRaises(MEDIA_ERROR):
            media.MediaIngest.extract(self.request(), payload_budget=1)
        with self.assertRaises(MEDIA_ERROR):
            media.MediaIngest.extract(self.request(), cancelled=lambda: True)
        files = media.FileExtraction()
        files.files.output_directory = self.root / "cancel-output"
        with self.assertRaises(MEDIA_ERROR):
            media.MediaIngest.extract_files(self.request(), files, cancelled=lambda: True)
        self.assertFalse(list(files.files.output_directory.glob("frame_*.png")))

    def resume_files(self, directory, file_format=None):
        files = media.FileExtraction()
        files.files.output_directory = directory
        files.write_metadata = True
        files.resume = True
        if file_format is not None:
            files.files.format = file_format
        return files

    def interrupt_resume(self, request, files):
        processed = []
        with self.assertRaises(MEDIA_ERROR):
            media.MediaIngest.extract_files(request, files,
                progress=lambda value: processed.append(value.processed),
                cancelled=lambda: bool(processed))
        self.assertTrue(processed)

    def test_resume_replays_selection_without_rewriting_verified_images(self):
        for label, file_format in (("png", media.FrameFileFormat.PNG), ("jpeg", media.FrameFileFormat.JPEG),
                                   ("exr", media.FrameFileFormat.EXR)):
            with self.subTest(format=label):
                request = self.request()
                if label == "exr":
                    request.output_format = media.FramePixelFormat.RGBFloat32
                    request.input_color.transfer = media.ColorTransfer.Linear
                    request.input_color.primaries = media.ColorPrimaries.Bt709
                files = self.resume_files(self.root / ("resume-" + label), file_format)
                self.interrupt_resume(request, files)
                previous = {p: (p.read_bytes(), p.stat().st_mtime_ns)
                            for p in files.files.output_directory.iterdir() if p.is_file()}
                report = media.MediaIngest.extract_files(request, files)
                self.assertEqual(report.frames_accepted, 4)
                for path, (data, modified) in previous.items():
                    self.assertEqual(path.read_bytes(), data)
                    self.assertEqual(path.stat().st_mtime_ns, modified)
                manifest = json.loads((files.files.output_directory / "extraction_metadata.json").read_text())
                self.assertEqual(len(manifest["frames"]), 4)
                self.assertEqual(len(list((files.files.output_directory / ".media-ingest-checkpoint").glob("frame-*.json"))), 4)

    def test_resume_sha256_matches_independent_input_pixels_and_files(self):
        request = self.request()
        _, frames = media.MediaIngest.extract(request)
        files = self.resume_files(self.root / "resume-sha256")
        media.MediaIngest.extract_files(request, files)
        checkpoint = files.files.output_directory / ".media-ingest-checkpoint"
        identity = json.loads((checkpoint / "identity.json").read_text())
        self.assertEqual(identity["input_sha256"], hashlib.sha256(request.input.read_bytes()).hexdigest())
        for index, frame in enumerate(frames):
            record = json.loads((checkpoint / f"frame-{index}.json").read_text())
            digest = hashlib.sha256()
            for row in range(frame.layout.height):
                start = row * frame.layout.row_stride
                digest.update(frame.pixels[start:start + frame.layout.width * 3])
            self.assertEqual(record["pixels_sha256"], digest.hexdigest())
            self.assertEqual(record["file_sha256"], hashlib.sha256(
                (files.files.output_directory / record["file"]).read_bytes()).hexdigest())

    def test_resume_manifest_preserves_selected_source_timestamps(self):
        for fixture, end in (("cfr-asymmetric", 0.35), ("vfr-asymmetric", 1.5)):
            for label, file_format in (("png", media.FrameFileFormat.PNG), ("exr", media.FrameFileFormat.EXR)):
                with self.subTest(fixture=fixture, format=label):
                    request = self.request()
                    request.input = self.corpus / (fixture + ".nut")
                    request.end_seconds = end
                    if label == "exr":
                        request.output_format = media.FramePixelFormat.RGBFloat32
                        request.input_color.transfer = media.ColorTransfer.Linear
                        request.input_color.primaries = media.ColorPrimaries.Bt709
                    _, selected = media.MediaIngest.extract(request)
                    files = self.resume_files(self.root / ("resume-timing-" + fixture + label), file_format)
                    files.files.exr.overwrite = True
                    self.interrupt_resume(request, files)
                    media.MediaIngest.extract_files(request, files)
                    manifest_path = files.files.output_directory / "extraction_metadata.json"
                    manifest = json.loads(manifest_path.read_text())
                    self.assertEqual(len(manifest["frames"]), len(selected))
                    for entry, surface in zip(manifest["frames"], selected):
                        self.assertEqual(entry["source_pts"], surface.info.source_timestamp.ticks)
                        self.assertEqual(entry["timestamp_origin"], "best_effort")
                        self.assertAlmostEqual(entry["timestamp"], surface.info.relative_seconds, places=6)
                    images = {entry["file"]: (files.files.output_directory / entry["file"]).read_bytes()
                              for entry in manifest["frames"]}
                    media.MediaIngest.extract_files(request, files)
                    self.assertEqual(json.loads(manifest_path.read_text())["frames"], manifest["frames"])
                    missing = files.files.output_directory / manifest["frames"][-1]["file"]
                    missing.unlink()
                    media.MediaIngest.extract_files(request, files)
                    self.assertEqual(json.loads(manifest_path.read_text())["frames"], manifest["frames"])
                    self.assertEqual(images, {name: (files.files.output_directory / name).read_bytes() for name in images})

    def test_resume_rejects_changed_request_and_corrupt_output(self):
        request = self.request()
        files = self.resume_files(self.root / "resume-reject")
        self.interrupt_resume(request, files)
        image = next(files.files.output_directory.glob("frame_*.png"))
        original = image.read_bytes()
        request.selection.interval = 2
        with self.assertRaises(MEDIA_ERROR):
            media.MediaIngest.extract_files(request, files)
        self.assertEqual(image.read_bytes(), original)
        request.selection.interval = 1
        image.write_bytes(original[:16])
        with self.assertRaises(MEDIA_ERROR):
            media.MediaIngest.extract_files(request, files)
        self.assertEqual(image.read_bytes(), original[:16])
        self.assertFalse((files.files.output_directory / "extraction_metadata.json").exists())

    def test_resume_recovers_empty_publication_reservations(self):
        for kind in ("identity", "record", "image"):
            with self.subTest(reservation=kind):
                files = self.resume_files(self.root / ("resume-empty-" + kind))
                checkpoint = files.files.output_directory / ".media-ingest-checkpoint"
                if kind == "identity":
                    checkpoint.mkdir(parents=True)
                    (checkpoint / "identity.json").touch()
                    (checkpoint / "identity.json.tmp-interrupted").write_text("partial")
                else:
                    self.interrupt_resume(self.request(), files)
                    image = next(files.files.output_directory.glob("frame_*.png"))
                    if kind == "record":
                        image.unlink()
                        (checkpoint / "frame-0.json").write_bytes(b"")
                    else:
                        image.write_bytes(b"")
                self.assertEqual(media.MediaIngest.extract_files(self.request(), files).frames_accepted, 4)
                self.assertTrue(all(path.stat().st_size > 0 for path in files.files.output_directory.glob("frame_*.png")))

    def test_resume_surplus_records_preserve_previous_manifest(self):
        files = self.resume_files(self.root / "resume-surplus")
        media.MediaIngest.extract_files(self.request(), files)
        checkpoint = files.files.output_directory / ".media-ingest-checkpoint"
        # Schema-valid extra record: only end-of-selection validation can reject it.
        shutil.copyfile(checkpoint / "frame-0.json", checkpoint / "frame-4.json")
        manifest = files.files.output_directory / "extraction_metadata.json"
        before = (manifest.read_bytes(), manifest.stat().st_mtime_ns)
        with self.assertRaises(MEDIA_ERROR):
            media.MediaIngest.extract_files(self.request(), files)
        self.assertEqual((manifest.read_bytes(), manifest.stat().st_mtime_ns), before)

    def test_resume_rejects_different_application_build(self):
        files = self.resume_files(self.root / "resume-build")
        self.interrupt_resume(self.request(), files)
        identity = files.files.output_directory / ".media-ingest-checkpoint" / "identity.json"
        value = json.loads(identity.read_text())
        self.assertTrue(value["application_build"])
        value["application_build"] += "-different-codecs"
        identity.write_text(json.dumps(value))
        with self.assertRaises(MEDIA_ERROR):
            media.MediaIngest.extract_files(self.request(), files)

    def test_completed_exr_resume_regenerates_manifest_without_image_overwrite(self):
        request = self.request()
        request.output_format = media.FramePixelFormat.RGBFloat32
        request.input_color.transfer = media.ColorTransfer.Linear
        request.input_color.primaries = media.ColorPrimaries.Bt709
        files = self.resume_files(self.root / "resume-complete-exr", media.FrameFileFormat.EXR)
        media.MediaIngest.extract_files(request, files)
        image = next(files.files.output_directory.glob("frame_*.exr"))
        before = (image.read_bytes(), image.stat().st_mtime_ns)
        manifest = files.files.output_directory / "extraction_metadata.json"
        original_manifest = json.loads(manifest.read_bytes())
        self.assertEqual(media.MediaIngest.extract_files(request, files).frames_accepted, 4)
        self.assertEqual(json.loads(manifest.read_bytes())["output"], original_manifest["output"])
        files.files.exr.overwrite = True
        self.assertEqual(media.MediaIngest.extract_files(request, files).frames_accepted, 4)
        self.assertEqual((image.read_bytes(), image.stat().st_mtime_ns), before)

    def test_resume_reconstructs_missing_image_and_pending_publication(self):
        for label, file_format, extension in (("png", media.FrameFileFormat.PNG, "png"),
                                               ("jpeg", media.FrameFileFormat.JPEG, "jpg"),
                                               ("exr", media.FrameFileFormat.EXR, "exr"),
                                               ("tiff16", media.FrameFileFormat.TIFF16, "tif")):
            with self.subTest(format=label):
                request = self.request()
                if label in ("exr", "tiff16"):
                    request.output_format = media.FramePixelFormat.RGBFloat32
                    request.input_color.transfer = media.ColorTransfer.Linear
                    request.input_color.primaries = media.ColorPrimaries.Bt709
                files = self.resume_files(self.root / ("resume-missing-" + label), file_format)
                self.interrupt_resume(request, files)
                image = next(files.files.output_directory.glob("frame_*." + extension))
                expected = image.read_bytes()
                image.unlink()
                report = media.MediaIngest.extract_files(request, files)
                self.assertEqual(report.frames_accepted, 4)
                self.assertEqual(image.read_bytes(), expected)

    def test_resume_recovers_record_published_before_image(self):
        request = self.request()
        files = self.resume_files(self.root / "resume-pending")
        record = files.files.output_directory / ".media-ingest-checkpoint" / "frame-0.json"
        with self.assertRaises(MEDIA_ERROR):
            media.MediaIngest.extract_files(request, files, cancelled=record.exists)
        self.assertTrue(record.is_file())
        self.assertFalse(list(files.files.output_directory.glob("frame_*.png")))
        self.assertEqual(media.MediaIngest.extract_files(request, files).frames_accepted, 4)

    def resume_child(self, directory, environment=None, exit_after_progress=True):
        # Exit only this disposable child, after a completed image publication.
        program = r'''
import os, sys
from pathlib import Path
handles = []
if os.name == "nt":
    for directory in sys.argv[4:]:
        if directory:
            handles.append(os.add_dll_directory(directory))
sys.path.insert(0, sys.argv[1])
import media_test_bindings as media
r = media.IngestRequest()
r.input = Path(sys.argv[2])
r.selection.mode = media.SelectionMode.Interval
r.selection.interval = 1
r.end_seconds = .35
r.allow_hardware_decode = False
f = media.FileExtraction()
f.files.output_directory = Path(sys.argv[3])
f.write_metadata = True
f.resume = True
media.MediaIngest.extract_files(r, f, progress=lambda _: os._exit(73))
'''
        if args.studio:
            program = program.replace("import media_test_bindings as media", "from lichtfeld import media")
        if not exit_after_progress:
            program = program.replace("progress=lambda _: os._exit(73)", "progress=lambda _: None")
        command = [sys.executable, "-c", program, str(args.module_dir.resolve()),
                   str(self.request().input), str(directory), str(args.runtime_dir.resolve()),
                   str(args.dependency_dir.resolve()) if args.dependency_dir else ""]
        return subprocess.run(command, capture_output=True, timeout=30, env=environment)

    def test_resume_after_abrupt_process_exit(self):
        directory = self.root / "resume-crash"
        child = self.resume_child(directory)
        self.assertEqual(child.returncode, 73, child.stderr.decode(errors="replace"))
        self.assertEqual(len(list(directory.glob("frame_*.png"))), 1)
        self.assertEqual(media.MediaIngest.extract_files(self.request(), self.resume_files(directory)).frames_accepted, 4)

    @unittest.skipUnless(sys.platform == "darwin" and os.environ.get("LFS_MEDIA_TEST_RESERVED_PUBLICATION") == "1",
                         "requires explicit macOS/exFAT reserved-publication qualification")
    def test_resume_after_exit_inside_reserved_publication(self):
        shim = Path(os.environ["LFS_MEDIA_TEST_RESERVATION_SHIM"])
        self.assertTrue(shim.is_file())
        for name in ("identity.json", "frame-0.json", "frame_1.png"):
            with self.subTest(reservation=name):
                directory = self.root / ("reserved-crash-" + name)
                environment = dict(os.environ, DYLD_INSERT_LIBRARIES=str(shim),
                                   LFS_MEDIA_TEST_RESERVED_FILENAME=name)
                child = self.resume_child(directory, environment, exit_after_progress=False)
                self.assertEqual(child.returncode, 73, child.stderr.decode(errors="replace"))
                reservation = directory / name if name.endswith(".png") else directory / ".media-ingest-checkpoint" / name
                self.assertEqual(reservation.stat().st_size, 0)
                self.assertEqual(media.MediaIngest.extract_files(self.request(), self.resume_files(directory)).frames_accepted, 4)

    def test_resume_refuses_to_adopt_legacy_output(self):
        request = self.request()
        directory = self.root / "resume-legacy"
        files = self.resume_files(directory)
        files.resume = False
        media.MediaIngest.extract_files(request, files)
        before = {p: p.read_bytes() for p in directory.iterdir() if p.is_file()}
        files.resume = True
        with self.assertRaises(MEDIA_ERROR):
            media.MediaIngest.extract_files(request, files)
        self.assertEqual(before, {p: p.read_bytes() for p in directory.iterdir() if p.is_file()})

    def test_resume_preserves_unowned_checkpoint_directory(self):
        files = self.resume_files(self.root / "resume-unowned")
        checkpoint = files.files.output_directory / ".media-ingest-checkpoint"
        checkpoint.mkdir(parents=True)
        sentinel = checkpoint / "frame_1.png"
        sentinel.write_bytes(b"unrelated photograph")
        with self.assertRaises(MEDIA_ERROR):
            media.MediaIngest.extract_files(self.request(), files)
        self.assertEqual(sentinel.read_bytes(), b"unrelated photograph")
        self.assertFalse((checkpoint / "identity.json").exists())

    def test_resume_vfr_and_sharpness_match_uninterrupted_extraction(self):
        for window in (False, True):
            with self.subTest(window=window):
                request = self.request()
                request.input = self.corpus / "vfr-asymmetric.nut"
                request.end_seconds = -1
                request.selection.mode = media.SelectionMode.FPS
                request.selection.fps = 3
                request.sharpness.enabled = True
                request.sharpness.window = window
                files = self.resume_files(self.root / ("resume-vfr-" + str(window)))
                self.interrupt_resume(request, files)
                actual = media.MediaIngest.extract_files(request, files)
                reference = self.resume_files(self.root / ("reference-vfr-" + str(window)))
                reference.resume = False
                expected = media.MediaIngest.extract_files(request, reference)
                self.assertEqual(actual.frames_accepted, expected.frames_accepted)
                for image in reference.files.output_directory.glob("frame_*.png"):
                    self.assertEqual((files.files.output_directory / image.name).read_bytes(), image.read_bytes())
                actual_metadata = json.loads((files.files.output_directory / "extraction_metadata.json").read_text())
                expected_metadata = json.loads((reference.files.output_directory / "extraction_metadata.json").read_text())
                self.assertEqual(actual_metadata["frames"], expected_metadata["frames"])

    def test_resume_changed_input_and_invalid_checkpoint_preserve_output(self):
        request = self.request()
        original_input = request.input
        request.input = self.root / "resume-input.nut"
        shutil.copyfile(original_input, request.input)
        files = self.resume_files(self.root / "resume-input-change")
        self.interrupt_resume(request, files)
        image = next(files.files.output_directory.glob("frame_*.png"))
        before = image.read_bytes()
        with request.input.open("ab") as source:
            source.write(b"changed identity")
        with self.assertRaises(MEDIA_ERROR):
            media.MediaIngest.extract_files(request, files)
        self.assertEqual(image.read_bytes(), before)
        shutil.copyfile(original_input, request.input)
        record = files.files.output_directory / ".media-ingest-checkpoint" / "frame-0.json"
        record.write_text('{"file":"../../outside.png"}', encoding="utf-8")
        with self.assertRaises(MEDIA_ERROR):
            media.MediaIngest.extract_files(request, files)
        self.assertEqual(image.read_bytes(), before)

    def test_callback_exception_preserves_python_error(self):
        def failed(*args):
            raise ValueError("callback sentinel")
        with self.assertRaisesRegex(ValueError, "callback sentinel"):
            media.MediaIngest.extract(self.request(), progress=failed)
        with self.assertRaisesRegex(ValueError, "callback sentinel"):
            media.MediaIngest.extract(self.request(), cancelled=failed)

    def test_float_owned_extraction_and_independent_image_output(self):
        import struct
        request = self.request()
        request.output_format = media.FramePixelFormat.RGBFloat32
        request.input_color.transfer = media.ColorTransfer.Linear
        request.input_color.primaries = media.ColorPrimaries.Bt709
        report, frames = media.MediaIngest.extract(request)
        self.assertEqual(report.frames_accepted, len(frames))
        self.assertEqual(frames[0].layout.row_stride, 64 * 12)
        self.assertEqual(frames[0].layout.color.transfer, media.ColorTransfer.Linear)
        self.assertEqual(frames[0].info.origin, media.FrameOrigin.Decoded)
        files = media.FileExtraction()
        files.files.output_directory = self.root / "float-files"
        files.files.format = media.FrameFileFormat.EXR
        files.files.exr.precision = media.ExrPrecision.Float
        self.assertEqual(media.MediaIngest.extract_files(request, files).frames_accepted, len(frames))
        manifest = json.loads((files.files.output_directory / "extraction_metadata.json").read_text())
        self.assertEqual(manifest["schema_version"], 2)
        self.assertEqual(manifest["output"]["bit_depth"], 32)
        self.assertFalse(manifest["processing"]["decoder"]["hardware_requested"])
        self.assertEqual(manifest["output"]["input_transfer"], "linear")
        self.assertTrue(manifest["output"]["transfer_override"])
        self.assertEqual(len(manifest["frames"]), len(frames))
        for entry, surface in zip(manifest["frames"], frames):
            self.assertEqual(entry["source_pts"], surface.info.source_timestamp.ticks)
            self.assertEqual(entry["timestamp_origin"], "best_effort")
            self.assertEqual(manifest["input"]["video"]["time_base"],
                             [surface.info.source_timestamp.time_base.numerator,
                              surface.info.source_timestamp.time_base.denominator])
        color = media.FrameColor()
        color.transfer = media.ColorTransfer.Linear
        color.primaries = media.ColorPrimaries.Bt709
        color.alpha = media.AlphaMode.NoAlpha
        info = media.FrameInfo()
        info.origin = media.FrameOrigin.Rendered
        raw = struct.pack("fff", -.25, 2.0, .1234567)
        surface = media.FrameSurface.from_bytes(1, 1, media.FramePixelFormat.RGBFloat32, raw, color=color, info=info)
        self.assertEqual(surface.pixels, raw)
        self.assertIsNone(surface.info.source_timestamp)
        options = media.ExrOutputOptions()
        options.precision = media.ExrPrecision.Float
        options.compression = media.ExrCompression.Uncompressed
        target = self.root / "renderer é 日本語.exr"
        media.ImageOutput.write_exr(target, surface, options)
        previous = target.read_bytes()
        with self.assertRaises(MEDIA_ERROR):
            media.ImageOutput.write_exr(target, surface, options)
        options.overwrite = True
        def failed():
            raise ValueError("EXR cancellation sentinel")
        with self.assertRaisesRegex(ValueError, "EXR cancellation sentinel"):
            media.ImageOutput.write_exr(target, surface, options, cancelled=failed)
        self.assertEqual(target.read_bytes(), previous)
        self.assertFalse(list(self.root.glob("*.tmp")))

    def test_callbacks_run_with_gil_and_other_thread_can_run(self):
        event = threading.Event()
        worker = threading.Thread(target=event.set)
        def progress(value):
            if not worker.is_alive() and not event.is_set():
                worker.start()
            self.assertTrue(event.wait(3), "callback must not block the other Python thread")
        media.MediaIngest.extract(self.request(), progress=progress)
        worker.join(3)
        self.assertTrue(event.is_set())


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--module-dir", type=Path, required=True)
    parser.add_argument("--runtime-dir", type=Path, required=True)
    parser.add_argument("--dependency-dir", type=Path)
    parser.add_argument("--cli", type=Path, help="Production media-ingest executable for metadata parity checks")
    parser.add_argument("--studio", action="store_true", help="Test the complete lichtfeld module")
    args, remaining = parser.parse_known_args()
    METADATA_CLI = args.cli or args.runtime_dir / ("media-ingest.exe" if os.name == "nt" else "media-ingest")
    dll_handles = []
    if os.name == "nt":
        for directory in (args.runtime_dir, args.dependency_dir):
            if directory and directory.is_dir():
                dll_handles.append(os.add_dll_directory(str(directory.resolve())))
    sys.path.insert(0, str(args.module_dir.resolve()))
    if args.studio:
        import lichtfeld
        assert "media" in lichtfeld.__all__, "Media Ingest must be exported by lichtfeld"
        media = lichtfeld.media
        MEDIA_ERROR = lichtfeld.Error
    else:
        import media_test_bindings as media
        MEDIA_ERROR = media.MediaError
    unittest.main(argv=[__file__, *remaining])
