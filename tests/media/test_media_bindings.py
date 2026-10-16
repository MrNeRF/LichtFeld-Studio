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
spec = importlib.util.spec_from_file_location("media_fixtures", ROOT / "scripts/prepare_media_fixtures.py")
fixtures = importlib.util.module_from_spec(spec)
spec.loader.exec_module(fixtures)
media = None
MEDIA_ERROR = None


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
                    source_exif_fixture.write_tiff_graph(path, big=big, little=little, subimages=70)
                    value = json.loads(media.MediaIngest.source_exif_json(path))
                    self.assertEqual(value["status"], "limited")
                    tags = {tag["id"]: tag for tag in value["tiff"]["directories"][0]["tags"]}
                    self.assertEqual(tags[274]["unsigned_values"], [6])
                    source_exif_fixture.write_tiff_graph(path, big=big, little=little, cycle=True)
                    value = json.loads(media.MediaIngest.source_exif_json(path))
                    self.assertEqual(value["status"], "malformed")
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
        metadata_before = metadata_path.read_bytes()
        with self.assertRaises(MEDIA_ERROR):
            media.MediaIngest.extract_files(request, output)
        self.assertEqual(metadata_path.read_bytes(), metadata_before)
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

    def test_completed_exr_resume_requires_manifest_replacement(self):
        request = self.request()
        request.output_format = media.FramePixelFormat.RGBFloat32
        request.input_color.transfer = media.ColorTransfer.Linear
        request.input_color.primaries = media.ColorPrimaries.Bt709
        files = self.resume_files(self.root / "resume-complete-exr", media.FrameFileFormat.EXR)
        media.MediaIngest.extract_files(request, files)
        image = next(files.files.output_directory.glob("frame_*.exr"))
        before = (image.read_bytes(), image.stat().st_mtime_ns)
        manifest = files.files.output_directory / "extraction_metadata.json"
        original_manifest = manifest.read_bytes()
        with self.assertRaises(MEDIA_ERROR):
            media.MediaIngest.extract_files(request, files)
        self.assertEqual(manifest.read_bytes(), original_manifest)
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
    parser.add_argument("--studio", action="store_true", help="Test the complete lichtfeld module")
    args, remaining = parser.parse_known_args()
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
