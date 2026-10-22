# SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
# SPDX-License-Identifier: GPL-3.0-or-later
"""Production CPU CLI contracts; synthetic offline input and legacy references."""
import argparse
import array
import ctypes
import importlib.util
import json
import os
import platform
import re
import signal
import shutil
from pathlib import Path
import subprocess
import sys
import tempfile
import struct
import unittest
import zlib
ROOT=Path(__file__).resolve().parents[2]
_exif_spec = importlib.util.spec_from_file_location("source_exif_fixture", ROOT / "tests/media/source_exif_fixture.py")
source_exif_fixture = importlib.util.module_from_spec(_exif_spec)
_exif_spec.loader.exec_module(source_exif_fixture)
spec=importlib.util.spec_from_file_location("media_fixtures",ROOT/"scripts/prepare_media_fixtures.py")
fixtures=importlib.util.module_from_spec(spec)
spec.loader.exec_module(fixtures)
FFMPEG=os.environ.get("LFS_MEDIA_TEST_FFMPEG","ffmpeg")
FFPROBE=os.environ.get("LFS_MEDIA_TEST_FFPROBE","ffprobe")
CLI=None
RUNNER=None
VERSION_HEADER=None
CONSUMERS=[]
SYMBOLS_TOOL=None
BUILD_DIR=None
BUILD_CONFIG=None

def defined_symbols(path, exports=False):
    if sys.platform == "win32":
        tool=SYMBOLS_TOOL or shutil.which("dumpbin")
        if not tool:
            raise RuntimeError("MSVC symbol inspection tool is required")
        command=[str(tool)]
        if Path(tool).name.lower() in ("link.exe", "link"):
            command.append("/dump")
        command.extend(["/exports", str(path)])
        output=subprocess.check_output(command,text=True,errors="replace")
        return set(re.findall(r"^\s+\d+\s+[0-9A-Fa-f]+\s+(?:[0-9A-Fa-f]+\s+)?([A-Za-z_][A-Za-z0-9_]*)",output,re.M))
    tool=SYMBOLS_TOOL or shutil.which("nm")
    if not tool:
        raise RuntimeError("nm is required for provider ownership contracts")
    flags=["-gU"] if sys.platform == "darwin" else (["-D","--defined-only"] if exports else ["--defined-only"])
    output=subprocess.check_output([str(tool),*flags,str(path)],text=True,errors="replace")
    names=set()
    for line in output.splitlines():
        fields=line.split()
        if len(fields)>=3:
            name=fields[-1].split("@")[0]
            if sys.platform == "darwin" and name.startswith("_"):
                name=name[1:]
            # Local static inline helpers in public FFmpeg headers are not
            # independent archive implementations in a consumer.
            if exports or fields[-2].isupper() or name.startswith("ff_"):
                names.add(name)
    return names

def ffmpeg_public(name):
    return re.match(r"^(av|avcodec|avformat|avutil|avfilter|avdevice|avio|swscale|swresample|sws|swr)_",name)
class IngestCLI(unittest.TestCase):
    @staticmethod
    def _xmp_jpeg(path, blocks):
        source_exif_fixture.write_jpeg(path)
        original = path.read_bytes()
        markers = b"".join(b"\xff\xe1" + struct.pack(">H", len(block) + 2) + block for block in blocks)
        path.write_bytes(original[:2] + markers + original[2:])

    @staticmethod
    def _xmp_png(path, before, after=(), *, bad_crc=False):
        source_exif_fixture.write_png(path)
        original = path.read_bytes()
        def chunk(payload, damage=False):
            crc = zlib.crc32(b"iTXt" + payload) ^ int(damage)
            return struct.pack(">I", len(payload)) + b"iTXt" + payload + struct.pack(">I", crc)
        offset = 8
        while original[offset + 4:offset + 8] != b"IDAT":
            offset += 12 + struct.unpack_from(">I", original, offset)[0]
        path.write_bytes(original[:offset] + b"".join(chunk(b, bad_crc and i == 0) for i, b in enumerate(before)) +
                         original[offset:-12] + b"".join(chunk(b) for b in after) + original[-12:])

    def _xmp_limited(self, path, **limits):
        request = self.root / "xmp-budget-request.json"
        request.write_text(json.dumps({"operation": "source-metadata", "input": str(path), **limits}), encoding="utf-8")
        result = subprocess.run([str(RUNNER), str(request)], capture_output=True, timeout=30)
        self.assertEqual(result.returncode, 0, result.stderr + result.stdout)
        return json.loads(result.stdout)

    def test_source_metadata_xmp_jpeg_originals_and_extended_segments(self):
        path = self.root / "XMP source é 日本語.jpeg"
        standard = b"http://ns.adobe.com/xap/1.0/\0"
        extended = b"http://ns.adobe.com/xmp/extension/\0"
        # Preserve arbitrary XML/encoding and unknown namespaces without parsing.
        packet = b'<?xml version="1.0"?><!DOCTYPE x SYSTEM "file:///not-read"><x xmlns="urn:unknown">\x00\xff</x>'
        segments = [extended + b"A" * 32 + struct.pack(">II", 6, offset) + payload
                    for offset, payload in [(3, b"def"), (0, b"abc")]]
        envelopes = [standard + packet, *segments, standard + packet]
        self._xmp_jpeg(path, envelopes)
        before, stamp = path.read_bytes(), path.stat().st_mtime_ns
        exif, _ = self.invoke("source-exif", path)
        result, stderr = self.invoke("source-metadata", path)
        self.assertEqual(stderr, b"")
        metadata = result["metadata"]
        self.assertEqual(metadata["image_exif"], exif["exif"])
        xmp = metadata["image_xmp"]
        self.assertEqual(xmp["status"], "collected")
        self.assertEqual(xmp["xml_validation"], "not_examined")
        self.assertEqual(xmp["extended_assembly"], "not_performed")
        self.assertEqual([bytes.fromhex(b["bytes"]) for b in xmp["blocks"]], envelopes)
        self.assertEqual([bytes.fromhex(b["payload_bytes"]) for b in xmp["blocks"]], [packet, b"def", b"abc", packet])
        self.assertEqual([b["payload_kind"] for b in xmp["blocks"]], ["packet", "extended_segment", "extended_segment", "packet"])
        self.assertEqual([r["block"] for r in metadata["records"] if r["kind"] == "xmp_block"], list(range(4)))
        self.assertEqual(path.read_bytes(), before)
        self.assertEqual(path.stat().st_mtime_ns, stamp)
        for malformed in [extended, extended + b"?" * 32 + struct.pack(">II", 3, 0) + b"abc",
                          extended + b"A" * 32 + struct.pack(">II", 3, 2) + b"abc"]:
            with self.subTest(malformed=malformed):
                self._xmp_jpeg(path, [standard + b"good", malformed, standard + b"last"])
                result, _ = self.invoke("source-metadata", path)
                xmp = result["metadata"]["image_xmp"]
                self.assertEqual(xmp["status"], "malformed")
                self.assertEqual([b["status"] for b in xmp["blocks"]], ["collected", "malformed", "collected"])
                self.assertEqual(bytes.fromhex(xmp["blocks"][1]["bytes"]), malformed)
                self.assertIsNone(xmp["blocks"][1]["payload_bytes"])

    def test_source_metadata_xmp_png_compression_crc_and_originals(self):
        path = self.root / "XMP source é 日本語.png"
        prefix = b"XML:com.adobe.xmp\0"
        packet = b'<x:xmpmeta xmlns:x="adobe:ns:meta/" xmlns:v="urn:vendor"><v:a>unknown</v:a></x:xmpmeta>'
        plain = prefix + b"\0\0\0\0" + packet
        compressed = prefix + b"\1\0en\0Translated\0" + zlib.compress(packet)
        self._xmp_png(path, [plain], [compressed, plain])
        before, stamp = path.read_bytes(), path.stat().st_mtime_ns
        exif, _ = self.invoke("source-exif", path)
        result, stderr = self.invoke("source-metadata", path)
        self.assertEqual(stderr, b"")
        self.assertEqual(result["metadata"]["image_exif"], exif["exif"])
        blocks = result["metadata"]["image_xmp"]["blocks"]
        self.assertEqual([bytes.fromhex(b["bytes"]) for b in blocks], [plain, compressed, plain])
        self.assertEqual([bytes.fromhex(b["payload_bytes"]) for b in blocks], [packet] * 3)
        self.assertEqual([b["compressed"] for b in blocks], [False, True, False])
        self.assertTrue(all(b["crc_valid"] for b in blocks))
        self.assertEqual(path.read_bytes(), before)
        self.assertEqual(path.stat().st_mtime_ns, stamp)
        self._xmp_png(path, [plain], [plain], bad_crc=True)
        result, _ = self.invoke("source-metadata", path)
        blocks = result["metadata"]["image_xmp"]["blocks"]
        self.assertEqual([b["crc_valid"] for b in blocks], [False, True])
        self.assertEqual([b["status"] for b in blocks], ["malformed", "collected"])
        for malformed in [prefix, prefix + b"\2\0\0\0x", prefix + b"\0\1\0\0x",
                          prefix + b"\0\0unterminated", prefix + b"\1\0\0\0bad zlib",
                          compressed + b"trailing"]:
            with self.subTest(malformed=malformed):
                self._xmp_png(path, [malformed], [plain])
                result, _ = self.invoke("source-metadata", path)
                xmp = result["metadata"]["image_xmp"]
                self.assertEqual(xmp["status"], "malformed")
                self.assertEqual(xmp["blocks"][0]["status"], "malformed")
                self.assertIsNone(xmp["blocks"][0]["payload_bytes"])
                self.assertEqual(bytes.fromhex(xmp["blocks"][1]["payload_bytes"]), packet)
        self._xmp_png(path, [plain])
        path.write_bytes(path.read_bytes()[:-10])
        result, _ = self.invoke("source-metadata", path)
        self.assertEqual(result["metadata"]["image_xmp"]["status"], "malformed")

    def test_source_metadata_xmp_budgets_and_absence(self):
        path = self.root / "xmp-budget.jpeg"
        envelope = b"http://ns.adobe.com/xap/1.0/\0" + b"x" * 31
        self._xmp_jpeg(path, [envelope, envelope])
        cases = [({"blocks": 1}, 1), ({"block_bytes": len(envelope) - 1}, 0),
                 ({"total_bytes": 2 * len(envelope) - 1}, 1), ({"blocks": 0}, 0)]
        for limits, count in cases:
            with self.subTest(limits=limits):
                result = self._xmp_limited(path, **limits)
                self.assertEqual(result["image_xmp"]["status"], "limited")
                self.assertEqual(len(result["image_xmp"]["blocks"]), count)
                self.assertEqual(len(result["image_exif"]["blocks"]), 2)
        exact = self._xmp_limited(path, block_bytes=len(envelope), total_bytes=2 * len(envelope), blocks=2)
        self.assertEqual(exact["image_xmp"]["status"], "collected")
        no_index = self._xmp_limited(path, records=0)
        self.assertEqual(no_index["index_status"], "limited")
        self.assertEqual(len(no_index["image_xmp"]["blocks"]), 2)
        self.assertEqual(no_index["records"], [])
        large_exif, small_exif = b"Exif\0\0" + b"x" * 32, b"Exif\0\0II"
        self._xmp_jpeg(path, [large_exif, small_exif, envelope])
        low_exif = self._xmp_limited(path, exif_block_bytes=16)
        self.assertEqual(low_exif["image_exif"], low_exif["legacy_exif"])
        self.assertEqual(low_exif["image_exif"]["blocks"], [])
        self.assertEqual(len(low_exif["image_xmp"]["blocks"]), 1)
        source_exif_fixture.write_camera_metadata(path, carrier="jpeg")
        original = path.read_bytes()
        path.write_bytes(original[:2] + b"\xff\xe1" + struct.pack(">H", len(envelope) + 2) + envelope + original[2:])
        full = self._xmp_limited(path)
        exif_records = sum(record["kind"] != "xmp_block" for record in full["records"])
        only_xmp_omitted = self._xmp_limited(path, records=exif_records)
        self.assertEqual(only_xmp_omitted["index_status"], "limited")
        self.assertEqual(only_xmp_omitted["test_summary"]["facts"], full["test_summary"]["facts"],
                         "Omitting only XMP references must not invalidate complete EXIF compound facts")
        exif_omitted = self._xmp_limited(path, records=exif_records - 1)
        self.assertTrue(any(fact["diagnostic"] == "Directory metadata is incomplete" for fact in exif_omitted["test_summary"]["facts"]))
        path = self.root / "xmp-bomb.png"
        packet = b"x" * 65536
        compressed = b"XML:com.adobe.xmp\0\1\0\0\0" + zlib.compress(packet)
        self._xmp_png(path, [compressed])
        for limits in [{"inflated_bytes": len(packet) - 1}, {"total_bytes": len(compressed) + len(packet) - 1}]:
            with self.subTest(limits=limits):
                limited = self._xmp_limited(path, **limits)["image_xmp"]
                self.assertEqual(limited["status"], "limited")
                self.assertEqual(limited["blocks"][0]["status"], "limited")
                self.assertEqual(bytes.fromhex(limited["blocks"][0]["bytes"]), compressed)
                self.assertIsNone(limited["blocks"][0]["payload_bytes"])
        exact = self._xmp_limited(path, inflated_bytes=len(packet), total_bytes=len(compressed) + len(packet))["image_xmp"]
        self.assertEqual(bytes.fromhex(exact["blocks"][0]["payload_bytes"]), packet)
        source_exif_fixture.write_png(path, include_exif=False)
        result, _ = self.invoke("source-metadata", path)
        self.assertEqual(result["metadata"]["image_xmp"]["status"], "absent")
        self._xmp_png(path, [b"Other\0\0\0\0\0" + b"x" * 20000], [b"XML:com.adobe.xmp\0\0\0\0\0abc"])
        result = self._xmp_limited(path, block_bytes=128)["image_xmp"]
        self.assertEqual(result["status"], "collected")
        self.assertEqual(len(result["blocks"]), 1)
        source_exif_fixture.write_tiff(path)
        result, _ = self.invoke("source-metadata", path)
        self.assertEqual(result["metadata"]["image_xmp"]["status"], "unsupported")
        result, _ = self.invoke("source-metadata", self.root / "missing-xmp.png")
        self.assertEqual(result["metadata"]["image_xmp"]["status"], "io_error")

    def test_source_container_metadata_reuses_probe_without_writing(self):
        path = self.source()
        before = path.read_bytes()
        before_stat = path.stat()
        probe, probe_stderr = self.invoke("probe", path)
        result, stderr = self.invoke("source-container-metadata", path)
        self.assertEqual(probe_stderr, b"")
        self.assertEqual(stderr, b"")
        self.assertTrue(result["success"])
        catalog = result["metadata"]
        self.assertEqual(catalog["scope"], "container_stream")
        self.assertEqual(catalog["index_status"], "complete")
        self.assertEqual(catalog["description"], probe["media"])
        self.assertNotIn("image_exif", catalog)
        retained = catalog["source_metadata"]
        self.assertFalse(retained["limited"])
        self.assertEqual(catalog["source_locator"], str(path))
        encoding = "utf-16-le" if os.name == "nt" else sys.getfilesystemencoding()
        self.assertEqual(catalog["source_locator_encoding"], "utf16le" if os.name == "nt" else "posix_bytes")
        self.assertEqual(bytes.fromhex(catalog["source_locator_bytes"]), str(path).encode(encoding, "surrogateescape"))
        version, _ = self.invoke("version")
        self.assertEqual(retained["provider"]["name"], "ffmpeg")
        self.assertEqual(retained["provider"]["version"], version["ffmpeg"])
        self.assertGreater(retained["provider"]["codec_api_version"], 0)

        self.assertEqual(catalog["identity_scope"], "in_memory_observation")
        self.assertFalse(result["summary"]["limited"])
        facts = result["summary"]["facts"]
        self.assertTrue(any(fact["semantic"] == "image.width" for fact in facts))
        for fact in facts:
            self.assertEqual(fact["temporal_scope"], "static")
            self.assertEqual(fact["origin"], "demuxer_observation")
            self.assertEqual(fact["source"]["kind"], "stream_description")
            stream = probe["media"]["streams"][fact["source"]["stream"]]
            if fact["semantic"] == "image.width":
                self.assertEqual(fact["value"], stream["width"])
                self.assertEqual(fact["unit"], "pixels")
            if fact["semantic"] == "frame_rate.nominal":
                self.assertEqual(fact["value"], {"numerator": stream["nominal_frame_rate"][0], "denominator": stream["nominal_frame_rate"][1]})
                self.assertEqual(fact["unit"], "frames_per_second")
        reference = json.loads(subprocess.check_output(
            [FFPROBE, "-v", "error", "-show_streams", "-show_format", "-of", "json", str(path)]))
        reference_streams = {stream["index"]: stream for stream in reference["streams"]}
        for stream in retained["streams"]:
            expected_tags = reference_streams[stream["stream_index"]].get("tags", {})
            self.assertEqual({tag["key"]: tag["value"] for tag in stream["tags"]}, expected_tags)
        self.assertEqual({tag["key"]: tag["value"] for tag in retained["tags"]}, reference["format"].get("tags", {}))
        indices = {stream["index"] for stream in probe["media"]["streams"]}
        for stream in retained["streams"]:
            self.assertIn(stream["stream_index"], indices)
            for tag in stream["tags"]:
                self.assertEqual(bytes.fromhex(tag["key_bytes"]).decode("utf-8", "replace"), tag["key"])
                self.assertEqual(bytes.fromhex(tag["value_bytes"]).decode("utf-8", "replace"), tag["value"])
        for reference in catalog["records"]:
            kind = reference["kind"]
            if kind == "container_tag":
                self.assertLess(reference["entry"], len(retained["tags"]))
            elif kind == "stream_description":
                self.assertLess(reference["stream"], len(probe["media"]["streams"]))
            elif kind == "stream_tag":
                self.assertLess(reference["entry"], len(retained["streams"][reference["stream"]]["tags"]))
            elif kind == "stream_side_data":
                self.assertLess(reference["block"], len(retained["streams"][reference["stream"]]["coded_side_data"]))
            else:
                self.fail(f"Unexpected container metadata reference: {kind}")
        self.assertEqual(path.read_bytes(), before)
        self.assertEqual(path.stat().st_mtime_ns, before_stat.st_mtime_ns)

    @unittest.skipUnless(sys.platform.startswith("linux"), "Linux filesystem fixture with non-UTF8 filename bytes")
    def test_source_container_metadata_preserves_non_utf8_filename_bytes(self):
        native = os.fsencode(self.root) + b"/metadata-native-\xff.mp4"
        shutil.copyfile(self.source(), native)
        path = os.fsdecode(native)
        result, stderr = self.invoke("source-container-metadata", path)
        self.assertEqual(stderr, b"")
        catalog = result["metadata"]
        self.assertEqual(catalog["source_locator"], native.decode("utf-8", "replace"))
        self.assertEqual(catalog["source_locator_encoding"], "posix_bytes")
        self.assertEqual(bytes.fromhex(catalog["source_locator_bytes"]), native)
        self.assertEqual(catalog["description"]["streams"][0]["width"], 64)
        probe, _ = self.invoke("probe", path)
        self.assertEqual(probe["media"]["streams"][0]["width"], 64)
        expected = self.root / "native-reference-frames"
        actual = self.root / "native-byte-frames"
        baseline, _ = self.invoke("extract", self.source(), "--output", expected,
                                  "--interval", "1", "--end", "0.35")
        result, _ = self.invoke("extract", path, "--output", actual,
                                "--interval", "1", "--end", "0.35")
        self.assertGreater(result["frames_accepted"], 0)
        self.assertEqual(result["frames_accepted"], baseline["frames_accepted"])
        expected_frames = {p.name: p.read_bytes() for p in expected.glob("*.png")}
        self.assertTrue(expected_frames)
        self.assertEqual({p.name: p.read_bytes() for p in actual.glob("*.png")}, expected_frames)

        request = self.root / "native-encode-request.json"
        request.write_text(json.dumps({"operation": "encode-session", "output": str(self.root / "unused.mp4"),
                                       "non_utf8_filename": True}), encoding="utf-8")
        encoded = subprocess.run([str(RUNNER), str(request)], capture_output=True, timeout=60)
        self.assertEqual(encoded.returncode, 0, encoded.stderr)
        self.assertTrue(json.loads(encoded.stdout)["success"])
        encoded_path = os.fsdecode(os.fsencode(self.root) + b"/encoded-\xff.mp4")
        encoded_probe, _ = self.invoke("probe", encoded_path)
        self.assertEqual(encoded_probe["media"]["streams"][0]["width"], 64)
        decoded, _ = self.invoke("extract", encoded_path, "--output", self.root / "native-encoded-frames",
                                 "--interval", "1")
        self.assertEqual(decoded["frames_accepted"], json.loads(encoded.stdout)["frames"])
        decoded_frames = list((self.root / "native-encoded-frames").glob("*.png"))
        self.assertEqual(len(decoded_frames), decoded["frames_accepted"])
        self.assertTrue(all(frame.stat().st_size > 0 for frame in decoded_frames))

    def test_source_container_metadata_failure_preserves_the_probe_error(self):
        path = self.root / "missing metadata input.bin"
        probe, _ = self.invoke("probe", path, code=3)
        catalog, _ = self.invoke("source-container-metadata", path, code=3)
        self.assertFalse(catalog["success"])
        self.assertEqual(catalog["error"]["code"], probe["error"]["code"])
        self.assertFalse(path.exists())

    def test_camera_metadata_summary_real_carriers(self):
        path = self.root / "Camera metadata é 日本語.bin"
        carriers = [(carrier, little, False) for carrier in ("jpeg", "png", "tiff") for little in (True, False)]
        carriers += [("tiff", little, True) for little in (True, False)]
        for carrier, little, big, padded in [(*case, padding) for case in carriers for padding in (False, True)]:
            with self.subTest(carrier=carrier, little=little, big=big, padded=padded):
                source_exif_fixture.write_camera_metadata(path, carrier=carrier, little=little, big=big, padded=padded)
                before, modified = path.read_bytes(), path.stat().st_mtime_ns
                result, stderr = self.invoke("source-metadata", path)
                self.assertEqual(stderr, b"")
                self.assertTrue(result["success"])
                self.assertEqual(result["metadata"]["image_exif"]["status"], "collected", result)
                summary = result["summary"]
                self.assertEqual(summary["scope"], "static_image_properties")
                self.assertFalse(summary["limited"])
                facts = {item["semantic"]: item for item in summary["facts"]}
                self.assertEqual(len(facts), len(summary["facts"]), "unexpected duplicate semantic in a single-image fixture")
                for semantic in ("camera.make", "camera.model", "lens.model", "lens.focal_length", "gps.latitude", "gps.longitude", "gps.altitude_sea_level", "time.original"):
                    self.assertIn(semantic, facts, result)
                    self.assertEqual(facts[semantic]["state"], "present", facts[semantic])
                    self.assertFalse(facts[semantic]["ambiguous"])
                self.assertEqual(facts["camera.make"]["value"], "Test Camera")
                self.assertEqual(facts["lens.focal_length"]["unit"], "millimetres")
                self.assertEqual(facts["gps.latitude"]["value"], -45.5)
                self.assertEqual(facts["gps.longitude"]["value"], -73.25)
                self.assertEqual(facts["gps.altitude_sea_level"]["value"], -12.5)
                self.assertEqual(facts["gps.utc_seconds_of_day"]["value"], 36240.5)
                self.assertEqual(facts["gps.utc_date"]["value"], "2024-02-29")
                self.assertEqual(facts["time.original"]["value"], "2024-02-29T12:34:56.000123+02:30")
                self.assertEqual(len(facts["time.original"]["inputs"]), 2)
                references = result["metadata"]["records"]
                for item in summary["facts"]:
                    self.assertIn(item["source"], references)
                    for operand in item["inputs"]:
                        self.assertIn(operand, references)
                self.assertEqual(path.read_bytes(), before)
                self.assertEqual(path.stat().st_mtime_ns, modified)

    def test_camera_metadata_review_real_carriers(self):
        path = self.root / "Camera review é 日本語.bin"
        carriers = [(carrier, little, False) for carrier in ("jpeg", "png", "tiff") for little in (True, False)]
        carriers += [("tiff", little, True) for little in (True, False)]
        for carrier, little, big in carriers:
            with self.subTest(carrier=carrier, little=little, big=big):
                source_exif_fixture.write_camera_metadata(path, carrier=carrier, little=little, big=big)
                baseline, _ = self.invoke("source-metadata", path)
                source_exif_fixture.write_camera_metadata(path, carrier=carrier, little=little, big=big, thumbnail=True, unknown_optics=True)
                original, modified = path.read_bytes(), path.stat().st_mtime_ns
                result, _ = self.invoke("source-metadata", path)
                raw, _ = self.invoke("source-exif", path)
                self.assertEqual(result["metadata"]["image_exif"], raw["exif"])
                self.assertEqual(raw["exif"]["status"], "collected", result)
                self.assertIn(b"Thumbnail Camera", original)
                facts = result["summary"]["facts"]
                by_semantic = {item["semantic"]: item for item in facts}
                self.assertEqual(len(by_semantic), len(facts))
                self.assertEqual(set(by_semantic), {item["semantic"] for item in baseline["summary"]["facts"]})
                for semantic in ("camera.make", "camera.model", "orientation.exif", "time.modified.civil"):
                    self.assertEqual(by_semantic[semantic]["value"], next(item["value"] for item in baseline["summary"]["facts"] if item["semantic"] == semantic))
                    self.assertFalse(by_semantic[semantic]["ambiguous"])
                for semantic in ("lens.f_number", "lens.focal_length"):
                    self.assertEqual(by_semantic[semantic]["state"], "absent")
                    self.assertEqual(by_semantic[semantic]["diagnostic"], "Source declares unknown value")
                for item in facts:
                    if carrier == "tiff":
                        directory = raw["exif"]["tiff"]["directories"][item["source"]["directory"]]
                        primary = raw["exif"]["tiff"]["directories"][0]
                        self.assertTrue(directory == primary or
                                        (directory["parent"] == primary["offset"] and directory["kind"] in ("exif", "gps")))
                    else:
                        entry = raw["exif"]["blocks"][item["source"]["block"]]["interpretation"]["entries"][item["source"]["entry"]]
                        self.assertEqual(entry["directory_path"][0], 0)
                self.assertEqual(path.read_bytes(), original)
                self.assertEqual(path.stat().st_mtime_ns, modified)
        # Exercise malformed ASCII through EXIF parsing, without suppressing
        # libtiff provider diagnostics for nonconforming TIFF fields.
        for carrier in ("jpeg", "png"):
            for little in (True, False):
                with self.subTest(carrier=carrier, little=little, unterminated=True):
                    source_exif_fixture.write_camera_metadata(path, carrier=carrier, little=little,
                                                              unterminated_refs=True, invalid_offset=True)
                    result, _ = self.invoke("source-metadata", path)
                    self.assertEqual(result["metadata"]["image_exif"]["status"], "collected")
                    facts = {item["semantic"]: item for item in result["summary"]["facts"]}
                    self.assertEqual(facts["gps.latitude"]["value"], -45.5)
                    self.assertEqual(facts["gps.longitude"]["value"], -73.25)
                    self.assertEqual(facts["time.original.offset"]["state"], "invalid")
                    self.assertEqual(facts["time.original"]["state"], "present")
                    self.assertEqual(facts["time.original"]["value"], "2024-02-29T12:34:56.000123")
                    self.assertEqual(facts["time.original"]["diagnostic"], "Timezone unspecified")

    def test_camera_metadata_summary_retains_tiff_provider_warnings(self):
        path = self.root / "TIFF padded warning é 日本語.bin"
        for big in (False, True):
            for little in (False, True):
                with self.subTest(big=big, little=little):
                    # libtiff diagnoses NULs inside the declared ASCII count.
                    # The projection must not turn that warning into success.
                    source_exif_fixture.write_camera_metadata(path, carrier="tiff", big=big, little=little, padded=True, nul_padding=True)
                    original, modified = path.read_bytes(), path.stat().st_mtime_ns
                    result, stderr = self.invoke("source-metadata", path)
                    self.assertEqual(stderr, b"")
                    self.assertEqual(result["metadata"]["image_exif"]["status"], "malformed")
                    self.assertTrue(result["metadata"]["image_exif"]["diagnostic"])
                    facts = {item["semantic"]: item for item in result["summary"]["facts"]}
                    for key in ("gps.latitude", "gps.longitude", "gps.altitude_sea_level", "time.original"):
                        self.assertEqual(facts[key]["state"], "unsupported")
                        self.assertEqual(facts[key]["diagnostic"], "Directory metadata is incomplete")
                    self.assertEqual(path.read_bytes(), original)
                    self.assertEqual(path.stat().st_mtime_ns, modified)

    def test_source_metadata_catalog_preserves_existing_descriptions(self):
        path = self.root / "Metadata catalog é 日本語.bin"
        writers = [lambda: source_exif_fixture.write_png(path, duplicate=True)]
        writers += [lambda big=big, little=little: source_exif_fixture.write_tiff(
            path, big=big, little=little, pages=2)
            for big in (False, True) for little in (False, True)]
        for ordinal, write in enumerate(writers):
            with self.subTest(fixture=ordinal):
                write()
                before = path.read_bytes()
                before_stat = path.stat()
                legacy, legacy_stderr = self.invoke("source-exif", path)
                result, stderr = self.invoke("source-metadata", path)
                self.assertEqual(legacy_stderr, b"")
                self.assertEqual(stderr, b"")
                self.assertTrue(result["success"])
                catalog = result["metadata"]
                self.assertEqual(catalog["scope"], "static_image_exif")
                self.assertEqual(catalog["index_status"], "complete")
                self.assertEqual(catalog["image_exif"], legacy["exif"])
                expected = []
                for block_index, block in enumerate(legacy["exif"]["blocks"]):
                    for entry_index, _ in enumerate(block["interpretation"]["entries"]):
                        expected.append({"kind": "exif_entry", "block": block_index, "entry": entry_index})
                tiff = legacy["exif"].get("tiff")
                if tiff:
                    for directory_index, directory in enumerate(tiff["directories"]):
                        for tag_index, _ in enumerate(directory["tags"]):
                            expected.append({"kind": "tiff_tag", "directory": directory_index, "tag": tag_index})
                self.assertEqual(catalog["records"], expected)
                self.assertEqual(path.read_bytes(), before)
                self.assertEqual(path.stat().st_mtime_ns, before_stat.st_mtime_ns)

    def test_tiff_decoded_tags_and_original_source_ranges(self):
        path = self.root / "TIFF metadata é 日本語.bin"
        for big in (False, True):
            for little in (False, True):
                with self.subTest(big=big, little=little):
                    payload = source_exif_fixture.write_tiff(path, big=big, little=little, pages=2)
                    result, stderr = self.invoke("source-exif", path)
                    self.assertEqual(stderr, b"")
                    value = result["exif"]
                    source_exif_fixture.check_tiff(self, value, path, payload, big=big, little=little, pages=2)
                    maker = source_exif_fixture.write_tiff_graph(path, big=big, little=little)
                    result, stderr = self.invoke("source-exif", path)
                    self.assertEqual(stderr, b"")
                    value = result["exif"]
                    source_exif_fixture.check_tiff_graph(self, value, path, maker)

    def test_png_idat_segmentation_and_ancillary_crc_recovery(self):
        path = self.root / "review PNG.png"
        for options, count, status in (({"idat_chunks": 5000}, 1, "collected"),
                                        ({"idat_chunks": 5000, "include_exif": False}, 0, "absent"),
                                        ({"damaged_ancillary": True}, 1, "malformed")):
            with self.subTest(options=options):
                payload = source_exif_fixture.write_png(path, **options)
                result, stderr = self.invoke("source-exif", path)
                self.assertEqual(stderr, b"")
                source_exif_fixture.check_png(self, result["exif"], payload, count=count, status=status)

    def test_png_source_exif_original_and_nonconforming_prefixes(self):
        path = self.root / "PNG EXIF é 日本語.png"
        for options in ({}, {"post_idat": True}, {"duplicate": True}):
            with self.subTest(options=options):
                payload = source_exif_fixture.write_png(path, **options)
                before = path.read_bytes()
                result, stderr = self.invoke("source-exif", path)
                self.assertEqual(stderr, b"")
                source_exif_fixture.check_png(self, result["exif"], payload,
                                              count=2 if options.get("duplicate") else 1,
                                              status="malformed" if options else "collected")
                self.assertEqual(path.read_bytes(), before)

    def test_source_exif_originals_unicode_and_status(self):
        path = self.root / "original EXIF é 日本語.jpg"
        payload = source_exif_fixture.write_jpeg(path)
        before = path.read_bytes()
        result, stderr = self.invoke("source-exif", path)
        self.assertTrue(result["success"])
        self.assertEqual(stderr, b"")
        source_exif_fixture.check(self, result["exif"], payload)
        self.assertEqual(path.read_bytes(), before)
        result, _ = self.invoke("source-exif", self.root / "missing.jpg")
        self.assertEqual(result["exif"]["status"], "io_error")
        self.invoke("source-exif", path, "--extra", code=2)

    def test_source_exif_parse_budget_is_shared_across_blocks(self):
        path = self.root / "shared-budget.jpg"
        source_exif_fixture.write_jpeg(path)
        # Preserve a valid JPEG header, adding two independent large EXIF blocks.
        count = 2500
        tiff = b"II" + struct.pack("<HIH", 42, 8, count)
        tiff += struct.pack("<HHII", 0xc7fe, 4, 1, 123) * count + b"\0" * 4
        payload = b"Exif\0\0" + tiff
        marker = b"\xff\xe1" + struct.pack(">H", len(payload) + 2) + payload
        jpeg = path.read_bytes()
        path.write_bytes(jpeg[:2] + marker + marker + jpeg[2:])
        result, _ = self.invoke("source-exif", path)
        blocks = result["exif"]["blocks"]
        self.assertEqual(result["exif"]["status"], "collected")
        self.assertEqual(blocks[0]["interpretation"]["status"], "parsed")
        self.assertEqual(len(blocks[0]["interpretation"]["entries"]), count)
        self.assertEqual(blocks[1]["interpretation"]["status"], "limited")
        self.assertEqual(bytes.fromhex(blocks[1]["bytes"]), payload)

    def test_source_exif_retains_prefix_on_large_header_warning_and_exhaustion(self):
        path = self.root / "large-header.jpg"
        for long_header in (False, True):
            with self.subTest(long_header=long_header):
                payload = source_exif_fixture.write_header_failure(path, long_header)
                result, _ = self.invoke("source-exif", path)
                value = result["exif"]
                self.assertEqual(value["status"], "limited" if long_header else "malformed")
                self.assertEqual(len(value["blocks"]), 2)
                for block in value["blocks"]:
                    self.assertEqual(bytes.fromhex(block["bytes"]), payload)
                    self.assertEqual(block["interpretation"]["status"], "parsed")

    def test_source_exif_invalid_utf8_diagnostic_is_valid_json(self):
        path = self.root / "invalid-diagnostic.avif"
        source_exif_fixture.write_invalid_heif(path)
        result, _ = self.invoke("source-exif", path)
        self.assertTrue(result["success"])
        self.assertEqual(result["exif"]["status"], "malformed")
        self.assertIn("\ufffd", result["exif"]["diagnostic"])

    @classmethod
    def setUpClass(cls):
        cls.temp=tempfile.TemporaryDirectory(prefix="lfs-cli-contratti-")
        cls.addClassCleanup(cls.temp.cleanup)
        cls.root=Path(cls.temp.name)
        cls.corpus=cls.root/"media é 日本語"
        fixtures.prepare(cls.corpus,FFMPEG,FFPROBE)
    def invoke(self,*args,code=0):
        result=subprocess.run([str(CLI),*map(str,args)],capture_output=True,timeout=30)
        self.assertEqual(result.returncode,code,result.stderr.decode("utf-8",errors="replace")+result.stdout.decode("utf-8",errors="replace"))
        return json.loads(result.stdout),result.stderr
    def source(self,name="cfr-asymmetric"):
        return self.corpus/(name+".nut")
    def test_capabilities_and_help(self):
        result,_=self.invoke("capabilities")
        self.assertTrue(result["software_decode"])
        self.assertEqual(result["hardware_decode"],sys.platform == "darwin")
        self.assertFalse(result["hdr_to_sdr"])
        version,_=self.invoke("version")
        self.assertTrue(version["ffmpeg_license"])
        self.assertTrue(version["version"])
        expected=re.search(r'^#define GIT_TAGGED_VERSION "(.*)"',VERSION_HEADER.read_text(encoding="utf-8"),re.MULTILINE)
        self.assertIsNotNone(expected)
        self.assertEqual(version["version"],expected.group(1))
        self.assertIn(b"--hdr-to-sdr",subprocess.check_output([str(CLI),"--help"]))
    @unittest.skipUnless(sys.platform.startswith("linux") and platform.machine().lower() in ("x86_64","amd64"),
                         "ELF x86 FFmpeg assembly symbol isolation")
    def test_elf_ffmpeg_assembly_data_stays_private(self):
        library=ctypes.CDLL(str(CLI.parent/"liblfs_media.so"))
        # These direct-reference assembly constants must not be interposable.
        # Public MediaIngest APIs remain exercised by the CLI contracts below.
        for symbol in ("ff_pw_9","ff_pw_512"):
            with self.subTest(symbol=symbol):
                with self.assertRaises(ValueError):
                    ctypes.c_uint64.in_dll(library,symbol)
    def test_ffmpeg_has_one_explicit_provider(self):
        # Release binaries may be stripped. Check the generated final link
        # commands as well as binary symbols; ownership cannot depend on nm
        # retaining a consumer's ordinary symbol table.
        reply=BUILD_DIR/".cmake/api/v1/reply"
        indexes=list(reply.glob("index-*.json"))
        self.assertTrue(indexes,"Root CMake codemodel is required")
        index=json.loads(max(indexes,key=lambda p:p.stat().st_mtime_ns).read_text(encoding="utf-8"))
        model_ref=next(item for item in index["objects"] if item["kind"]=="codemodel")
        model=json.loads((reply/model_ref["jsonFile"]).read_text(encoding="utf-8"))
        configuration=next(item for item in model["configurations"] if item["name"]==BUILD_CONFIG)
        providers=set()
        allowed_providers={"lfs_media","media_extraction_runner","hdr_tonemap_contracts","lichtfeld_tests","lichtfeld_format_tests"}
        archives=re.compile(r"(?:lib)?(?:avcodec|avformat|avutil|avfilter|avdevice|swscale|swresample)d?\.(?:a|lib)\b|(?<![\w-])-l(?:avcodec|avformat|avutil|avfilter|avdevice|swscale|swresample)\b",re.I)
        for ref in configuration["targets"]:
            target=json.loads((reply/ref["jsonFile"]).read_text(encoding="utf-8"))
            fragments=target.get("link",{}).get("commandFragments",[])
            ffmpeg=[item["fragment"] for item in fragments if archives.search(item["fragment"])]
            if ffmpeg:
                providers.add(target["name"])
                if target["name"]=="lichtfeld_native_third_party_tests":
                    # This Windows-only test host exercises the native FFmpeg
                    # file boundary directly, outside the production provider.
                    self.assertEqual(sys.platform,"win32")
                    self.assertEqual(target["type"],"EXECUTABLE")
                    self.assertIn("tests/test_native_third_party_paths.cpp",
                                  {source["path"] for source in target.get("sources",[])})
                    allowed_providers.add(target["name"])
                if target["name"]=="media_apple_hevc_fallback_runner":
                    # The Apple codec-open fault seam compiles the session into
                    # this test host; production still has one media provider.
                    self.assertEqual(sys.platform,"darwin")
                    self.assertEqual(target["type"],"EXECUTABLE")
                    self.assertIn("tests/media/extractor/apple_hevc_fallback_runner.cpp",
                                  {source["path"] for source in target.get("sources",[])})
                    allowed_providers.add(target["name"])
                if target["name"]=="media_av1_provider_runner":
                    # This test host compiles the codec-open fault seam directly;
                    # application consumers must still use the production provider.
                    self.assertEqual(target["type"],"EXECUTABLE")
                    self.assertTrue({"tests/media/extractor/av1_provider_contracts.cpp",
                                     "tests/media/extractor/encode_session_contracts.cpp"}.issubset(
                                         {source["path"] for source in target.get("sources",[])}))
                    allowed_providers.add(target["name"])
                self.assertIn(target["name"],allowed_providers,f"FFmpeg linked independently by {target['name']}: {ffmpeg}")
        self.assertIn("lfs_media",providers)
        filename="lfs_media.dll" if sys.platform=="win32" else ("liblfs_media.dylib" if sys.platform=="darwin" else "liblfs_media.so")
        provider=CLI.parent/filename
        exports=defined_symbols(provider,exports=True)
        self.assertFalse({name for name in exports if ffmpeg_public(name)},
                         "FFmpeg public API must stay private to media")
        self.assertFalse({name for name in exports if name.startswith(("ff_","avpriv_"))},
                         "Internal FFmpeg symbols must not escape the provider")
        self.assertTrue(CONSUMERS,"Root application consumers must be supplied")
        for consumer in [CLI,*CONSUMERS]:
            with self.subTest(consumer=consumer.name):
                self.assertTrue(consumer.is_file(),f"Build the root media_contracts target: missing {consumer}")
                symbols=defined_symbols(consumer)
                copies={name for name in symbols if ffmpeg_public(name) or name.startswith(("ff_","avpriv_"))}
                self.assertFalse(copies,f"FFmpeg implementation leaked into {consumer}: {sorted(copies)[:10]}")
                if sys.platform=="win32":
                    tool=SYMBOLS_TOOL or shutil.which("dumpbin")
                    command=[str(tool)]
                    if Path(tool).name.lower() in ("link.exe","link"):
                        command.append("/dump")
                    imports=subprocess.check_output([*command,"/imports",str(consumer)],text=True,errors="replace")
                    self.assertFalse(re.findall(r"\b(?:avcodec|avformat|avutil|avfilter|avdevice|swscale|swresample)-\d+\.dll\b",imports,re.I),
                                     "Product consumers must not import FFmpeg")
    def test_probe_rational_inventory_and_unicode_no_output(self):
        before={p.name:p.read_bytes() for p in self.corpus.iterdir()}
        result,_=self.invoke("probe",self.source())
        media=result["media"]
        ref=json.loads(subprocess.check_output([FFPROBE,"-v","error","-show_streams","-of","json",str(self.source())]))["streams"][0]
        self.assertEqual(media["streams"][0]["time_base"],[int(x) for x in ref["time_base"].split("/")])
        self.assertEqual(media["streams"][0]["width"],64)
        headers,_=self.invoke("probe",self.source(),"--headers-only")
        self.assertFalse(headers["media"]["stream_info_probed"])
        maximum_timeout,_=self.invoke("probe",self.source(),"--timeout-ms",str((1<<63)-1))
        self.assertEqual(maximum_timeout["media"],media)
        self.assertEqual(before,{p.name:p.read_bytes() for p in self.corpus.iterdir()})
    def test_extract_matches_legacy_pixels_names_metadata(self):
        cases=[("cfr-asymmetric",[],{}),("vfr-asymmetric",[],{}),
               ("cfr-asymmetric",["--rotate","90"],{"rotation":90}),
               ("cfr-asymmetric",["--scale","0.5"],{"scale":0.5}),
               ("cfr-asymmetric",["--size","40","24"],{"width":40,"height":24}),
               ("cfr-asymmetric",["--start","0.1"],{"start":0.1}),
               ("cfr-asymmetric",["--format","jpeg"],{"format":"jpg"}),
               ("cfr-asymmetric",["--window"],{"sharpness":True,"window":True}),
               ("cfr-asymmetric",["--window","--algorithm","laplacian"],{"sharpness":True,"window":True,"algorithm":"laplacian"}),
               ("cfr-asymmetric",["--window","--algorithm","tenengrad"],{"sharpness":True,"window":True,"algorithm":"tenengrad"})]
        for name,options,legacy_options in cases:
            with self.subTest(options=options):
                work=Path(tempfile.mkdtemp(dir=self.root));output=work/"CLI é 日本語";legacy=work/"legacy"
                actual,stderr=self.invoke("extract",self.source(name),"--output",output,"--interval","1","--end","0.35","--name","frame_%03d","--metadata",*options)
                request={"input":str(self.source(name)),"output":str(legacy),"interval":1,"end":0.35,**legacy_options}
                path=work/"request.json";path.write_text(json.dumps(request,ensure_ascii=False),encoding="utf-8")
                ref=subprocess.run([str(RUNNER),str(path)],capture_output=True,timeout=30)
                self.assertEqual(ref.returncode,0,ref.stderr)
                self.assertTrue(json.loads(ref.stdout)["success"])
                self.assertEqual({p.name:p.read_bytes() for p in output.iterdir() if p.suffix in (".png",".jpg")},
                                 {p.name:p.read_bytes() for p in legacy.iterdir() if p.suffix in (".png",".jpg")})
                metadata=json.loads((output/"extraction_metadata.json").read_text(encoding="utf-8"))
                reference=json.loads((legacy/"extraction_metadata.json").read_text(encoding="utf-8"))
                self.assertEqual(metadata["frames"],reference["frames"])
                self.assertEqual(actual["frames_accepted"],len(metadata["frames"]))
                progress=[json.loads(line) for line in stderr.splitlines() if line.startswith(b'{')]
                self.assertTrue(progress)
    def test_jxl_uint16_transfer_and_option_validation(self):
        for transfer in ("linear", "srgb"):
            output = self.root / ("jxl-u16-" + transfer)
            args = ("extract", self.source(), "--output", output, "--interval", "1", "--end", "0.35",
                    "--format", "jxl", "--jxl-precision", "uint16", "--jxl-transfer", transfer,
                    "--jxl-effort", "1", "--input-transfer", "linear", "--input-primaries", "bt709", "--resume")
            result, _ = self.invoke(*args)
            self.assertEqual(result["frames_accepted"], 4)
            metadata = json.loads((output / "extraction_metadata.json").read_text())
            self.assertEqual(metadata["output"]["bit_depth"], 16)
            self.assertEqual(metadata["output"]["transfer"], transfer)
            self.assertEqual(metadata["output"]["compression"], "lossless-integer-codes")
            before = {p: (p.read_bytes(), p.stat().st_mtime_ns) for p in output.glob("*.jxl")}
            self.invoke(*args, "--overwrite")
            self.assertEqual({p: (p.read_bytes(), p.stat().st_mtime_ns) for p in output.glob("*.jxl")}, before)
        for i, flags in enumerate((("--format", "png", "--jxl-precision", "uint16"),
                                    ("--format", "jxl", "--jxl-precision", "half"),
                                    ("--format", "jxl", "--jxl-transfer", "srgb"),
                                    ("--format", "jxl", "--jxl-precision", "uint16", "--jxl-transfer", "bt709"))):
            destination = self.root / f"jxl-u16-invalid-{i}"
            self.invoke("extract", self.source(), "--output", destination, *flags, code=2)
            self.assertFalse(destination.exists())

    def test_jxl_missing_primaries_uses_format_neutral_diagnostic(self):
        destination = self.root / "jxl-no-primaries"
        result, _ = self.invoke("extract", self.source(), "--output", destination,
                                "--format", "jxl", "--input-transfer", "linear", code=3)
        message = result["error"]["message"]
        self.assertIn("primaries", message)
        self.assertNotIn("EXR", message)
        self.assertFalse(destination.exists())

    def test_completed_high_precision_resume_preserves_frames_without_overwrite(self):
        modes = (("exr", (), "exr"), ("tiff16", (), "tiff"),
                 ("jxl", (), "jxl"), ("jxl", ("--jxl-precision", "uint16"), "jxl"),
                 ("avif", (), "avif"), ("png", (), "png"))
        for index, (fmt, flags, extension) in enumerate(modes):
            with self.subTest(format=fmt, flags=flags):
                output = self.root / f"completed-resume-{index}"
                color = () if fmt == "png" else ("--input-transfer", "linear", "--input-primaries", "bt709")
                args = ("extract", self.source(), "--output", output, "--interval", "1", "--end", "0.35",
                        "--format", fmt, "--metadata", "--resume", *color, *flags)
                self.invoke(*args)
                images = sorted(output.glob("*." + extension))
                # TIFF uses the shared sink's .tif suffix.
                if fmt == "tiff16":
                    images = sorted(output.glob("*.tif*"))
                self.assertEqual(len(images), 4)
                original = {p: (p.read_bytes(), p.stat().st_mtime_ns) for p in images}
                for _ in range(2):
                    result, _ = self.invoke(*args)
                    self.assertEqual(result["frames_accepted"], 4)
                    for p, (data, modified) in original.items():
                        self.assertEqual(p.read_bytes(), data)
                        self.assertEqual(p.stat().st_mtime_ns, modified)
                manifest = output / "extraction_metadata.json"
                before = manifest.read_bytes()
                images[-1].write_bytes(b"corrupted")
                self.invoke(*args, code=4)
                self.assertEqual(manifest.read_bytes(), before)
                images[-1].write_bytes(original[images[-1]][0])
                self.invoke(*args)
                images[1].unlink()
                self.invoke(*args)
                self.assertEqual(images[1].read_bytes(), original[images[1]][0])

    def test_resume_does_not_adopt_untracked_metadata(self):
        for fmt in ("exr", "tiff16", "jxl", "avif", "png"):
            with self.subTest(format=fmt):
                output = self.root / ("untracked-metadata-" + fmt)
                output.mkdir()
                manifest = output / "extraction_metadata.json"
                manifest.write_bytes(b"unrelated metadata")
                color = () if fmt == "png" else ("--input-transfer", "linear", "--input-primaries", "bt709")
                self.invoke("extract", self.source(), "--output", output, "--format", fmt,
                            "--interval", "1", "--end", "0.35", "--resume", *color, code=4)
                self.assertEqual(manifest.read_bytes(), b"unrelated metadata")
                self.assertEqual(list(output.glob("frame_*")), [])

    def test_avif_manifest_reports_integer_storage_at_every_depth(self):
        for depth in (8, 10, 12):
            with self.subTest(depth=depth):
                output = self.root / f"avif-manifest-{depth}"
                self.invoke("extract", self.source(), "--output", output, "--interval", "1", "--end", "0.05",
                            "--format", "avif", "--avif-depth", str(depth),
                            "--input-transfer", "linear", "--input-primaries", "bt709")
                metadata = json.loads((output / "extraction_metadata.json").read_text(encoding="utf-8"))
                self.assertEqual(metadata["output"]["bit_depth"], depth)
                self.assertEqual(metadata["output"]["pixel_format"], "rgb_uint8" if depth == 8 else "rgb_uint16")

    def test_avif_independent_decoder_matrix_chroma_and_integer_reference(self):
        def pixels(path, depth=16, identity_format=None):
            if identity_format:
                # MC=0 stores G, B, R planes. Read integer planes directly so
                # the external reader's GBR/YUV naming cannot invoke a YUV matrix.
                data = subprocess.check_output([FFMPEG, "-v", "error", "-i", str(path), "-frames:v", "1",
                                                "-pix_fmt", identity_format, "-f", "rawvideo", "-"], timeout=20)
                values = array.array("B" if depth == 8 else "H", data)
                if depth > 8 and sys.byteorder != "little":
                    values.byteswap()
                count = len(values) // 3
                self.assertEqual(len(values), count * 3)
                scale = 65535 / ((1 << depth) - 1)
                return [round(values[plane * count + i] * scale)
                        for i in range(count) for plane in (2, 0, 1)]
            data = subprocess.check_output([FFMPEG, "-v", "error", "-i", str(path), "-frames:v", "1",
                                            "-pix_fmt", "rgb24" if depth == 8 else "rgb48le", "-f", "rawvideo", "-"], timeout=20)
            if depth == 8:
                # Compare normalized 8-bit codes, not swscale's 8-to-16 expansion.
                return [value * 257 for value in data]
            values = array.array("H", data)
            if sys.byteorder != "little":
                values.byteswap()
            return values
        for primaries in ("bt709", "bt2020"):
            reference_dir = self.root / ("avif-reference-" + primaries)
            args = ("extract", self.source(), "--interval", "1", "--end", "0.05",
                    "--input-transfer", "linear", "--input-primaries", primaries)
            self.invoke(*args, "--output", reference_dir, "--format", "tiff16")
            reference = pixels(next(reference_dir.glob("*.tif*")))
            for depth in (8, 10, 12):
                for lossless in (False, True):
                    with self.subTest(primaries=primaries, depth=depth, lossless=lossless):
                        output = self.root / f"avif-independent-{primaries}-{depth}-{lossless}"
                        quality = () if lossless else ("--avif-quality", "90")
                        self.invoke(*args, "--output", output, "--format", "avif", "--avif-depth", str(depth), *quality)
                        path = next(output.glob("*.avif"))
                        probe = json.loads(subprocess.check_output([FFPROBE, "-v", "error", "-show_streams", "-show_frames",
                                                                   "-read_intervals", "%+#1", "-of", "json", str(path)]))
                        stream = probe["streams"][0]
                        self.assertEqual(stream["color_space"], "gbr" if lossless else "bt709" if primaries == "bt709" else "bt2020nc")
                        # Readers can name identity-matrix planes GBR or YUV444.
                        # Preserve exact depth/chroma checks and verify RGB values below.
                        suffix = f"{depth}le" if depth > 8 else ""
                        formats = {"yuv444p" + suffix}
                        if lossless:
                            formats.add("gbrp" + suffix)
                        self.assertIn(stream["pix_fmt"], formats)
                        self.assertEqual(len(probe["frames"]), 1)
                        self.assertIn(probe["frames"][0]["pix_fmt"], formats)
                        self.assertEqual(stream["color_range"], "pc")
                        self.assertEqual(stream["color_transfer"], "iec61966-2-1")
                        actual = pixels(path, depth, probe["frames"][0]["pix_fmt"] if lossless else None)
                        self.assertEqual(len(actual), len(reference))
                        errors = [abs(a-b)/65535 for a,b in zip(actual, reference)]
                        if lossless:
                            self.assertLessEqual(max(errors), .51/((1 << depth)-1) + .0005)
                        else:
                            self.assertLess(sum(errors)/len(errors), .04)  # Fixture-specific quality check.

    def test_avif_lossy_resume_rejects_previous_matrix_policy(self):
        output = self.root / "avif-lossy-resume"
        args = ("extract", self.source(), "--output", output, "--interval", "1", "--end", "0.35",
                "--format", "avif", "--avif-quality", "75", "--resume",
                "--input-transfer", "linear", "--input-primaries", "bt709")
        self.invoke(*args)
        self.invoke(*args)
        images = {p: (p.read_bytes(), p.stat().st_mtime_ns) for p in output.glob("*.avif")}
        manifest = output / "extraction_metadata.json"
        before = manifest.read_bytes()
        identity = output / ".media-ingest-checkpoint" / "identity.json"
        original = identity.read_bytes()
        previous = json.loads(original)
        self.assertEqual(previous["avif"][0], "avif-sdr-ycbcr444-v2")
        previous["avif"][0] = "avif-sdr-v1"
        identity.write_text(json.dumps(previous), encoding="utf-8")
        self.invoke(*args, code=4)
        self.assertEqual(manifest.read_bytes(), before)
        for path, (data, modified) in images.items():
            self.assertEqual(path.read_bytes(), data)
            self.assertEqual(path.stat().st_mtime_ns, modified)
        identity.write_bytes(original)
        self.invoke(*args)

    def test_avif_output_options_and_verified_resume(self):
        output=self.root/"avif-cli"
        args=("extract",self.source(),"--output",output,"--interval","1","--end","0.35",
              "--format","avif","--input-transfer","linear","--input-primaries","bt709","--resume")
        result,_=self.invoke(*args)
        self.assertEqual(result["frames_accepted"],4)
        paths=sorted(output.glob("*.avif"))
        self.assertEqual(len(paths),4)
        original={p.name:p.read_bytes() for p in paths}
        metadata=json.loads((output/"extraction_metadata.json").read_text(encoding="utf-8"))
        self.assertEqual(metadata["output"]["format"],"avif")
        self.assertEqual(metadata["output"]["bit_depth"],8)
        self.assertEqual(metadata["output"]["compression"],"lossless-integer-codes")
        self.assertTrue(metadata["output"]["provider"])
        self.assertIsNone(metadata["output"]["avif_quality"])
        self.invoke(*args,"--overwrite")
        self.assertEqual({p.name:p.read_bytes() for p in output.glob("*.avif")},original)
        paths[0].unlink()
        self.invoke(*args,"--overwrite")
        self.assertEqual({p.name:p.read_bytes() for p in output.glob("*.avif")},original)
        self.invoke(*args,"--overwrite","--avif-quality","70",code=4)
        for index,flags in enumerate((("--format","png","--avif-speed","1"),
                                      ("--format","avif","--avif-quality","101"),
                                      ("--format","avif","--avif-speed","10"),
                                      ("--format","avif","--exr-precision","float"),
                                      ("--format","avif","--hdr-to-sdr"),
                                      ("--format","avif","--quality","95"))):
            destination=self.root/f"avif-invalid-{index}"
            self.invoke("extract",self.source(),"--output",destination,*flags,code=2)
            self.assertFalse(destination.exists())

    def test_jxl_output_options_and_verified_resume(self):
        output=self.root/"jxl-cli"
        args=("extract",self.source(),"--output",output,"--interval","1","--end","0.35",
              "--format","jxl","--input-transfer","linear","--input-primaries","bt709","--resume")
        result,_=self.invoke(*args)
        self.assertEqual(result["frames_accepted"],4)
        paths=sorted(output.glob("*.jxl"))
        self.assertEqual(len(paths),4)
        original={p.name:p.read_bytes() for p in paths}
        metadata=json.loads((output/"extraction_metadata.json").read_text(encoding="utf-8"))
        self.assertEqual(metadata["output"]["format"],"jxl")
        self.assertEqual(metadata["output"]["bit_depth"],32)
        self.assertEqual(metadata["output"]["compression"],"lossless")
        self.assertGreater(metadata["output"]["libjxl_version"],0)
        self.assertIsNone(metadata["output"]["jxl_distance"])
        self.invoke(*args,"--overwrite")
        self.assertEqual({p.name:p.read_bytes() for p in output.glob("*.jxl")},original)
        paths[0].unlink()
        self.invoke(*args,"--overwrite")
        self.assertEqual({p.name:p.read_bytes() for p in output.glob("*.jxl")},original)
        self.invoke(*args,"--overwrite","--jxl-distance","1",code=4)
        for index,flags in enumerate((("--format","png","--jxl-effort","1"),
                                      ("--format","jxl","--jxl-distance","0"),
                                      ("--format","jxl","--jxl-effort","10"),
                                      ("--format","jxl","--jxl-effort","1"),
                                      ("--format","jxl","--jxl-effort","2"),
                                      ("--format","jxl","--jxl-effort","3"),
                                      ("--format","jxl","--exr-precision","float"),
                                      ("--format","jxl","--hdr-to-sdr"),
                                      ("--format","jxl","--quality","95"))):
            destination=self.root/f"jxl-invalid-{index}"
            self.invoke("extract",self.source(),"--output",destination,*flags,code=2)
            self.assertFalse(destination.exists())

    def test_tiff16_output_and_option_validation(self):
        output=self.root/"tiff16-cli"
        result,_=self.invoke("extract",self.source(),"--output",output,"--interval","1","--end","0.35",
                             "--format","tiff16","--tiff16-transfer","linear",
                             "--input-transfer","linear","--input-primaries","bt709")
        self.assertEqual(result["frames_accepted"],4)
        self.assertEqual(len(list(output.glob("*.tif"))),4)
        metadata=json.loads((output/"extraction_metadata.json").read_text(encoding="utf-8"))
        self.assertEqual(metadata["output"]["bit_depth"],16)
        self.assertEqual(metadata["output"]["transfer"],"linear")
        for index,flags in enumerate(( ("--format","png","--tiff16-transfer","linear"),
                                      ("--format","tiff16","--exr-precision","half"),
                                      ("--format","tiff16","--quality","95") )):
            destination=self.root/f"tiff16-invalid-{index}"
            self.invoke("extract",self.source(),"--output",destination,*flags,code=2)
            self.assertFalse(destination.exists())

    def test_resume_is_opt_in_and_preserves_verified_image_files(self):
        output=self.root/"resume-cli"
        command=("extract",self.source(),"--output",output,"--interval","1","--end","0.35","--metadata","--resume","--quiet")
        first,_=self.invoke(*command)
        images={p:(p.read_bytes(),p.stat().st_mtime_ns) for p in output.glob("*.png")}
        second,_=self.invoke(*command)
        self.assertEqual(first["frames_accepted"],second["frames_accepted"])
        self.assertEqual(len(images),4)
        for path,(data,modified) in images.items():
            self.assertEqual(path.read_bytes(),data)
            self.assertEqual(path.stat().st_mtime_ns,modified)
        self.assertTrue((output/".media-ingest-checkpoint"/"identity.json").is_file())

    def test_fps_vfr_quiet_and_metadata_off(self):
        output=self.root/"fps-output"
        actual,stderr=self.invoke("extract",self.source("vfr-asymmetric"),"--output",output,"--fps","5","--end","1.5","--quiet")
        self.assertTrue(actual["success"])
        self.assertFalse((output/"extraction_metadata.json").exists())
        self.assertNotIn(b'"event":"progress"',stderr)
        self.assertEqual(actual["frames_accepted"],len(list(output.glob("*.png"))))
    def test_invalid_arguments_fail_before_output(self):
        for flags in (["--fps","0"],["--fps","nan"],["--interval","0"],["--scale","0"],
                      ["--size","0","32"],["--rotate","45"],["--quality","101"],
                      ["--fps","1","--interval","1"],["--scale","1","--size","64","32"],
                      ["--format","other"],["--unknown"],["--fps"]):
            with self.subTest(flags=flags):
                output=self.root/"invalid-output"
                result,_=self.invoke("extract",self.source(),"--output",output,*flags,code=2)
                self.assertEqual(result["error"]["code"],"InvalidArgument")
                self.assertFalse(output.exists())
    def test_missing_input_and_hdr_capability(self):
        result,_=self.invoke("probe",self.root/"missing.nut",code=3)
        self.assertEqual(result["error"]["code"],"NotFound")
        output=self.root/"unavailable-hdr"
        result,_=self.invoke("extract",self.source(),"--output",output,"--hdr-to-sdr",code=3)
        self.assertEqual(result["error"]["code"],"Unsupported")
        self.assertFalse(output.exists())
    def test_writer_failure_retains_count_and_files(self):
        output=self.root/"blocked-file";output.mkdir();(output/"frame_2.png").mkdir()
        result,_=self.invoke("extract",self.source(),"--output",output,"--interval","1","--end","0.35",code=3)
        self.assertTrue((output/"frame_1.png").is_file())
        fields=[c["fields"] for c in result["error"]["context"]]
        self.assertEqual(next(f["frames_accepted"] for f in fields if "frames_accepted" in f),1)
    def test_filesystem_error_keeps_utf8_native_status(self):
        output=self.root/"blocked é 日本語";output.write_bytes(b"keep")
        result,_=self.invoke("extract",self.source(),"--output",output,code=3)
        self.assertIn(str(output),result["error"]["message"])
        self.assertIn("native",result["error"])
        self.assertEqual(output.read_bytes(),b"keep")
    @unittest.skipIf(sys.platform == "win32", "Windows terminate() does not deliver a CRT SIGTERM")
    def test_sigterm_cancels_and_preserves_written_frames(self):
        source=self.root/"long.nut"
        subprocess.run([FFMPEG,"-v","error","-f","lavfi","-i","color=size=64x48:rate=100:duration=120",
                        "-c:v","ffv1",str(source)],check=True,timeout=30)
        output=self.root/"cancelled"
        with tempfile.TemporaryFile() as stderr:
            process=subprocess.Popen([str(CLI),"extract",str(source),"--output",str(output),"--interval","1"],
                                     stdout=subprocess.PIPE,stderr=stderr)
            try:
                import time
                deadline=time.monotonic()+20
                while not (output/"frame_1.png").exists() and process.poll() is None and time.monotonic()<deadline:
                    time.sleep(.01)
                self.assertTrue((output/"frame_1.png").exists(),"extraction must start before cancellation")
                process.send_signal(signal.SIGTERM)
                stdout,_=process.communicate(timeout=20)
                self.assertEqual(process.returncode,130,stdout)
                result=json.loads(stdout)
                self.assertEqual(result["error"]["code"],"Cancelled")
                fields=[c["fields"] for c in result["error"]["context"]]
                accepted=next(f["frames_accepted"] for f in fields if "frames_accepted" in f)
                self.assertEqual(accepted,len(list(output.glob("*.png"))))
                self.assertGreater(accepted,0)
            finally:
                if process.poll() is None:
                    process.kill();process.communicate()
if __name__=="__main__":
    parser=argparse.ArgumentParser();parser.add_argument("--cli",type=Path,required=True);parser.add_argument("--runner",type=Path,required=True)
    parser.add_argument("--version-header",type=Path,required=True)
    parser.add_argument("--application",type=Path,required=True)
    parser.add_argument("--visualizer",type=Path,required=True)
    parser.add_argument("--python-module",type=Path,required=True)
    parser.add_argument("--build-dir",type=Path,required=True)
    parser.add_argument("--config",required=True)
    parser.add_argument("--symbols-tool",type=Path)
    args,remaining=parser.parse_known_args();CLI=args.cli.resolve();RUNNER=args.runner.resolve();VERSION_HEADER=args.version_header.resolve()
    CONSUMERS=[args.application.resolve(),args.visualizer.resolve(),args.python_module.resolve()]
    SYMBOLS_TOOL=args.symbols_tool
    BUILD_DIR=args.build_dir.resolve();BUILD_CONFIG=args.config
    unittest.main(argv=[__file__,*remaining])
