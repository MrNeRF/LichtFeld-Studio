# SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
# SPDX-License-Identifier: GPL-3.0-or-later
"""Production CPU CLI contracts; synthetic offline input and legacy references."""
import argparse
import array
import ctypes
import hashlib
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
        for malformed in [prefix, prefix + b"\2\0\0\0x", prefix + b"\1\1\0\0x",
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

    def test_source_metadata_xmp_png_method_and_inflate_boundaries(self):
        path = self.root / "xmp-boundaries.png"
        prefix = b"XML:com.adobe.xmp\0"
        for method in [1, 255]:
            envelope = prefix + bytes([0, method, 0, 0]) + b"<x/>"
            self._xmp_png(path, [envelope])
            block = self._xmp_limited(path)["image_xmp"]["blocks"][0]
            self.assertEqual(block["status"], "collected")
            self.assertEqual(bytes.fromhex(block["bytes"]), envelope)
            self.assertEqual(bytes.fromhex(block["payload_bytes"]), b"<x/>")
        for size in [0, 1, 8191, 8192, 8193, 65536]:
            packet = b"x" * size
            envelope = prefix + b"\1\0\0\0" + zlib.compress(packet)
            self._xmp_png(path, [envelope])
            exact = self._xmp_limited(path, inflated_bytes=size)["image_xmp"]["blocks"][0]
            self.assertEqual(exact["status"], "collected", size)
            self.assertEqual(bytes.fromhex(exact["payload_bytes"]), packet)
            if size:
                limited = self._xmp_limited(path, inflated_bytes=size - 1)["image_xmp"]["blocks"][0]
                self.assertEqual(limited["status"], "limited")
                self.assertIsNone(limited["payload_bytes"])
            for damaged in [envelope[:-1], envelope + b"trailing", envelope + zlib.compress(b"extra")]:
                self._xmp_png(path, [damaged])
                block = self._xmp_limited(path)["image_xmp"]["blocks"][0]
                self.assertEqual(block["status"], "malformed")
                self.assertIsNone(block["payload_bytes"])

    def test_source_metadata_xmp_png_exif_prefix_and_collection_limits(self):
        path = self.root / "xmp-exif-budget.png"
        envelope = b"XML:com.adobe.xmp\0\0\0\0\0<x/>"
        self._xmp_png(path, [envelope])
        for limits in [{"block_bytes": 1}, {"total_bytes": 1}]:
            result = self._xmp_limited(path, collection_limits=limits)
            self.assertEqual(result["image_exif"], result["legacy_exif"])
            self.assertEqual(len(result["image_xmp"]["blocks"]), 1)
            self.assertEqual(result["image_xmp"]["blocks"][0]["status"], "collected")
        original = path.read_bytes()
        offset = 8
        while original[offset + 4:offset + 8] != b"eXIf":
            offset += 12 + struct.unpack_from(">I", original, offset)[0]
        end = offset + 12 + struct.unpack_from(">I", original, offset)[0]
        def exif_chunk(payload):
            return struct.pack(">I", len(payload)) + b"eXIf" + payload + struct.pack(">I", zlib.crc32(b"eXIf" + payload))
        # Once a block is omitted, a later smaller EXIF block must remain omitted.
        path.write_bytes(original[:offset] + exif_chunk(b"x" * 1048577) +
                         original[offset:end] + original[end:])
        result = self._xmp_limited(path, collection_limits={})
        self.assertEqual(result["image_exif"], result["legacy_exif"])
        self.assertEqual(result["image_exif"]["blocks"], [])
        self.assertEqual(len(result["image_xmp"]["blocks"]), 1)
        path.write_bytes(original[:end] + original[offset:end] + original[end:])
        result = self._xmp_limited(path, collection_limits={"blocks": 1})
        # This collector-only adapter supplies placeholder interpretations to the catalog.
        for block in result["image_exif"]["blocks"]:
            block["interpretation"] = None
        self.assertEqual(result["image_exif"], result["legacy_exif"])
        self.assertEqual(len(result["image_exif"]["blocks"]), 1)
        self.assertEqual(len(result["image_xmp"]["blocks"]), 1)
        path.write_bytes(original)
        for limits in [{"blocks": 0}, {"block_bytes": 0}, {"total_bytes": 0},
                       {"png_chunks": 0}, {"png_chunks": 65537},
                       {"png_scan_bytes": 0}, {"png_scan_bytes": 1073741825}]:
            result = self._xmp_limited(path, collection_limits=limits)
            self.assertEqual(result["image_xmp"]["status"], "limited", limits)
            self.assertNotIn("supports JPEG", result["image_xmp"]["diagnostic"])
        result = self._xmp_limited(path, collection_limits={"png_scan_bytes": 33})
        self.assertEqual(result["image_xmp"]["status"], "limited")
        self.assertEqual(result["image_xmp"]["blocks"], [])

    def test_source_metadata_png_exif_status_freezes_after_omission(self):
        path = self.root / "exif-frozen.png"
        source_exif_fixture.write_png(path, include_exif=False)
        original = path.read_bytes()
        def chunk(kind, data, bad_crc=False):
            return struct.pack(">I", len(data)) + kind + data + struct.pack(">I", zlib.crc32(kind + data) ^ int(bad_crc))
        packet = b"XML:com.adobe.xmp\0\0\0\0\0<after/>"
        large = chunk(b"eXIf", b"x" * (1536 * 1024))
        small = chunk(b"eXIf", b"II*\0")
        for damage, expected, diagnostic in [
                (chunk(b"tEXt", b"key\0value", True), "malformed", "CRC mismatch"),
                (b"\0\0\0", "malformed", "Truncated"),
                (chunk(b"tEXt", b"k\0v"), "limited", "EXIF collection budget exhausted")]:
            with self.subTest(expected=expected, diagnostic=diagnostic):
                path.write_bytes(original[:33] + large + small + chunk(b"iTXt", packet) + damage +
                                 (b"" if damage == bytes(3) else original[33:]))
                exif, _ = self.invoke("source-exif", path)
                result, _ = self.invoke("source-metadata", path)
                self.assertEqual(result["metadata"]["image_exif"], exif["exif"])
                self.assertEqual(exif["exif"]["status"], "limited")
                self.assertEqual(exif["exif"]["blocks"], [])
                xmp = result["metadata"]["image_xmp"]
                self.assertEqual(xmp["status"], expected, xmp)
                self.assertIn(diagnostic, xmp["diagnostic"])
                self.assertEqual(bytes.fromhex(xmp["blocks"][0]["payload_bytes"]), b"<after/>")
        # Actual scan exhaustion is reported separately from the frozen EXIF result.
        path.write_bytes(original[:33] + large + small + chunk(b"iTXt", packet) + original[33:])
        result = self._xmp_limited(path, exif_block_bytes=16, collection_limits={"png_chunks": 2})
        self.assertEqual(result["image_exif"], result["legacy_exif"])
        self.assertIn("PNG non-IDAT chunk count", result["image_xmp"]["diagnostic"])
        self.assertNotIn("PNG non-IDAT", result["image_exif"]["diagnostic"])

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
        tiff_records = sum(record["kind"] != "xmp_block" for record in exact["records"])
        omitted = self._xmp_limited(path, records=tiff_records)
        self.assertEqual(omitted["index_status"], "limited")
        self.assertEqual(omitted["test_summary"]["facts"], exact["test_summary"]["facts"])
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
        self.assertEqual(result["metadata"]["image_xmp"]["status"], "absent")
        result, _ = self.invoke("source-metadata", self.root / "missing-xmp.png")
        self.assertEqual(result["metadata"]["image_xmp"]["status"], "io_error")

    def _xmp_jxl(self, path, before=(), after=()):
        def box(kind, data):
            return struct.pack(">I4s", len(data) + 8, kind) + data
        codestream = (ROOT / "tests/data/jxl/rgb8.jxl").read_bytes()
        path.write_bytes(box(b"JXL ", b"\x0d\x0a\x87\x0a") + box(b"ftyp", b"jxl " + bytes(4) + b"jxl ") +
                         b"".join(box(kind, data) for kind, data in before) + box(b"jxlc", codestream) +
                         b"".join(box(kind, data) for kind, data in after))

    @staticmethod
    def _rdf_packet(body, attributes=""):
        return ('<r:RDF xmlns:r="http://www.w3.org/1999/02/22-rdf-syntax-ns#">'
                '<r:Description r:about="" xmlns:t="http://ns.adobe.com/tiff/1.0/" '
                'xmlns:e="http://ns.adobe.com/exif/1.0/" ' + attributes + '>' + body +
                '</r:Description></r:RDF>')

    def test_source_metadata_xmp_interpretation_carriers(self):
        packet = self._rdf_packet('<t:Model>Body &amp; lens</t:Model><e:FNumber>28/10</e:FNumber>',
                                  't:Make="Caméra 日本語" xmlns:d="http://www.dji.com/drone-dji/1.0/" d:RelativeAltitude="-0.75"').encode()
        for carrier in ("jpeg", "png", "jxl", "tiff", "avif", "heic", "exr"):
            with self.subTest(carrier=carrier):
                path = self.root / ("interpreted." + carrier)
                if carrier == "jpeg":
                    self._xmp_jpeg(path, [b"http://ns.adobe.com/xap/1.0/\x00" + packet])
                elif carrier == "png":
                    self._xmp_png(path, [b"XML:com.adobe.xmp\x00\x01\x00\x00\x00" + zlib.compress(packet)])
                elif carrier == "jxl":
                    self._xmp_jxl(path, [(b"xml ", packet)])
                elif carrier == "tiff":
                    source_exif_fixture.write_tiff(path, xmp_packets=[packet])
                elif carrier == "exr":
                    # Header-only metadata fixture; complete EXR images and pixels
                    # are exercised by the codec and extraction contracts.
                    path.write_bytes(struct.pack("<II", 20000630, 2) + b"xmp\0string\0" +
                                     struct.pack("<i", len(packet)) + packet + b"\0")
                else:
                    source_exif_fixture.write_heif_metadata(path, [{"data": zlib.compress(packet), "encoding": "compress_zlib"}], carrier=carrier)
                original, stamp = path.read_bytes(), path.stat().st_mtime_ns
                result, _ = self.invoke("source-metadata", path)
                summary = result["summary"]
                self.assertEqual([p["state"] for p in summary["xmp_packets"]], ["parsed"])
                facts = {f["semantic"]: f for f in summary["facts"] if "xmp_property" in f}
                self.assertEqual(len(facts), 4)
                self.assertEqual(facts["camera.make"]["value"], "Caméra 日本語")
                self.assertEqual(facts["camera.model"]["value"], "Body & lens")
                self.assertEqual(facts["lens.f_number"]["value"], {"numerator": 28, "denominator": 10})
                self.assertEqual((facts["vendor.dji.altitude.relative"]["value"], facts["vendor.dji.altitude.relative"]["unit"],
                                  facts["vendor.dji.altitude.relative"]["origin"]), (-0.75, "metres", "interpreted_source"))
                self.assertTrue(all(f["source"] == {"kind": "xmp_block", "block": 0} for f in facts.values()))
                self.assertEqual(facts["camera.make"]["xmp_property"], {"namespace": "http://ns.adobe.com/tiff/1.0/", "name": "Make", "description": 0, "ordinal": 1})
                self.assertEqual((path.read_bytes(), path.stat().st_mtime_ns), (original, stamp))

    def test_source_metadata_dji_named_photo_subject(self):
        path = self.root / "dji-photo.jpg"
        packet = self._rdf_packet('<d:RelativeAltitude>-1.25</d:RelativeAltitude><e:GPSAltitude>5/1</e:GPSAltitude>',
                                  'xmlns:d="http://www.dji.com/drone-dji/1.0/" t:Make="DJI"')
        packet = packet.replace('r:about=""', 'r:about="DJI Meta Data"').encode()
        self._xmp_jpeg(path, [b"http://ns.adobe.com/xap/1.0/\0" + packet])
        result, _ = self.invoke("source-metadata", path)
        fact, = [f for f in result["summary"]["facts"] if "xmp_property" in f]
        self.assertEqual((fact["semantic"], fact["value"], fact["origin"]),
                         ("vendor.dji.altitude.relative", -1.25, "interpreted_source"))
        self.assertEqual(fact["xmp_property"]["rdf_about"], "DJI Meta Data")
        self.assertEqual(result["summary"]["xmp_packets"][0]["state"], "parsed")
        self.assertEqual(bytes.fromhex(result["metadata"]["image_xmp"]["blocks"][0]["bytes"]),
                         b"http://ns.adobe.com/xap/1.0/\0" + packet)

    def test_source_metadata_xmp_agreement_and_independent_limits(self):
        path = self.root / "xmp-agreement.jpg"
        for orientation in ("6", "3", "invalid"):
            with self.subTest(orientation=orientation):
                source_exif_fixture.write_camera_metadata(path, carrier="jpeg")
                packet = self._rdf_packet("", f't:Orientation="{orientation}" t:Make="Test Camera"').encode()
                envelope = b"http://ns.adobe.com/xap/1.0/\0" + packet
                original = path.read_bytes()
                path.write_bytes(original[:2] + b"\xff\xe1" + struct.pack(">H", len(envelope) + 2) + envelope + original[2:])
                saved = path.read_bytes(), path.stat().st_mtime_ns
                result, _ = self.invoke("source-metadata", path)
                for semantic, conflict in (("orientation.exif", orientation != "6"), ("camera.make", False)):
                    facts = [f for f in result["summary"]["facts"] if f["semantic"] == semantic]
                    self.assertEqual(len(facts), 2)
                    self.assertTrue(all(f["ambiguous"] == conflict for f in facts))
                    self.assertEqual(sum("xmp_property" in f for f in facts), 1)
                self.assertEqual((path.read_bytes(), path.stat().st_mtime_ns), saved)
        good = self._rdf_packet('<t:Make>Unique</t:Make>').encode()
        attrs = " ".join(f'a{i}="x"' for i in range(129))
        failures = (self._rdf_packet("", attrs).encode(), b"x" * (1024 * 1024 + 1))
        for bad in failures:
            for packets in ((good, bad), (bad, good)):
                self._xmp_jxl(path, [(b"xml ", packet) for packet in packets])
                result, _ = self.invoke("source-metadata", path)
                summary = result["summary"]
                self.assertTrue(summary["limited"])
                fact, = summary["facts"]
                self.assertEqual(fact["value"], "Unique")
                self.assertFalse(fact["ambiguous"])
    @staticmethod
    def _extended_xmp(packet, step=19, guid=None):
        guid = guid or hashlib.md5(packet, usedforsecurity=False).hexdigest().upper()
        prefix = b"http://ns.adobe.com/xmp/extension/\0" + guid.encode("ascii")
        return guid, [prefix + struct.pack(">II", len(packet), offset) + packet[offset:offset + step]
                      for offset in range(0, len(packet), step)]

    def _extended_standard(self, guid, body="", element=False):
        note = 'xmlns:n="http://ns.adobe.com/xmp/note/" '
        if element:
            body = '<n:HasExtendedXMP>' + guid + '</n:HasExtendedXMP>' + body
        else:
            note += 'n:HasExtendedXMP="' + guid + '" '
        return b"http://ns.adobe.com/xap/1.0/\0" + self._rdf_packet(body, note + 't:Make="standard"').encode()

    def test_source_metadata_jpeg_extended_xmp_reassembly(self):
        packet = self._rdf_packet('<t:Model>Body 日本語 🎥</t:Model><e:FNumber>28/10</e:FNumber>').encode()
        guid, fragments = self._extended_xmp(packet)
        path = self.root / "Extended XMP é 日本語.jpg"
        for element in (False, True):
            for standard_first in (False, True):
                with self.subTest(element=element, standard_first=standard_first):
                    standard = self._extended_standard(guid, element=element)
                    blocks = [standard, *reversed(fragments)] if standard_first else [*reversed(fragments), standard]
                    self._xmp_jpeg(path, blocks)
                    original, stamp = path.read_bytes(), path.stat().st_mtime_ns
                    result, _ = self.invoke("source-metadata", path)
                    facts = {f["semantic"]: f for f in result["summary"]["facts"] if "xmp_property" in f}
                    self.assertEqual(facts["camera.make"]["value"], "standard")
                    self.assertEqual(facts["camera.model"]["value"], "Body 日本語 🎥")
                    self.assertEqual(facts["lens.f_number"]["value"], {"numerator": 28, "denominator": 10})
                    self.assertEqual(facts["camera.model"]["xmp_packet_kind"], "jpeg_extended_assembly")
                    packets = [p for p in result["summary"]["xmp_packets"] if "extended_xmp" in p]
                    self.assertEqual(len(packets), 1)
                    self.assertEqual(packets[0]["state"], "parsed")
                    assembly = next(a for a in result["summary"]["xmp_assemblies"] if a["guid"] == packets[0]["extended_xmp"])
                    self.assertEqual((assembly["guid"], assembly["declared_bytes"], assembly["md5_verified"]), (guid, len(packet), True))
                    ordered = assembly["fragments"]
                    self.assertEqual([f["offset"] for f in ordered], list(range(0, len(packet), 19)))
                    self.assertEqual(sum(f["bytes"] for f in ordered), len(packet))
                    raw = result["metadata"]["image_xmp"]
                    self.assertEqual([bytes.fromhex(b["bytes"]) for b in raw["blocks"]], blocks)
                    self.assertEqual(raw["extended_assembly"], "not_performed")
                    legacy, _ = self.invoke("source-exif", path)
                    self.assertEqual(result["metadata"]["image_exif"], legacy["exif"])
                    self.assertEqual((path.read_bytes(), path.stat().st_mtime_ns), (original, stamp))

    def test_source_metadata_jpeg_extended_xmp_rejections(self):
        packet = self._rdf_packet('<t:Model>extended</t:Model>').encode()
        guid, fragments = self._extended_xmp(packet, step=len(packet) // 2)
        standard = self._extended_standard(guid)
        path = self.root / "broken-extended.jpg"
        corrupt = fragments.copy()
        corrupt[-1] = corrupt[-1][:-1] + bytes([corrupt[-1][-1] ^ 1])
        overlap = fragments.copy()
        overlap[1] = overlap[1][:71] + struct.pack(">I", 1) + overlap[1][75:]
        inconsistent = fragments.copy()
        inconsistent[1] = inconsistent[1][:67] + struct.pack(">I", len(packet) + 1) + inconsistent[1][71:]
        for blocks, expected in [
                ([standard, *fragments[:-1]], "malformed"),
                ([standard, *fragments, fragments[0]], "malformed"),
                ([standard, *corrupt], "malformed"),
                ([standard, *overlap], "malformed"),
                ([standard, *inconsistent], "malformed"),
                ([standard], "malformed"),
                ([*fragments], "unsupported"),
                ([standard, standard, *fragments], "unsupported"),
                ([self._extended_standard(guid, '<n:HasExtendedXMP>' + guid + '</n:HasExtendedXMP>'), *fragments], "unsupported"),
        ]:
            with self.subTest(expected=expected, blocks=len(blocks)):
                self._xmp_jpeg(path, blocks)
                result, _ = self.invoke("source-metadata", path)
                facts = [f for f in result["summary"]["facts"] if "xmp_property" in f]
                self.assertTrue(all(f["semantic"] == "camera.make" for f in facts), result)
                packets = [p for p in result["summary"]["xmp_packets"] if "extended_xmp" in p]
                self.assertEqual(packets[-1]["state"], expected, result)
        self._xmp_jpeg(path, [standard, *fragments])
        for options in ({"records": 2}, {"blocks": 2}):
            limited = self._xmp_limited(path, **options)
            self.assertTrue(limited["test_summary"]["limited"], limited)
            self.assertFalse(any(f["semantic"] == "camera.model" for f in limited["test_summary"]["facts"]))

    def test_source_metadata_jpeg_extended_xmp_cross_portion_altitude(self):
        path = self.root / "split-altitude.jpg"
        for altitude_in_extension in (False, True):
            altitude, reference = '<e:GPSAltitude>25/2</e:GPSAltitude>', '<e:GPSAltitudeRef>1</e:GPSAltitudeRef>'
            packet = self._rdf_packet(altitude if altitude_in_extension else reference).encode()
            guid, fragments = self._extended_xmp(packet)
            standard = self._extended_standard(guid, reference if altitude_in_extension else altitude)
            self._xmp_jpeg(path, [standard, *fragments])
            result, _ = self.invoke("source-metadata", path)
            fact = next(f for f in result["summary"]["facts"] if f["semantic"] == "gps.altitude_sea_level")
            self.assertEqual((fact["value"], fact["state"], fact["origin"]), (-12.5, "present", "derived"))
            self.assertNotEqual(fact["xmp_inputs"][0]["source"], fact["source"])
            self.assertEqual(fact["xmp_inputs"][0]["source"], fact["inputs"][0])
            self.assertEqual(fact["xmp_assembly"]["guid"], guid)
        packet = self._rdf_packet(reference).encode()
        guid, fragments = self._extended_xmp(packet)
        self._xmp_jpeg(path, [self._extended_standard(guid, altitude + reference), *fragments])
        result, _ = self.invoke("source-metadata", path)
        fact = next(f for f in result["summary"]["facts"] if f["semantic"] == "gps.altitude_sea_level")
        self.assertEqual(fact["state"], "unsupported")
        self.assertTrue(fact["ambiguous"])

        packet = self._rdf_packet('<e:GPSAltitudeRef>2</e:GPSAltitudeRef>').encode()
        guid, fragments = self._extended_xmp(packet)
        self._xmp_jpeg(path, [self._extended_standard(guid, altitude), *fragments])
        result, _ = self.invoke("source-metadata", path)
        fact = next(f for f in result["summary"]["facts"] if f["semantic"] == "gps.altitude_sea_level")
        self.assertEqual((fact["state"], fact["value"], fact["origin"]),
                         ("unsupported", "25/2", "interpreted_source"))
        self.assertEqual(fact["inputs"], [])

    def test_source_metadata_jpeg_extended_xmp_association_and_xml(self):
        path = self.root / "association.jpg"
        packet = self._rdf_packet('<t:Model>extended</t:Model>').encode()
        guid, fragments = self._extended_xmp(packet)
        standard = self._extended_standard(guid)
        for changed in (
                standard.replace(b'http://ns.adobe.com/xmp/note/', b'urn:unrelated/'),
                self._extended_standard(guid.lower()),
                self._extended_standard('0' * 32)):
            self._xmp_jpeg(path, [changed, *fragments])
            result, _ = self.invoke("source-metadata", path)
            self.assertFalse(any(f["semantic"] == "camera.model" for f in result["summary"]["facts"]))
            self.assertEqual(next(f for f in result["summary"]["facts"] if f["semantic"] == "camera.make")["value"], "standard")
        # A different serialization of the same text still requires a different digest.
        replacement = packet.replace(b'extended', b'exten&#100;ed')
        _, altered = self._extended_xmp(replacement, guid=guid)
        self._xmp_jpeg(path, [standard, *altered])
        result, _ = self.invoke("source-metadata", path)
        self.assertEqual(result["summary"]["xmp_assemblies"][0]["md5_verified"], False)
        self.assertFalse(any(f["semantic"] == "camera.model" for f in result["summary"]["facts"]))
        for bad, expected in ((packet[:-1], "malformed"),
                              (packet.replace(b'extended', b'ext\0ended'), "malformed"),
                              (b'<!DOCTYPE x SYSTEM "file:///must-not-open">' + packet, "unsupported"),
                              (b'<!DOCTYPE x [<!ENTITY a "expanded">]>' + packet, "unsupported")):
            bad_guid, bad_fragments = self._extended_xmp(bad)
            self._xmp_jpeg(path, [self._extended_standard(bad_guid), *bad_fragments])
            result, _ = self.invoke("source-metadata", path)
            assembly_packet = next(p for p in result["summary"]["xmp_packets"] if "extended_xmp" in p)
            self.assertEqual(assembly_packet["state"], expected, result)
            self.assertTrue(result["summary"]["xmp_assemblies"][0]["md5_verified"])
            self.assertFalse(any(f["semantic"] == "camera.model" for f in result["summary"]["facts"]))

        # The completed markers survive a later truncated, unidentified APP1.
        blocks = [standard, *fragments]
        self._xmp_jpeg(path, blocks)
        prefix_bytes = 2 + sum(4 + len(block) for block in blocks)
        path.write_bytes(path.read_bytes()[:prefix_bytes] + b"\xff\xe1\x00\xfftruncated")
        result, _ = self.invoke("source-metadata", path)
        self.assertEqual(result["metadata"]["image_xmp"]["status"], "malformed")
        self.assertFalse(any(f["semantic"] == "camera.model" for f in result["summary"]["facts"]))
        self.assertEqual(next(p for p in result["summary"]["xmp_packets"] if "extended_xmp" in p)["state"], "limited")

    def test_source_metadata_jpeg_extended_xmp_group_isolation(self):
        path = self.root / "separate-groups.jpg"
        packet_a = self._rdf_packet('<t:Model>A</t:Model>').encode()
        packet_b = self._rdf_packet('<t:Model>B</t:Model><e:GPSAltitudeRef>1</e:GPSAltitudeRef>').encode()
        guid_a, fragments_a = self._extended_xmp(packet_a)
        guid_b, fragments_b = self._extended_xmp(packet_b)
        standard_a = self._extended_standard(guid_a, '<e:GPSAltitude>25/2</e:GPSAltitude>')
        self._xmp_jpeg(path, [standard_a, self._extended_standard(guid_b), *fragments_b, *fragments_a])
        result, _ = self.invoke("source-metadata", path)
        models = [f for f in result["summary"]["facts"] if f["semantic"] == "camera.model"]
        self.assertEqual(sorted(f["value"] for f in models), ["A", "B"])
        self.assertTrue(all(f["ambiguous"] for f in models))
        altitude = next(f for f in result["summary"]["facts"] if f["semantic"] == "gps.altitude_sea_level")
        self.assertEqual(altitude["state"], "unsupported")
        self.assertEqual(altitude["inputs"], [])
        corrupted = fragments_a.copy()
        corrupted[-1] = corrupted[-1][:-1] + bytes([corrupted[-1][-1] ^ 1])
        self._xmp_jpeg(path, [standard_a, self._extended_standard(guid_b), *corrupted, *fragments_b])
        result, _ = self.invoke("source-metadata", path)
        self.assertEqual([f["value"] for f in result["summary"]["facts"] if f["semantic"] == "camera.model"], ["B"])

    def test_source_metadata_jpeg_extended_xmp_index_limits(self):
        path = self.root / "output-limits.jpg"
        packet = self._rdf_packet('<e:GPSAltitudeRef>1</e:GPSAltitudeRef>').encode()
        guid, fragments = self._extended_xmp(packet)
        self._xmp_jpeg(path, [self._extended_standard(guid, '<e:GPSAltitude>25/2</e:GPSAltitude>'), *fragments])
        # Index omissions must not create dangling assembled provenance. Image
        # summary fact/text quotas have separate C++ contracts.
        for limit in (0, 1, 2, 4096):
            observed = self._xmp_limited(path, records=limit)
            summary = observed["test_summary"]
            self.assertLessEqual(len(observed["records"]), limit)
            for fact in summary["facts"]:
                if "xmp_assembly" in fact:
                    self.assertTrue(any(a["guid"] == fact["xmp_assembly"]["guid"] for a in summary["xmp_assemblies"]))

    def test_source_metadata_jpeg_extended_xmp_digest_policy(self):
        path = self.root / "digest-policy.jpg"
        packet = self._rdf_packet('<t:Model>extended</t:Model>').encode()
        guid, fragments = self._extended_xmp(packet)
        self._xmp_jpeg(path, [self._extended_standard(guid), *fragments])
        config = self.root / "openssl.cnf"
        config.write_text('openssl_conf = init\n[init]\nalg_section = algorithms\n[algorithms]\ndefault_properties = fips=yes\n')
        env = dict(os.environ, OPENSSL_CONF=str(config))
        completed = subprocess.run([str(CLI), "source-metadata", str(path)], capture_output=True, text=True, env=env)
        self.assertEqual(completed.returncode, 0, completed.stderr)
        result = json.loads(completed.stdout)
        assembly = next(p for p in result["summary"]["xmp_packets"] if "extended_xmp" in p)
        self.assertEqual(assembly["state"], "unsupported", assembly)
        value = next(a for a in result["summary"]["xmp_assemblies"] if a["guid"] == assembly["extended_xmp"])
        self.assertFalse(value["md5_verified"])
        self.assertFalse(any(f["semantic"] == "camera.model" for f in result["summary"]["facts"]))

    def test_jpeg_standard_export_does_not_require_md5(self):
        source = self.root / "standard-export.mkv"
        subprocess.run([FFMPEG, "-v", "error", "-y", "-i", str(self.source()),
                        "-map", "0:v:0", "-c:v", "copy", "-metadata", "comment=" + "A" * 16000,
                        "-write_crc32", "0", str(source)], check=True, capture_output=True, timeout=30)
        config = self.root / "no-md5-export.cnf"
        config.write_text('openssl_conf = init\n[init]\nalg_section = algorithms\n[algorithms]\ndefault_properties = fips=yes\n', encoding="utf-8")
        output = self.root / "standard-without-md5"
        completed = subprocess.run([str(CLI), "extract", str(source), "--output", str(output),
                                    "--format", "jpeg", "--interval", "1", "--end", "0.2",
                                    "--embed-source-metadata"], capture_output=True, text=True,
                                   env=dict(os.environ, OPENSSL_CONF=str(config)), timeout=30)
        self.assertEqual(completed.returncode, 0, completed.stderr)
        self.assertGreater(json.loads(completed.stdout)["frames_accepted"], 0)
        for image in output.glob("*.jpg"):
            data = image.read_bytes()
            self.assertIn(b"http://ns.adobe.com/xap/1.0/\0", data)
            self.assertNotIn(b"http://ns.adobe.com/xmp/extension/\0", data)

    def test_source_metadata_xmp_interpretation_secondary_and_invalid(self):
        packet = self._rdf_packet('<t:Make>primary</t:Make>').encode()
        child = self._rdf_packet('<t:Make>secondary</t:Make>').encode()
        path = self.root / "xmp-directory.tif"
        source_exif_fixture.write_tiff_graph(path, blobs=[(700, packet)], subimage_blobs=[(700, child)])
        result, _ = self.invoke("source-metadata", path)
        self.assertEqual([p["state"] for p in result["summary"]["xmp_packets"]], ["parsed", "unsupported"])
        self.assertEqual([f["value"] for f in result["summary"]["facts"] if "xmp_property" in f], ["primary"])
        path = self.root / "xmp-crc.png"
        self._xmp_png(path, [b"XML:com.adobe.xmp\x00\x00\x00\x00\x00" + packet], bad_crc=True)
        result, _ = self.invoke("source-metadata", path)
        self.assertEqual(result["summary"]["xmp_packets"][0]["state"], "malformed")
        self.assertFalse(any("xmp_property" in f for f in result["summary"]["facts"]))

    def test_source_metadata_xmp_interpretation_packet_failures(self):
        path = self.root / "unsafe-xmp.jxl"
        good = self._rdf_packet('<t:Model>valid</t:Model>').encode()
        for packet, expected in (
                (good[:-1], "malformed"),
                (b'<!DOCTYPE x SYSTEM "file:///must-not-open">' + good, "unsupported"),
                (b'<!DOCTYPE x [<!ENTITY a "expanded">]>' + good, "unsupported"),
                (good.replace(b"valid", b"\xff"), "malformed"),
                (good.replace(b'r:about=""', b'r:about="other"'), "unsupported"),
                (good.replace(b'<t:Model>valid</t:Model>', b'<u>' * 40 + b'</u>' * 40), "limited")):
            with self.subTest(expected=expected, packet=packet[:50]):
                self._xmp_jxl(path, [(b"xml ", packet), (b"xml ", good)])
                result, _ = self.invoke("source-metadata", path)
                self.assertEqual([p["state"] for p in result["summary"]["xmp_packets"]], [expected, "parsed"])
                self.assertEqual(len(result["summary"]["facts"]), 1)
                self.assertEqual(result["summary"]["facts"][0]["source"]["block"], 1)
                self.assertEqual(bytes.fromhex(result["metadata"]["image_xmp"]["blocks"][0]["payload_bytes"]), packet)

    def test_source_metadata_xmp_interpretation_values(self):
        path = self.root / "xmp-values.jxl"
        cases = [("GPSLatitude", "90,0N", "present", 90.0),
                 ("GPSLatitude", "90,0.1N", "invalid", "90,0.1N"),
                 ("GPSLongitude", "45,30,0W", "present", -45.5),
                 ("GPSLongitude", "45.5W", "invalid", "45.5W"),
                 ("GPSLongitude", "45,3e1W", "invalid", "45,3e1W"),
                 ("GPSLatitude", "-0,30N", "invalid", "-0,30N"),
                 ("ExposureTime", "1/125", "present", {"numerator": 1, "denominator": 125}),
                 ("ExposureTime", "1/0", "invalid", "1/0"),
                 ("FocalLength", "NaN", "invalid", "NaN"),
                 ("DateTimeOriginal", "2024-02", "present", "2024-02"),
                 ("DateTimeOriginal", "2023-02-29", "invalid", "2023-02-29"),
                 ("DateTimeOriginal", "2016-12-31T23:59:60Z", "unsupported", "2016-12-31T23:59:60Z")]
        for name, value, state, expected in cases:
            with self.subTest(name=name, value=value):
                packet = self._rdf_packet(f"<e:{name}>{value}</e:{name}>").encode()
                self._xmp_jxl(path, [(b"xml ", packet)])
                result, _ = self.invoke("source-metadata", path)
                fact, = result["summary"]["facts"]
                self.assertEqual((fact["state"], fact["value"]), (state, expected))

    def test_source_metadata_xmp_interpretation_utf16_and_duplicates(self):
        path = self.root / "utf16-xmp.jxl"
        xml = self._rdf_packet('<t:Make>日本語</t:Make><t:Make>other</t:Make><z:Make xmlns:z="urn:wrong">ignored</z:Make>')
        packet = ('<?xml version="1.0" encoding="UTF-16"?>' + xml).encode("utf-16")
        self._xmp_jxl(path, [(b"xml ", packet)])
        result, _ = self.invoke("source-metadata", path)
        self.assertEqual([f["value"] for f in result["summary"]["facts"]], ["日本語", "other"])
        self.assertTrue(all(f["ambiguous"] for f in result["summary"]["facts"]))
        self.assertEqual([f["xmp_property"]["ordinal"] for f in result["summary"]["facts"]], [1, 2])

    def test_source_metadata_exif_jxl_envelopes_and_facts(self):
        path = self.root / "EXIF JPEG XL \u00e9 \u65e5\u672c\u8a9e.jxl"
        fixture = self.root / "camera-jxl.png"
        for little in (False, True):
            source_exif_fixture.write_camera_metadata(fixture, little=little)
            data = fixture.read_bytes()
            start = data.index(b"eXIf")
            tiff = data[start + 4:start + 4 + struct.unpack(">I", data[start - 4:start])[0]]
            envelope = struct.pack(">I", 3) + b"pad" + tiff
            self._xmp_jxl(path, [(b"Exif", envelope), (b"xml ", b"<opaque/>")], [(b"Exif", envelope)])
            original, stamp = path.read_bytes(), path.stat().st_mtime_ns
            legacy, _ = self.invoke("source-exif", path)
            result, _ = self.invoke("source-metadata", path)
            metadata = result["metadata"]
            self.assertEqual(metadata["image_exif"], legacy["exif"])
            exif = metadata["image_exif"]
            self.assertEqual(exif["status"], "collected", exif)
            self.assertEqual(exif["scope"], "jxl-container-exif")
            self.assertEqual(len(exif["blocks"]), 2)
            for block in exif["blocks"]:
                self.assertEqual(block["envelope"], "jxl_offset")
                self.assertEqual(block["representation"], "libjxl_decoded_bytes")
                self.assertEqual(bytes.fromhex(block["bytes"]), envelope)
                self.assertEqual(block["interpretation"]["status"], "parsed")
                self.assertFalse(block["jxl_box"]["brotli_compressed"])
            reference, _ = self.invoke("source-metadata", fixture)
            # Compare every occurrence, including compound inputs and its source block.
            expected = []
            for ordinal in range(2):
                for fact in reference["summary"]["facts"]:
                    duplicate = json.loads(json.dumps(fact))
                    for source in [duplicate["source"], *duplicate["inputs"]]:
                        self.assertEqual(source["kind"], "exif_entry")
                        source["block"] = ordinal
                    expected.append(duplicate)
            self.assertCountEqual(result["summary"]["facts"], expected)
            self.assertEqual([block["jxl_box"]["ordinal"] for block in exif["blocks"]], [2, 5])
            self.assertEqual(metadata["image_xmp"]["blocks"][0]["jxl_box"]["ordinal"], 3)
            self.assertEqual(path.read_bytes(), original)
            self.assertEqual(path.stat().st_mtime_ns, stamp)
        # Independent Brotli fixture: four-byte zero offset and a little-endian orientation IFD.
        compressed = bytes.fromhex("1b1d00000471633b82982dd1d44c4a60f12503273dfc9f790164538216")
        decoded = bytes.fromhex("0000000049492a0008000000010012010300010000000600000000000000")
        self._xmp_jxl(path, [(b"brob", b"Exif" + compressed)])
        result, _ = self.invoke("source-metadata", path)
        block = result["metadata"]["image_exif"]["blocks"][0]
        self.assertEqual(bytes.fromhex(block["bytes"]), decoded)
        self.assertTrue(block["jxl_box"]["brotli_compressed"])
        self.assertEqual(block["interpretation"]["status"], "parsed")
        self.assertEqual(block["interpretation"]["entries"][0]["unsigned_values"], [6])
        self._xmp_jxl(path, [(b"Exif", b""), (b"Exif", bytes.fromhex("ffffffff"))])
        result, _ = self.invoke("source-metadata", path)
        self.assertEqual(result["metadata"]["image_exif"]["status"], "collected")
        self.assertEqual(len(result["metadata"]["image_exif"]["blocks"]), 2)
        self.assertTrue(all(b["interpretation"]["status"] == "malformed" for b in result["metadata"]["image_exif"]["blocks"]))

    def test_source_metadata_exif_jxl_independent_quotas(self):
        def collection_only(exif):
            return {**exif, "blocks": [{k: v for k, v in block.items() if k != "interpretation"}
                                      for block in exif["blocks"]]}
        path = self.root / "exif-quotas.jxl"
        compressed = bytes.fromhex("1bff1f0024f0e2b140722f0000")  # Brotli 8192 b"x" bytes.
        self._xmp_jxl(path, [(b"Exif", b"first"), (b"brob", b"Exif" + compressed)], [(b"xml ", b"later"), (b"Exif", b"last")])
        for budget in (8191, 8192, 8193):
            result = self._xmp_limited(path, collection_limits={"block_bytes": budget})
            exif = result["image_exif"]
            self.assertEqual(collection_only(exif), collection_only(result["legacy_exif"]))
            self.assertEqual(exif["status"], "limited" if budget == 8191 else "collected", exif)
            self.assertEqual([bytes.fromhex(b["bytes"]) for b in exif["blocks"]],
                             [b"first"] if budget == 8191 else [b"first", b"x" * 8192, b"last"])
            self.assertEqual(result["image_xmp"]["status"], "collected")
            self.assertEqual(bytes.fromhex(result["image_xmp"]["blocks"][0]["bytes"]), b"later")
            if budget >= 8192:
                self.assertTrue(exif["blocks"][1]["jxl_box"]["brotli_compressed"])
        self._xmp_jxl(path, [(b"xml ", b"first"), (b"brob", b"xml " + compressed)], [(b"Exif", b"later")])
        result = self._xmp_limited(path, inflated_bytes=8191)
        self.assertEqual(result["image_xmp"]["status"], "limited")
        self.assertEqual(result["image_exif"]["status"], "collected")
        self.assertEqual(bytes.fromhex(result["image_exif"]["blocks"][0]["bytes"]), b"later")
        self._xmp_jxl(path, [(b"Exif", b"one"), (b"Exif", b"two")], [(b"xml ", b"last")])
        result = self._xmp_limited(path, collection_limits={"blocks": 1})
        self.assertEqual(collection_only(result["image_exif"]), collection_only(result["legacy_exif"]))
        self.assertEqual(result["image_exif"]["status"], "limited")
        self.assertEqual(result["image_xmp"]["status"], "collected")
        self._xmp_jxl(path, [(b"Exif", b"x" * 129), (b"Exif", b"last"), (b"xml ", b"last")])
        result = self._xmp_limited(path, collection_limits={"block_bytes": 128})
        self.assertEqual(result["image_exif"]["status"], "limited")
        self.assertEqual([bytes.fromhex(b["bytes"]) for b in result["image_exif"]["blocks"]], [b"last"])
        self.assertEqual(result["image_xmp"]["status"], "collected")

        self._xmp_jxl(path, [(b"Exif", b"one"), (b"Exif", b"two")], [(b"xml ", b"tail")])
        for budget in (5, 6):
            result = self._xmp_limited(path, collection_limits={"total_bytes": budget})
            self.assertEqual(result["image_exif"]["status"], "limited" if budget == 5 else "collected")
            self.assertEqual(len(result["image_exif"]["blocks"]), 1 if budget == 5 else 2)
            self.assertEqual(result["image_xmp"]["status"], "collected")
        result = self._xmp_limited(path, collection_limits={"block_bytes": (16 << 20) + 1})
        self.assertEqual(result["image_exif"]["status"], "limited")
        self.assertEqual(result["image_exif"]["blocks"], [])
        self.assertEqual(result["image_xmp"]["status"], "collected")

    def test_source_metadata_exif_jxl_damage_and_scan_boundaries(self):
        path = self.root / "damaged-exif.jxl"
        compressed = bytes.fromhex("1bff1f0024f0e2b140722f0000")
        for bad in (b"broken", compressed[:-1]):
            self._xmp_jxl(path, [(b"Exif", b"prefix"), (b"brob", b"Exif" + bad)])
            result, _ = self.invoke("source-metadata", path)
            exif = result["metadata"]["image_exif"]
            self.assertEqual(exif["status"], "malformed", exif)
            self.assertEqual([bytes.fromhex(b["bytes"]) for b in exif["blocks"]], [b"prefix"])
        # A carrier stopped at its quota does not claim later malformed boxes were examined.
        self._xmp_jxl(path, [(b"Exif", b"one"), (b"Exif", b"two")], [(b"xml ", b"last")])
        path.write_bytes(path.read_bytes() + struct.pack(">I4s", 7, b"junk"))
        result = self._xmp_limited(path, collection_limits={"blocks": 1})
        self.assertEqual(result["image_exif"]["status"], "limited")
        self.assertEqual(result["image_xmp"]["status"], "malformed")
        self.assertEqual([bytes.fromhex(b["bytes"]) for b in result["image_xmp"]["blocks"]], [b"last"])
        self._xmp_jxl(path)
        original = path.read_bytes()
        envelope = bytes(4) + b"II" + struct.pack("<HIH", 42, 8, 0) + bytes(4)
        for box in (struct.pack(">I4sQ", 1, b"Exif", len(envelope) + 16) + envelope,
                    struct.pack(">I4s", 0, b"Exif") + envelope):
            path.write_bytes(original + box)
            result, _ = self.invoke("source-metadata", path)
            exif = result["metadata"]["image_exif"]
            self.assertEqual(exif["status"], "collected", exif)
            self.assertEqual(bytes.fromhex(exif["blocks"][0]["bytes"]), envelope)
        self._xmp_jxl(path, [(b"brob", b"Exif" + bytes.fromhex("9bffff3ff825f0e2b14020f7fe07"))], [(b"xml ", b"tail")])
        result = self._xmp_limited(path, collection_limits={"block_bytes": 4096})
        self.assertEqual(result["image_exif"]["status"], "limited")
        self.assertEqual(result["image_exif"]["blocks"], [])
        self.assertEqual(result["image_xmp"]["status"], "collected")
        self.assertEqual(bytes.fromhex(result["image_xmp"]["blocks"][0]["bytes"]), b"tail")
        for count in (4092, 4094):
            self._xmp_jxl(path, [(b"junk", b"")] * count)
            result, _ = self.invoke("source-metadata", path)
            self.assertEqual(result["metadata"]["image_exif"]["status"], "absent" if count == 4092 else "limited")
        path.write_bytes((ROOT / "tests/data/jxl/rgb8.jxl").read_bytes())
        result, _ = self.invoke("source-metadata", path)
        self.assertEqual(result["metadata"]["image_exif"]["status"], "absent")

    def test_source_metadata_jxl_completed_boxes_survive_bad_next_header(self):
        path = self.root / "complete-box-before-error.jxl"
        exif = bytes.fromhex("0000000049492a0008000000010012010300010000000600000000000000")
        cases = [
            (b"Exif", exif, bytes.fromhex("1b1d00000471633b82982dd1d44c4a60f12503273dfc9f790164538216"), "image_exif"),
            (b"xml ", b"<xmp/>\x00\xff", bytes.fromhex("8b03803c786d702f3e00ff03"), "image_xmp"),
        ]
        for kind, decoded, brotli, carrier in cases:
            for compressed in (False, True):
                for tail in (struct.pack(">I4s", 7, b"junk"), struct.pack(">I4sI", 1, b"junk", 0)):
                    with self.subTest(kind=kind, compressed=compressed, tail=tail):
                        self._xmp_jxl(path, after=[(b"brob", kind + brotli)] if compressed else [(kind, decoded)])
                        path.write_bytes(path.read_bytes() + tail)
                        result, _ = self.invoke("source-metadata", path)
                        observed = result["metadata"][carrier]
                        self.assertEqual(observed["status"], "malformed", observed)
                        self.assertEqual(len(observed["blocks"]), 1)
                        block = observed["blocks"][0]
                        self.assertEqual(bytes.fromhex(block["bytes"]), decoded)
                        self.assertEqual(block["jxl_box"], {"ordinal": 3, "brotli_compressed": compressed})
                        if carrier == "image_exif":
                            self.assertEqual(block["interpretation"]["status"], "parsed")
                            legacy, _ = self.invoke("source-exif", path)
                            self.assertEqual(legacy["exif"], observed)

    def test_source_metadata_jxl_damaged_carrier_recovers_other(self):
        path = self.root / "damaged carrier \u00e9 \u65e5\u672c\u8a9e.jxl"
        exif = bytes.fromhex("0000000049492a0008000000010012010300010000000600000000000000")
        carriers = [
            (b"Exif", exif, bytes.fromhex("1b1d00000471633b82982dd1d44c4a60f12503273dfc9f790164538216"), "image_exif"),
            (b"xml ", b"<xmp/>\x00\xff", bytes.fromhex("8b03803c786d702f3e00ff03"), "image_xmp"),
        ]
        for index, (damaged_kind, damaged_value, damaged_brotli, damaged_key) in enumerate(carriers):
            healthy_kind, healthy_value, healthy_brotli, healthy_key = carriers[1 - index]
            for bad in (b"broken", damaged_brotli[:-1]):
                for compressed in (False, True):
                    with self.subTest(damaged=damaged_kind, bad=bad, compressed=compressed):
                        healthy_box = (b"brob", healthy_kind + healthy_brotli) if compressed else (healthy_kind, healthy_value)
                        self._xmp_jxl(path, [healthy_box, (damaged_kind, damaged_value), (b"brob", damaged_kind + bad)],
                                      [healthy_box, (damaged_kind, b"not examined")])
                        original, stamp = path.read_bytes(), path.stat().st_mtime_ns
                        result, _ = self.invoke("source-metadata", path)
                        metadata = result["metadata"]
                        damaged, healthy = metadata[damaged_key], metadata[healthy_key]
                        self.assertEqual(damaged["status"], "malformed", damaged)
                        self.assertEqual([bytes.fromhex(b["bytes"]) for b in damaged["blocks"]], [damaged_value])
                        self.assertEqual(healthy["status"], "collected", healthy)
                        self.assertEqual([bytes.fromhex(b["bytes"]) for b in healthy["blocks"]], [healthy_value, healthy_value])
                        self.assertEqual([b["jxl_box"] for b in healthy["blocks"]],
                                         [{"ordinal": 2, "brotli_compressed": compressed}, {"ordinal": 6, "brotli_compressed": compressed}])
                        legacy, _ = self.invoke("source-exif", path)
                        self.assertEqual(metadata["image_exif"], legacy["exif"])
                        self.assertEqual(path.read_bytes(), original)
                        self.assertEqual(path.stat().st_mtime_ns, stamp)

    def test_source_metadata_jxl_damaged_carrier_retry_limits(self):
        path = self.root / "retry-limits.jxl"
        for damaged_kind, healthy_kind, healthy_key in [(b"Exif", b"xml ", "image_xmp"), (b"xml ", b"Exif", "image_exif")]:
            self._xmp_jxl(path, [(healthy_kind, b"one"), (b"brob", damaged_kind + b"broken")], [(healthy_kind, b"two")])
            options = {"blocks": 1} if healthy_key == "image_xmp" else {"collection_limits": {"blocks": 1}}
            result = self._xmp_limited(path, **options)
            self.assertEqual(result[healthy_key]["status"], "limited", result)
            self.assertEqual([bytes.fromhex(b["bytes"]) for b in result[healthy_key]["blocks"]], [b"one"])
            # Container framing errors remain failures even after skipping the damaged carrier.
            path.write_bytes(path.read_bytes() + struct.pack(">I4s", 7, b"junk"))
            result, _ = self.invoke("source-metadata", path)
            self.assertEqual(result["metadata"][healthy_key]["status"], "malformed", result)
            self.assertEqual([bytes.fromhex(b["bytes"]) for b in result["metadata"][healthy_key]["blocks"]], [b"one", b"two"])
        # A second damaged carrier fails without triggering an unbounded retry.
        self._xmp_jxl(path, [(b"Exif", b"one"), (b"xml ", b"two"), (b"brob", b"Exifbroken")], [(b"brob", b"xml broken")])
        result, _ = self.invoke("source-metadata", path)
        for key, prefix in [("image_exif", b"one"), ("image_xmp", b"two")]:
            self.assertEqual(result["metadata"][key]["status"], "malformed", result)
            self.assertEqual([bytes.fromhex(b["bytes"]) for b in result["metadata"][key]["blocks"]], [prefix])

    def test_source_metadata_jxl_damaged_carrier_refill_and_totals(self):
        path = self.root / "retry-refill.jxl"
        for damaged_kind, healthy_kind, healthy_key in [(b"Exif", b"xml ", "image_xmp"), (b"xml ", b"Exif", "image_exif")]:
            # Both passes must refill beyond the initial 64 KiB input window.
            self._xmp_jxl(path, [(b"junk", bytes(70000)), (healthy_kind, b"one"), (b"brob", damaged_kind + b"broken")],
                          [(b"junk", bytes(70000)), (healthy_kind, b"two")])
            for budget in (5, 6):
                options = {"total_bytes": budget} if healthy_key == "image_xmp" else {"collection_limits": {"total_bytes": budget}}
                result = self._xmp_limited(path, **options)
                healthy = result[healthy_key]
                self.assertEqual(healthy["status"], "limited" if budget == 5 else "collected", healthy)
                self.assertEqual([bytes.fromhex(b["bytes"]) for b in healthy["blocks"]], [b"one"] if budget == 5 else [b"one", b"two"])
                self.assertEqual([b["jxl_box"]["ordinal"] for b in healthy["blocks"]], [3] if budget == 5 else [3, 7])
            # A carrier already stopped at its quota keeps that snapshot.
            self._xmp_jxl(path, [(healthy_kind, b"one"), (healthy_kind, b"two"), (b"brob", damaged_kind + b"broken")],
                          [(healthy_kind, b"three")])
            options = {"blocks": 1} if healthy_key == "image_xmp" else {"collection_limits": {"blocks": 1}}
            result = self._xmp_limited(path, **options)
            self.assertEqual(result[healthy_key]["status"], "limited", result)
            self.assertEqual([bytes.fromhex(b["bytes"]) for b in result[healthy_key]["blocks"]], [b"one"])

    def test_source_metadata_xmp_jxl_boxes_and_legacy(self):
        path = self.root / "XMP JPEG XL \u00e9 \u65e5\u672c\u8a9e.jxl"
        packet = b'<x xmlns="urn:unknown">\x00\xff</x>'
        compressed = bytes.fromhex("8b03803c786d702f3e00ff03")  # Brotli b"<xmp/>\x00\xff".
        self._xmp_jxl(path, [(b"xml ", packet), (b"Exif", b"opaque"), (b"junk", b"ignored")],
                      [(b"brob", b"xml " + compressed), (b"xml ", packet), (b"xml ", b"")])
        original, stamp = path.read_bytes(), path.stat().st_mtime_ns
        legacy, _ = self.invoke("source-exif", path)
        result, _ = self.invoke("source-metadata", path)
        metadata = result["metadata"]
        self.assertEqual(metadata["image_exif"], legacy["exif"])
        xmp = metadata["image_xmp"]
        self.assertEqual(xmp["status"], "collected", xmp)
        self.assertEqual([bytes.fromhex(b["bytes"]) for b in xmp["blocks"]], [packet, b"<xmp/>\x00\xff", packet, b""])
        self.assertEqual([b["jxl_box"]["brotli_compressed"] for b in xmp["blocks"]], [False, True, False, False])
        ordinals = [b["jxl_box"]["ordinal"] for b in xmp["blocks"]]
        self.assertEqual(ordinals, sorted(set(ordinals)))
        for block in xmp["blocks"]:
            self.assertEqual(block["envelope"], "jxl_xml")
            self.assertEqual(block["representation"], "libjxl_decoded_bytes")
            self.assertEqual(block["bytes"], block["payload_bytes"])
            self.assertFalse(block["compressed"])
        self.assertEqual(path.read_bytes(), original)
        self.assertEqual(path.stat().st_mtime_ns, stamp)
        self.assertEqual(self._xmp_limited(path, records=0)["image_xmp"], xmp)
        for raw in (False, True):
            if raw:
                path.write_bytes((ROOT / "tests/data/jxl/rgb8.jxl").read_bytes())
            else:
                self._xmp_jxl(path)
            result, _ = self.invoke("source-metadata", path)
            self.assertEqual(result["metadata"]["image_xmp"]["status"], "absent")

    def test_source_metadata_xmp_jxl_budgets_and_damage(self):
        path = self.root / "limited-xmp.jxl"
        self._xmp_jxl(path, [(b"xml ", b"x" * 129), (b"xml ", b"last")])
        limited = self._xmp_limited(path, block_bytes=128)["image_xmp"]
        self.assertEqual(limited["status"], "limited")
        self.assertEqual([bytes.fromhex(b["bytes"]) for b in limited["blocks"]], [b"last"])
        compressed = bytes.fromhex("1bff1f0024f0e2b140722f0000")  # Brotli 8192 b"x" bytes.
        self._xmp_jxl(path, [(b"xml ", b"first"), (b"brob", b"xml " + compressed)], [(b"xml ", b"later")])
        for budget in (8191, 8192, 8193):
            xmp = self._xmp_limited(path, inflated_bytes=budget)["image_xmp"]
            self.assertEqual(xmp["status"], "limited" if budget == 8191 else "collected", xmp)
            self.assertEqual(bytes.fromhex(xmp["blocks"][0]["bytes"]), b"first")
            self.assertEqual([bytes.fromhex(b["bytes"]) for b in xmp["blocks"]],
                             [b"first"] if budget == 8191 else [b"first", b"x" * 8192, b"later"])
            self.assertTrue(all(b["bytes"] == b["payload_bytes"] for b in xmp["blocks"]))
        self._xmp_jxl(path, [(b"xml ", b"one"), (b"xml ", b"two")])
        for budget in (5, 6):
            xmp = self._xmp_limited(path, total_bytes=budget)["image_xmp"]
            self.assertEqual(xmp["status"], "limited" if budget == 5 else "collected", xmp)
            self.assertEqual([bytes.fromhex(b["bytes"]) for b in xmp["blocks"]],
                             [b"one"] if budget == 5 else [b"one", b"two"])
        xmp = self._xmp_limited(path, blocks=1)["image_xmp"]
        self.assertEqual(xmp["status"], "limited")
        self.assertEqual(len(xmp["blocks"]), 1)
        for bad in (b"broken", compressed[:-1]):
            self._xmp_jxl(path, [(b"xml ", b"first"), (b"brob", b"xml " + bad)])
            result, _ = self.invoke("source-metadata", path)
            xmp = result["metadata"]["image_xmp"]
            self.assertEqual(xmp["status"], "malformed", xmp)
            self.assertEqual([bytes.fromhex(b["bytes"]) for b in xmp["blocks"]], [b"first"])
        self._xmp_jxl(path, [(b"xml ", b"first")], [(b"xml ", b"truncated")])
        path.write_bytes(path.read_bytes()[:-3])
        result, _ = self.invoke("source-metadata", path)
        self.assertEqual(result["metadata"]["image_xmp"]["status"], "malformed")

    def test_source_metadata_xmp_jxl_box_framing_and_scan_limits(self):
        path = self.root / "framing-xmp.jxl"
        self._xmp_jxl(path)
        original = path.read_bytes()
        # Extended-size and EOF-sized boxes, independent of the application's parser.
        packet = b"<opaque/>\x00\xff"
        for box in (struct.pack(">I4sQ", 1, b"xml ", len(packet) + 16) + packet,
                    struct.pack(">I4s", 0, b"xml ") + packet):
            path.write_bytes(original + box)
            result, _ = self.invoke("source-metadata", path)
            self.assertEqual(result["metadata"]["image_xmp"]["status"], "collected", result)
            self.assertEqual(bytes.fromhex(result["metadata"]["image_xmp"]["blocks"][0]["bytes"]), packet)
        for box in (struct.pack(">I4sQ", 1, b"xml ", 15), struct.pack(">I4s", 7, b"xml "),
                    struct.pack(">I4sQ", 1, b"xml ", (1 << 64) - 1),
                    struct.pack(">I4sQ", 1, b"xml ", (1 << 64) - len(original))):
            path.write_bytes(original + box)
            result, _ = self.invoke("source-metadata", path)
            self.assertEqual(result["metadata"]["image_xmp"]["status"], "malformed")
        large = b"x" * 70000 + b"\x00\xff"
        self._xmp_jxl(path, [(b"xml ", large)])
        result, _ = self.invoke("source-metadata", path)
        self.assertEqual(bytes.fromhex(result["metadata"]["image_xmp"]["blocks"][0]["bytes"]), large)
        # Split the codestream into two jxlp boxes, keeping XML between them.
        codestream = (ROOT / "tests/data/jxl/rgb8.jxl").read_bytes()
        def box(kind, data):
            return struct.pack(">I4s", len(data) + 8, kind) + data
        prefix = original[:32]
        midpoint = len(codestream) // 2
        path.write_bytes(prefix + box(b"jxlp", bytes(4) + codestream[:midpoint]) +
                         box(b"xml ", packet) + box(b"jxlp", struct.pack(">I", 0x80000001) + codestream[midpoint:]))
        result, _ = self.invoke("source-metadata", path)
        self.assertEqual(result["metadata"]["image_xmp"]["status"], "collected", result)
        self.assertEqual(bytes.fromhex(result["metadata"]["image_xmp"]["blocks"][0]["bytes"]), packet)
        for count in (4092, 4094):
            self._xmp_jxl(path, [(b"junk", b"")] * count)
            result, _ = self.invoke("source-metadata", path)
            self.assertEqual(result["metadata"]["image_xmp"]["status"], "absent" if count == 4092 else "limited", result)
        self._xmp_jxl(path, [(b"xml ", b"one")], [(b"junk", b"ignored")])
        self.assertEqual(self._xmp_limited(path, blocks=1)["image_xmp"]["status"], "collected")
        self._xmp_jxl(path, [(b"brob", b"xml " + bytes.fromhex("8b03803c786d702f3e00ff03"))])
        self.assertEqual(self._xmp_limited(path, inflated_bytes=0)["image_xmp"]["status"], "limited")

    def test_source_metadata_xmp_jxl_external_entities_stay_opaque(self):
        marker = self.root / "entity-marker.txt"
        marker.write_bytes(b"external entity must stay unread")
        packet = ('<!DOCTYPE x [<!ENTITY e SYSTEM "' + marker.as_uri() + '">]><x>&e;</x>').encode()
        path = self.root / "opaque-entity.jxl"
        self._xmp_jxl(path, [(b"xml ", packet)])
        watch = None
        if sys.platform.startswith("linux"):
            libc = ctypes.CDLL(None, use_errno=True)
            libc.inotify_init1.argtypes = [ctypes.c_int]
            libc.inotify_init1.restype = ctypes.c_int
            libc.inotify_add_watch.argtypes = [ctypes.c_int, ctypes.c_char_p, ctypes.c_uint32]
            libc.inotify_add_watch.restype = ctypes.c_int
            watch = libc.inotify_init1(os.O_NONBLOCK | os.O_CLOEXEC)
            self.assertGreaterEqual(watch, 0)
            self.addCleanup(os.close, watch)
            self.assertGreaterEqual(libc.inotify_add_watch(watch, os.fsencode(marker), 0x21), 0)
        result, _ = self.invoke("source-metadata", path)
        xmp = result["metadata"]["image_xmp"]
        self.assertEqual(xmp["status"], "collected", xmp)
        self.assertEqual([bytes.fromhex(b["bytes"]) for b in xmp["blocks"]], [packet])
        if watch is not None:
            with self.assertRaises(BlockingIOError):
                os.read(watch, 4096)

    def test_source_metadata_xmp_jxl_high_expansion(self):
        path = self.root / "expansion.jxl"
        # Independently encoded Brotli: 4 MiB of b"x", without a test-time Brotli dependency.
        compressed = bytes.fromhex("9bffff3ff825f0e2b14020f7fe07")
        self._xmp_jxl(path, [(b"xml ", b"prefix"), (b"brob", b"xml " + compressed)], [(b"xml ", b"tail")])
        result = self._xmp_limited(path, inflated_bytes=4096)
        xmp = result["image_xmp"]
        self.assertEqual(xmp["status"], "limited", xmp)
        self.assertEqual([bytes.fromhex(b["bytes"]) for b in xmp["blocks"]], [b"prefix"])
        result = self._xmp_limited(path, block_bytes=4 << 20, inflated_bytes=4 << 20, total_bytes=(4 << 20) + 10)
        xmp = result["image_xmp"]
        self.assertEqual(xmp["status"], "collected", xmp)
        self.assertEqual([bytes.fromhex(b["bytes"]) for b in xmp["blocks"]], [b"prefix", b"x" * (4 << 20), b"tail"])

    @unittest.skipUnless(sys.platform.startswith("linux"), "Sparse-file regression uses Linux filesystem semantics")
    def test_source_metadata_xmp_jxl_encoded_size_ceiling(self):
        path = self.root / "sparse-limit.jxl"
        self._xmp_jxl(path)
        with path.open("r+b") as stream:
            stream.truncate((2 << 30) + 1)
        result, _ = self.invoke("source-metadata", path)
        xmp = result["metadata"]["image_xmp"]
        self.assertEqual(xmp["status"], "limited", xmp)
        self.assertIn("encoded-input limit", xmp["diagnostic"])
        self.assertEqual(xmp["blocks"], [])
        self.assertEqual(result["metadata"]["image_exif"]["status"], "limited")
        self.assertEqual(result["metadata"]["image_exif"]["blocks"], [])

    def test_source_metadata_xmp_heif_primary_items(self):
        packet = b'<x xmlns="urn:unknown">\0\xff</x>'
        for carrier in ["avif", "heic"]:
            with self.subTest(carrier=carrier):
                path = self.root / ("XMP primaire é 日本語." + carrier)
                ids = source_exif_fixture.write_heif_metadata(path, [
                    {"data": packet},
                    {"data": b"secondary", "images": [1]},
                    {"data": b"unassociated", "images": []},
                    {"data": packet, "images": [1, 2]},
                    {"data": b"other", "content_type": "application/xml"},
                    {"data": b"parameterized", "content_type": " Application/RDF+XML ; charset=UTF-8"},
                ], carrier=carrier)
                original, stamp = path.read_bytes(), path.stat().st_mtime_ns
                legacy, _ = self.invoke("source-exif", path)
                result, _ = self.invoke("source-metadata", path)
                metadata = result["metadata"]
                self.assertEqual(metadata["image_exif"], legacy["exif"])
                xmp = metadata["image_xmp"]
                self.assertEqual(xmp["status"], "collected", xmp)
                self.assertEqual(xmp["scope"], "heif-primary-image-mime")
                self.assertEqual([b["heif_item"]["metadata_item"] for b in xmp["blocks"]], [ids[0], ids[3], ids[5]])
                self.assertEqual([bytes.fromhex(b["bytes"]) for b in xmp["blocks"]], [packet, packet, b"parameterized"])
                for block in xmp["blocks"]:
                    self.assertEqual(block["envelope"], "heif_mime")
                    self.assertEqual(block["representation"], "libheif_decoded_bytes")
                    self.assertEqual(block["heif_item"]["image_item"], 2)
                    self.assertEqual(block["heif_item"]["content_encoding"], "")
                    self.assertEqual(block["bytes"], block["payload_bytes"])
                self.assertEqual(xmp["blocks"][2]["heif_item"]["content_type"], " Application/RDF+XML ; charset=UTF-8")
                self.assertEqual(path.read_bytes(), original)
                self.assertEqual(path.stat().st_mtime_ns, stamp)

    def test_source_metadata_xmp_heif_provider_compression_policy(self):
        packet = b"<opaque/>" * 1024 + b"\0\xff"
        exif = self.root / "compressed-exif.png"
        source_exif_fixture.write_png(exif)
        png = exif.read_bytes()
        offset = 8
        while png[offset + 4:offset + 8] != b"eXIf":
            offset += 12 + struct.unpack_from(">I", png, offset)[0]
        size = struct.unpack_from(">I", png, offset)[0]
        exif_payload = png[offset + 8:offset + 8 + size]
        for carrier in ("avif", "heic"):
            path = self.root / ("compressed-xmp." + carrier)
            source_exif_fixture.write_heif_metadata(path, carrier=carrier)
            baseline, _ = self.invoke("source-exif", path)
            baseline_exif = [bytes.fromhex(b["bytes"]) for b in baseline["exif"]["blocks"]]
            for encoding, window in (("compress_zlib", 15), ("deflate", -15)):
                with self.subTest(carrier=carrier, encoding=encoding):
                    compressor = zlib.compressobj(wbits=window)
                    compressed = compressor.compress(packet) + compressor.flush()
                    source_exif_fixture.write_heif_metadata(path, [
                        {"type": b"Exif", "data": bytes(4) + exif_payload},
                        {"data": compressed, "encoding": encoding}, {"data": b"later"}], carrier=carrier)
                    original, stamp = path.read_bytes(), path.stat().st_mtime_ns
                    legacy, _ = self.invoke("source-exif", path)
                    result, _ = self.invoke("source-metadata", path)
                    self.assertEqual(result["metadata"]["image_exif"], legacy["exif"])
                    self.assertEqual(legacy["exif"]["status"], "collected", legacy)
                    self.assertEqual([bytes.fromhex(b["bytes"]) for b in legacy["exif"]["blocks"]], baseline_exif + [bytes(4) + exif_payload])
                    xmp = result["metadata"]["image_xmp"]
                    self.assertEqual(xmp["status"], "collected", xmp)
                    self.assertEqual([bytes.fromhex(b["bytes"]) for b in xmp["blocks"]], [packet, b"later"])
                    self.assertEqual(xmp["blocks"][0]["heif_item"]["content_encoding"], encoding)
                    self.assertTrue(all(b["bytes"] == b["payload_bytes"] and not b["compressed"] for b in xmp["blocks"]))
                    self.assertEqual(path.read_bytes(), original)
                    self.assertEqual(path.stat().st_mtime_ns, stamp)
                    for damaged in (compressed[:-1], b"broken", compressed + b"tail"):
                        source_exif_fixture.write_heif_metadata(path, [{"data": damaged, "encoding": encoding}], carrier=carrier)
                        result, _ = self.invoke("source-metadata", path)
                        self.assertEqual(result["metadata"]["image_xmp"]["status"], "malformed", result)

    def test_source_metadata_xmp_heif_inflate_bounds(self):
        path = self.root / "inflate-limit.avif"
        for window, encoding in ((15, "compress_zlib"), (-15, "deflate")):
            for size in (0, (16 << 20), (16 << 20) + 1):
                with self.subTest(encoding=encoding, size=size):
                    compressor = zlib.compressobj(wbits=window)
                    data = compressor.compress(b"x" * size) + compressor.flush()
                    source_exif_fixture.write_heif_metadata(path, [{"data": data, "encoding": encoding}, {"data": b"tail"}])
                    result, _ = self.invoke("source-metadata", path)
                    xmp = result["metadata"]["image_xmp"]
                    if size > (16 << 20):
                        self.assertEqual(xmp["status"], "malformed", xmp)
                        self.assertIn("decompression output limit", xmp["diagnostic"])
                        self.assertEqual(xmp["blocks"], [])
                    elif size:
                        # Provider accepts the exact boundary; default collector skips it and keeps the next item.
                        self.assertEqual(xmp["status"], "limited", xmp)
                        self.assertEqual([bytes.fromhex(b["bytes"]) for b in xmp["blocks"]], [b"tail"])
                    else:
                        self.assertEqual(xmp["status"], "collected", xmp)
                        self.assertEqual([bytes.fromhex(b["bytes"]) for b in xmp["blocks"]], [b"", b"tail"])

    def test_source_metadata_xmp_heif_inflate_aggregate(self):
        path = self.root / "inflate-aggregate.avif"
        for window, encoding in ((15, "compress_zlib"), (-15, "deflate")):
            compressor = zlib.compressobj(wbits=window)
            compressed = compressor.compress(b"x" * (16 << 20)) + compressor.flush()
            for extra in (b"", b"x"):
                with self.subTest(encoding=encoding, extra=len(extra)):
                    compressor = zlib.compressobj(wbits=window)
                    last = compressor.compress(extra) + compressor.flush()
                    items = [{"data": compressed, "encoding": encoding}] * 4
                    items += [{"data": last, "encoding": encoding}, {"data": b"tail"}]
                    source_exif_fixture.write_heif_metadata(path, items)
                    result, _ = self.invoke("source-metadata", path)
                    xmp = result["metadata"]["image_xmp"]
                    if extra:
                        self.assertEqual(xmp["status"], "malformed", xmp)
                        self.assertIn("decompression aggregate limit", xmp["diagnostic"])
                        self.assertEqual(xmp["blocks"], [])
                    else:
                        self.assertEqual(xmp["status"], "limited", xmp)
                        self.assertEqual([bytes.fromhex(b["bytes"]) for b in xmp["blocks"]], [b"", b"tail"])

    def test_source_metadata_xmp_heif_limits_and_absence(self):
        path = self.root / "limited-xmp.avif"
        packet = b"<x/>"
        header = len("application/rdf+xml")
        source_exif_fixture.write_heif_metadata(path, [{"data": packet}, {"data": b""}])
        for budget, count in [(header + len(packet), 2), (header + len(packet) - 1, 1)]:
            result = self._xmp_limited(path, block_bytes=budget)
            self.assertEqual(len(result["image_xmp"]["blocks"]), count)
            self.assertEqual(result["image_xmp"]["status"], "collected" if count == 2 else "limited")
        legacy, _ = self.invoke("source-exif", path)
        for budget, count in [(2 * header + len(packet), 2), (2 * header + len(packet) - 1, 1)]:
            result = self._xmp_limited(path, total_bytes=budget)
            self.assertEqual(len(result["image_xmp"]["blocks"]), count)
            self.assertEqual(result["image_exif"], legacy["exif"])
        result = self._xmp_limited(path, blocks=1)
        self.assertEqual(len(result["image_xmp"]["blocks"]), 1)
        self.assertEqual(result["image_xmp"]["status"], "limited")
        source_exif_fixture.write_heif_metadata(path, [{"data": b"x" * (1536 * 1024)}, {"data": packet}])
        result = self._xmp_limited(path)
        self.assertEqual(result["image_xmp"]["status"], "limited")
        self.assertEqual([bytes.fromhex(b["bytes"]) for b in result["image_xmp"]["blocks"]], [packet])
        result = self._xmp_limited(path, collection_limits={"block_bytes": 1})
        self.assertEqual(result["image_exif"], result["legacy_exif"])
        self.assertEqual(result["image_exif"]["status"], "limited")
        self.assertEqual(bytes.fromhex(result["image_xmp"]["blocks"][0]["bytes"]), packet)
        source_exif_fixture.write_heif_metadata(path, [{"data": packet, "images": [1]},
                                                     {"data": packet, "content_type": "application/xml"}])
        result, _ = self.invoke("source-metadata", path)
        self.assertEqual(result["metadata"]["image_xmp"]["status"], "absent")

    def test_source_metadata_xmp_heif_count_index_and_header_budgets(self):
        path = self.root / "many-xmp.avif"
        ids = source_exif_fixture.write_heif_metadata(path, [{"data": str(i).encode()} for i in range(70)])
        result, _ = self.invoke("source-metadata", path)
        xmp = result["metadata"]["image_xmp"]
        self.assertEqual(xmp["status"], "limited")
        self.assertEqual(len(xmp["blocks"]), 64)
        self.assertEqual([b["heif_item"]["metadata_item"] for b in xmp["blocks"]], ids[:64])
        indexed = self._xmp_limited(path, records=0)
        self.assertEqual(indexed["index_status"], "limited")
        self.assertEqual(indexed["records"], [])
        self.assertEqual(indexed["image_xmp"], xmp)
        source_exif_fixture.write_heif_metadata(path, [
            {"data": b"first", "content_type": "application/rdf+xml; padding=" + "x" * 1024},
            {"data": b"last"},
        ])
        limited = self._xmp_limited(path, block_bytes=128)["image_xmp"]
        self.assertEqual(limited["status"], "limited")
        self.assertEqual([bytes.fromhex(b["bytes"]) for b in limited["blocks"]], [b"last"])
        source_exif_fixture.write_heif_metadata(path, [{"data": b"payload"}])
        path.write_bytes(path.read_bytes()[:-3])
        legacy, _ = self.invoke("source-exif", path)
        result, _ = self.invoke("source-metadata", path)
        self.assertEqual(result["metadata"]["image_exif"], legacy["exif"])
        self.assertEqual(result["metadata"]["image_xmp"]["status"], "malformed")
        self.assertTrue(result["metadata"]["image_xmp"]["diagnostic"])

    def test_source_metadata_xmp_heif_mime_bytes_and_count_boundary(self):
        path = self.root / "mime-boundary.avif"
        content_type = b"application/rdf+xml; title=Z"
        source_exif_fixture.write_heif_metadata(path, [{"data": b"packet", "content_type": content_type.decode()}])
        data = path.read_bytes()
        self.assertEqual(data.count(content_type), 1)
        path.write_bytes(data.replace(content_type, content_type[:-1] + b"\xff"))
        result, _ = self.invoke("source-metadata", path)
        mime = result["metadata"]["image_xmp"]["blocks"][0]["heif_item"]
        self.assertEqual(bytes.fromhex(mime["content_type_bytes"]), content_type[:-1] + b"\xff")
        self.assertEqual(mime["content_encoding_bytes"], "")
        self.assertTrue(mime["content_type"].endswith("\ufffd"))
        for extra_xmp in (False, True):
            items = [{"data": b"packet"} for _ in range(64)]
            items += [{"data": b"unrelated", "content_type": "application/xml"}]
            if extra_xmp:
                items += [{"data": b"omitted"}]
            source_exif_fixture.write_heif_metadata(path, items)
            result, _ = self.invoke("source-metadata", path)
            xmp = result["metadata"]["image_xmp"]
            self.assertEqual(len(xmp["blocks"]), 64)
            self.assertEqual(xmp["status"], "limited" if extra_xmp else "collected")
        for count in (997, 998):
            source_exif_fixture.write_heif_metadata(path, [{"data": b"x"} for _ in range(count)])
            result, _ = self.invoke("source-metadata", path)
            xmp = result["metadata"]["image_xmp"]
            self.assertEqual(xmp["status"], "limited" if count == 997 else "malformed")
            self.assertEqual(len(xmp["blocks"]), 64 if count == 997 else 0)
            if count == 998:
                self.assertIn("security limit", xmp["diagnostic"].lower())

    def test_source_metadata_xmp_heif_provider_failures(self):
        path = self.root / "invalid-xmp.avif"
        for data, encoding in [(b"broken", "compress_zlib"), (b"unknown", "unknown")]:
            with self.subTest(encoding=encoding):
                source_exif_fixture.write_heif_metadata(path, [{"data": data, "encoding": encoding}])
                legacy, _ = self.invoke("source-exif", path)
                result, _ = self.invoke("source-metadata", path)
                metadata = result["metadata"]
                self.assertEqual(metadata["image_exif"], legacy["exif"])
                self.assertEqual(metadata["image_xmp"]["status"], "malformed")
                self.assertTrue(metadata["image_xmp"]["diagnostic"])
                self.assertEqual(metadata["image_xmp"]["blocks"], [])

    def test_source_metadata_xmp_tiff_byte_packets_and_directory_identity(self):
        path = self.root / "XMP TIFF é 日本語.tiff"
        for little in [True, False]:
            for big in [False, True]:
                for packet in [b"a", b"<x/>", b"abcdefgh", b'<x xmlns="urn:vendor">\0\xffopaque</x>' + b" " * 37]:
                    with self.subTest(little=little, big=big, size=len(packet)):
                        packets = [packet, None, packet]
                        source_exif_fixture.write_tiff(path, little=little, big=big, pages=3, xmp_packets=packets)
                        before, stamp = path.read_bytes(), path.stat().st_mtime_ns
                        legacy, _ = self.invoke("source-exif", path)
                        result, stderr = self.invoke("source-metadata", path)
                        self.assertEqual(stderr, b"")
                        metadata = result["metadata"]
                        self.assertEqual(metadata["image_exif"], legacy["exif"])
                        xmp = metadata["image_xmp"]
                        self.assertEqual(xmp["status"], "collected", xmp)
                        self.assertEqual(xmp["scope"], "tiff-libtiff-observed-tag700")
                        blocks = xmp["blocks"]
                        self.assertEqual(len(blocks), 2)
                        self.assertEqual([bytes.fromhex(b["bytes"]) for b in blocks], [packet, packet])
                        for block, directory_index in zip(blocks, [0, 2]):
                            self.assertEqual(block["envelope"], "tiff_tag700")
                            self.assertEqual(block["representation"], "libtiff_decoded_bytes")
                            self.assertEqual(bytes.fromhex(block["payload_bytes"]), packet)
                            self.assertEqual(block["payload_offset"], 0)
                            self.assertIsNone(block["crc_valid"])
                            ref = block["tiff_tag"]
                            self.assertEqual(ref["directory"], directory_index)
                            directory = metadata["image_exif"]["tiff"]["directories"][directory_index]
                            self.assertEqual(ref["directory_offset"], directory["offset"])
                            self.assertEqual(directory["tags"][ref["tag"]]["id"], 700)
                            self.assertEqual(bytes.fromhex(directory["tags"][ref["tag"]]["bytes"]), packet)
                        self.assertEqual(path.read_bytes(), before)
                        self.assertEqual(path.stat().st_mtime_ns, stamp)
        packet = b"<subimage/>"
        source_exif_fixture.write_tiff_graph(path, blobs=[(700, packet)], subimage_blobs=[(700, b"<child/>")])
        result, _ = self.invoke("source-metadata", path)
        metadata = result["metadata"]
        directories = metadata["image_exif"]["tiff"]["directories"]
        packets = {directories[b["tiff_tag"]["directory"]]["kind"]: bytes.fromhex(b["bytes"])
                   for b in metadata["image_xmp"]["blocks"]}
        self.assertEqual(packets, {"image": packet, "subimage": b"<child/>"})

    def test_source_metadata_xmp_tiff_limits_and_observation_failures(self):
        path = self.root / "xmp-budget.tif"
        packet = b"<x/>" + b" " * 4096
        source_exif_fixture.write_tiff(path, pages=2, xmp_packets=[packet, packet])
        for limits, count in [({"blocks": 1}, 1), ({"block_bytes": len(packet) - 1}, 0),
                              ({"total_bytes": 2 * len(packet) - 1}, 1)]:
            observed = self._xmp_limited(path, **limits)
            self.assertEqual(observed["image_xmp"]["status"], "limited")
            self.assertEqual(len(observed["image_xmp"]["blocks"]), count)
            self.assertEqual(sum(tag["id"] == 700 for d in observed["image_exif"]["tiff"]["directories"] for tag in d["tags"]), 2)
        exact = self._xmp_limited(path, blocks=2, block_bytes=len(packet), total_bytes=2 * len(packet), inflated_bytes=0)
        self.assertEqual(exact["image_xmp"]["status"], "collected")
        tiff_records = sum(record["kind"] != "xmp_block" for record in exact["records"])
        omitted = self._xmp_limited(path, records=tiff_records)
        self.assertEqual(omitted["index_status"], "limited")
        self.assertEqual(omitted["test_summary"]["facts"], exact["test_summary"]["facts"])
        no_index = self._xmp_limited(path, records=0)
        self.assertEqual(no_index["records"], [])
        self.assertEqual(len(no_index["image_xmp"]["blocks"]), 2)
        for limits in [{"tiff_directories": 1}, {"tiff_entries": 1}, {"tiff_value_bytes": 1},
                       {"tiff_read_bytes": 16}, {"tiff_reads": 1}, {"tiff_reads": 0}]:
            observed = self._xmp_limited(path, collection_limits=limits)
            self.assertEqual(observed["image_exif"], observed["legacy_exif"])
            self.assertEqual(observed["image_xmp"]["status"], "limited", limits)
        source_exif_fixture.write_tiff(path)
        result, _ = self.invoke("source-metadata", path)
        self.assertEqual(result["metadata"]["image_xmp"]["status"], "absent")
        source_exif_fixture.write_tiff(path, xmp_packets=[packet])
        path.write_bytes(path.read_bytes()[:32])
        result, _ = self.invoke("source-metadata", path)
        self.assertEqual(result["metadata"]["image_xmp"]["status"], "malformed")
        self.assertEqual(result["metadata"]["image_xmp"]["blocks"], [])
        source_exif_fixture.write_tiff(path, cycle=True, xmp_packets=[b"<x/>"])
        result, _ = self.invoke("source-metadata", path)
        self.assertNotEqual(result["metadata"]["image_xmp"]["status"], "absent")

    def test_source_metadata_xmp_tiff_provider_representation(self):
        path = self.root / "xmp-provider.tif"
        for little in [True, False]:
            order = "<" if little else ">"
            for type_, encoded, decoded in [(2, b"<x/>\0\xff", b"<x/>\0\xff"),
                                            (7, b"<x/>\0\xff", b"<x/>\0\xff"),
                                            (3, struct.pack(order + "4H", 60, 120, 47, 62), b"<x/>")]:
                with self.subTest(little=little, type=type_):
                    source_exif_fixture.write_tiff(path, little=little, xmp_packets=[encoded], xmp_type=type_)
                    result, _ = self.invoke("source-metadata", path)
                    block = result["metadata"]["image_xmp"]["blocks"][0]
                    self.assertEqual(bytes.fromhex(block["bytes"]), decoded)
                    self.assertEqual(block["representation"], "libtiff_decoded_bytes")
                    for span in result["metadata"]["image_exif"]["tiff"]["source_ranges"]:
                        raw = bytes.fromhex(span["bytes"])
                        self.assertEqual(path.read_bytes()[span["offset"]:span["offset"] + len(raw)], raw)
        source_exif_fixture.write_tiff(path, pages=2, xmp_packets=[b"", b"<last/>"])
        result, _ = self.invoke("source-metadata", path)
        self.assertEqual(result["metadata"]["image_xmp"]["status"], "malformed")
        self.assertTrue(result["metadata"]["image_xmp"]["diagnostic"])
        block = result["metadata"]["image_xmp"]["blocks"][0]
        self.assertEqual(block["bytes"], "")
        self.assertEqual(block["status"], "malformed")
        self.assertTrue(block["diagnostic"])
        self.assertIsNone(block["payload_bytes"])
        self.assertEqual(bytes.fromhex(result["metadata"]["image_xmp"]["blocks"][1]["payload_bytes"]), b"<last/>")
        source_exif_fixture.write_tiff_graph(path, blobs=[(700, b"first"), (700, b"second")])
        result, _ = self.invoke("source-metadata", path)
        metadata = result["metadata"]
        observed = [tag["bytes"] for directory in metadata["image_exif"]["tiff"]["directories"]
                    for tag in directory["tags"] if tag["id"] == 700]
        self.assertEqual([block["bytes"] for block in metadata["image_xmp"]["blocks"]], observed)

    def test_source_metadata_xmp_tiff_main_ifd_numeric_types(self):
        path = self.root / "xmp-numeric.tif"
        for little in [True, False]:
            order = "<" if little else ">"
            for big in [False, True]:
                for type_, encoded, payload in [
                        (3, struct.pack(order + "4H", 60, 63, 120, 112), b"<?xp"),
                        (4, struct.pack(order + "4I", 60, 63, 120, 112), b"<?xp"),
                        (3, struct.pack(order + "4H", 60, 256, 120, 112), None),
                        (11, struct.pack(order + "4f", 60, 63, 120, 112), None),
                        (12, struct.pack(order + "4d", 60, 63, 120, 112), None)]:
                    with self.subTest(little=little, big=big, type=type_, payload=payload):
                        source_exif_fixture.write_tiff(path, little=little, big=big,
                            xmp_packets=[encoded], xmp_type=type_)
                        original, modified = path.read_bytes(), path.stat().st_mtime_ns
                        legacy, _ = self.invoke("source-exif", path)
                        result, _ = self.invoke("source-metadata", path)
                        metadata = result["metadata"]
                        self.assertEqual(metadata["image_exif"], legacy["exif"])
                        xmp = metadata["image_xmp"]
                        if payload is None:
                            self.assertEqual(metadata["image_exif"]["status"], "malformed")
                            self.assertEqual(xmp["status"], "malformed")
                            self.assertIn("tag ignored", xmp["diagnostic"].lower())
                            self.assertEqual(xmp["blocks"], [])
                        else:
                            self.assertEqual(xmp["status"], "collected")
                            self.assertEqual(len(xmp["blocks"]), 1)
                            block = xmp["blocks"][0]
                            self.assertEqual(bytes.fromhex(block["payload_bytes"]), payload)
                            ref = block["tiff_tag"]
                            tag = metadata["image_exif"]["tiff"]["directories"][ref["directory"]]["tags"][ref["tag"]]
                            self.assertEqual(tag["type"], 1)
                        self.assertEqual(path.read_bytes(), original)
                        self.assertEqual(path.stat().st_mtime_ns, modified)

    def test_source_metadata_xmp_tiff_child_byte_representations(self):
        path = self.root / "xmp-child.tif"
        packet = b"<child/>\0\xff"
        for little in [True, False]:
            for big in [False, True]:
                for kind in ["exif", "gps", "interoperability"]:
                    for type_ in [1, 7]:
                        with self.subTest(little=little, big=big, kind=kind, type=type_):
                            source_exif_fixture.write_tiff_graph(path, little=little, big=big,
                                directory_tags={kind: [(700, type_, len(packet), packet)]})
                            legacy, _ = self.invoke("source-exif", path)
                            result, _ = self.invoke("source-metadata", path)
                            metadata = result["metadata"]
                            self.assertEqual(metadata["image_exif"], legacy["exif"])
                            xmp = metadata["image_xmp"]
                            self.assertEqual(xmp["status"], "collected", xmp)
                            self.assertEqual(len(xmp["blocks"]), 1)
                            block = xmp["blocks"][0]
                            self.assertEqual(bytes.fromhex(block["payload_bytes"]), packet)
                            ref = block["tiff_tag"]
                            directory = metadata["image_exif"]["tiff"]["directories"][ref["directory"]]
                            self.assertEqual(directory["kind"], kind)
                            self.assertEqual(directory["tags"][ref["tag"]]["type"], type_)

    def test_source_metadata_xmp_tiff_scan_failure_overrides_packet_support(self):
        path = self.root / "xmp-incomplete.tif"
        packet = b"<child/>"
        for type_, encoded in [(3, struct.pack("<4H", 60, 120, 47, 62)), (7, packet)]:
            source_exif_fixture.write_tiff_graph(path,
                directory_tags={"gps": [(700, type_, len(encoded) // (2 if type_ == 3 else 1), encoded)]})
            full = self._xmp_limited(path)
            self.assertEqual(full["image_xmp"]["status"], "collected" if type_ == 7 else "unsupported")
            limited = self._xmp_limited(path, collection_limits={"tiff_directories": 4})
            self.assertEqual(limited["image_exif"], limited["legacy_exif"])
            self.assertEqual(limited["image_exif"]["status"], "limited")
            self.assertEqual(limited["image_xmp"]["status"], "limited")
            self.assertIn("TIFF directory budget exhausted", limited["image_xmp"]["diagnostic"])
            if type_ == 7:
                self.assertEqual(bytes.fromhex(limited["image_xmp"]["blocks"][0]["payload_bytes"]), packet)
            else:
                self.assertIn("no byte-preserving representation", limited["image_xmp"]["diagnostic"])
            # An invalid later pointer leaves the earlier GPS packet intact.
            data = bytearray(path.read_bytes())
            directory = next(d for d in full["image_exif"]["tiff"]["directories"] if d["kind"] == "exif")
            offset = directory["offset"]
            for index in range(struct.unpack_from("<H", data, offset)[0]):
                entry = offset + 2 + index * 12
                if struct.unpack_from("<H", data, entry)[0] == 40965:
                    struct.pack_into("<I", data, entry + 8, len(data) + 1024)
                    break
            else:
                self.fail("Fixture must contain an interoperability pointer")
            path.write_bytes(data)
            damaged = self._xmp_limited(path)
            self.assertEqual(damaged["image_exif"]["status"], "malformed")
            self.assertEqual(damaged["image_xmp"]["status"], "malformed")
            if type_ == 3:
                self.assertIn("no byte-preserving representation", damaged["image_xmp"]["diagnostic"])
            else:
                self.assertEqual(bytes.fromhex(damaged["image_xmp"]["blocks"][0]["payload_bytes"]), packet)

    def test_source_metadata_xmp_heif_invalid_container(self):
        path = self.root / "header-only.heic"
        path.write_bytes(struct.pack(">I", 20) + b"ftypheic" + bytes(4) + b"heic")
        result = self._xmp_limited(path)
        self.assertEqual(result["image_exif"]["status"], "malformed")
        self.assertEqual(result["image_xmp"]["status"], "malformed")
        self.assertEqual(result["image_xmp"]["blocks"], [])

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
        embedded = self.root / "native-byte-jpeg"
        self.invoke("extract", path, "--output", embedded, "--format", "jpeg",
                    "--interval", "1", "--end", "0.35", "--embed-source-metadata")
        spec = importlib.util.spec_from_file_location("jpeg_metadata_fixture", ROOT / "tests/media/jpeg_metadata_fixture.py")
        reader = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(reader)
        images = list(embedded.glob("*.jpg"))
        self.assertTrue(images)
        _, _, _, _, retained, _ = reader.read(images[0].read_bytes())
        self.assertEqual(bytes.fromhex(retained["metadata"]["source_locator_bytes"]), native)

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
    def test_source_metadata_preserved_in_extraction_manifest(self):
        directory = self.root / "source-metadata-manifest"
        result, _ = self.invoke("extract", self.source(), "--output", directory,
                                "--interval", "1", "--end", "0.2", "--preserve-source-metadata")
        self.assertTrue(result["success"])
        manifest = json.loads((directory / "extraction_metadata.json").read_text(encoding="utf-8"))
        self.assertEqual(manifest["source_metadata"]["metadata"]["source_locator"], str(self.source()))
        self.assertEqual(manifest["source_metadata_policy"], "preserve_static_container_v1")
        self.assertEqual(manifest["source_metadata"]["metadata"]["scope"], "container_stream")
        self.assertEqual(manifest["source_metadata"]["metadata"]["source_metadata"]["provider"]["name"], "ffmpeg")
        self.assertEqual(manifest["source_metadata"]["summary"]["scope"], "static_stream_properties")
    def test_embedded_jpeg_metadata_cli(self):
        directory = self.root / "embedded-jpeg"
        result, _ = self.invoke("extract", self.source(), "--output", directory,
                                "--format", "jpg", "--interval", "1", "--end", "0.2", "--embed-source-metadata")
        self.assertTrue(result["success"])
        manifest = json.loads((directory / "extraction_metadata.json").read_text(encoding="utf-8"))
        self.assertEqual(manifest["embedded_metadata_policy"], "jpeg-static-source-xmp-v1")
        self.assertTrue(list(directory.glob("*.jpg")))
        self.assertIn(b"Exif\0\0", next(directory.glob("*.jpg")).read_bytes())

    def test_embedded_jxl_metadata_cli(self):
        import xml.etree.ElementTree as ET
        directory = self.root / "embedded-jxl"
        source = self.root / "jxl-source-tags.mkv"
        subprocess.run([FFMPEG, "-v", "error", "-y", "-i", str(self.source()), "-map", "0:v:0",
                        "-c:v", "copy", "-metadata", "comment=" + 'A<&"' * 4000,
                        "-write_crc32", "0", str(source)], check=True, capture_output=True, timeout=30)
        result, _ = self.invoke("extract", source, "--output", directory,
                                "--format", "jxl", "--input-transfer", "linear", "--input-primaries", "bt709", "--interval", "1", "--end", "0.2", "--embed-source-metadata")
        self.assertTrue(result["success"])
        manifest = json.loads((directory / "extraction_metadata.json").read_text(encoding="utf-8"))
        self.assertEqual(manifest["embedded_metadata_policy"], "jxl-static-source-xmp-v1")
        paths = list(directory.glob("*.jxl"))
        self.assertTrue(paths)
        for path in paths:
            _, _, packets = source_exif_fixture.read_export_jxl(path.read_bytes())
            self.assertEqual(len(packets), 1)
            self.assertGreater(len(packets[0]), 65535)
            facts = {node.tag.rsplit("}", 1)[-1]: node.text for node in ET.fromstring(packets[0]).iter()
                     if node.tag.startswith("{https://lichtfeld.io/ns/media/1.0/}")}
            self.assertEqual(json.loads(facts["SourceMetadata"]), manifest["source_metadata"])

    def test_source_metadata_sidecars_cli(self):
        import xml.etree.ElementTree as ET
        directory = self.root / "source-sidecars é 日本語"
        result, _ = self.invoke("extract", self.source(), "--output", directory,
                                "--format", "png", "--interval", "1", "--end", "0.2", "--source-metadata-sidecars")
        self.assertTrue(result["success"])
        manifest = json.loads((directory / "extraction_metadata.json").read_text(encoding="utf-8"))
        self.assertEqual(manifest["sidecar_metadata_policy"], "image-static-source-xmp-v1")
        self.assertTrue(manifest["frames"])
        before = {}
        for entry in manifest["frames"]:
            self.assertEqual(entry["sidecar"], entry["file"] + ".xmp")
            image, sidecar = directory / entry["file"], directory / entry["sidecar"]
            ET.fromstring(sidecar.read_bytes())
            before[image.name] = image.read_bytes(), image.stat().st_mtime_ns
            before[sidecar.name] = sidecar.read_bytes(), sidecar.stat().st_mtime_ns
        result, _ = self.invoke("extract", self.source(), "--output", directory,
                                "--format", "png", "--interval", "1", "--end", "0.2", "--source-metadata-sidecars")
        self.assertTrue(result["success"])
        self.assertEqual(before, {name: ((directory / name).read_bytes(), (directory / name).stat().st_mtime_ns) for name in before})

    def test_source_metadata_sidecars_cli_recovers_and_refuses_corruption(self):
        directory = self.root / "sidecar-cli-recovery"
        args = ("extract", self.source(), "--output", directory, "--format", "png",
                "--interval", "1", "--end", "0.01", "--source-metadata-sidecars")
        self.invoke(*args)
        manifest = json.loads((directory / "extraction_metadata.json").read_text(encoding="utf-8"))
        self.assertTrue(manifest["frames"])
        entry = manifest["frames"][0]
        image, sidecar = directory / entry["file"], directory / entry["sidecar"]
        image_snapshot = image.read_bytes(), image.stat().st_mtime_ns
        packet = sidecar.read_bytes()
        sidecar.unlink()
        self.invoke(*args)
        self.assertEqual(image_snapshot, (image.read_bytes(), image.stat().st_mtime_ns))
        self.assertEqual(sidecar.read_bytes(), packet)
        sidecar_snapshot = sidecar.read_bytes(), sidecar.stat().st_mtime_ns
        image.unlink()
        self.invoke(*args)
        self.assertEqual(image.read_bytes(), image_snapshot[0])
        self.assertEqual(sidecar_snapshot, (sidecar.read_bytes(), sidecar.stat().st_mtime_ns))
        sidecar.write_bytes(packet + b" ")
        image_snapshot = image.read_bytes(), image.stat().st_mtime_ns
        result, _ = self.invoke(*args, code=4)
        self.assertEqual(result["error"]["code"], "FailedPrecondition")
        self.assertEqual(sidecar.read_bytes(), packet + b" ")
        self.assertEqual(image_snapshot, (image.read_bytes(), image.stat().st_mtime_ns))

    def test_embedded_avif_metadata_cli(self):
        import xml.etree.ElementTree as ET
        directory = self.root / "embedded-avif"
        source = self.root / "avif-source-tags.mkv"
        subprocess.run([FFMPEG, "-v", "error", "-y", "-i", str(self.source()), "-map", "0:v:0",
                        "-c:v", "copy", "-metadata", "comment=" + 'A<&"' * 4000,
                        "-write_crc32", "0", str(source)], check=True, capture_output=True, timeout=30)
        result, _ = self.invoke("extract", source, "--output", directory,
                                "--format", "avif", "--input-transfer", "linear", "--input-primaries", "bt709", "--interval", "1", "--end", "0.2", "--embed-source-metadata")
        self.assertTrue(result["success"])
        manifest = json.loads((directory / "extraction_metadata.json").read_text(encoding="utf-8"))
        self.assertEqual(manifest["embedded_metadata_policy"], "avif-static-source-xmp-v1")
        paths = list(directory.glob("*.avif"))
        self.assertTrue(paths)
        for path in paths:
            _, _, packets = source_exif_fixture.read_export_avif(path.read_bytes())
            self.assertEqual(len(packets), 1)
            self.assertGreater(len(packets[0]), 65535)
            facts = {node.tag.rsplit("}", 1)[-1]: node.text for node in ET.fromstring(packets[0]).iter()
                     if node.tag.startswith("{https://lichtfeld.io/ns/media/1.0/}")}
            self.assertEqual(json.loads(facts["SourceMetadata"]), manifest["source_metadata"])

    def test_embedded_png_metadata_cli_and_md5_independence(self):
        source = self.root / "png-source-tags.mkv"
        subprocess.run([FFMPEG, "-v", "error", "-y", "-i", str(self.source()), "-map", "0:v:0",
                        "-c:v", "copy", "-metadata", "comment=" + 'A<&"' * 4000,
                        "-write_crc32", "0", str(source)], check=True, capture_output=True, timeout=30)
        config = self.root / "png-no-md5.cnf"
        config.write_text('openssl_conf = init\n[init]\nalg_section = algorithms\n[algorithms]\ndefault_properties = fips=yes\n', encoding="utf-8")
        output = self.root / "embedded-png"
        result = subprocess.run([str(CLI), "extract", str(source), "--output", str(output),
                                 "--format", "png", "--interval", "1", "--end", "0.2", "--embed-source-metadata"],
                                capture_output=True, env=dict(os.environ, OPENSSL_CONF=str(config)), timeout=30)
        self.assertEqual(result.returncode, 0, result.stderr)
        manifest = json.loads((output / "extraction_metadata.json").read_text(encoding="utf-8"))
        self.assertEqual(manifest["embedded_metadata_policy"], "png-static-source-xmp-v1")
        spec = importlib.util.spec_from_file_location("png_metadata_fixture", ROOT / "tests/media/png_metadata_fixture.py")
        reader = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(reader)
        files = list(output.glob("*.png"))
        self.assertTrue(files)
        for image in files:
            _, _, _, catalog = reader.read(image.read_bytes())
            self.assertEqual(catalog, manifest["source_metadata"])
            self.assertGreater(image.stat().st_size, 65535)

    def test_embedded_tiff_metadata_cli(self):
        import xml.etree.ElementTree as ET
        source = self.root / "tiff-source-tags.mkv"
        subprocess.run([FFMPEG, "-v", "error", "-y", "-i", str(self.source()), "-map", "0:v:0",
                        "-c:v", "copy", "-metadata", "comment=" + 'A<&"' * 4000,
                        "-write_crc32", "0", str(source)], check=True, capture_output=True, timeout=30)
        result, _ = self.invoke("extract", source, "--output", self.root / "embedded-tiff", "--format", "tiff16", "--input-transfer", "linear", "--input-primaries", "bt709",
                                "--interval", "1", "--end", "0.2", "--embed-source-metadata")
        self.assertTrue(result["success"])
        manifest = json.loads((self.root / "embedded-tiff/extraction_metadata.json").read_text(encoding="utf-8"))
        self.assertEqual(manifest["embedded_metadata_policy"], "tiff16-static-source-xmp-v1")
        paths = list((self.root / "embedded-tiff").glob("*.tif"))
        self.assertTrue(paths)
        for path in paths:
            _, _, packet = source_exif_fixture.read_export_tiff(path.read_bytes())
            facts = {node.tag.rsplit("}", 1)[-1]: node.text for node in ET.fromstring(packet).iter()
                     if node.tag.startswith("{https://lichtfeld.io/ns/media/1.0/}")}
            self.assertEqual(json.loads(facts["SourceMetadata"]), manifest["source_metadata"])
            self.assertGreater(len(packet), 65535)

    def test_embedded_exr_metadata_cli(self):
        import xml.etree.ElementTree as ET
        source = self.root / "exr-source-tags.mkv"
        subprocess.run([FFMPEG, "-v", "error", "-y", "-i", str(self.source()), "-map", "0:v:0",
                        "-c:v", "copy", "-metadata", "comment=" + 'A<&"' * 4000,
                        "-write_crc32", "0", str(source)], check=True, capture_output=True, timeout=30)
        result, _ = self.invoke("extract", source, "--output", self.root / "embedded-exr", "--format", "exr",
                                "--input-transfer", "linear", "--input-primaries", "bt709",
                                "--interval", "1", "--end", "0.2", "--embed-source-metadata")
        self.assertTrue(result["success"])
        manifest = json.loads((self.root / "embedded-exr/extraction_metadata.json").read_text(encoding="utf-8"))
        self.assertEqual(manifest["embedded_metadata_policy"], "exr-static-source-xmp-v1")
        paths = list((self.root / "embedded-exr").glob("*.exr"))
        self.assertTrue(paths)
        for path in paths:
            _, _, packet = source_exif_fixture.read_export_exr(path.read_bytes())
            facts = {node.tag.rsplit("}", 1)[-1]: node.text for node in ET.fromstring(packet).iter()
                     if node.tag.startswith("{https://lichtfeld.io/ns/media/1.0/}")}
            self.assertEqual(json.loads(facts["SourceMetadata"]), manifest["source_metadata"])
            self.assertGreater(len(packet), 65535)

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
