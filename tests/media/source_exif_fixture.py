# SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
# SPDX-License-Identifier: GPL-3.0-or-later
"""One independent TIFF fixture shared by the headless consumer contracts."""
import struct
import base64
import zlib
from pathlib import Path


def write_png(path, *, post_idat=False, duplicate=False, idat_chunks=1, damaged_ancillary=False, include_exif=True):
    tiff = b"II" + struct.pack("<HIH", 42, 8, 1)
    tiff += struct.pack("<HHIHHI", 0x112, 3, 1, 6, 0, 0)

    def chunk(kind, data):
        return struct.pack(">I", len(data)) + kind + data + struct.pack(">I", zlib.crc32(kind + data))

    header = chunk(b"IHDR", struct.pack(">IIBBBBB", 1, 1, 8, 2, 0, 0, 0))
    pixels = chunk(b"IDAT", b"") * (idat_chunks - 1) + chunk(b"IDAT", zlib.compress(b"\0\x14\x50\xa0"))
    metadata = chunk(b"eXIf", tiff) * (2 if duplicate else 1) if include_exif else b""
    ancillary = bytearray(chunk(b"tEXt", b"key\0value")) if damaged_ancillary else bytearray()
    if ancillary:
        ancillary[-1] ^= 1
    path.write_bytes(b"\x89PNG\r\n\x1a\n" + header + ancillary + (pixels + metadata if post_idat else metadata + pixels) + chunk(b"IEND", b""))
    return tiff


def check_png(test, result, payload, *, count=1, status="collected"):
    test.assertEqual(result["status"], status)
    test.assertEqual(result["scope"], "png-chunks-through-iend")
    test.assertEqual(len(result["blocks"]), count)
    for block in result["blocks"]:
        test.assertEqual(block["envelope"], "png_exif")
        test.assertEqual(bytes.fromhex(block["bytes"]), payload)
        test.assertIsNone(block["image_item"])
        test.assertIsNone(block["metadata_item"])
        test.assertEqual(block["interpretation"]["status"], "parsed")
        entry, = block["interpretation"]["entries"]
        test.assertEqual((entry["id"], entry["type"], entry["unsigned_values"]), (0x112, 3, [6]))


def write_jpeg(path):
    # Original procedural 16x16 red JPEG, generated with FFmpeg 9.0.2:
    # -f lavfi -i color=c=red:s=16x16 -frames:v 1 -c:v mjpeg
    # No runtime encoder or external image is required for metadata-only tests.
    original = base64.b64decode(
        "/9j/4AAQSkZJRgABAgAAAQABAAD//gAPTGF2YzYzLjEuMTAyAP/bAEMACAQEBAQEBQUFBQUFBgYGBgYGBgYGBgYGBgcHBwgICAcHBwYGBwcICAgICQkJCAgICAkJCgoKDAwLCw4ODhERFP/EAEwAAQEAAAAAAAAAAAAAAAAAAAAGAQEBAAAAAAAAAAAAAAAAAAAGBxABAAAAAAAAAAAAAAAAAAAAABEBAAAAAAAAAAAAAAAAAAAAAP/AABEIABAAEAMBIgACEQADEQD/2gAMAwEAAhEDEQA/AIsATX9//9k="
    )
    # A little-endian classic TIFF with orientation=6; no pixel transformation.
    tiff = b"II" + struct.pack("<HIH", 42, 8, 1)
    tiff += struct.pack("<HHIHHI", 0x112, 3, 1, 6, 0, 0)
    payload = b"Exif\0\0" + tiff
    marker = b"\xff\xe1" + struct.pack(">H", len(payload) + 2) + payload
    path.write_bytes(original[:2] + marker + marker + original[2:])
    return payload


def check(test, result, payload):
    test.assertEqual(result["status"], "collected")
    test.assertEqual(result["scope"], "jpeg-before-sos")
    test.assertEqual(result["binary_encoding"], "hex")
    test.assertEqual(len(result["blocks"]), 2)
    for block in result["blocks"]:
        test.assertEqual(bytes.fromhex(block["bytes"]), payload)
        test.assertIsNone(block["image_item"])
        test.assertEqual(block["interpretation"]["status"], "parsed")
        entry, = block["interpretation"]["entries"]
        test.assertEqual((entry["id"], entry["type"], entry["unsigned_values"]), (0x112, 3, [6]))


def write_header_failure(path, long_header=False):
    payload = write_jpeg(path)
    original = path.read_bytes()
    prefix_end = 2 + 2 * (len(payload) + 4)
    extra = (b"\xff\xe2\xff\xff" + b"\0" * 65533) * 21 if long_header else b"\x01\x02\x03"
    changed = original[:prefix_end] + extra + original[prefix_end:]
    path.write_bytes(changed + b"\0" * (9 * 1024 * 1024 - len(changed)))
    return payload


def write_invalid_heif(path):
    # Keep the existing essential irot property's size/association intact, but
    # make its fourcc unknown and invalid UTF-8 in libheif's diagnostic.
    original = bytearray((Path(__file__).resolve().parents[1] / "data/heif/rotated.avif").read_bytes())
    original[original.index(b"irot")] = 255
    path.write_bytes(original)
