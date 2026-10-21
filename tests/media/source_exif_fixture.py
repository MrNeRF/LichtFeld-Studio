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


def write_camera_metadata(path, *, carrier="png", little=True, big=False, padded=False, nul_padding=None):
    """Independent byte-level camera/GPS fixture, including nested IFDs."""
    if big and carrier != "tiff":
        raise ValueError("EXIF envelopes use classic TIFF")
    order = "<" if little else ">"
    slot, count_width, entry_width = (8, 8, 20) if big else (4, 2, 12)
    byte_order = "little" if little else "big"
    integer = lambda n: n.to_bytes(slot, byte_order)
    short = lambda n: struct.pack(order + "H", n)
    long = lambda n: struct.pack(order + "I", n)
    rational = lambda n, d: struct.pack(order + "II", n, d)
    header_size = 16 if big else 8
    directories = [
        [(256, 4, 1, long(1)), (257, 4, 1, long(1)),
         (258, 3, 3, short(8) * 3), (259, 3, 1, short(1)),
         (262, 3, 1, short(2)), (271, 2, 12, b"Test Camera\0"),
         (272, 2, 8, b"Model A\0"), (273, 16 if big else 4, 1, b""),
         (274, 3, 1, short(6)), (277, 3, 1, short(3)),
         (278, 4, 1, long(1)), (279, 4, 1, long(3)),
         (0x8769, 18 if big else 4, 1, b""),
         (0x8825, 18 if big else 4, 1, b"")],
        [(0x829a, 5, 1, rational(1, 125)),
         (0x829d, 5, 1, rational(28, 10)),
         (0x8827, 3, 1, short(200)), (0x9000, 7, 4, b"0232"),
         (0x9003, 2, 20, b"2024:02:29 12:34:56\0"),
         (0x9011, 2, 7, b"+02:30\0"),
         (0x9101, 7, 4, b"\x01\x02\x03\0"),
         (0x920a, 5, 1, rational(350, 10)),
         (0x9291, 2, 7, b"000123\0"),
         (0xa434, 2, 7, b"Lens A\0")],
        [(0, 1, 4, b"\x02\x03\0\0"), (1, 2, 2, b"S\0"),
         (2, 5, 3, rational(45, 1) + rational(30, 1) + rational(0, 1)),
         (3, 2, 2, b"W\0"),
         (4, 5, 3, rational(73, 1) + rational(15, 1) + rational(0, 1)),
         (5, 1, 1, b"\x01"), (6, 5, 1, rational(125, 10)),
         (7, 5, 3, rational(10, 1) + rational(4, 1) + rational(1, 2)),
         (18, 2, 7, b"WGS-84\0"), (29, 2, 11, b"2024:02:29\0")]]
    if padded:
        for index, tags in enumerate(directories):
            for ordinal, (id_, type_, count, value) in enumerate(tags):
                if id_ in (271, 272, 0xa434) and (nul_padding if nul_padding is not None else carrier != "tiff"):
                    value += b"\0" * 8
                elif id_ == 0x9291:
                    value = value[:-1] + b"   \0"
                else:
                    continue
                directories[index][ordinal] = (id_, type_, len(value), value)
    offsets = []
    next_offset = header_size
    for tags in directories:
        offsets.append(next_offset)
        next_offset += count_width + entry_width * len(tags) + slot
    payload = bytearray()
    tables = bytearray()
    pixel_slot = None
    for directory, tags in enumerate(directories):
        tables.extend(len(tags).to_bytes(count_width, byte_order))
        for id_, type_, count, value in sorted(tags):
            if directory == 0 and id_ in (0x8769, 0x8825):
                value = integer(offsets[1 if id_ == 0x8769 else 2])
            if directory == 0 and id_ == 273:
                pixel_slot = header_size + len(tables) + (12 if big else 8)
            tables.extend(struct.pack(order + "HH" + ("Q" if big else "I"), id_, type_, count))
            if len(value) > slot:
                payload.extend(bytes((-(next_offset + len(payload))) % (8 if big else 2)))
                tables.extend(integer(next_offset + len(payload)))
                payload.extend(value)
            else:
                tables.extend(value.ljust(slot, b"\0"))
        tables.extend(bytes(slot))
    header = (b"II" if little else b"MM") + short(43 if big else 42)
    header += short(8) + short(0) + integer(header_size) if big else integer(header_size)
    data = bytearray(header + tables + payload)
    data[pixel_slot:pixel_slot + slot] = integer(len(data))
    data.extend(b"\x14\x50\xa0")
    if carrier == "tiff":
        path.write_bytes(data)
    elif carrier == "jpeg":
        previous = write_jpeg(path)
        base = path.read_bytes()
        metadata = b"Exif\0\0" + data
        path.write_bytes(base[:2] + b"\xff\xe1" + struct.pack(">H", len(metadata) + 2) + metadata + base[2 + 2 * (len(previous) + 4):])
    elif carrier == "png":
        previous = write_png(path)
        base = path.read_bytes()
        marker = b"eXIf" + previous
        start = base.index(marker) - 4
        replacement = struct.pack(">I", len(data)) + b"eXIf" + data
        replacement += struct.pack(">I", zlib.crc32(b"eXIf" + data))
        path.write_bytes(base[:start] + replacement + base[start + len(previous) + 12:])
    else:
        raise ValueError(carrier)


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


def write_tiff(path, *, little=True, big=False, pages=1, pixel_offset=None, cycle=False, short_datetime=False):
    """Independent classic/BigTIFF producer; no libtiff writer is used."""
    order = "<" if little else ">"
    slot = 8 if big else 4
    header = (b"II" if little else b"MM") + struct.pack(order + "H", 43 if big else 42)
    if big:
        header += struct.pack(order + "HHQ", 8, 0, 16)
    else:
        header += struct.pack(order + "I", 8)
    data = bytearray(header)
    payload = b"camera\0\xffopaque\0"
    for page in range(pages):
        offset = len(data)
        tags = [(256, 4, 1, struct.pack(order + "I", 1)),
                (257, 4, 1, struct.pack(order + "I", 1)),
                (258, 3, 1, struct.pack(order + "H", 8)),
                (259, 3, 1, struct.pack(order + "H", 1)),
                (262, 3, 1, struct.pack(order + "H", 2)),
                (270, 2, 12, b"TIFF camera\0"),
                (273, 16 if big else 4, 1, b""),
                (274, 3, 1, struct.pack(order + "H", 6)),
                (277, 3, 1, struct.pack(order + "H", 3)),
                (278, 4, 1, struct.pack(order + "I", 1)),
                (279, 4, 1, struct.pack(order + "I", 3)),
                (336, 3, 2, struct.pack(order + "HH", 0, 255)),
                (65000, 7, len(payload), payload),
                (65001, 5, 1, struct.pack(order + "II", 0xffffffff, 2))]
        tags.extend([(65002, 6, 2, struct.pack(order + "bb", -128, 127)),
                     (65003, 8, 3, struct.pack(order + "hhh", -32768, -1, 32767)),
                     (65004, 9, 1, struct.pack(order + "i", -2147483648)),
                     (65005, 10, 1, struct.pack(order + "ii", -1, 2)),
                     (65006, 11, 3, struct.pack(order + "fff", float("nan"), float("inf"), -float("inf"))),
                     (65007, 12, 1, struct.pack(order + "d", 0.25))])
        if big:
            tags.extend([(65008, 16, 2, struct.pack(order + "QQ", 2**63 + 7, 2**64 - 1)),
                         (65009, 17, 2, struct.pack(order + "qq", -2**63, -1))])
        if short_datetime:
            tags.append((306, 3, 1, struct.pack(order + "H", 2)))
        tags.sort(key=lambda tag: tag[0])
        count_width, entry_width = (8, 20) if big else (2, 12)
        tail = offset + count_width + entry_width * len(tags) + slot
        extra = bytearray()
        encoded = []
        pixel_slot = None
        for id_, type_, count, value in tags:
            if id_ == 273:
                pixel_slot = offset + count_width + len(encoded) * entry_width + (12 if big else 8)
                value = bytes(slot)
            if len(value) > slot:
                alignment = 8 if big else 2
                extra.extend(bytes((-(tail + len(extra))) % alignment))
                stored = (tail + len(extra)).to_bytes(slot, "little" if little else "big")
                extra.extend(value)
            else:
                stored = value.ljust(slot, b"\0")
            encoded.append(struct.pack(order + "HH" + ("Q" if big else "I"), id_, type_, count) + stored)
        next_offset = tail + len(extra) + 3
        next_offset += (-next_offset) % (8 if big else 2)
        data.extend(len(tags).to_bytes(count_width, "little" if little else "big"))
        data.extend(b"".join(encoded))
        following = (len(header) if cycle else next_offset) if page + 1 < pages or cycle else 0
        data.extend(following.to_bytes(slot, "little" if little else "big"))
        data.extend(extra)
        pixel = pixel_offset if pixel_offset is not None else len(data)
        data[pixel_slot:pixel_slot + slot] = pixel.to_bytes(slot, "little" if little else "big")
        data.extend(b"\x14\x50\xa0")
        data.extend(bytes(next_offset - len(data)))
    if pixel_offset is not None:
        with path.open("wb") as file:
            file.write(data)
            file.seek(pixel_offset)
            file.write(b"\x14\x50\xa0")
    else:
        path.write_bytes(data)
    return payload


def check_tiff(test, result, path, payload, *, big=False, little=True, pages=1):
    test.assertEqual(result["status"], "collected", result)
    test.assertEqual(result["scope"], "tiff-libtiff-observed-metadata")
    test.assertEqual(result["blocks"], [])
    tiff = result["tiff"]
    test.assertFalse(tiff["complete_metadata_archive"])
    test.assertFalse(tiff["relocatable_exif"])
    test.assertEqual(tiff["big_tiff"], big)
    test.assertEqual(tiff["big_endian"], not little)
    test.assertEqual(len(tiff["directories"]), pages)
    with path.open("rb") as file:
        for span in tiff["source_ranges"]:
            original = bytes.fromhex(span["bytes"])
            file.seek(span["offset"])
            test.assertEqual(original, file.read(len(original)))
    for directory in tiff["directories"]:
        tags = {tag["id"]: tag for tag in directory["tags"]}
        test.assertEqual(tags[274]["unsigned_values"], [6])
        test.assertNotIn(318, tags)
        test.assertNotIn(319, tags)
        test.assertEqual(tags[336]["unsigned_values"], [0, 255])
        test.assertEqual(tags[65002]["signed_values"], [-128, 127])
        test.assertEqual(tags[65003]["signed_values"], [-32768, -1, 32767])
        test.assertEqual(tags[65004]["signed_values"], [-2147483648])
        test.assertEqual(tags[65005]["floating_values"], [-0.5])
        test.assertEqual(tags[65006]["floating_values"], ["nan", "+inf", "-inf"])
        test.assertEqual(tags[65007]["floating_values"], [0.25])
        if big:
            test.assertEqual(tags[65008]["unsigned_values"], [2**63 + 7, 2**64 - 1])
            test.assertEqual(tags[65009]["signed_values"], [-2**63, -1])
        test.assertEqual(bytes.fromhex(tags[65000]["bytes"]), payload)
        expected = float(0xffffffff) / 2
        if tags[65001]["decoded_storage_bytes"] == 4:
            expected = struct.unpack("f", struct.pack("f", expected))[0]
        test.assertEqual(tags[65001]["floating_values"], [expected])



def write_tiff_graph(path, *, little=True, big=False, cycle=False, blobs=(), subimages=1, alpha=False):
    """Image -> EXIF/GPS/SubIFD, plus EXIF -> interoperability, all independent."""
    order = "<" if little else ">"
    byte_order = "little" if little else "big"
    slot, count_width, entry_width = (8, 8, 20) if big else (4, 2, 12)
    align = 8 if big else 2
    pointer_type = 18 if big else 13
    def scalar(value, width):
        return value.to_bytes(width, byte_order)
    base = [(256, 4, 1, scalar(1, 4)), (257, 4, 1, scalar(1, 4)),
            (258, 3, 1, scalar(8, 2)), (259, 3, 1, scalar(1, 2)),
            (262, 3, 1, scalar(2, 2)), (273, 16 if big else 4, 1, scalar(1048576, slot)),
            (274, 3, 1, scalar(6, 2)), (277, 3, 1, scalar(3, 2)), (279, 4, 1, scalar(3, 4))]
    base += [(318, 5, 2, b"".join(scalar(n, 4) + scalar(10000, 4) for n in (3457, 3585))),
             (319, 5, 6, b"".join(scalar(n, 4) + scalar(10000, 4) for n in (7080, 2920, 1700, 7970, 1310, 460)))]
    maker = b"II*\0\x08\0\0\0maker\xff\0"
    nodes = [base + [(330, pointer_type, 1, "subimage"), (34665, pointer_type, 1, "exif"), (34853, pointer_type, 1, "gps")],
             [(33434, 5, 1, scalar(1, 4) + scalar(125, 4)), (37500, 7, len(maker), maker), (40965, pointer_type, 1, "interoperability")],
             [(0, 1, 4, b"\x02\x03\0\0"), (1, 2, 2, b"N\0"),
              (2, 5, 3, b"".join(scalar(n, 4) + scalar(d, 4) for n, d in [(45, 1), (30, 1), (1, 2)]))],
             base,
             [(1, 2, 4, b"R98\0"), (2, 7, 4, b"0100")]]
    if cycle == "self-subimage":
        nodes[3] = base + [(330, pointer_type, 1, "subimage")]
    nodes[0].extend((id_, 1 if id_ in (700, 34377) else 7, len(value), value) for id_, value in blobs)
    if alpha:
        nodes[0] = [tag for tag in nodes[0] if tag[0] not in (277, 279)]
        nodes[0].extend([(277, 3, 1, scalar(4, 2)), (279, 4, 1, scalar(4, 4)), (338, 3, 1, scalar(2, 2)),
                         (254, 4, 1, scalar(1, 4)), (266, 3, 1, scalar(1, 2))])
    nodes[0] = [(id_, type_, subimages if id_ == 330 else count, value)
                for id_, type_, count, value in nodes[0]]
    names = ["image", "exif", "gps", "subimage", "interoperability"]
    header = (b"II" if little else b"MM") + scalar(43 if big else 42, 2)
    header += scalar(8, 2) + scalar(0, 2) + scalar(16, 8) if big else scalar(8, 4)
    data = bytearray(header)
    offsets = {}
    for name, tags in zip(names, nodes):
        data.extend(bytes((-len(data)) % align))
        offsets[name] = len(data)
        data.extend(bytes(count_width + len(tags) * entry_width + slot))
    for name, tags in zip(names, nodes):
        table = offsets[name]
        data[table:table + count_width] = scalar(len(tags), count_width)
        for index, (id_, type_, count, value) in enumerate(sorted(tags)):
            if isinstance(value, str):
                value = scalar(offsets["image"] if (cycle is True and value == "subimage") or cycle == value else offsets[value], slot) * count
            if len(value) > slot:
                data.extend(bytes((-len(data)) % align))
                stored = scalar(len(data), slot)
                data.extend(value)
            else:
                stored = value.ljust(slot, b"\0")
            encoded = scalar(id_, 2) + scalar(type_, 2) + scalar(count, 8 if big else 4) + stored
            at = table + count_width + index * entry_width
            data[at:at + entry_width] = encoded
    path.write_bytes(data)
    return maker


def check_tiff_graph(test, result, path, maker):
    test.assertEqual(result["status"], "collected", result)
    dirs = {node["kind"]: node for node in result["tiff"]["directories"]}
    test.assertEqual(set(dirs), {"image", "exif", "gps", "subimage", "interoperability"})
    for kind in ("exif", "gps", "subimage"):
        test.assertEqual(dirs[kind]["parent"], dirs["image"]["offset"])
    test.assertEqual(dirs["interoperability"]["parent"], dirs["exif"]["offset"])
    for kind in ("image", "subimage"):
        tags = {tag["id"]: tag for tag in dirs[kind]["tags"]}
        for id_, numerators in ((318, (3457, 3585)), (319, (7080, 2920, 1700, 7970, 1310, 460))):
            tag = tags[id_]
            test.assertEqual(tag["type"], 5)
            test.assertEqual(tag["count"], len(numerators))
            test.assertEqual(tag["decoded_storage_bytes"], 4)
            expected = [struct.unpack("f", struct.pack("f", n / 10000))[0] for n in numerators]
            test.assertEqual(tag["floating_values"], expected)
    exif = {tag["id"]: tag for tag in dirs["exif"]["tags"]}
    test.assertEqual(bytes.fromhex(exif[37500]["bytes"]), maker)
    test.assertAlmostEqual(exif[33434]["floating_values"][0], 1 / 125, places=8)
    gps = {tag["id"]: tag for tag in dirs["gps"]["tags"]}
    test.assertEqual(gps[2]["floating_values"], [45, 30, 0.5])
    with path.open("rb") as file:
        for span in result["tiff"]["source_ranges"]:
            original = bytes.fromhex(span["bytes"])
            file.seek(span["offset"])
            test.assertEqual(original, file.read(len(original)))
