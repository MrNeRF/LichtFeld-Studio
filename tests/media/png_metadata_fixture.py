# SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
# SPDX-License-Identifier: GPL-3.0-or-later
"""Independent framing/CRC/XML oracle; pixel decoding is checked separately."""
import json
import struct
import xml.etree.ElementTree as ET
import zlib


def read(data):
    if data[:8] != b"\x89PNG\r\n\x1a\n":
        raise AssertionError("not PNG")
    original = bytearray(data[:8])
    packet = dimensions = None
    saw_idat = ended = False
    offset = 8
    while offset < len(data):
        if len(data) - offset < 12:
            raise AssertionError("truncated PNG chunk")
        size, = struct.unpack_from(">I", data, offset)
        if size > 0x7fffffff or size > len(data) - offset - 12:
            raise AssertionError("invalid PNG chunk extent")
        kind = data[offset + 4:offset + 8]
        payload = data[offset + 8:offset + 8 + size]
        crc, = struct.unpack_from(">I", data, offset + 8 + size)
        if zlib.crc32(kind + payload) != crc:
            raise AssertionError("invalid PNG CRC")
        if dimensions is None:
            if kind != b"IHDR" or size != 13:
                raise AssertionError("missing IHDR")
            dimensions = struct.unpack_from(">II", payload)
        elif kind == b"IHDR":
            raise AssertionError("duplicate IHDR")
        is_xmp = kind == b"iTXt" and payload.startswith(b"XML:com.adobe.xmp\0")
        if is_xmp:
            if packet is not None or saw_idat or not payload.startswith(b"XML:com.adobe.xmp\0\0\0\0\0"):
                raise AssertionError("invalid XMP carrier/order")
            packet = payload[22:]
        else:
            original.extend(data[offset:offset + 12 + size])
        if kind == b"IDAT":
            saw_idat = True
        if kind == b"IEND":
            if size or not saw_idat or offset + 12 != len(data):
                raise AssertionError("invalid PNG end")
            ended = True
            break
        offset += 12 + size
    if not ended or packet is None:
        raise AssertionError("missing PNG pixels/end/XMP")
    if not packet.endswith(b'<?xpacket end="r"?>'):
        raise AssertionError("PNG XMP must be read-only")
    tree = ET.fromstring(packet)
    facts = {node.tag.rsplit("}", 1)[-1]: node.text for node in tree.iter()
             if node.tag.startswith("{https://lichtfeld.io/ns/media/1.0/}")}
    catalog = json.loads(facts["SourceMetadata"])
    return bytes(original), dimensions, facts, catalog
