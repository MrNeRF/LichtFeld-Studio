# SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
# SPDX-License-Identifier: GPL-3.0-or-later
"""Independent JPEG framing, TIFF and XML reader for export contracts."""
import hashlib
import json
import struct
import xml.etree.ElementTree as ET


def read(data):
    if data[:2] != b"\xff\xd8":
        raise AssertionError("not JPEG")
    standard = extension = exif = None
    chunks = []
    original = bytearray(data[:2])
    offset = 2
    while offset < len(data):
        marker = data[offset:offset + 2]
        if marker == b"\xff\xda":
            original.extend(data[offset:])
            break
        length = int.from_bytes(data[offset + 2:offset + 4], "big")
        if length < 2 or offset + 2 + length > len(data):
            raise AssertionError("bad JPEG marker extent")
        payload = data[offset + 4:offset + 2 + length]
        if marker == b"\xff\xe1" and payload.startswith(b"Exif\0\0"):
            if exif is not None:
                raise AssertionError("duplicate EXIF")
            exif = payload[6:]
        elif marker == b"\xff\xe1" and payload.startswith(b"http://ns.adobe.com/xap/1.0/\0"):
            if standard is not None:
                raise AssertionError("duplicate Standard XMP")
            standard = ET.fromstring(payload[29:])
        elif marker == b"\xff\xe1" and payload.startswith(b"http://ns.adobe.com/xmp/extension/\0"):
            chunks.append((payload[35:67].decode("ascii"), *struct.unpack(">II", payload[67:75]), payload[75:]))
        else:
            original.extend(data[offset:offset + 2 + length])
        offset += 2 + length
    if exif is None or standard is None:
        raise AssertionError("missing EXIF/XMP")
    if exif[:4] != b"II*\0":
        raise AssertionError("bad TIFF header")

    def directory(at):
        count, = struct.unpack_from("<H", exif, at)
        result = {}
        for index in range(count):
            tag, kind, size, value = struct.unpack_from("<HHII", exif, at + 2 + 12 * index)
            if tag == 0x9000 and kind == 7 and size == 4:
                continue
            if size != 1 or kind not in (3, 4) or tag in result:
                raise AssertionError("bad output EXIF entry")
            result[tag] = value
        return result

    first = directory(struct.unpack_from("<I", exif, 4)[0])
    dimensions = directory(first[0x8769])
    link = standard.find(".//{http://ns.adobe.com/xmp/note/}HasExtendedXMP")
    if link is not None:
        parts = bytearray()
        for guid, total, at, chunk in chunks:
            if guid != link.text or at != len(parts):
                raise AssertionError("bad Extended XMP linkage")
            parts.extend(chunk)
        if not chunks or len(parts) != total or hashlib.md5(parts).hexdigest().upper() != link.text:
            raise AssertionError("bad Extended XMP digest/extent")
        extension = ET.fromstring(parts)
    elif chunks:
        raise AssertionError("unlinked Extended XMP")
    facts = {node.tag.rsplit("}", 1)[-1]: node.text for node in standard.iter()
             if node.tag.startswith("{https://lichtfeld.io/ns/media/1.0/}")}
    catalog_tree = extension if extension is not None else standard
    catalog = json.loads(catalog_tree.find(".//{https://lichtfeld.io/ns/media/1.0/}SourceMetadata").text)
    return bytes(original), first[0x112], (dimensions[0xa002], dimensions[0xa003]), facts, catalog, len(chunks)
