# SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
# SPDX-License-Identifier: GPL-3.0-or-later
"""Validation shared by native gallery projects and their preparation files."""
import math
import os
import stat
import struct

MAX_NODES = 4096
MAX_SPLATS = 100_000_000
MAX_MANIFEST_BYTES = 4 * 1024 * 1024
MAX_HEADER_BYTES = 64 * 1024
MAX_READ_BYTES = 8 * 1024 * 1024
CHUNK_BYTES = 1024 * 1024
MAX_ENVIRONMENT_PIXELS = 8_388_608

def environment_header(source, size):
    header = source.read(16)
    _require(len(header) == 16, "Incomplete HDR background.")
    magic, width, height = struct.unpack("<8sII", header)
    _require(magic == b"LFSENV1\0" and 0 < width <= 8192 and 0 < height <= 8192
             and width * height <= MAX_ENVIRONMENT_PIXELS and size == 16 + width * height * 12,
             "Invalid HDR background dimensions or size.")
    return width, height


def _environment_values(chunk):
    _require(len(chunk) % 4 == 0 and all(math.isfinite(x[0]) for x in struct.iter_unpack("<f", chunk)),
             "HDR background contains invalid pixels.")


def _require(condition, message):
    if not condition:
        raise ValueError(message)


def _object(pairs):
    result = {}
    for key, value in pairs:
        _require(key not in result, "Duplicate gallery metadata key.")
        result[key] = value
    return result


def _constant(value):
    raise ValueError("Non-finite gallery metadata.")


def _matrix(value):
    _require(isinstance(value, (list, tuple)) and len(value) == 4, "Invalid node transform.")
    result = []
    for row in value:
        _require(isinstance(row, (list, tuple)) and len(row) == 4, "Invalid node transform.")
        converted = []
        for component in row:
            _require(type(component) in (int, float), "Invalid node transform.")
            try:
                component = float(component)
            except OverflowError:
                raise ValueError("Invalid node transform.") from None
            _require(math.isfinite(component) and abs(component) <= 3.4028234663852886e38,
                     "Invalid node transform.")
            converted.append(component)
        result.append(converted)
    _require(result[3] == [0, 0, 0, 1], "Node transform must be affine.")
    return result


def _degree(value):
    _require(type(value) is int and 0 <= value <= 3, "Invalid node SH degree.")
    return value


def _properties(degree):
    return ["x", "y", "z", "nx", "ny", "nz", "f_dc_0", "f_dc_1", "f_dc_2",
            *[f"f_rest_{i}" for i in range(3 * ((degree + 1) ** 2 - 1))],
            "opacity", "scale_0", "scale_1", "scale_2", "rot_0", "rot_1", "rot_2", "rot_3"]


def _ply_header(source, size):
    """Check allocation-relevant PLY fields without materializing vertex data."""
    header = bytearray()
    while len(header) <= MAX_HEADER_BYTES:
        line = source.readline(MAX_HEADER_BYTES + 1 - len(header))
        _require(line and line.endswith(b"\n"), "Incomplete gallery PLY header.")
        header.extend(line)
        if line == b"end_header\n":
            break
    _require(len(header) <= MAX_HEADER_BYTES, "Gallery PLY header is too large.")
    try:
        lines = header.decode("utf-8").splitlines()
    except UnicodeError:
        raise ValueError("Invalid gallery PLY header.") from None
    _require(lines[:2] == ["ply", "format binary_little_endian 1.0"] and lines[-1:] == ["end_header"],
             "Gallery nodes require binary float PLY data.")
    count, properties = None, []
    for line in lines[2:-1]:
        if line.startswith("comment "):
            continue
        words = line.split()
        if len(words) == 3 and words[:2] == ["element", "vertex"] and count is None and not properties:
            try:
                count = int(words[2])
            except ValueError:
                raise ValueError("Invalid gallery PLY count.") from None
            _require(0 < count <= MAX_SPLATS, "Invalid gallery PLY count.")
        elif len(words) == 3 and words[:2] == ["property", "float"] and count is not None:
            properties.append(words[2])
        else:
            raise ValueError("Unsupported gallery PLY schema.")
    degree = next((d for d in range(4) if properties == _properties(d)), None)
    _require(count is not None and degree is not None, "Unsupported gallery PLY schema.")
    _require(size == len(header) + count * len(properties) * 4, "Gallery PLY size does not match its count.")
    return count, degree


def _stamp(source):
    value = os.fstat(source.fileno())
    _require(stat.S_ISREG(value.st_mode), "Gallery node source must be a regular file.")
    return value.st_dev, value.st_ino, value.st_size, value.st_mtime_ns, value.st_ctime_ns
