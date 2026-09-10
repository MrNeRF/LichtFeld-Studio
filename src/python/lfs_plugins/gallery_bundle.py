# SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
# SPDX-License-Identifier: GPL-3.0-or-later
"""Visible-scene transport preserving local Gaussians and composed node affines.

This standard-library-only codec is mirrored in LichtFeld-portal/gallery/bundle.py.
It never extracts archive paths. Callers own their staging files and must finish
verification before handing any imported geometry to the native scene.
"""
from __future__ import annotations

from contextlib import contextmanager
import hashlib
import io
import json
import math
import os
from pathlib import Path
import stat
import struct
import tempfile
from zipfile import BadZipFile, ZIP_STORED, ZipFile, ZipInfo

FORMAT = "lichtfeld-gallery"
VERSION = 1
MAX_NODES = 4096
MAX_SPLATS = 100_000_000
MAX_MANIFEST_BYTES = 4 * 1024 * 1024
MAX_HEADER_BYTES = 64 * 1024
MAX_READ_BYTES = 8 * 1024 * 1024
MAX_DIRECTORY_BYTES = 2 * 1024 * 1024
CHUNK_BYTES = 1024 * 1024
MAX_ENVIRONMENT_PIXELS = 8_388_608
MAX_ENVIRONMENT_BYTES = 16 + MAX_ENVIRONMENT_PIXELS * 12


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


class BundleError(ValueError):
    pass


def _require(condition, message):
    if not condition:
        raise BundleError(message)


def _object(pairs):
    result = {}
    for key, value in pairs:
        _require(key not in result, "Duplicate gallery metadata key.")
        result[key] = value
    return result


def _constant(value):
    raise BundleError("Non-finite gallery metadata.")


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
                raise BundleError("Invalid node transform.") from None
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
        raise BundleError("Invalid gallery PLY header.") from None
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
                raise BundleError("Invalid gallery PLY count.") from None
            _require(0 < count <= MAX_SPLATS, "Invalid gallery PLY count.")
        elif len(words) == 3 and words[:2] == ["property", "float"] and count is not None:
            properties.append(words[2])
        else:
            raise BundleError("Unsupported gallery PLY schema.")
    degree = next((d for d in range(4) if properties == _properties(d)), None)
    _require(count is not None and degree is not None, "Unsupported gallery PLY schema.")
    _require(size == len(header) + count * len(properties) * 4, "Gallery PLY size does not match its count.")
    return count, degree


class _BoundedReader(io.RawIOBase):
    """Bound ZipFile's central-directory read BEFORE it allocates its entry list."""
    def __init__(self, source):
        self.source = source
        position = source.tell()
        self.size = source.seek(0, io.SEEK_END)
        source.seek(position)

    def readable(self):
        return True

    def seekable(self):
        return True

    def tell(self):
        return self.source.tell()

    def seek(self, offset, whence=io.SEEK_SET):
        position = offset + (0 if whence == io.SEEK_SET else self.tell() if whence == io.SEEK_CUR else self.size)
        _require(whence in (io.SEEK_SET, io.SEEK_CUR, io.SEEK_END) and 0 <= position <= self.size,
                 "Invalid gallery archive offset.")
        return self.source.seek(position)

    def read(self, size=-1):
        remaining = self.size - self.tell()
        size = remaining if size < 0 else min(size, remaining)
        _require(size <= MAX_READ_BYTES, "Gallery archive directory is too large.")
        return self.source.read(size)


def _directory_preflight(source):
    """Check EOCD/ZIP64 counts before ZipFile creates any ZipInfo objects."""
    def record(offset, layout):
        source.seek(offset)
        size = struct.calcsize(layout)
        data = source.read(size)
        _require(len(data) == size, "Incomplete gallery archive directory.")
        return struct.unpack(layout, data)

    end = source.size - 22
    _require(end >= 0, "Incomplete gallery archive directory.")
    sig, disk, cd_disk, disk_count, count, cd_size, cd_offset, comment = record(end, "<4s4H2IH")
    _require(sig == b"PK\x05\x06" and disk == cd_disk == comment == 0,
             "Unsupported gallery archive directory.")
    # Python may emit ZIP64 end records even when the ordinary fields fit
    # (e.g. a forced-ZIP64 writer). Inspect the locator as well as sentinels.
    locator = record(end - 20, "<4sIQI") if end >= 20 else (b"", 0, 0, 0)
    if locator[0] == b"PK\x06\x07":
        _, zip64_disk, offset, disks = locator
        _require(zip64_disk == 0 and disks == 1 and offset + 56 == end - 20,
                 "Unsupported gallery ZIP64 directory.")
        fields = record(offset, "<4sQ2H2I4Q")
        _require(fields[0] == b"PK\x06\x06" and fields[1] == 44 and fields[4] == fields[5] == 0,
                 "Unsupported gallery ZIP64 directory.")
        actual_disk_count, actual_count, actual_size, actual_offset = fields[6:]
        _require(disk_count in (actual_disk_count, 0xffff) and count in (actual_count, 0xffff)
                 and cd_size in (actual_size, 0xffffffff) and cd_offset in (actual_offset, 0xffffffff),
                 "Inconsistent gallery ZIP64 directory.")
        disk_count, count, cd_size, cd_offset = fields[6:]
        end = offset
    _require(2 <= count <= MAX_NODES + 2 and disk_count == count, "Invalid gallery archive entry count.")
    _require(0 < cd_size <= MAX_DIRECTORY_BYTES and cd_offset + cd_size == end,
             "Invalid gallery archive directory size.")
    source.seek(cd_offset)
    directory = source.read(cd_size)
    _require(len(directory) == cd_size, "Incomplete gallery archive directory.")
    cursor, actual_count = 0, 0
    while cursor < cd_size:
        _require(cursor + 46 <= cd_size and actual_count < count, "Inconsistent gallery archive entry count.")
        fields = struct.unpack_from("<4s6H3I5H2I", directory, cursor)
        _require(fields[0] == b"PK\x01\x02", "Invalid gallery central directory.")
        name_length, extra_length, comment_length = fields[10:13]
        limit = cursor + 46 + name_length + extra_length + comment_length
        _require(limit <= cd_size, "Invalid gallery central directory.")
        _zip64_extra(directory[cursor + 46 + name_length:cursor + 46 + name_length + extra_length])
        cursor = limit
        actual_count += 1
    _require(actual_count == count, "Inconsistent gallery archive entry count.")
    source.seek(0)
    return count, cd_offset


def _zip64_extra(extra):
    if not extra:
        return b""
    _require(len(extra) >= 4, "Invalid gallery archive extra field.")
    kind, size = struct.unpack_from("<HH", extra)
    _require(kind == 1 and size in (8, 16, 24, 28) and size + 4 == len(extra),
             "Unsupported gallery archive extra field.")
    return extra[4:]


def _local_records(archive, directory_offset):
    """Reject overlap, hidden prefix/gaps, and disagreements with local headers."""
    position, ranges = 0, {}
    for entry in sorted(archive.infolist(), key=lambda entry: entry.header_offset):
        _require(entry.header_offset == position and position + 30 <= directory_offset,
                 "Overlapping or misplaced gallery archive member.")
        archive.fp.seek(position)
        header = archive.fp.read(30)
        _require(len(header) == 30, "Incomplete gallery local header.")
        fields = struct.unpack("<4s5H3I2H", header)
        _require(fields[0] == b"PK\x03\x04" and fields[1] == entry.extract_version
                 and fields[2] == entry.flag_bits and fields[3] == entry.compress_type
                 and fields[6] == entry.CRC, "Inconsistent gallery local header.")
        name_length, extra_length = fields[9:]
        name = archive.fp.read(name_length)
        extra = archive.fp.read(extra_length)
        _require(name == entry.filename.encode("ascii") and len(extra) == extra_length,
                 "Inconsistent gallery local member name.")
        extended = _zip64_extra(extra)
        compressed, size = fields[7:9]
        for is_size in (True, False):
            value = size if is_size else compressed
            if value == 0xffffffff:
                _require(len(extended) >= 8, "Missing gallery ZIP64 member size.")
                value, = struct.unpack("<Q", extended[:8])
                extended = extended[8:]
                if is_size:
                    size = value
                else:
                    compressed = value
        _require(not extended and size == entry.file_size and compressed == entry.compress_size,
                 "Inconsistent gallery local member size.")
        data_offset = position + 30 + name_length + extra_length
        ranges[entry.filename] = (data_offset, size, entry.CRC)
        position = data_offset + size
        _require(position <= directory_offset, "Overlapping gallery archive member.")
    _require(position == directory_offset, "Gallery archive contains unreferenced data.")
    return ranges


class Bundle:
    def __init__(self, archive, directory_offset):
        self._archive = archive
        entries = archive.infolist()
        _require(2 <= len(entries) <= MAX_NODES + 2, "Invalid gallery archive entry count.")
        _require(not archive.comment, "Unsupported gallery archive comment.")
        names = set()
        for entry in entries:
            mode = stat.S_IFMT(entry.external_attr >> 16)
            _require(entry.filename == entry.orig_filename and entry.filename not in names,
                     "Duplicate or invalid gallery archive member.")
            _require(not entry.is_dir() and mode in (0, stat.S_IFREG) and not entry.comment,
                     "Unsupported gallery archive member.")
            _require(entry.compress_type == ZIP_STORED and entry.compress_size == entry.file_size
                     and not (entry.flag_bits & ~0x800) and entry.extract_version <= 45,
                     "Gallery archive must use unencrypted stored members.")
            names.add(entry.filename)
        _require("manifest.json" in names, "Gallery manifest is missing.")
        manifest_entry = archive.getinfo("manifest.json")
        _require(0 < manifest_entry.file_size <= MAX_MANIFEST_BYTES, "Gallery manifest is too large.")
        try:
            manifest = json.loads(archive.read(manifest_entry), object_pairs_hook=_object, parse_constant=_constant)
        except (UnicodeError, json.JSONDecodeError, RecursionError):
            raise BundleError("Invalid gallery manifest.") from None
        has_environment = isinstance(manifest, dict) and "environment" in manifest
        _require(isinstance(manifest, dict) and set(manifest) == ({"format", "version", "nodes", "environment"} if has_environment else {"format", "version", "nodes"})
                 and manifest["format"] == FORMAT and type(manifest["version"]) is int
                 and manifest["version"] == (2 if has_environment else VERSION), "Unsupported gallery manifest.")
        nodes = manifest["nodes"]
        _require(isinstance(nodes, list) and 0 < len(nodes) <= MAX_NODES, "Invalid gallery node count.")
        expected_names = {"manifest.json"} | {f"nodes/{i:06d}.ply" for i in range(len(nodes))}
        if has_environment:
            environment = manifest["environment"]
            _require(isinstance(environment, dict) and set(environment) == {"file", "width", "height", "sha256"}
                     and environment["file"] == "environment.lfsenv", "Invalid HDR background metadata.")
            _require(type(environment["width"]) is int and type(environment["height"]) is int,
                     "Invalid HDR background dimensions.")
            digest = environment["sha256"]
            _require(isinstance(digest, str) and len(digest) == 64 and all(c in "0123456789abcdef" for c in digest),
                     "Invalid HDR background digest.")
            entry = archive.getinfo("environment.lfsenv")
            with archive.open(entry) as source:
                dimensions = environment_header(source, entry.file_size)
            _require(dimensions == (environment["width"], environment["height"]), "HDR background dimensions disagree.")
            expected_names.add("environment.lfsenv")
        _require(names == expected_names, "Gallery archive contains unexpected members.")
        self._ranges = _local_records(archive, directory_offset)
        total = 0
        for i, node in enumerate(nodes):
            _require(isinstance(node, dict) and set(node) == {"file", "count", "shDegree", "transform", "sha256"},
                     "Invalid gallery node metadata.")
            _require(node["file"] == f"nodes/{i:06d}.ply", "Invalid gallery node member.")
            _require(type(node["count"]) is int and 0 < node["count"] <= MAX_SPLATS, "Invalid gallery node count.")
            _degree(node["shDegree"])
            node["transform"] = _matrix(node["transform"])
            digest = node["sha256"]
            _require(isinstance(digest, str) and len(digest) == 64
                     and all(c in "0123456789abcdef" for c in digest), "Invalid gallery node digest.")
            entry = archive.getinfo(node["file"])
            with archive.open(entry) as source:
                count, degree = _ply_header(source, entry.file_size)
            _require(count == node["count"] and node["shDegree"] <= degree, "Gallery node disagrees with its PLY data.")
            total += count
            _require(total <= MAX_SPLATS, "Gallery contains too many splats.")
        self.manifest = manifest

    def environment_storage(self):
        _require("environment" in self.manifest, "This scene has no HDR background.")
        return self._ranges["environment.lfsenv"]

    def copy_environment(self, output, *, progress=None):
        _, size, _ = self.environment_storage()
        digest = hashlib.sha256()
        with self._archive.open("environment.lfsenv") as source:
            environment_header(source, size)
            source.seek(0)
            header = source.read(16)
            output.write(header)
            digest.update(header)
            completed = 16
            while chunk := source.read(CHUNK_BYTES):
                _environment_values(chunk)
                output.write(chunk)
                digest.update(chunk)
                completed += len(chunk)
                if progress is not None:
                    progress(completed)
        _require(completed == size and digest.hexdigest() == self.manifest["environment"]["sha256"],
                 "HDR background integrity check failed.")
        return completed

    def node_storage(self, index):
        """Validated STORED byte offset, size and CRC; tied to this immutable archive."""
        _require(type(index) is int and 0 <= index < len(self.manifest["nodes"]), "Invalid gallery node index.")
        return self._ranges[self.manifest["nodes"][index]["file"]]

    def copy_node(self, index, output, *, progress=None):
        """Stream to caller-owned staging storage; do not import until this returns."""
        _require(type(index) is int and 0 <= index < len(self.manifest["nodes"]), "Invalid gallery node index.")
        node = self.manifest["nodes"][index]
        digest = hashlib.sha256()
        completed = 0
        with self._archive.open(node["file"]) as source:
            while chunk := source.read(CHUNK_BYTES):
                digest.update(chunk)
                output.write(chunk)
                completed += len(chunk)
                if progress is not None:
                    progress(completed)
        _require(digest.hexdigest() == node["sha256"], "Gallery node integrity check failed.")
        return completed


@contextmanager
def open_bundle(source):
    """Read an already-open seekable file, including the portal's bounded R2 reader."""
    try:
        bounded = _BoundedReader(source)
        count, directory_offset = _directory_preflight(bounded)
        with ZipFile(bounded) as archive:
            _require(len(archive.infolist()) == count, "Inconsistent gallery archive entry count.")
            yield Bundle(archive, directory_offset)
    except (BadZipFile, EOFError, KeyError, NotImplementedError, UnicodeError, struct.error, OverflowError) as exc:
        raise BundleError("Gallery archive is incomplete or invalid.") from exc


def _stamp(source):
    value = os.fstat(source.fileno())
    _require(stat.S_ISREG(value.st_mode), "Gallery node source must be a regular file.")
    return value.st_dev, value.st_ino, value.st_size, value.st_mtime_ns, value.st_ctime_ns


def write_bundle(destination, nodes, *, progress=None, environment=None):
    """Package private export snapshots, atomically. Each node supplies path/transform/shDegree.

    The caller must snapshot visible, undeleted geometry before this worker runs.
    Metadata is constructed from the allowlist here, never copied from a project.
    Progress may raise to cancel; the previous destination then remains intact.
    """
    _require(isinstance(nodes, (list, tuple)) and 0 < len(nodes) <= MAX_NODES, "Invalid gallery node count.")
    destination = Path(destination)
    prepared = [(Path(node["path"]), _matrix(node["transform"]), _degree(node["shDegree"])) for node in nodes]
    manifest = {"format": FORMAT, "version": VERSION, "nodes": []}
    total, completed = 0, 0
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(dir=destination.parent, prefix=".gallery-bundle-", delete=False) as output:
            temporary = Path(output.name)
            with ZipFile(output, "w", compression=ZIP_STORED, allowZip64=True) as archive:
                for i, (path, transform, degree) in enumerate(prepared):
                    with path.open("rb") as source:
                        stamp = _stamp(source)
                        count, stored_degree = _ply_header(source, stamp[2])
                        _require(degree <= stored_degree, "Gallery node exceeds its stored SH degree.")
                        total += count
                        _require(total <= MAX_SPLATS, "Gallery contains too many splats.")
                        source.seek(0)
                        name = f"nodes/{i:06d}.ply"
                        entry = ZipInfo(name)
                        entry.file_size = stamp[2]
                        entry.external_attr = (stat.S_IFREG | 0o600) << 16
                        digest = hashlib.sha256()
                        copied = 0
                        with archive.open(entry, "w", force_zip64=stamp[2] >= 2**31) as member:
                            while chunk := source.read(min(CHUNK_BYTES, stamp[2] - copied + 1)):
                                copied += len(chunk)
                                _require(copied <= stamp[2], "Gallery node changed during preparation.")
                                digest.update(chunk)
                                member.write(chunk)
                                completed += len(chunk)
                                if progress is not None:
                                    progress(completed)
                        _require(copied == stamp[2] and _stamp(source) == stamp, "Gallery node changed during preparation.")
                        manifest["nodes"].append({"file": name, "count": count, "shDegree": degree,
                                                  "transform": transform, "sha256": digest.hexdigest()})
                if environment is not None:
                    with Path(environment).open("rb") as source:
                        stamp = _stamp(source)
                        width, height = environment_header(source, stamp[2])
                        source.seek(0)
                        header = source.read(16)
                        digest = hashlib.sha256(header)
                        entry = ZipInfo("environment.lfsenv")
                        entry.external_attr = (stat.S_IFREG | 0o600) << 16
                        with archive.open(entry, "w") as member:
                            member.write(header)
                            copied = 16
                            while chunk := source.read(min(CHUNK_BYTES, stamp[2] - copied + 1)):
                                copied += len(chunk)
                                _require(copied <= stamp[2], "HDR background changed during preparation.")
                                _environment_values(chunk)
                                digest.update(chunk)
                                member.write(chunk)
                                if progress is not None:
                                    progress(completed + copied)
                        _require(copied == stamp[2] and _stamp(source) == stamp, "HDR background changed during preparation.")
                        manifest["version"] = 2
                        manifest["environment"] = {"file": "environment.lfsenv", "width": width, "height": height,
                                                   "sha256": digest.hexdigest()}
                encoded = json.dumps(manifest, separators=(",", ":"), allow_nan=False).encode()
                _require(len(encoded) <= MAX_MANIFEST_BYTES, "Gallery manifest is too large.")
                entry = ZipInfo("manifest.json")
                entry.external_attr = (stat.S_IFREG | 0o600) << 16
                archive.writestr(entry, encoded)
            output.flush()
            os.fsync(output.fileno())
        os.replace(temporary, destination)
        temporary = None
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)
    return manifest
