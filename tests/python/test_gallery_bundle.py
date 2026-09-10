# SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
# SPDX-License-Identifier: GPL-3.0-or-later
"""Shared transport corpus; mirrored in the portal's test_bundle.py."""
import copy
import hashlib
import io
import json
import random
from pathlib import Path
import stat
import struct
import tempfile
import unittest
from unittest.mock import patch
import warnings
import zipfile
import zlib

from lfs_plugins import gallery_bundle as bundle


IDENTITY = [[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]]
SHEAR = [[1, 1, 0, 2], [0, -1, 0.4, 3], [0, 0, 2, 4], [0, 0, 0, 1]]


def ply(count=2, degree=3):
    names = ["x", "y", "z", "nx", "ny", "nz", "f_dc_0", "f_dc_1", "f_dc_2"]
    names += [f"f_rest_{i}" for i in range(3 * ((degree + 1) ** 2 - 1))]
    names += ["opacity", "scale_0", "scale_1", "scale_2", "rot_0", "rot_1", "rot_2", "rot_3"]
    header = "ply\nformat binary_little_endian 1.0\ncomment minimal build stamp\nelement vertex " + str(count) + "\n"
    header += "".join("property float " + name + "\n" for name in names) + "end_header\n"
    return header.encode() + b"".join(struct.pack("<" + "f" * len(names),
        *[float(i + row * len(names)) / 17 for i in range(len(names))]) for row in range(count))


def manifest(data, *, degree=1, count=2):
    return {"format": "lichtfeld-gallery", "version": 1, "nodes": [{
        "file": "nodes/000000.ply", "count": count, "shDegree": degree,
        "transform": copy.deepcopy(SHEAR), "sha256": hashlib.sha256(data).hexdigest()}]}


def archive_bytes(metadata=None, data=None, *, extras=(), compression=zipfile.ZIP_STORED, node_info=None):
    data = ply() if data is None else data
    metadata = manifest(data) if metadata is None else metadata
    out = io.BytesIO()
    with warnings.catch_warnings(), zipfile.ZipFile(out, "w", compression=compression) as archive:
        warnings.simplefilter("ignore", UserWarning)
        archive.writestr(node_info or "nodes/000000.ply", data)
        archive.writestr("manifest.json", metadata if isinstance(metadata, bytes) else json.dumps(metadata))
        for name, content in extras:
            archive.writestr(name, content)
    return out.getvalue()


class GalleryBundleTests(unittest.TestCase):
    def test_hdr_roundtrip_keeps_only_pixels_and_dimensions(self):
        with tempfile.TemporaryDirectory() as root:
            root = Path(root)
            node = root / "private-node.ply"
            node.write_bytes(ply())
            background = root / "private-home-location.lfsenv"
            pixels = b"LFSENV1\0" + struct.pack("<II6f", 2, 1, -0.2, 0, 12, 1, 2, 3)
            background.write_bytes(pixels)
            target = root / "scene.lfsg"
            metadata = bundle.write_bundle(target, [{"path": node, "transform": IDENTITY, "shDegree": 3}], environment=background)
            self.assertEqual(metadata["version"], 2)
            self.assertEqual(metadata["environment"], {"file": "environment.lfsenv", "width": 2, "height": 1, "sha256": hashlib.sha256(pixels).hexdigest()})
            self.assertNotIn(b"private-home-location", target.read_bytes())
            self.assertNotIn(str(root).encode(), target.read_bytes())
            with target.open("rb") as source, bundle.open_bundle(source) as value:
                result = io.BytesIO()
                value.copy_environment(result)
                self.assertEqual(result.getvalue(), pixels)
                offset, size, crc = value.environment_storage()
                self.assertEqual(target.read_bytes()[offset:offset+size], pixels)
                self.assertEqual(crc, zlib.crc32(pixels))

    def test_hdr_rejects_invalid_metadata_and_payloads(self):
        pixels = b"LFSENV1\0" + struct.pack("<II3f", 1, 1, 1, 2, 3)
        valid = manifest(ply())
        valid.update(version=2, environment={"file": "environment.lfsenv", "width": 1, "height": 1, "sha256": hashlib.sha256(pixels).hexdigest()})
        cases = []
        for field, value in (("file", "../private.hdr"), ("width", True), ("height", 0), ("width", 8193), ("sha256", "0"*64), ("path", "private")):
            metadata = copy.deepcopy(valid)
            metadata["environment"][field] = value
            cases.append((metadata, pixels))
        for data in (pixels[:-1], pixels+b"private metadata", b"OTHER000"+pixels[8:], pixels[:8]+struct.pack("<II", 8192, 8192)+pixels[16:],
                     pixels[:16]+struct.pack("<3f", float("nan"), 0, 0), pixels[:16]+struct.pack("<3f", 0, float("inf"), 0)):
            metadata = copy.deepcopy(valid)
            metadata["environment"]["sha256"] = hashlib.sha256(data).hexdigest()
            cases.append((metadata, data))
        old = copy.deepcopy(valid); old["version"] = 1
        cases.append((old, pixels))
        for metadata, data in cases:
            with self.subTest(metadata=metadata), self.assertRaises(bundle.BundleError):
                with bundle.open_bundle(io.BytesIO(archive_bytes(metadata, extras=[("environment.lfsenv", data)]))) as value:
                    value.copy_environment(io.BytesIO())

    def read(self, encoded):
        with bundle.open_bundle(io.BytesIO(encoded)) as value:
            output = io.BytesIO()
            value.copy_node(0, output)
            return value.manifest, output.getvalue()

    def test_preserves_local_data_shear_reflection_and_active_sh(self):
        with tempfile.TemporaryDirectory() as root:
            root = Path(root)
            nodes = []
            payloads = [ply(degree=degree) for degree in (0, 1, 2, 3)]
            for i, payload in enumerate(payloads):
                path = root / f"private-node-{i}.ply"
                path.write_bytes(payload)
                nodes.append({"path": path, "transform": SHEAR if i % 2 else IDENTITY,
                              "shDegree": min(i, 1), "privateName": "not-published", "credential": "not-published"})
            target = root / "scene.lfsg"
            progress = []
            expected = bundle.write_bundle(target, nodes, progress=progress.append)
            self.assertEqual(progress[-1], sum(map(len, payloads)))
            encoded = target.read_bytes()
            self.assertNotIn(b"not-published", encoded)
            self.assertNotIn(str(root).encode(), encoded)
            with target.open("rb") as source, bundle.open_bundle(source) as result:
                self.assertEqual(result.manifest, expected)
                for i, payload in enumerate(payloads):
                    output = io.BytesIO()
                    self.assertEqual(result.copy_node(i, output), len(payload))
                    self.assertEqual(output.getvalue(), payload)
                    offset, size, crc = result.node_storage(i)
                    self.assertEqual(encoded[offset:offset + size], payload)
                    self.assertEqual(crc, zlib.crc32(payload))
                    self.assertEqual(result.manifest["nodes"][i]["shDegree"], min(i, 1))
            # Repack the same snapshots byte-for-byte, without source filenames/timestamps.
            bundle.write_bundle(target, nodes)
            self.assertEqual(target.read_bytes(), encoded)

    def test_standard_zip64_records(self):
        # Exercise ZIP64 EOCD/locator and per-entry extended lengths without a
        # multi-gigabyte allocation. This uses the real standard-library writer.
        with patch.object(zipfile, "ZIP64_LIMIT", 100):
            encoded = archive_bytes()
        self.assertIn(b"PK\x06\x06", encoded)
        self.assertEqual(self.read(encoded)[1], ply())
        with bundle.open_bundle(io.BytesIO(encoded)) as value:
            offset, size, crc = value.node_storage(0)
            self.assertEqual(encoded[offset:offset + size], ply())
            self.assertEqual(crc, zlib.crc32(ply()))
            for invalid in (-1, 1, True, "0", None):
                with self.assertRaises(bundle.BundleError):
                    value.node_storage(invalid)

    def test_member_hash_is_checked_before_copy_returns(self):
        metadata = manifest(ply())
        metadata["nodes"][0]["sha256"] = "0" * 64
        output = io.BytesIO()
        with bundle.open_bundle(io.BytesIO(archive_bytes(metadata))) as value:
            with self.assertRaisesRegex(bundle.BundleError, "integrity"):
                value.copy_node(0, output)
        self.assertTrue(output.getvalue())  # Caller must stage, not mutate the live scene as bytes arrive.

    def test_crc_damage_is_rejected(self):
        encoded = bytearray(archive_bytes())
        offset = encoded.index(b"end_header\n") + len(b"end_header\n")
        encoded[offset] ^= 0xff
        with self.assertRaises(bundle.BundleError):
            self.read(encoded)

    def test_manifest_is_strict_and_does_not_ignore_private_fields(self):
        cases = []
        for key, value in [("version", True), ("version", 2), ("format", "other"), ("nodes", []),
                           ("nodes", {}), ("credentials", "private")]:
            metadata = manifest(ply())
            metadata[key] = value
            cases.append(metadata)
        cases.extend([b'{"format":"lichtfeld-gallery","format":"lichtfeld-gallery","version":1,"nodes":[]}',
                      b'{"format": NaN}', b'{"format": Infinity}', b'[[[[', b'\xff'])
        for metadata in cases:
            with self.subTest(metadata=metadata), self.assertRaises(bundle.BundleError):
                self.read(archive_bytes(metadata))

    def test_node_metadata_rejects_invalid_counts_degrees_digests_and_paths(self):
        for field, value in [("count", True), ("count", 0), ("count", 3), ("count", bundle.MAX_SPLATS + 1),
                             ("shDegree", True), ("shDegree", -1), ("shDegree", 4),
                             ("sha256", "no"), ("sha256", "A" * 64), ("sha256", []),
                             ("file", "../outside.ply"), ("file", "nodes/000001.ply"),
                             ("file", "/nodes/000000.ply"), ("sourcePath", "private")]:
            metadata = manifest(ply())
            metadata["nodes"][0][field] = value
            with self.subTest(field=field, value=value), self.assertRaises(bundle.BundleError):
                self.read(archive_bytes(metadata))

    def test_invalid_affine_transforms(self):
        matrices = [None, [], [1] * 16, [[1] * 4] * 3]
        for value in [True, "1", float("inf"), 1e39, 10**400]:
            matrix = copy.deepcopy(IDENTITY)
            matrix[0][0] = value
            matrices.append(matrix)
        for index in range(4):
            matrix = copy.deepcopy(IDENTITY)
            matrix[3][index] += 1
            matrices.append(matrix)
        for matrix in matrices:
            metadata = manifest(ply())
            metadata["nodes"][0]["transform"] = matrix
            with self.subTest(matrix=matrix), self.assertRaises(bundle.BundleError):
                self.read(archive_bytes(metadata))

    def test_degenerate_but_finite_affines_are_not_silently_rewritten(self):
        # Renderer visibility rules decide the result; the transport must not
        # turn a zero scale into a different nonzero geometry.
        metadata = manifest(ply())
        metadata["nodes"][0]["transform"][0] = [0, 0, 0, 0]
        self.assertEqual(self.read(archive_bytes(metadata))[0], metadata)

    def test_rejects_extra_duplicate_traversal_and_directory_members(self):
        for name in ["secret.txt", "../outside.ply", "nodes/", "nodes/000000.ply", "manifest.json"]:
            with self.subTest(name=name), self.assertRaises(bundle.BundleError):
                self.read(archive_bytes(extras=[(name, b"hidden")]))

    def test_rejects_symlinks_special_files_and_compression(self):
        for mode in [stat.S_IFLNK, stat.S_IFDIR, stat.S_IFIFO, stat.S_IFSOCK, stat.S_IFCHR]:
            info = zipfile.ZipInfo("nodes/000000.ply")
            info.create_system = 3
            info.external_attr = (mode | 0o600) << 16
            with self.subTest(mode=mode), self.assertRaises(bundle.BundleError):
                self.read(archive_bytes(node_info=info))
        with self.assertRaises(bundle.BundleError):
            self.read(archive_bytes(compression=zipfile.ZIP_DEFLATED))

    def test_rejects_encryption_and_descriptor_flags(self):
        for flag in (1, 8, 0x40):
            encoded = bytearray(archive_bytes())
            central = encoded.index(b"PK\x01\x02")
            struct.pack_into("<H", encoded, central + 8, flag)
            with self.subTest(flag=flag), self.assertRaises(bundle.BundleError):
                self.read(encoded)

    def test_rejects_embedded_nul_in_member_name(self):
        encoded = archive_bytes().replace(b"nodes/000000.ply", b"nodes\x0000000.ply")
        with self.assertRaises(bundle.BundleError):
            self.read(encoded)

    def test_rejects_overlapping_members(self):
        encoded = bytearray(archive_bytes())
        first = encoded.index(b"PK\x01\x02")
        second = encoded.index(b"PK\x01\x02", first + 4)
        # Make the second member point to the first local header.
        struct.pack_into("<I", encoded, second + 42, 0)
        with self.assertRaises(bundle.BundleError):
            self.read(encoded)

    def test_limits_directory_before_zipfile_allocates_entries(self):
        encoded = bytearray(archive_bytes())
        end = len(encoded) - 22
        for field, value, layout in [(8, bundle.MAX_NODES + 2, "<H"),
                                     (12, bundle.MAX_DIRECTORY_BYTES + 1, "<I")]:
            damaged = encoded.copy()
            struct.pack_into(layout, damaged, end + field, value)
            with self.subTest(field=field), patch.object(bundle, "ZipFile") as constructor:
                with self.assertRaises(bundle.BundleError):
                    self.read(damaged)
                constructor.assert_not_called()

    def test_forged_small_eocd_count_cannot_bypass_preallocation_limit(self):
        encoded = bytearray(archive_bytes(extras=[("extra", b"hidden")]))
        end = len(encoded) - 22
        struct.pack_into("<HH", encoded, end + 8, 2, 2)
        with patch.object(bundle, "ZipFile") as constructor:
            with self.assertRaises(bundle.BundleError):
                self.read(encoded)
            constructor.assert_not_called()

    def test_local_headers_must_match_central_directory(self):
        for offset, layout, value in [(6, "<H", 8), (8, "<H", 8), (14, "<I", 0), (18, "<I", 1), (22, "<I", 1)]:
            encoded = bytearray(archive_bytes())
            struct.pack_into(layout, encoded, offset, value)
            with self.subTest(offset=offset), self.assertRaises(bundle.BundleError):
                self.read(encoded)

    def test_unknown_extra_fields_and_hidden_prefix_are_rejected(self):
        info = zipfile.ZipInfo("nodes/000000.ply")
        info.extra = struct.pack("<HH", 0x5455, 5) + b"\x01\x00\x00\x00\x00"
        with self.assertRaises(bundle.BundleError):
            self.read(archive_bytes(node_info=info))
        out = io.BytesIO()
        out.write(b"hidden prefix")
        with zipfile.ZipFile(out, "w") as archive:
            archive.writestr("nodes/000000.ply", ply())
            archive.writestr("manifest.json", json.dumps(manifest(ply())))
        with self.assertRaises(bundle.BundleError):
            self.read(out.getvalue())

    def test_manifest_size_limit(self):
        encoded = archive_bytes(b" " * (bundle.MAX_MANIFEST_BYTES + 1))
        with self.assertRaisesRegex(bundle.BundleError, "manifest is too large"):
            self.read(encoded)

    def test_corruption_corpus_cannot_change_verified_geometry(self):
        original = archive_bytes()
        random_source = random.Random(735092)
        for _ in range(300):
            damaged = bytearray(original)
            offset = random_source.randrange(len(damaged))
            damaged[offset] ^= 1 << random_source.randrange(8)
            with self.subTest(offset=offset):
                try:
                    metadata, data = self.read(damaged)
                except bundle.BundleError:
                    continue
                # Benign DOS timestamp / permission-bit changes may be valid.
                self.assertEqual(metadata, manifest(ply()))
                self.assertEqual(data, ply())

    def test_invalid_utf8_zip_filename_is_reported_as_bundle_error(self):
        encoded = bytearray(archive_bytes())
        central = encoded.index(b"PK\x01\x02")
        struct.pack_into("<H", encoded, central + 8, 0x800)
        encoded[central + 46] = 0xff
        with self.assertRaises(bundle.BundleError):
            self.read(encoded)

    def test_rejects_trailing_archive_comment_and_truncation(self):
        encoded = archive_bytes()
        for candidate in (encoded[:-1], encoded + b"extra", b"", encoded[:16]):
            with self.subTest(length=len(candidate)), self.assertRaises(bundle.BundleError):
                self.read(candidate)

    def test_rejects_unsupported_ply_schema_and_size(self):
        valid = ply()
        cases = [valid[:-1], valid + b"extra", valid.replace(b"property float x", b"property double x"),
                 valid.replace(b"property float y", b"property float x"),
                 valid.replace(b"element vertex 2", b"element vertex 0"),
                 valid.replace(b"element vertex 2", b"element vertex 100000001"),
                 valid.replace(b"format binary_little_endian", b"format ascii"),
                 valid.replace(b"end_header\n", b"element face 0\nend_header\n"),
                 valid.replace(b"end_header\n", b"comment " + b"x" * bundle.MAX_HEADER_BYTES + b"\nend_header\n")]
        for data in cases:
            with self.subTest(length=len(data)), self.assertRaises(bundle.BundleError):
                self.read(archive_bytes(manifest(data), data))

    def test_active_degree_cannot_exceed_storage(self):
        data = ply(degree=0)
        with self.assertRaises(bundle.BundleError):
            self.read(archive_bytes(manifest(data, degree=1), data))

    def test_cancel_keeps_previous_destination_and_removes_partial_archive(self):
        with tempfile.TemporaryDirectory() as root:
            root = Path(root)
            source, target = root / "source.ply", root / "scene.lfsg"
            source.write_bytes(ply())
            target.write_bytes(b"previous")
            def cancel(_):
                raise InterruptedError("paused")
            with self.assertRaises(InterruptedError):
                bundle.write_bundle(target, [{"path": source, "transform": IDENTITY, "shDegree": 0}], progress=cancel)
            self.assertEqual(target.read_bytes(), b"previous")
            self.assertEqual(set(root.iterdir()), {source, target})

    def test_source_change_keeps_previous_destination(self):
        with tempfile.TemporaryDirectory() as root:
            root = Path(root)
            source, target = root / "source.ply", root / "scene.lfsg"
            source.write_bytes(ply())
            target.write_bytes(b"previous")
            def change(_):
                source.write_bytes(ply() + b"changed")
            with self.assertRaisesRegex(bundle.BundleError, "changed"):
                bundle.write_bundle(target, [{"path": source, "transform": IDENTITY, "shDegree": 0}], progress=change)
            self.assertEqual(target.read_bytes(), b"previous")
            self.assertEqual(set(root.iterdir()), {source, target})

    def test_large_members_are_streamed_with_bounded_reads(self):
        data = ply(count=20_000)
        reads = []
        class Reader(io.BytesIO):
            def read(self, size=-1):
                reads.append(size)
                return super().read(size)
        with bundle.open_bundle(Reader(archive_bytes(manifest(data, count=20_000), data))) as value:
            output = io.BytesIO()
            progress = []
            value.copy_node(0, output, progress=progress.append)
            self.assertEqual(output.getvalue(), data)
            self.assertGreater(len(progress), 1)
        self.assertLessEqual(max(reads), bundle.CHUNK_BYTES)


if __name__ == "__main__":
    unittest.main()
