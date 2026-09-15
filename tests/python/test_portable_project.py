import gzip
import io
import json
from pathlib import Path
import struct
import unittest


def rewrite_chapter(data, kind, change):
    """Mutate semantic data while maintaining valid native container checksums."""
    data = bytearray(data)
    head_offset = next(offset for offset in (4096, 8192) if data[offset:offset+8] == b'LFSHEAD\0')
    commit_offset = struct.unpack_from('<Q', data, head_offset + 80)[0]
    index_offset, index_size = struct.unpack_from('<QQ', data, commit_offset + 136)
    count = struct.unpack_from('<Q', data, index_offset + 16)[0]
    row_offset = next(index_offset+64+i*96 for i in range(count) if data[index_offset+64+i*96:index_offset+68+i*96] == kind)
    header_offset, payload_offset, size = struct.unpack_from('<QQQ', data, row_offset + 32)
    value = json.loads(data[payload_offset:payload_offset+size])
    change(value)
    payload = json.dumps(value, separators=(',',':')).encode()
    assert len(payload) <= size
    payload = payload.ljust(size,b' ')
    data[payload_offset:payload_offset+size] = payload
    checksum = codec._crc(payload)
    struct.pack_into('<I',data,row_offset+72,checksum)
    struct.pack_into('<I',data,header_offset+56,checksum)
    checksum = codec._crc(data[header_offset:header_offset+60])
    struct.pack_into('<I',data,header_offset+60,checksum)
    struct.pack_into('<I',data,row_offset+76,checksum)
    checksum = codec._crc(data[index_offset:index_offset+index_size])
    struct.pack_into('<II',data,commit_offset+160,checksum,checksum)
    checksum = codec._crc(data[commit_offset:commit_offset+252])
    struct.pack_into('<I',data,commit_offset+252,checksum)
    struct.pack_into('<I',data,head_offset+104,checksum)
    struct.pack_into('<I',data,head_offset+4092,codec._crc(data[head_offset:head_offset+4092]))
    return bytes(data)


class PortableProjectTests(unittest.TestCase):
    def fixture(self, format='sog'):
        return (FIXTURES / ('portable-'+format+'.licht')).read_bytes()

    def test_real_native_exports_keep_compression_and_hdr(self):
        for format in ('ply','sog','ssog'):
            with self.subTest(format=format):
                project = codec.ProjectFile(io.BytesIO(self.fixture(format)))
                self.assertEqual(project.manifest['nodes'][0]['count'],64)
                self.assertTrue(project.manifest['nodes'][0]['file'].endswith('.'+format))
                output = io.BytesIO()
                project.copy_node(0,output)
                self.assertTrue(output.getvalue().startswith(b'ply\n' if format == 'ply' else b'PK'))
                output = io.BytesIO()
                project.copy_environment(output)
                self.assertTrue(output.getvalue().startswith(b'LFSENV1\0'))
                self.assertNotIn(b'CKPT', project.chapters)
                self.assertEqual(project.chapters[b'EDTR']['open_files'],[])
                self.assertIsNone(project.chapters[b'PROJ']['dataset_reference_uuid'])

    def test_native_camera_path_and_multiple_objects(self):
        project = codec.ProjectFile(io.BytesIO(self.fixture('multi')))
        self.assertEqual(len(project.manifest['nodes']), 2)
        self.assertIn('environment', project.manifest)
        sequencer = project.chapters[b'SEQR']
        self.assertEqual(len(sequencer['timeline']['keyframes']), 2)
        self.assertEqual(sequencer['loop_mode'], 'ping_pong')
        self.assertEqual(sequencer['playback_speed'], 1.5)
        self.assertEqual(sequencer['ply_sequences'], [])

    def test_tampered_embedded_bytes_fail_integrity(self):
        stream=io.BytesIO(self.fixture())
        project=codec.ProjectFile(stream)
        offset=project._nodes[0]['offset']+100
        stream.seek(offset); byte=stream.read(1)
        stream.seek(offset); stream.write(bytes([byte[0]^1]))
        with self.assertRaisesRegex(ValueError,'checksum'):
            project.copy_node(0,io.BytesIO())

    def test_editor_buffers_are_rejected_even_with_valid_checksums(self):
        data=rewrite_chapter(self.fixture(),b'EDTR',lambda value:value.update(open_files=[{'buffer':'private'}]))
        with self.assertRaisesRegex(ValueError,'Editor'):
            codec.ProjectFile(io.BytesIO(data))

    def test_dataset_reference_is_rejected_even_with_valid_checksums(self):
        data=rewrite_chapter(self.fixture(),b'PROJ',lambda value:value.update(dataset_reference_uuid='00000000-0000-4000-8000-000000000001'))
        with self.assertRaisesRegex(ValueError,'Training'):
            codec.ProjectFile(io.BytesIO(data))

    def test_absolute_asset_reference_is_rejected(self):
        def absolute(value): value['references'][0]['locator']['absolute_fallback']='/private/file'
        data=rewrite_chapter(self.fixture(),b'REFS',absolute)
        with self.assertRaises(ValueError):codec.ProjectFile(io.BytesIO(data))

    def test_unreferenced_history_bytes_are_rejected(self):
        with self.assertRaises(ValueError):codec.ProjectFile(io.BytesIO(self.fixture()+b'CKPT private history'))

    def test_missing_compatibility_gate_is_rejected(self):
        data=bytearray(self.fixture())
        offset=next(offset for offset in (4096,8192) if data[offset:offset+8]==b'LFSHEAD\0')
        commit=struct.unpack_from('<Q',data,offset+80)[0]
        data[commit+192:commit+208]=bytes(16)
        checksum=codec._crc(data[commit:commit+252])
        struct.pack_into('<I',data,commit+252,checksum)
        struct.pack_into('<I',data,offset+104,checksum)
        struct.pack_into('<I',data,offset+4092,codec._crc(data[offset:offset+4092]))
        with self.assertRaisesRegex(ValueError,'compatibility'):codec.ProjectFile(io.BytesIO(data))

    def test_asset_ranges_cannot_escape_container(self):
        stream=codec.SliceReader(io.BytesIO(b'privatePUBLICprivate'),7,6)
        self.assertEqual(stream.read(100),b'PUBLIC')
        with self.assertRaises(ValueError):stream.seek(-1)
        with self.assertRaises(ValueError):stream.seek(7)

from lfs_plugins import portable_project as codec
FIXTURES = Path(__file__).parents[1] / "data"
NATIVE_SPZ = Path(__file__).parents[1] / "data" / "spz"


def _validate(payload, count):
    return codec.validate_spz(codec.SliceReader(io.BytesIO(payload), 0, len(payload)), count)


def _spz_v4(count, sh_degree=0, *, flags=0, coord=None, unknown_ext=None, compressed=None):
    expected = codec.spz_stream_sizes(count, sh_degree)
    streams = len(expected)
    extensions = b''
    if unknown_ext is not None:
        extensions += struct.pack('<II', 0xDEAD0001, len(unknown_ext)) + unknown_ext
        flags |= 2
    if coord is not None:
        extensions += struct.pack('<II I', 0xADBE0003, 4, coord)
        flags |= 2
    toc_offset = 32 + len(extensions)
    header = struct.pack('<III BBBB I 12s', 0x5053474e, 4, count, sh_degree, 12, flags, streams, toc_offset, b'\x00' * 12)
    if compressed is None:
        compressed = [1] * streams
    toc = b''.join(struct.pack('<QQ', size, expected[i]) for i, size in enumerate(compressed))
    payload = b''.join(bytes(size) for size in compressed)
    return header + extensions + toc + payload


class SpzPortableValidationTests(unittest.TestCase):
    def test_v4_header_count_degree_aa_and_coordinate_extension(self):
        payload = _spz_v4(64, 2, flags=1, coord=6)
        self.assertEqual(_validate(payload, 64), 2)
        self.assertEqual(_validate(_spz_v4(8, 0, coord=0), 8), 0)

    def test_exact_stream_counts_for_sh0_and_sh3(self):
        self.assertEqual(codec.spz_stream_sizes(64, 0), [576, 64, 192, 192, 256])
        self.assertEqual(len(codec.spz_stream_sizes(64, 0)), 5)
        self.assertEqual(codec.spz_stream_sizes(4096, 3)[-1], 4096 * 15 * 3)
        self.assertEqual(len(codec.spz_stream_sizes(4096, 3)), 6)
        self.assertEqual(_validate(_spz_v4(8, 0), 8), 0)
        self.assertEqual(_validate(_spz_v4(8, 3), 8), 3)

    def test_rejects_wrong_stream_count_and_uncompressed_toc(self):
        payload = bytearray(_spz_v4(8, 0))
        payload[15] = 1
        with self.assertRaises(ValueError):
            _validate(bytes(payload), 8)
        payload = bytearray(_spz_v4(8, 0))
        struct.pack_into('<Q', payload, 40, 10**12)
        with self.assertRaises(ValueError):
            _validate(bytes(payload), 8)
        payload = bytearray(_spz_v4(8, 0))
        struct.pack_into('<Q', payload, 32, 0)
        with self.assertRaises(ValueError):
            _validate(bytes(payload), 8)

    def test_rejects_count_mismatch_unknown_version_truncated_and_legacy_gzip(self):
        payload = _spz_v4(8, 0)
        with self.assertRaises(ValueError):
            _validate(payload, 9)
        broken = bytearray(payload)
        struct.pack_into('<I', broken, 4, 99)
        with self.assertRaises(ValueError):
            _validate(bytes(broken), 8)
        with self.assertRaises(ValueError):
            _validate(payload[:20], 8)
        gzipped = gzip.compress(struct.pack('<III BBB x', 0x5053474e, 3, 8, 1, 12, 1))
        with self.assertRaises(ValueError):
            _validate(gzipped, 8)

    def test_rejects_unknown_coordinate_duplicate_coord_and_flag_mismatch(self):
        with self.assertRaises(ValueError):
            _validate(_spz_v4(8, 0, coord=99), 8)
        dup = _spz_v4(8, 0, coord=6)
        extra = struct.pack('<II I', 0xADBE0003, 4, 4)
        toc = 32 + 12
        patched = bytearray(dup)
        patched[16:20] = struct.pack('<I', toc + len(extra))
        patched = patched[:toc] + extra + patched[toc:]
        with self.assertRaises(ValueError):
            _validate(bytes(patched), 8)
        flagged = bytearray(_spz_v4(8, 0))
        flagged[14] = 2
        with self.assertRaises(ValueError):
            _validate(bytes(flagged), 8)

    def test_skips_unknown_extensions_without_materializing_payload(self):
        blob = b'\x11' * 4096
        payload = _spz_v4(8, 1, unknown_ext=blob, coord=4)
        self.assertEqual(_validate(payload, 8), 1)

    def test_highly_compressed_exact_toc_is_accepted(self):
        count = 1_000_000
        payload = _spz_v4(count, 0)
        uncompressed = sum(codec.spz_stream_sizes(count, 0))
        self.assertGreater(uncompressed, 8 * 1024 * 1024)
        self.assertGreater(uncompressed / len(payload), 20)
        self.assertEqual(_validate(payload, count), 0)

    def test_native_writer_fixtures_validate(self):
        for name, count, degree in [('native-v4.spz', 64, 0), ('reference-sh3-v4.spz', 4096, 3)]:
            data = (NATIVE_SPZ / name).read_bytes()
            self.assertEqual(_validate(data, count), degree, name)
        # Native writer emits one Adobe coordinate record before the TOC.
        data = bytearray((NATIVE_SPZ / 'reference-sh3-v4.spz').read_bytes())
        self.assertEqual(struct.unpack_from('<I', data, 32)[0], 0xADBE0003)
        for coordinate in (0, 4, 6, 9, 14, 16):
            struct.pack_into('<I', data, 40, coordinate)
            self.assertEqual(_validate(bytes(data), 4096), 3)

    def test_does_not_require_python_zstd(self):
        self.assertFalse(hasattr(codec, 'zstd') or hasattr(codec, 'zstandard'))
