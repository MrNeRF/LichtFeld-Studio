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
