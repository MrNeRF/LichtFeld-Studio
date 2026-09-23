"""Rebuild small, deterministic samples of the gallery publishing subset.

The base chapters and payloads come from the native writer fixtures beside this
directory. Container records are packed afresh to keep the corpus compact.
"""
import copy
import gzip
import io
import json
from pathlib import Path
import struct
import sys
import uuid
from zipfile import ZIP_DEFLATED, ZIP_STORED, ZipFile, ZipInfo

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[2] / 'src' / 'python'))
from lfs_plugins import portable_project as codec

SOURCE = HERE.parent
PROJECT_ID = uuid.UUID('10000000-0000-4000-8000-000000000001').bytes
FILE_ID = uuid.UUID('20000000-0000-4000-8000-000000000002').bytes
COMMIT_ID = uuid.UUID('30000000-0000-4000-8000-000000000003').bytes
NODE_ID = '40000000-0000-4000-8000-000000000004'
ENV_ID = '50000000-0000-4000-8000-000000000005'


def align(value):
    return (value + 63) & ~63


def crc(data):
    return codec._crc(data)


def seal(blob):
    struct.pack_into('<I', blob, len(blob) - 4, crc(blob[:-4]))


def base(kind):
    source = io.BytesIO((SOURCE / f'portable-{kind}.licht').read_bytes())
    chapters, assets = codec.read_project(source)
    node = chapters[b'SCNG']['nodes'][0]
    raw = assets[node['uuid']]
    source.seek(raw['offset'])
    payload = source.read(raw['size'])
    environment = next(asset for key, asset in assets.items() if key != node['uuid'])
    source.seek(environment['offset'])
    return chapters, payload, source.read(environment['size'])


def zip_payload(kind, *, license_name=None, license_method=ZIP_STORED, license_data=b'License: Example',
                extra=None, extra_method=ZIP_STORED, bad_count=False, count=1):
    output = io.BytesIO()
    name = 'meta.json' if kind == 'sog' else 'lod-meta.json'
    manifest = {'count': count + 1 if bad_count else count} if kind == 'sog' else {
        'counts': [count + 1 if bad_count else count], 'filenames': [name]}
    with ZipFile(output, 'w') as archive:
        archive.writestr(ZipInfo(name), json.dumps(manifest, sort_keys=True), compress_type=ZIP_STORED)
        if license_name:
            archive.writestr(ZipInfo(license_name), license_data, compress_type=license_method)
        if extra:
            archive.writestr(ZipInfo(extra), b'extra', compress_type=extra_method)
    return output.getvalue()


def native_with_license(kind, name, method):
    payload = base(kind)[1]
    output = io.BytesIO()
    with ZipFile(io.BytesIO(payload)) as original, ZipFile(output, 'w') as archive:
        for entry in original.infolist():
            archive.writestr(ZipInfo(entry.filename), original.read(entry), compress_type=entry.compress_type)
        archive.writestr(ZipInfo(name), b'License: Example', compress_type=method)
    return output.getvalue()


def project(kind, *, environment=False, license=False, payload=None, history=False,
            training=False, extra_asset=False, degree=0):
    chapters, native_payload, env_payload = base('sog' if kind == 'spz' else kind)
    chapters = copy.deepcopy(chapters)
    payload = native_payload if payload is None else payload
    node = chapters[b'SCNG']['nodes'][0]
    node['uuid'] = NODE_ID
    node['name'] = 'sample'
    node['payload']['instance_uuid'] = NODE_ID
    node['payload']['source_kind'] = kind
    node['publication']['count'] = 64
    node['publication']['sh_degree'] = degree
    chapters[b'PROJ']['project_uuid'] = str(uuid.UUID(bytes=PROJECT_ID))
    chapters[b'PROJ']['created_at_unix_ns'] = 0
    chapters[b'PROJ']['modified_at_unix_ns'] = 0
    if license:
        chapters[b'PROJ']['license'] = {'identifier': 'CC-BY-4.0', 'notice': 'Sample credit'}
    if training:
        chapters[b'PROJ']['project_lineage'] = [str(uuid.UUID(bytes=COMMIT_ID))]
    if environment:
        ref = chapters[b'REFS']['references'][0]
        ref['uuid'] = ENV_ID
        ref['locator']['preferred'] = ENV_ID + '.lfsenv'
        chapters[b'VIEW']['render_settings']['environment_reference_uuid'] = ENV_ID
    else:
        chapters[b'REFS']['references'] = []
        chapters[b'VIEW']['render_settings']['environment_reference_uuid'] = None
    records = [(name, json.dumps(value, sort_keys=True, separators=(',', ':')).encode()
                if isinstance(value, (dict, list)) else value, PROJECT_ID)
               for name, value in sorted(chapters.items())]
    records.append((b'DSRC', payload, uuid.UUID(NODE_ID).bytes))
    if environment:
        records.append((b'DSRC', env_payload, uuid.UUID(ENV_ID).bytes))
    if extra_asset:
        records.append((b'DSRC', b'unreferenced', uuid.UUID('60000000-0000-4000-8000-000000000006').bytes))
    data = bytearray(65536)
    rows = []
    for kind_code, content, identity in records:
        start = align(len(data))
        data.extend(bytes(start - len(data)))
        header = bytearray(64)
        header[:4] = kind_code
        struct.pack_into('<HH', header, 4, 1, 64)
        header[8:24] = identity
        struct.pack_into('<QQ', header, 32, len(content), len(content))
        struct.pack_into('<I', header, 56, crc(content))
        seal(header)
        data.extend(header)
        data.extend(content)
        row = bytearray(96)
        row[:4] = kind_code
        struct.pack_into('<H', row, 4, 1)
        row[16:32] = identity
        struct.pack_into('<QQQQQ', row, 32, start, start + 64, len(content), len(content), 1)
        struct.pack_into('<II', row, 72, crc(content), struct.unpack_from('<I', header, 60)[0])
        rows.append(row)
    index_offset = align(len(data))
    data.extend(bytes(index_offset - len(data)))
    index = bytearray(64 + len(rows) * 96)
    index[:8] = b'LFSINDEX'
    struct.pack_into('<HHHH', index, 8, 1, 64, 96, 1)
    struct.pack_into('<QQ', index, 16, len(rows), 1)
    index[32:48] = COMMIT_ID
    for i, row in enumerate(rows):
        index[64 + i * 96:160 + i * 96] = row
    data.extend(index)
    commit_offset = align(len(data))
    data.extend(bytes(commit_offset - len(data)))
    commit = bytearray(256)
    commit[:8] = b'LFSCOMIT'
    struct.pack_into('<HH', commit, 8, 256, 1)
    commit[16:32], commit[32:48], commit[48:64] = PROJECT_ID, FILE_ID, COMMIT_ID
    struct.pack_into('<Q', commit, 64, 2 if history else 1)
    struct.pack_into('<QQQ', commit, 136, index_offset, len(index), len(index))
    struct.pack_into('<II', commit, 160, crc(index), crc(index))
    struct.pack_into('<Q', commit, 176, commit_offset + 256)
    commit[193] = 2
    seal(commit)
    data.extend(commit)
    head = bytearray(4096)
    head[:8] = b'LFSHEAD\0'
    struct.pack_into('<IIQQ', head, 8, 0, 4096, 1, 1)
    head[32:48], head[48:64], head[64:80] = PROJECT_ID, FILE_ID, COMMIT_ID
    struct.pack_into('<QQQQ', head, 80, commit_offset, 256, len(data), 0)
    struct.pack_into('<I', head, 104, struct.unpack_from('<I', commit, 252)[0])
    seal(head)
    data[4096:8192] = head
    superblock = bytearray(256)
    superblock[:8] = b'\x89LFS\r\n\x1a\n'
    struct.pack_into('<HHII', superblock, 8, 1, 1, 0x01020304, 256)
    superblock[24:40], superblock[40:56] = PROJECT_ID, FILE_ID
    struct.pack_into('<QQIIQ', superblock, 56, 4096, 8192, 4096, 2, 65536)
    seal(superblock)
    data[:256] = superblock
    return bytes(data)


def samples():
    spz = (SOURCE / 'spz' / 'native-v4.spz').read_bytes()
    yield 'valid_ply.licht', project('ply'), 'accept', None
    yield 'valid_sog_environment.licht', project('sog', environment=True, license=True,
        payload=native_with_license('sog', 'license.txt', ZIP_STORED)), 'accept', None
    yield 'valid_ssog_license.licht', project('ssog', payload=native_with_license(
        'ssog', 'LICENSE.md', ZIP_DEFLATED)), 'accept', None
    yield 'valid_spz.licht', project('spz', payload=spz), 'accept', None
    yield 'history.licht', project('ply', history=True), 'reject', 'history'
    yield 'training.licht', project('ply', training=True), 'reject', 'training'
    yield 'unreferenced_asset.licht', project('ply', extra_asset=True), 'reject', 'unreferenced_asset'
    yield 'degree_over.licht', project('sog', degree=1), 'reject', 'degree'
    for name, kw, verdict, reason in (
        ('license_case.sog', {'license_name': 'LiCeNsE'}, 'accept', None),
        ('license_deflated.sog', {'license_name': 'LICENSE.md', 'license_method': ZIP_DEFLATED}, 'accept', None),
        ('nested_license.sog', {'license_name': 'sub/license.txt'}, 'reject', 'unreferenced_member'),
        ('oversized_license.sog', {'license_name': 'license.txt', 'license_data': b'x' * 65537}, 'reject', 'license_size'),
        ('unreferenced_member.ssog', {'extra': 'readme.txt'}, 'reject', 'unreferenced_member'),
        ('deflated_texture.sog', {'extra': 'texture.webp', 'extra_method': ZIP_DEFLATED}, 'reject', 'compressed_texture'),
        ('bad_manifest_count.sog', {'bad_count': True}, 'reject', 'manifest_count'),
    ):
        yield name, zip_payload(name.rsplit('.', 1)[1], **kw), verdict, reason
    yield 'valid_spz_extension.spz', spz, 'accept', None
    bad = bytearray(spz)
    bad[14] |= 0x80
    yield 'spz_flags.spz', bytes(bad), 'reject', 'spz_flags'
    bad = bytearray(spz)
    struct.pack_into('<I', bad, 40, 99)
    yield 'spz_extension.spz', bytes(bad), 'reject', 'spz_extension'
    bad = bytearray(spz)
    struct.pack_into('<I', bad, 4, 3)
    yield 'spz_v3.spz', bytes(bad), 'reject', 'spz_version'
    legacy = struct.pack('<III BBB x', 0x5053474e, 3, 64, 0, 12, 1)
    yield 'spz_v3_gzip.spz', gzip.compress(legacy, mtime=0), 'reject', 'spz_legacy'


def main():
    expected = {}
    for name, payload, verdict, reason in samples():
        (HERE / name).write_bytes(payload)
        expected[name] = {'verdict': verdict, **({'reason': reason} if reason else {})}
    (HERE / 'expected.json').write_text(json.dumps(expected, indent=2, sort_keys=True) + '\n')


if __name__ == '__main__':
    main()
