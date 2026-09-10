# SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
# SPDX-License-Identifier: GPL-3.0-or-later
"""Portable native projects: a fresh .licht container with embedded splat and HDR assets.

Mirrored in the portal. Only the bounded, stored, single-generation publishing
subset is admitted here; this is deliberately not a general project importer.
"""
import hashlib
import json
import io
import uuid
import zlib
import stat
import struct
from zipfile import ZIP_STORED, ZipFile

from . import gallery_bundle as codec

MAX_PROJECT_BYTES = 8 * 1024 * 1024
CHAPTERS = {b'PROJ', b'REFS', b'SCNG', b'PRMS', b'SELM', b'VIEW', b'GUIL', b'EDTR', b'SEQR', b'METR'}


def _check(value, message='Invalid portable LichtFeld project.'):
    codec._require(value, message)


def _crc(data):
    # Castagnoli, as used by the native container (not ZIP's IEEE CRC).
    value = 0xffffffff
    for byte in data:
        value = _CRC_TABLE[(value ^ byte) & 255] ^ (value >> 8)
    return value ^ 0xffffffff


def _crc_table():
    result = []
    for byte in range(256):
        for _ in range(8):
            byte = (byte >> 1) ^ (0x82f63b78 if byte & 1 else 0)
        result.append(byte)
    return tuple(result)


_CRC_TABLE = _crc_table()


def read_project(source):
    """Validate container identity, checksums, live chunks and ALL unused bytes.

    Reject history, compressed/opaque chunks, stale trailing payloads, external
    references and editor buffers before allowing native project opening.
    """
    size_total = source.seek(0, 2)
    _check(65536 < size_total <= 100 * 1024**3)
    occupied = []

    def region(start, size, read=True):
        _check(0 <= start <= size_total and 0 <= size <= size_total - start)
        occupied.append((start, start + size))
        if not read:
            return None
        _check(size <= MAX_PROJECT_BYTES)
        source.seek(start)
        data = source.read(size)
        _check(len(data) == size)
        return data

    def number(blob, offset, kind='Q'):
        return struct.unpack_from('<' + kind, blob, offset)[0]

    def checksum(blob):
        _check(_crc(blob[:-4]) == number(blob, len(blob) - 4, 'I'), 'Project checksum failed.')

    superblock = region(0, 256)
    _check(superblock[:8] == b'\x89LFS\r\n\x1a\n')
    checksum(superblock)
    _check(number(superblock, 8, 'H') == 1 and number(superblock, 10, 'H') <= 1)
    _check(number(superblock, 12, 'I') == 0x01020304 and number(superblock, 16, 'I') == 256)
    _check(number(superblock, 20, 'I') == 0)
    project_id, file_id = superblock[24:40], superblock[40:56]
    _check(any(project_id) and any(file_id))
    _check(struct.unpack_from('<QQIIQ', superblock, 56) == (4096, 8192, 4096, 2, 65536))
    _check(not any(superblock[96:252]), 'Project contains recovery or private metadata.')
    heads = []
    for slot, offset in enumerate((4096, 8192)):
        head = region(offset, 4096)
        if not any(head):
            continue
        _check(head[:8] == b'LFSHEAD\0')
        checksum(head)
        _check(number(head, 8, 'I') == slot and number(head, 12, 'I') == 4096)
        _check(number(head, 16) == 1 and number(head, 24) == 1, 'Publish a fresh project without history.')
        _check(head[32:48] == project_id and head[48:64] == file_id)
        _check(number(head, 88) == 256 and number(head, 96) == size_total)
        _check(not any(head[108:4092]), 'Project contains unexpected preview or session bytes.')
        heads.append(head)
    _check(len(heads) == 1, 'Publish a fresh project without history.')
    head = heads[0]
    commit_offset = number(head, 80)
    commit = region(commit_offset, 256)
    checksum(commit)
    _check(commit[:8] == b'LFSCOMIT' and struct.unpack_from('<HH', commit, 8) == (256, 1))
    _check(commit[16:32] == project_id and commit[32:48] == file_id and commit[48:64] == head[64:80])
    _check(number(commit, 64) == 1 and not any(commit[72:96]), 'Project history is excluded from publishing.')
    _check(number(commit, 176) == size_total == commit_offset + 256)
    _check(int.from_bytes(commit[192:208], 'little') & (1 << 9), 'Project is missing its embedded-asset compatibility requirement.')
    _check(number(commit, 252, 'I') == number(head, 104, 'I'))
    _check(number(commit, 168, 'I') == 0 and number(commit, 172, 'I') == 0, 'Unsupported project index compression.')
    index_offset, size, decoded_size = struct.unpack_from('<QQQ', commit, 136)
    _check(size == decoded_size and 64 <= size <= 64 + (len(CHAPTERS) + codec.MAX_NODES + 1) * 96)
    index = region(index_offset, size)
    _check(_crc(index) == number(commit, 160, 'I') == number(commit, 164, 'I'))
    _check(index[:8] == b'LFSINDEX' and struct.unpack_from('<HHHH', index, 8) == (1, 64, 96, 1))
    count = number(index, 16)
    _check(len(CHAPTERS) < count <= len(CHAPTERS) + codec.MAX_NODES + 1 and size == 64 + count * 96)
    _check(number(index, 24) == 1 and index[32:48] == commit[48:64])
    _check(number(index, 48, 'I') <= 3 and not any(index[52:64]))
    chapters, assets = {}, {}
    for i in range(count):
        row = index[64 + i * 96:160 + i * 96]
        kind = row[:4]
        _check((kind in CHAPTERS and kind not in chapters) or kind == b'DSRC', 'Checkpoints and unknown project chapters cannot be published.')
        flags = number(row, 8, 'I')
        _check(struct.unpack_from('<HBB', row, 4) == (1, 0, 0) and flags in (0, 2))
        _check(flags == 0 or kind == b'DSRC')
        _check((kind == b'DSRC' or row[16:32] == project_id) and not any(row[12:16]) and not any(row[80:]))
        header_offset, payload_offset, stored, decoded, generation = struct.unpack_from('<QQQQQ', row, 32)
        _check(generation == 1 and stored == decoded and 0 < stored <= (100 * 1024**3 if kind == b'DSRC' else MAX_PROJECT_BYTES))
        _check(header_offset >= 65536 and header_offset % 64 == payload_offset % 64 == 0)
        _check(payload_offset + stored <= index_offset)
        header = region(header_offset, 64)
        checksum(header)
        _check(header[:4] == kind and struct.unpack_from('<HH', header, 4) == (1, 64) and header[8:24] == row[16:32])
        _check(number(header, 24, 'I') == flags and number(header, 28, 'H') == 0)
        if flags == 2:
            _check(number(header, 30, 'H') == 1 and number(header, 48) == header_offset + 64)
            table_offset = header_offset + 64
            table = region(table_offset, 64)
            checksum(table)
            _check(table[:8] == b'LFSBCRC\0' and struct.unpack_from('<HHHHIIQQQ', table, 8) ==
                   (1, 64, 4, 1, 4 * 1024**2, 0, payload_offset, stored, (stored + 4 * 1024**2 - 1) // (4 * 1024**2)))
            _check(not any(table[52:60]))
            entries_size = number(table, 40) * 4
            _check(payload_offset == (table_offset + 64 + entries_size + 63) // 64 * 64)
            entries = region(table_offset + 64, entries_size)
            _check(_crc(entries) == number(table, 48, 'I'), 'Project block table checksum failed.')
        else:
            _check(not any(header[30:32]) and not any(header[48:56]) and payload_offset == header_offset + 64)
            _check(stored < 1024**3, 'Large assets require native block checksums.')
        _check(struct.unpack_from('<QQ', header, 32) == (stored, stored))
        if kind == b'DSRC':
            identity = str(uuid.UUID(bytes=row[16:32]))
            _check(identity not in assets)
            _check(number(row, 72, 'I') == number(header, 56, 'I') and number(row, 76, 'I') == number(header, 60, 'I'))
            assets[identity] = {'offset': payload_offset, 'size': stored, 'crc32c': number(row, 72, 'I')}
            region(payload_offset, stored, read=False)
            continue
        payload = region(payload_offset, stored)
        _check(_crc(payload) == number(row, 72, 'I') == number(header, 56, 'I'))
        _check(number(row, 76, 'I') == number(header, 60, 'I'))
        if kind in (b'SELM', b'METR'):
            chapters[kind] = payload
        else:
            try:
                chapters[kind] = json.loads(payload, object_pairs_hook=codec._object, parse_constant=codec._constant)
            except (ValueError, UnicodeError, RecursionError):
                raise codec.BundleError('Invalid project chapter JSON.') from None
    cursor = 0
    for start, end in sorted(occupied):
        _check(cursor <= start, 'Overlapping project records.')
        source.seek(cursor)
        while cursor < start:
            gap = source.read(min(codec.CHUNK_BYTES, start - cursor))
            _check(gap and not any(gap), 'Project contains unreferenced data or history.')
            cursor += len(gap)
        cursor = end
    _check(cursor == size_total and chapters.keys() == CHAPTERS)
    project = chapters[b'PROJ']
    _check(project.get('dataset_reference_uuid') is None and project.get('project_lineage') == []
           and all(project.get(key) == [] for key in ('embed_decisions', 'provenance', 'embedded_payloads')),
           'Training sources, history and import provenance cannot be published.')
    _check(chapters[b'METR'] == b'LFMETR\r\n' + struct.pack('<I', 1) + bytes(40),
           'Training metrics cannot be published.')
    _check(chapters[b'SELM'] == bytes.fromhex('4c53454c010040000000000000000000000000000001000040000000000000004000000000000000400000000000000040000000000000004000000000000000'),
           'Private selection state cannot be published.')
    _check(chapters[b'PRMS'].get('dataset', {}).get('timelapse_images') == []
           and 'embedded_dataset' not in chapters[b'PRMS'], 'Training images cannot be published.')
    editor = chapters[b'EDTR']
    _check(editor.get('open_files') == [] and editor.get('active_file') is None and editor.get('contains_embedded_secrets') is False,
           'Editor files and buffers cannot be published.')
    graph = chapters[b'SCNG']
    _check(graph.get('training_model_uuid') is None, 'Training state cannot be published.')
    return chapters, assets



class SliceReader(io.RawIOBase):
    def __init__(self, source, offset, size):
        self.source, self.offset, self.size, self.position = source, offset, size, 0
    def readable(self): return True
    def seekable(self): return True
    def tell(self): return self.position
    def seek(self, offset, whence=0):
        position = offset + (0 if whence == 0 else self.position if whence == 1 else self.size)
        _check(whence in (0, 1, 2) and 0 <= position <= self.size)
        self.position = position
        return position
    def read(self, size=-1):
        size = self.size - self.position if size < 0 else min(size, self.size - self.position)
        _check(size <= codec.MAX_READ_BYTES)
        self.source.seek(self.offset + self.position)
        value = self.source.read(size)
        self.position += len(value)
        return value


class ProjectFile:
    def __init__(self, source):
        self.source = source
        self.chapters, self.assets = read_project(source)
        graph = self.chapters[b'SCNG']
        nodes = graph['nodes']
        _check(isinstance(nodes, list) and 0 < len(nodes) <= codec.MAX_NODES)
        self._nodes, manifest_nodes, bound, total = [], [], set(), 0
        for i, node in enumerate(nodes):
            _check(node['type'] == 'splat' and node.get('parent_uuid') is None and node['visible'] is True)
            _check(node.get('training_enabled') is False)
            payload = node['payload']
            extension = payload['source_kind']
            _check(payload['fourcc'] == 'DSRC' and extension in ('ply', 'sog', 'ssog') and payload.get('reference_uuid') is None)
            identity = payload['instance_uuid']
            _check(identity == node['uuid'] and identity in self.assets and identity not in bound)
            bound.add(identity)
            asset = self.assets[identity]
            transform = node['local_transform']
            _check(isinstance(transform, list) and len(transform) == 16)
            matrix = codec._matrix([[transform[column * 4 + row] for column in range(4)] for row in range(4)])
            publication = node['publication']
            count, degree = publication['count'], publication['sh_degree']
            _check(type(count) is int and 0 < count <= codec.MAX_SPLATS)
            codec._degree(degree)
            stream = SliceReader(source, asset['offset'], asset['size'])
            if extension == 'ply':
                actual, stored_degree = codec._ply_header(stream, asset['size'])
                _check(actual == count and stored_degree >= degree)
            else:
                validate_compressed(stream, extension, count)
            total += count
            _check(total <= codec.MAX_SPLATS)
            self._nodes.append(asset)
            manifest_nodes.append({'file': f'nodes/{i:06d}.{extension}', 'count': count, 'shDegree': degree,
                                   'transform': matrix, 'sha256': '', 'bytes': asset['size']})
        self.manifest = {'format': codec.FORMAT, 'version': 1, 'nodes': manifest_nodes}
        settings = self.chapters[b'VIEW']['render_settings']
        references = self.chapters[b'REFS']['references']
        _check(isinstance(references, list) and len(references) <= 1)
        environment_id = settings.get('environment_reference_uuid')
        if environment_id:
            _check(len(references) == 1 and references[0]['uuid'] == environment_id)
            ref = references[0]
            _check(ref['kind'] == 'environment_map' and ref['locator']['base'] == 'project'
                   and ref['locator'].get('absolute_fallback') is None
                   and ref['locator']['preferred'] == environment_id + '.lfsenv')
            _check(environment_id in self.assets and environment_id not in bound)
            bound.add(environment_id)
            self._environment = self.assets[environment_id]
            stream = SliceReader(source, self._environment['offset'], self._environment['size'])
            width, height = codec.environment_header(stream, self._environment['size'])
            self.manifest.update(version=2, environment={'file': 'environment.lfsenv', 'width': width, 'height': height, 'sha256': ''})
        else:
            _check(not references)
        _check(bound == self.assets.keys(), 'Unreferenced embedded assets cannot be published.')

    def node_storage(self, index):
        value = self._nodes[index]
        return value['offset'], value['size'], value['crc32c']

    def environment_storage(self):
        value = self._environment
        return value['offset'], value['size'], value['crc32c']

    def _copy(self, asset, output, metadata, progress, environment=False):
        try:
            import google_crc32c
            crc = google_crc32c.Checksum()
        except ImportError:
            crc = None
        digest, completed, fallback, ieee = hashlib.sha256(), 0, 0xffffffff, 0
        stream = SliceReader(self.source, asset['offset'], asset['size'])
        while chunk := stream.read(codec.CHUNK_BYTES):
            if environment:
                codec._environment_values(chunk[16:] if completed == 0 else chunk)
            if crc:
                crc.update(chunk)
            else:
                for byte in chunk:
                    fallback = _CRC_TABLE[(fallback ^ byte) & 255] ^ (fallback >> 8)
            ieee = zlib.crc32(chunk, ieee)
            digest.update(chunk)
            output.write(chunk)
            completed += len(chunk)
            if progress: progress(completed)
        actual_crc = int.from_bytes(crc.digest(), 'big') if crc else fallback ^ 0xffffffff
        _check(completed == asset['size'] and actual_crc == asset['crc32c'], 'Embedded asset checksum failed.')
        metadata['sha256'] = digest.hexdigest()
        asset['crc32'] = ieee
        return completed

    def copy_node(self, index, output, *, progress=None):
        return self._copy(self._nodes[index], output, self.manifest['nodes'][index], progress)

    def copy_environment(self, output, *, progress=None):
        return self._copy(self._environment, output, self.manifest['environment'], progress, environment=True)


def validate_compressed(stream, extension, count):
    # Check ZIP structure and referenced texture paths before browser decoding.
    with ZipFile(stream) as archive:
        entries = archive.infolist()
        _check(1 <= len(entries) <= 100000)
        names = set()
        expanded = 0
        for entry in entries:
            name = entry.filename
            _check(name == entry.orig_filename and name not in names and not entry.is_dir()
                   and not any(c in name for c in '\\:%?#\x00') and not name.startswith('/')
                   and all(part not in ('', '.', '..') for part in name.split('/')))
            _check((entry.compress_type == ZIP_STORED or (entry.compress_type == 8 and name.endswith('.json') and entry.file_size <= codec.MAX_MANIFEST_BYTES)) and not (entry.flag_bits & 1)
                   and stat.S_IFMT(entry.external_attr >> 16) in (0, stat.S_IFREG))
            names.add(name)
            expanded += entry.file_size
            _check(expanded <= max(stream.size * 20, codec.MAX_READ_BYTES))
        manifest_name = 'meta.json' if extension == 'sog' else 'lod-meta.json'
        _check(manifest_name in names and archive.getinfo(manifest_name).file_size <= codec.MAX_MANIFEST_BYTES)
        data = json.loads(archive.read(manifest_name), object_pairs_hook=codec._object, parse_constant=codec._constant)
        _check(isinstance(data, dict))
        if extension == 'ssog':
            _check(isinstance(data.get('counts'), list) and 1 <= len(data['counts']) <= 16)
        actual = data.get('count') if extension == 'sog' else data['counts'][0]
        _check(actual == count)
        metadata_files = [manifest_name] if extension == 'sog' else data['filenames']
        _check(isinstance(metadata_files, list) and 1 <= len(metadata_files) <= len(entries))
        used = {manifest_name}
        for filename in metadata_files:
            _check(isinstance(filename, str))
            used.add(filename)
            _check(filename in names and filename.endswith('meta.json') and archive.getinfo(filename).file_size <= codec.MAX_MANIFEST_BYTES)
            meta = json.loads(archive.read(filename), object_pairs_hook=codec._object, parse_constant=codec._constant)
            prefix = filename[:-len('meta.json')]
            for key in ('means', 'quats', 'scales', 'sh0', 'shN'):
                for texture in meta.get(key, {}).get('files', []):
                    _check(isinstance(texture, str) and '/' not in texture and '\\' not in texture
                           and texture.endswith('.webp') and prefix + texture in names)

                    used.add(prefix + texture)
        _check(used == names, 'Unreferenced files cannot be published inside compressed splats.')
