# SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
# SPDX-License-Identifier: GPL-3.0-or-later
"""Bounded saved-project content evidence for deciding PATCH versus replace.

This is not a project loader. Unsupported index encodings return no evidence,
which requires native replacement. Display freshness still
uses commit UUID alone (a save is never described as a geometry edit).
"""
import hashlib
import struct
from pathlib import Path


def saved_content_stamp(path):
    from .portable_project import _crc
    try:
        with Path(path).open('rb') as source:
            size = source.seek(0, 2)
            def read(offset, count):
                if offset < 0 or count < 0 or count > 8 * 1024 * 1024 or offset + count > size:
                    raise ValueError('Invalid range')
                source.seek(offset)
                data = source.read(count)
                if len(data) != count:
                    raise ValueError('Short read')
                return data
            def valid_record(data, magic):
                return data.startswith(magic) and _crc(data[:-4]) == struct.unpack_from('<I',data,len(data)-4)[0]
            superblock = read(0,256)
            if not valid_record(superblock,b'\x89LFS\r\n\x1a\n'):
                return ''
            heads = [read(slot,4096) for slot in (4096,8192)]
            heads = [h for h in heads if valid_record(h,b'LFSHEAD\0') and h[32:64] == superblock[24:56]]
            if not heads:
                return ''
            head = max(heads,key=lambda h: struct.unpack_from('<Q',h,16)[0])
            commit = read(struct.unpack_from('<Q',head,80)[0],256)
            if not valid_record(commit,b'LFSCOMIT') or commit[48:64] != head[64:80]:
                return ''
            offset,stored,decoded = struct.unpack_from('<QQQ',commit,136)
            if stored != decoded or struct.unpack_from('<II',commit,168) != (0,0):
                return ''
            index = read(offset,stored)
            if index[:8] != b'LFSINDEX' or _crc(index) != struct.unpack_from('<I',commit,160)[0]:
                return ''
            if struct.unpack_from('<HHH',index,8) != (1,64,96):
                return ''
            count = struct.unpack_from('<Q',index,16)[0]
            if len(index) != 64 + count * 96:
                return ''
            records, view_records = [], []
            for i in range(count):
                row=index[64+i*96:160+i*96]
                if row[:4] in (b'SCNG',b'REFS',b'DSRC',b'SPLT',b'CKPT',b'SELM'):
                    # Include decoded payload and native chunk-header checksums;
                    # ignore index/chapter generation and file layout offsets.
                    records.append(row[:32]+row[48:64]+row[72:80])
                elif row[:4] in (b'VIEW', b'SEQR'):
                    view_records.append(row[:32]+row[48:64]+row[72:80])
            if not any(r[:4] == b'SCNG' for r in records):
                return ''
            if len(view_records) != 2:
                return ''
            return ':'.join(hashlib.sha256(b''.join(sorted(rows))).hexdigest() for rows in (records, view_records))
    except (OSError, ValueError, struct.error):
        return ''
