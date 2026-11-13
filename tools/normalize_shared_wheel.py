#!/usr/bin/env python3
# SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
# SPDX-License-Identifier: GPL-3.0-or-later
"""Canonicalize wheel metadata with the standard wheel packer."""
import os
from pathlib import Path
import subprocess
import sys
import tempfile


def normalize(path):
    path = Path(path).resolve()
    with tempfile.TemporaryDirectory(dir=path.parent) as directory:
        staging = Path(directory)
        subprocess.run([sys.executable, '-m', 'wheel', 'unpack', str(path), '-d', str(staging)], check=True)
        unpacked = next(staging.iterdir())
        metadata = next(unpacked.glob('*.dist-info/METADATA'))
        metadata.write_text(metadata.read_text(encoding='utf-8'), encoding='utf-8', newline='\n')
        output = staging / 'packed'
        output.mkdir()
        subprocess.run([sys.executable, '-m', 'wheel', 'pack', str(unpacked), '-d', str(output)],
                       check=True, env={**os.environ, 'SOURCE_DATE_EPOCH': '315532800'})
        os.replace(output / path.name, path)


if __name__ == '__main__':
    normalize(sys.argv[1])
