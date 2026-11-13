# SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
# SPDX-License-Identifier: GPL-3.0-or-later
"""The shared client wheel has the same text payload on Windows and Linux."""
import importlib.util
import base64
import csv
import hashlib
import io
from pathlib import Path
from unittest.mock import patch
from zipfile import ZipFile
import pytest


def test_portal_client_stages_text_with_lf(tmp_path, monkeypatch):
    repository = Path(__file__).resolve().parents[2]
    generator = repository / 'tools' / 'package_portal_client.py'
    monkeypatch.syspath_prepend(str(generator.parent))
    spec = importlib.util.spec_from_file_location('portal_client_packager', generator)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    source = tmp_path / 'source'
    module.__file__ = str(source / 'tools' / generator.name)
    for name in [*(f'src/python/lfs_plugins/{item}.py' for item in module.MODULES),
                 'src/python/_lfs_path_io.py', 'LICENSE']:
        path = source / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(b'# text\r\n')
    (source / 'src/python/lfs_plugins/portal_account.py').write_bytes(b'from _lfs_path_io import Path\r\n')
    (source / 'src/python/lfs_plugins/distrusted_cas.pem').write_bytes(b'certificate\r\n')
    with patch.object(module, 'normalize'), patch.object(module.subprocess, 'run') as build:
        def inspect(*args, cwd, **kwargs):
            root = Path(cwd)
            for path in [*root.rglob('*.py'), *root.rglob('*.pem'), root / 'LICENSE', root / 'pyproject.toml']:
                assert b'\r' not in path.read_bytes(), path
            assert (root / 'lichtfeld_portal/distrusted_cas.pem').read_bytes() == b'certificate\n'
            assert (root / 'lichtfeld_portal/portal_account.py').read_text() == 'from .path_io import Path\n'
            (root / 'dist').mkdir()
            (root / 'dist/lichtfeld_portal_client-0.1.3-py3-none-any.whl').write_bytes(b'wheel')
        build.side_effect = inspect
        module.build(tmp_path, '0.1.3')
        build.assert_called_once()
    output = tmp_path / 'lichtfeld_portal_client-0.1.3-py3-none-any.whl'
    assert output.read_bytes() == b'wheel'
    with patch.object(module.subprocess, 'run', side_effect=inspect), \
         patch.object(module, 'normalize', side_effect=OSError('normalization failed')):
        with pytest.raises(OSError, match='normalization failed'):
            module.build(tmp_path, '0.1.3')
    assert output.read_bytes() == b'wheel'


def test_normalization_rebuilds_record_and_preserves_name(tmp_path, monkeypatch):
    from wheel.wheelfile import WheelFile
    repository = Path(__file__).resolve().parents[2]
    monkeypatch.syspath_prepend(str(repository / 'tools'))
    from normalize_shared_wheel import normalize
    path = tmp_path / 'example-1.0-py3-none-any.whl'
    with WheelFile(path, 'w') as archive:
        archive.writestr('example/__init__.py', b'')
        archive.writestr('example-1.0.dist-info/WHEEL', b'Wheel-Version: 1.0\nRoot-Is-Purelib: true\nTag: py3-none-any\n')
        archive.writestr('example-1.0.dist-info/METADATA', b'Metadata-Version: 2.4\r\nName: example\r\nVersion: 1.0\r\n')
    normalize(path)
    with ZipFile(path) as archive:
        name = 'example-1.0.dist-info/METADATA'
        data = archive.read(name)
        assert b'\r' not in data
        record = dict((row[0], (row[1], row[2])) for row in csv.reader(io.StringIO(
            archive.read('example-1.0.dist-info/RECORD').decode())))
        digest = base64.urlsafe_b64encode(hashlib.sha256(data).digest()).rstrip(b'=').decode()
        assert record[name] == ('sha256=' + digest, str(len(data)))
