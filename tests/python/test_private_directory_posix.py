# SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
# SPDX-License-Identifier: GPL-3.0-or-later
"""POSIX storage contracts; also runnable with standard-library unittest."""
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

repo = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(repo / 'src/python'))
from lfs_plugins.private_directory import mkdir_private
from lfs_plugins.credential_storage import FileBackend
from lfs_plugins.portal_account import _locked_sidecar

@unittest.skipUnless(os.name == 'posix', 'POSIX storage contracts')
class PosixStorageContracts(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory(prefix='lfs-posix-storage-')
        self.root = Path(self.temporary.name)
    def tearDown(self):
        self.temporary.cleanup()
    def test_private_permissions_and_file_roundtrip(self):
        account = self.root / 'account'
        mkdir_private(account)
        self.assertEqual(account.stat().st_mode & 0o777, 0o700)
        backend = FileBackend(account / 'dummy')
        backend.write(b'dummy-posix')
        self.assertEqual(backend.read(), b'dummy-posix')
        self.assertEqual(backend.path.stat().st_mode & 0o777, 0o600)
        with _locked_sidecar(account / 'dummy.lock'):
            self.assertEqual((account / 'dummy.lock').stat().st_mode & 0o777, 0o600)
    def test_existing_directory_permissions_and_contents_are_preserved(self):
        directory = self.root / 'existing'
        directory.mkdir()
        directory.chmod(0o750)
        (directory / 'marker').write_text('preserve')
        mkdir_private(directory, exist_ok=True)
        self.assertEqual(directory.stat().st_mode & 0o777, 0o750)
        self.assertEqual((directory / 'marker').read_text(), 'preserve')
    def test_parent_creation_and_path_collisions(self):
        private = self.root / 'parent' / 'private'
        mkdir_private(private, parents=True)
        self.assertEqual(private.stat().st_mode & 0o777, 0o700)
        with self.assertRaises(FileExistsError):
            mkdir_private(private)
        file = self.root / 'file'
        file.write_bytes(b'dummy')
        with self.assertRaises(FileExistsError):
            mkdir_private(file, exist_ok=True)
        with self.assertRaises(FileNotFoundError):
            mkdir_private(self.root / 'missing' / 'private')
    def test_lock_excludes_another_process(self):
        path = self.root / 'private' / 'credentials.lock'
        code = '''from pathlib import Path
import sys
from lfs_plugins.portal_account import _locked_sidecar
try:
    with _locked_sidecar(Path(sys.argv[1]), blocking=False):
        sys.exit(1)
except BlockingIOError:
    sys.exit(0)
'''
        environment = dict(os.environ, PYTHONPATH=str(repo / 'src/python'))
        with _locked_sidecar(path):
            child = subprocess.run([sys.executable, '-c', code, str(path)], env=environment,
                                   capture_output=True, text=True, timeout=10)
            self.assertEqual(child.returncode, 0, child.stderr)

if __name__ == '__main__':
    unittest.main(verbosity=2)
