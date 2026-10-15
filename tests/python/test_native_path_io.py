# SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
# SPDX-License-Identifier: GPL-3.0-or-later
import ctypes
import json
import os
import shutil
import tempfile
import unittest
import subprocess
import sys
from _lfs_path_io import Path, native_path, logical_path, open, walk, TemporaryDirectory, executable_kwargs


class NativePathIo(unittest.TestCase):
    def setUp(self):
        if os.name == "nt" and os.environ.get("LFS_REQUIRE_LONG_PATHS_DISABLED"):
            query = ctypes.WinDLL("ntdll").RtlAreLongPathsEnabled
            query.restype = ctypes.c_ubyte
            self.assertFalse(query(), "test interpreter must have executable opt-in disabled")
        self.root = Path(tempfile.mkdtemp(prefix="lfs-python-paths-"))
        self.addCleanup(lambda: shutil.rmtree(native_path(self.root, force_extended=True)))
        self.deep = self.root
        for _ in range(12):
            self.deep /= "directory_\u6e2c\u8a66_01234567890123456789"
        self.deep.mkdir(parents=True)

    @unittest.skipUnless(os.name == "nt", "Windows executable boundary")
    def test_launch_executable_from_long_directory(self):
        source = Path(sys.executable)
        target = self.deep / "python.exe"
        shutil.copyfile(source, target)
        dll_name = f"python{sys.version_info.major}{sys.version_info.minor}.dll"
        shutil.copyfile(source.parent / dll_name, self.deep / dll_name)
        command = [str(target), "--version"]
        result = subprocess.run(command, **executable_kwargs(command), capture_output=True, text=True, check=True)
        self.assertIn("Python", result.stdout)
        self.assertEqual(executable_kwargs(["python", "--version"]), {})

    def test_open_stat_replace_and_persisted_names(self):
        file = self.deep / "project_\u6e2c\u8a66.json"
        with open(str(file), "w", encoding="utf-8") as stream:
            json.dump({"path": str(file)}, stream)
        self.assertTrue(file.is_file())
        self.assertGreater(file.stat().st_size, 0)
        self.assertEqual(json.loads(file.read_text(encoding="utf-8"))["path"], str(file))
        target = file.with_name("replacement.json")
        self.assertEqual(file.replace(str(target)), target)
        self.assertTrue(target.is_file())
        self.assertFalse(str(target.resolve()).startswith("\\\\?\\"))
        self.assertEqual(list(self.deep.glob("*.json")), [target])
        self.assertEqual(Path(native_path(target)), target)
        self.assertEqual(logical_path(native_path(target)), str(target))
        if os.name == "nt":
            self.assertTrue(os.fspath(target).startswith("\\\\?\\"))

    def test_walk_and_copy_keep_short_root_with_long_descendants_working(self):
        file = self.deep / "payload.bin"
        file.write_bytes(b"payload")
        files = [str(Path(root) / name) for root, _, names in walk(self.root) for name in names]
        self.assertEqual(files, [str(file)])
        self.assertFalse(any(name.startswith("\\\\?\\") for name in files))
        with TemporaryDirectory(prefix="lfs-python-copy-") as copy:
            shutil.copytree(native_path(self.root, force_extended=True), native_path(Path(copy) / "copy", force_extended=True))
            self.assertEqual((Path(copy) / "copy" / self.deep.relative_to(self.root) / "payload.bin").read_bytes(), b"payload")

    def test_asset_catalog_roundtrip_in_long_directory(self):
        from lfs_plugins.asset_index import AssetIndex
        library = self.deep / ("x" * 250 + ".json")
        index = AssetIndex(library, self.deep / "default")
        self.assertTrue(index.load(), index.last_error)
        folder = self.deep / "assets"
        folder.mkdir()
        self.assertIsNotNone(index.add_folder(str(folder)))
        self.assertTrue(library.is_file())
        storage = library.read_text(encoding="utf-8")
        self.assertNotIn("\\\\\\\\?", storage)
        reloaded = AssetIndex(library, self.deep / "default")
        self.assertTrue(reloaded.load(), reloaded.last_error)
        self.assertEqual(reloaded.folders, index.folders)

    @unittest.skipUnless(os.name == "nt", "Windows verbatim filename semantics")
    def test_verbatim_names_keep_their_file_identity(self):
        ordinary = self.root / "name"
        ordinary.write_bytes(b"ordinary")
        for name in ["name.", "name ", "CON.licht"]:
            path = self.root / name
            path.write_bytes(b"verbatim")
            self.assertEqual(Path(native_path(path)).read_bytes(), b"verbatim")
            path.unlink()
            self.assertEqual(ordinary.read_bytes(), b"ordinary")
        self.assertEqual(native_path("NUL"), "NUL")

    def test_fd_open_is_preserved(self):
        fd, name = tempfile.mkstemp(dir=native_path(self.deep))
        with open(fd, "wb") as stream:
            stream.write(b"fd")
        self.assertEqual(Path(name).read_bytes(), b"fd")


if __name__ == "__main__":
    unittest.main()
