# SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
# SPDX-License-Identifier: GPL-3.0-or-later
"""Requires the compiled lichtfeld extension; no window or GPU scene is needed."""
import json
import os
import ctypes
import shutil
import tempfile
import unittest
import lichtfeld as lf
from _lfs_path_io import Path, native_path


class NativeBindingPathIo(unittest.TestCase):
    def test_pathlike_inputs_keep_inspector_metadata_logical(self):
        if os.name == "nt" and os.environ.get("LFS_REQUIRE_LONG_PATHS_DISABLED"):
            query = ctypes.WinDLL("ntdll").RtlAreLongPathsEnabled
            query.restype = ctypes.c_ubyte
            self.assertFalse(query())
        root = Path(tempfile.mkdtemp(prefix="lfs-native-path-api-"))
        self.addCleanup(lambda: shutil.rmtree(native_path(root, force_extended=True)))
        deep = root
        for _ in range(16):
            deep /= "directory_\u6e2c\u8a66_01234567890123456789"
        deep.mkdir(parents=True)
        path = deep / "damaged_\u6e2c\u8a66.licht"
        path.write_bytes(b"damaged project fixture")
        from_path = lf.io.inspect_project_card(path)
        from_string = lf.io.inspect_project_card(str(path))
        self.assertEqual(str(from_path.path), str(path))
        self.assertEqual(str(from_path.path), str(from_string.path))
        self.assertNotIn("\\\\?\\", json.dumps({"path": str(from_path.path)}))
        self.assertEqual(str(lf.io.inspect_project_card(native_path(path)).path), str(path))


if __name__ == "__main__":
    unittest.main()
