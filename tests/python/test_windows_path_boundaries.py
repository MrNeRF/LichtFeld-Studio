# SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
# SPDX-License-Identifier: GPL-3.0-or-later
import importlib.util
from pathlib import Path
import unittest

spec = importlib.util.spec_from_file_location("path_boundaries", Path(__file__).resolve().parents[2] / "tools/check_windows_path_boundaries.py")
checker = importlib.util.module_from_spec(spec)
spec.loader.exec_module(checker)


class WindowsPathBoundaries(unittest.TestCase):
    def test_detects_filesystem_bypasses(self):
        for source in ["std::filesystem::exists(file);", "namespace fs = std::filesystem;", "std::ifstream stream(file);", "auto stream = std::ofstream(file);", "std::filesystem::recursive_directory_iterator(root);"]:
            with self.subTest(source=source):
                self.assertTrue(checker.violations(source))

    def test_accepts_logical_paths_and_adapted_io(self):
        self.assertFalse(checker.violations("std::filesystem::path file;\nlfs::core::fs::exists(file);\nstd::ifstream& input;\n// std::filesystem::exists(file);"))

    def test_repository_uses_shared_boundary(self):
        self.assertEqual(checker.main(), 0)


if __name__ == "__main__":
    unittest.main()
