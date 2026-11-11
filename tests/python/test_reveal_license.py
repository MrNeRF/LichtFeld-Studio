"""Check the generated distribution notices for the reveal implementation."""

import argparse
import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "tools"))
from generate_third_party_licenses import generate, license_id_from_text


class RevealLicenseTest(unittest.TestCase):
    def test_wrapped_apache_titles_and_specific_exceptions(self):
        for name in ("LLMDET-LICENSE.txt", "TRANSFORMERS-LICENSE.txt"):
            text = (ROOT / "src/python/lfs_plugins/assets/bert" / name).read_text(encoding="utf-8")
            with self.subTest(name=name):
                self.assertEqual(license_id_from_text(text, "unknown"), "Apache-2.0")
        self.assertEqual(license_id_from_text("Apache-2.0 WITH LLVM-exception", "unknown"), "Apache-2.0 WITH LLVM-exception")
        self.assertEqual(license_id_from_text("Apache License, Version 1.0", "unknown"), "unknown")

    def test_distribution_contains_spark_notice(self):
        args = argparse.Namespace(
            source=ROOT,
            triplet_dir=ROOT / "build-tests-release/vcpkg_installed/x64-windows",
            install_prefix=ROOT / "dist-fast",
            vcpkg_root=None,
        )
        notice = generate(args)
        self.assertTrue("Spark" in notice, "Spark is missing from generated notices")
        self.assertTrue("WORLD LABS TECHNOLOGIES, INC." in notice, "Spark copyright is missing")

    def test_distribution_contains_html_viewer_notices(self):
        args = argparse.Namespace(
            source=ROOT,
            triplet_dir=ROOT / "build-tests-release/vcpkg_installed/x64-windows",
            install_prefix=ROOT / "dist-fast",
            vcpkg_root=None,
        )
        notice = generate(args)
        self.assertIn("PlayCanvas Ltd.", notice, "PlayCanvas copyright is missing")
        self.assertIn("Arseny Kapoulkine", notice, "meshoptimizer copyright is missing")


if __name__ == "__main__":
    unittest.main()
