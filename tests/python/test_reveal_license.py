"""Check the generated distribution notices for the reveal implementation."""

import argparse
import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "tools"))
from generate_third_party_licenses import generate


class RevealLicenseTest(unittest.TestCase):
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


if __name__ == "__main__":
    unittest.main()
