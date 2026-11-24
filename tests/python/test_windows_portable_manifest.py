# SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
# SPDX-License-Identifier: GPL-3.0-or-later
"""Exercise installed DLL inventory without a compiler or Windows SDK."""

from pathlib import Path
import shutil
import subprocess
import tempfile
import unittest
import xml.etree.ElementTree as ET


ROOT = Path(__file__).resolve().parents[2]
CMAKE = shutil.which("cmake")


@unittest.skipUnless(CMAKE, "CMake is required for the install-script fixture")
class PortableManifestInventoryTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory(prefix="lfs portable ")
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name).resolve()
        self.package = self.root / "package à"
        self.runtime = self.package / "bin"
        self.runtime.mkdir(parents=True)

    def run_install(self, prefix):
        script = self.root / "install.cmake"
        script.write_text(
            f'set(CMAKE_INSTALL_PREFIX [=[{prefix}]=])\n'
            'set(PROJECT_NAME LichtFeld-Studio)\n'
            'set(CMAKE_EXECUTABLE_SUFFIX .exe)\n'
            # CMake deliberately rejects SDK arguments. Inventory must succeed
            # before that boundary; this fixture does not validate PE editing.
            f'set(CMAKE_MT [=[{Path(CMAKE).as_posix()}]=])\n'
            f'configure_file([=[{(ROOT / "cmake/InstallWindowsPortableManifest.cmake.in").as_posix()}]=] '
            'installed.cmake @ONLY)\n'
            'include(installed.cmake)\n',
            encoding="utf-8",
        )
        return subprocess.run(
            [CMAKE, "-P", str(script)], cwd=self.root,
            capture_output=True, text=True, encoding="utf-8", errors="replace",
            timeout=30,
        )

    def check_inventory(self, prefix):
        (self.runtime / "runtime.dll").touch()
        (self.runtime / "nvcuda.dll").touch()
        result = self.run_install(prefix)
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("Cannot extract the installed application manifest", result.stderr)
        assembly = ET.parse(self.runtime / "bin.manifest")
        files = assembly.findall("{urn:schemas-microsoft-com:asm.v1}file")
        self.assertEqual([file.attrib["name"] for file in files], ["runtime.dll"])

    def test_relative_install_prefix_finds_dlls(self):
        self.check_inventory(self.package.name)

    def test_absolute_install_prefix_finds_dlls(self):
        self.check_inventory(self.package.as_posix())

    def test_empty_runtime_fails_before_sdk_invocation(self):
        result = self.run_install(self.package.name)
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("runtime contains no DLLs", result.stderr)
        self.assertFalse((self.runtime / "bin.manifest").exists())
