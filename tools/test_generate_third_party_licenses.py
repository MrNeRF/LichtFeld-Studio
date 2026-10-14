import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

from generate_third_party_licenses import ffmpeg_details, license_id_from_text, validate_coverage


class ThirdPartyNoticeCoverageTest(unittest.TestCase):
    def test_conditional_ffmpeg_features_use_installed_status_for_license(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            triplet = root / "arm64-osx"
            triplet.mkdir()
            (root / "vcpkg.json").write_text(json.dumps({"dependencies": [{
                "name": "ffmpeg", "features": ["avcodec", {"name": "nvcodec", "platform": "!osx"},
                                                {"name": "x264", "platform": "windows"}]
            }]}))
            details = ffmpeg_details(SimpleNamespace(source=root, triplet_dir=triplet, vcpkg_root=None))
            self.assertIn("nvcodec (!osx)", details)
            self.assertIn("x264 (windows)", details)
            self.assertIn("installed status is authoritative", details)
            self.assertIn("x264 is not installed", details)
            self.assertNotIn("GPL (x264 feature is installed)", details)
            status = root / "vcpkg" / "status"
            status.parent.mkdir()
            status.write_text("Package: ffmpeg\nFeature: avcodec\nStatus: install ok installed\n\n"
                              "Package: ffmpeg\nFeature: x264\nStatus: install ok installed\n")
            details = ffmpeg_details(SimpleNamespace(source=root, triplet_dir=triplet, vcpkg_root=None))
            self.assertIn("GPL (x264 feature is installed)", details)
            self.assertNotIn("installed status differs", details)

    def test_readme_license_labels_use_spdx_ids(self):
        self.assertEqual(license_id_from_text("License: MIT", "LicenseRef-example"), "MIT")

    def test_installed_ports_and_shared_libraries_must_be_listed(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            metadata_dir = root / "triplet" / "share" / "sample-port"
            metadata_dir.mkdir(parents=True)
            (metadata_dir / "vcpkg.spdx.json").write_text(json.dumps({
                "packages": [{"SPDXID": "SPDXRef-port", "name": "sample-port"}]
            }))
            prefix = root / "prefix"
            (prefix / "lib").mkdir(parents=True)
            (prefix / "lib" / "libsample.so.1").touch()

            self.assertEqual(validate_coverage("| sample-port |\nlibsample.so.1", root / "triplet", prefix), [])
            errors = validate_coverage("| unrelated |", root / "triplet", prefix)
            self.assertEqual(len(errors), 2)
            self.assertIn("installed vcpkg port missing from notices: sample-port", errors)
            self.assertIn("installed shared library missing from notices: libsample.so.1", errors)

    def test_nvjpeg2000_runtime_requires_its_supplement_notice(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            (root / "triplet" / "share").mkdir(parents=True)
            runtime = root / "prefix" / "lib" / "libnvjpeg2k.so.0.9.0"
            runtime.parent.mkdir(parents=True)
            runtime.touch()
            notice = "libnvjpeg2k.so.0.9.0 LicenseRef-NVIDIA-nvJPEG2000 runtime files .so and .h, nvjpeg2k_0.dll"

            self.assertEqual(validate_coverage(notice, root / "triplet", root / "prefix"), [])
            errors = validate_coverage("libnvjpeg2k.so.0.9.0", root / "triplet", root / "prefix")
            self.assertEqual(len(errors), 1)
            self.assertIn("installed NVIDIA runtime missing its file and license notice", errors[0])


if __name__ == "__main__":
    unittest.main()
