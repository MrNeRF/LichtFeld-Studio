# SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
# SPDX-License-Identifier: GPL-3.0-or-later
"""Offline integrity and failure-path tests for the generated media corpus."""

import importlib.util
import json
import os
from pathlib import Path
import tempfile
import unittest
from unittest import mock

ROOT = Path(__file__).resolve().parents[2]
MODULE_SPEC = importlib.util.spec_from_file_location("media_fixtures", ROOT / "scripts/prepare_media_fixtures.py")
fixtures = importlib.util.module_from_spec(MODULE_SPEC)
MODULE_SPEC.loader.exec_module(fixtures)
FFMPEG = os.environ.get("LFS_MEDIA_TEST_FFMPEG", "ffmpeg")
FFPROBE = os.environ.get("LFS_MEDIA_TEST_FFPROBE", "ffprobe")


class PreparationFailures(unittest.TestCase):
    def test_existing_destination_is_never_overwritten(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            sentinel = root / "existing.txt"
            sentinel.write_text("keep", encoding="utf-8")
            with self.assertRaises(FileExistsError):
                fixtures.prepare(root, "unavailable-ffmpeg", "unavailable-ffprobe")
            self.assertEqual(sentinel.read_text(encoding="utf-8"), "keep")

    def test_missing_tool_does_not_publish_partial_corpus(self):
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "corpus"
            with self.assertRaises(FileNotFoundError):
                fixtures.prepare(output, str(Path(directory) / "missing-ffmpeg"), FFPROBE)
            self.assertFalse(output.exists())
            self.assertEqual(list(Path(directory).iterdir()), [])


class GeneratedCorpus(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.temp = tempfile.TemporaryDirectory(prefix="lfs-media-test-")
        cls.addClassCleanup(cls.temp.cleanup)
        cls.output = Path(cls.temp.name) / "media with spaces é"
        cls.manifest = fixtures.prepare(cls.output, FFMPEG, FFPROBE)

    def test_manifest_and_checksums_identify_complete_corpus(self):
        saved = json.loads((self.output / "manifest.json").read_text(encoding="utf-8"))
        self.assertEqual(saved, self.manifest)
        self.assertEqual({item["id"] for item in saved["fixtures"]},
                         {"cfr-asymmetric", "vfr-asymmetric", "panorama-layout"})
        for item in saved["fixtures"]:
            path = self.output / item["file"]
            self.assertEqual(path.stat().st_size, item["size_bytes"])
            self.assertEqual(fixtures.digest(path.read_bytes()), item["sha256"])

    def test_timestamps_and_full_pixels_round_trip(self):
        for spec in fixtures.SPECS:
            with self.subTest(fixture=spec["id"]):
                verified = fixtures.verify_fixture(self.output / (spec["id"] + ".nut"),
                                                   spec, FFMPEG, FFPROBE)
                self.assertEqual(verified["timestamps_seconds"], [tick / 10 for tick in spec["ticks"]])

    def test_changed_expected_time_is_rejected(self):
        spec = dict(fixtures.SPECS[1], ticks=[0, 1, 2, 3, 4])
        with self.assertRaisesRegex(RuntimeError, "unexpected timestamps"):
            fixtures.verify_fixture(self.output / "vfr-asymmetric.nut", spec, FFMPEG, FFPROBE)

    def test_changed_expected_geometry_is_rejected(self):
        spec = dict(fixtures.SPECS[0], width=32)
        with self.assertRaisesRegex(RuntimeError, "unexpected dimensions"):
            fixtures.verify_fixture(self.output / "cfr-asymmetric.nut", spec, FFMPEG, FFPROBE)

    def test_late_failure_never_publishes_partial_files(self):
        destination = Path(self.temp.name) / "failed-corpus"
        original_verify = fixtures.verify_fixture

        def fail_second(path, spec, ffmpeg, ffprobe):
            if spec["id"] == "vfr-asymmetric":
                raise RuntimeError("injected verification failure")
            return original_verify(path, spec, ffmpeg, ffprobe)

        with mock.patch.object(fixtures, "verify_fixture", side_effect=fail_second):
            with self.assertRaisesRegex(RuntimeError, "injected verification failure"):
                fixtures.prepare(destination, FFMPEG, FFPROBE)
        self.assertFalse(destination.exists())
        self.assertEqual(list(Path(self.temp.name).iterdir()), [self.output])


if __name__ == "__main__":
    unittest.main()
