# SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
# SPDX-License-Identifier: GPL-3.0-or-later
"""Production Media Ingest bindings: GIL, ownership, file parity and callbacks."""
import argparse
import importlib.util
import os
from pathlib import Path
import sys
import tempfile
import threading
import unittest

ROOT = Path(__file__).resolve().parents[2]
spec = importlib.util.spec_from_file_location("media_fixtures", ROOT / "scripts/prepare_media_fixtures.py")
fixtures = importlib.util.module_from_spec(spec)
spec.loader.exec_module(fixtures)
media = None
MEDIA_ERROR = None


class MediaBindings(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.temp = tempfile.TemporaryDirectory(prefix="lfs-python-media-")
        cls.addClassCleanup(cls.temp.cleanup)
        cls.root = Path(cls.temp.name)
        cls.corpus = cls.root / "é 日本語"
        fixtures.prepare(cls.corpus, os.environ.get("LFS_MEDIA_TEST_FFMPEG", "ffmpeg"),
                         os.environ.get("LFS_MEDIA_TEST_FFPROBE", "ffprobe"))

    def request(self):
        request = media.IngestRequest()
        request.input = self.corpus / "cfr-asymmetric.nut"
        request.selection.mode = media.SelectionMode.Interval
        request.selection.interval = 1
        request.end_seconds = 0.35
        request.allow_hardware_decode = False
        return request

    def test_probe_optional_inventory_unicode(self):
        self.assertTrue(media.MediaIngest.codec_build_info().ffmpeg_license)
        source = media.MediaIngest.probe(self.request().input)
        self.assertEqual(source.streams[0].width, 64)
        self.assertIsNotNone(source.streams[0].time_base)
        self.assertFalse(media.MediaIngest.probe(self.request().input, headers_only=True).stream_info_probed)
        with self.assertRaises(MEDIA_ERROR):
            media.MediaIngest.probe(self.root / "missing.nut")

    def test_nested_options_owned_frames_and_file_report(self):
        request = self.request()
        request.geometry.clockwise_rotation = 90
        progress = []
        report, frames = media.MediaIngest.extract(request, progress=lambda value: progress.append(value.processed))
        self.assertEqual(report.frames_accepted, len(frames))
        self.assertTrue(progress)
        before = frames[0].pixels
        self.assertEqual(len(before), frames[0].layout.row_stride * frames[0].layout.height)
        files = media.FileExtraction()
        files.files.output_directory = self.root / "output é 日本語"
        files.write_metadata = True
        file_report = media.MediaIngest.extract_files(request, files)
        self.assertEqual(file_report.frames_accepted, len(frames))
        self.assertEqual(len(list(files.files.output_directory.glob("*.png"))), len(frames))
        del request, report, file_report
        self.assertEqual(before, frames[0].pixels)

    def test_budget_and_cancellation_failures(self):
        with self.assertRaises(MEDIA_ERROR):
            media.MediaIngest.extract(self.request(), payload_budget=1)
        with self.assertRaises(MEDIA_ERROR):
            media.MediaIngest.extract(self.request(), cancelled=lambda: True)
        files = media.FileExtraction()
        files.files.output_directory = self.root / "cancel-output"
        with self.assertRaises(MEDIA_ERROR):
            media.MediaIngest.extract_files(self.request(), files, cancelled=lambda: True)
        self.assertFalse(list(files.files.output_directory.glob("*.png")))

    def test_callback_exception_preserves_python_error(self):
        def failed(*args):
            raise ValueError("callback sentinel")
        with self.assertRaisesRegex(ValueError, "callback sentinel"):
            media.MediaIngest.extract(self.request(), progress=failed)
        with self.assertRaisesRegex(ValueError, "callback sentinel"):
            media.MediaIngest.extract(self.request(), cancelled=failed)

    def test_callbacks_run_with_gil_and_other_thread_can_run(self):
        event = threading.Event()
        worker = threading.Thread(target=event.set)
        def progress(value):
            if not worker.is_alive() and not event.is_set():
                worker.start()
            self.assertTrue(event.wait(3), "callback must not block the other Python thread")
        media.MediaIngest.extract(self.request(), progress=progress)
        worker.join(3)
        self.assertTrue(event.is_set())


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--module-dir", type=Path, required=True)
    parser.add_argument("--runtime-dir", type=Path, required=True)
    parser.add_argument("--dependency-dir", type=Path)
    parser.add_argument("--studio", action="store_true", help="Test the complete lichtfeld module")
    args, remaining = parser.parse_known_args()
    dll_handles = []
    if os.name == "nt":
        for directory in (args.runtime_dir, args.dependency_dir):
            if directory and directory.is_dir():
                dll_handles.append(os.add_dll_directory(str(directory.resolve())))
    sys.path.insert(0, str(args.module_dir.resolve()))
    if args.studio:
        import lichtfeld
        assert "media" in lichtfeld.__all__, "Media Ingest must be exported by lichtfeld"
        media = lichtfeld.media
        MEDIA_ERROR = lichtfeld.Error
    else:
        import media_test_bindings as media
        MEDIA_ERROR = media.MediaError
    unittest.main(argv=[__file__, *remaining])
