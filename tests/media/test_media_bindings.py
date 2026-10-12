# SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
# SPDX-License-Identifier: GPL-3.0-or-later
"""Production Media Ingest bindings: GIL, ownership, file parity and callbacks."""
import argparse
import errno
import importlib.util
import json
import os
import shutil
from pathlib import Path
import sys
import subprocess
import tempfile
import threading
import unicodedata
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
        cls.addClassCleanup(cls.cleanup_temp)
        cls.root = Path(cls.temp.name)
        cls.corpus = cls.root / "é 日本語"
        fixtures.prepare(cls.corpus, os.environ.get("LFS_MEDIA_TEST_FFMPEG", "ffmpeg"),
                         os.environ.get("LFS_MEDIA_TEST_FFPROBE", "ffprobe"))

    @classmethod
    def cleanup_temp(cls):
        try:
            cls.temp.cleanup()
        except OSError as error:
            if sys.platform != "darwin" or error.errno != errno.ENOTEMPTY:
                raise
            # macOS/exFAT can enumerate Unicode fixture names in NFD while
            # lookup requires the NFC spelling used to create them. Restrict
            # this fallback to normalized children of our own temporary root.
            for entry in cls.root.iterdir():
                normalized = entry.with_name(unicodedata.normalize("NFC", entry.name))
                if normalized == entry or not normalized.exists():
                    continue
                if normalized.is_symlink() or not normalized.is_dir():
                    normalized.unlink()
                else:
                    shutil.rmtree(normalized)
            cls.temp.cleanup()

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
        self.assertEqual(len(list(files.files.output_directory.glob("frame_*.png"))), len(frames))
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
        self.assertFalse(list(files.files.output_directory.glob("frame_*.png")))

    def resume_files(self, directory, file_format=None):
        files = media.FileExtraction()
        files.files.output_directory = directory
        files.write_metadata = True
        files.resume = True
        if file_format is not None:
            files.files.format = file_format
        return files

    def interrupt_resume(self, request, files):
        processed = []
        with self.assertRaises(MEDIA_ERROR):
            media.MediaIngest.extract_files(request, files,
                progress=lambda value: processed.append(value.processed),
                cancelled=lambda: bool(processed))
        self.assertTrue(processed)

    def test_resume_replays_selection_without_rewriting_verified_images(self):
        for label, file_format in (("png", media.FrameFileFormat.PNG), ("jpeg", media.FrameFileFormat.JPEG),
                                   ("exr", media.FrameFileFormat.EXR)):
            with self.subTest(format=label):
                request = self.request()
                if label == "exr":
                    request.output_format = media.FramePixelFormat.RGBFloat32
                    request.input_color.transfer = media.ColorTransfer.Linear
                    request.input_color.primaries = media.ColorPrimaries.Bt709
                files = self.resume_files(self.root / ("resume-" + label), file_format)
                self.interrupt_resume(request, files)
                previous = {p: (p.read_bytes(), p.stat().st_mtime_ns)
                            for p in files.files.output_directory.iterdir() if p.is_file()}
                report = media.MediaIngest.extract_files(request, files)
                self.assertEqual(report.frames_accepted, 4)
                for path, (data, modified) in previous.items():
                    self.assertEqual(path.read_bytes(), data)
                    self.assertEqual(path.stat().st_mtime_ns, modified)
                manifest = json.loads((files.files.output_directory / "extraction_metadata.json").read_text())
                self.assertEqual(len(manifest["frames"]), 4)
                self.assertEqual(len(list((files.files.output_directory / ".media-ingest-checkpoint").glob("frame-*.json"))), 4)

    def test_resume_rejects_changed_request_and_corrupt_output(self):
        request = self.request()
        files = self.resume_files(self.root / "resume-reject")
        self.interrupt_resume(request, files)
        image = next(files.files.output_directory.glob("frame_*.png"))
        original = image.read_bytes()
        request.selection.interval = 2
        with self.assertRaises(MEDIA_ERROR):
            media.MediaIngest.extract_files(request, files)
        self.assertEqual(image.read_bytes(), original)
        request.selection.interval = 1
        image.write_bytes(original[:16])
        with self.assertRaises(MEDIA_ERROR):
            media.MediaIngest.extract_files(request, files)
        self.assertEqual(image.read_bytes(), original[:16])
        self.assertFalse((files.files.output_directory / "extraction_metadata.json").exists())

    def test_resume_reconstructs_missing_image_and_pending_publication(self):
        for label, file_format, extension in (("png", media.FrameFileFormat.PNG, "png"),
                                               ("jpeg", media.FrameFileFormat.JPEG, "jpg"),
                                               ("exr", media.FrameFileFormat.EXR, "exr")):
            with self.subTest(format=label):
                request = self.request()
                if label == "exr":
                    request.output_format = media.FramePixelFormat.RGBFloat32
                    request.input_color.transfer = media.ColorTransfer.Linear
                    request.input_color.primaries = media.ColorPrimaries.Bt709
                files = self.resume_files(self.root / ("resume-missing-" + label), file_format)
                self.interrupt_resume(request, files)
                image = next(files.files.output_directory.glob("frame_*." + extension))
                expected = image.read_bytes()
                image.unlink()
                report = media.MediaIngest.extract_files(request, files)
                self.assertEqual(report.frames_accepted, 4)
                self.assertEqual(image.read_bytes(), expected)

    def test_resume_recovers_record_published_before_image(self):
        request = self.request()
        files = self.resume_files(self.root / "resume-pending")
        record = files.files.output_directory / ".media-ingest-checkpoint" / "frame-0.json"
        with self.assertRaises(MEDIA_ERROR):
            media.MediaIngest.extract_files(request, files, cancelled=record.exists)
        self.assertTrue(record.is_file())
        self.assertFalse(list(files.files.output_directory.glob("frame_*.png")))
        self.assertEqual(media.MediaIngest.extract_files(request, files).frames_accepted, 4)

    def test_resume_after_abrupt_process_exit(self):
        directory = self.root / "resume-crash"
        # Exit only this disposable child, after a completed image publication.
        program = r'''
import os, sys
from pathlib import Path
handles = []
if os.name == "nt":
    for directory in sys.argv[4:]:
        if directory:
            handles.append(os.add_dll_directory(directory))
sys.path.insert(0, sys.argv[1])
import media_test_bindings as media
r = media.IngestRequest()
r.input = Path(sys.argv[2])
r.selection.mode = media.SelectionMode.Interval
r.selection.interval = 1
r.end_seconds = .35
r.allow_hardware_decode = False
f = media.FileExtraction()
f.files.output_directory = Path(sys.argv[3])
f.write_metadata = True
f.resume = True
media.MediaIngest.extract_files(r, f, progress=lambda _: os._exit(73))
'''
        if args.studio:
            program = program.replace("import media_test_bindings as media", "from lichtfeld import media")
        command = [sys.executable, "-c", program, str(args.module_dir.resolve()),
                   str(self.request().input), str(directory), str(args.runtime_dir.resolve()),
                   str(args.dependency_dir.resolve()) if args.dependency_dir else ""]
        child = subprocess.run(command, capture_output=True, timeout=30)
        self.assertEqual(child.returncode, 73, child.stderr.decode(errors="replace"))
        self.assertEqual(len(list(directory.glob("frame_*.png"))), 1)
        self.assertEqual(media.MediaIngest.extract_files(self.request(), self.resume_files(directory)).frames_accepted, 4)

    def test_resume_refuses_to_adopt_legacy_output(self):
        request = self.request()
        directory = self.root / "resume-legacy"
        files = self.resume_files(directory)
        files.resume = False
        media.MediaIngest.extract_files(request, files)
        before = {p: p.read_bytes() for p in directory.iterdir() if p.is_file()}
        files.resume = True
        with self.assertRaises(MEDIA_ERROR):
            media.MediaIngest.extract_files(request, files)
        self.assertEqual(before, {p: p.read_bytes() for p in directory.iterdir() if p.is_file()})

    def test_resume_preserves_unowned_checkpoint_directory(self):
        files = self.resume_files(self.root / "resume-unowned")
        checkpoint = files.files.output_directory / ".media-ingest-checkpoint"
        checkpoint.mkdir(parents=True)
        sentinel = checkpoint / "frame_1.png"
        sentinel.write_bytes(b"unrelated photograph")
        with self.assertRaises(MEDIA_ERROR):
            media.MediaIngest.extract_files(self.request(), files)
        self.assertEqual(sentinel.read_bytes(), b"unrelated photograph")
        self.assertFalse((checkpoint / "identity.json").exists())

    def test_resume_vfr_and_sharpness_match_uninterrupted_extraction(self):
        for window in (False, True):
            with self.subTest(window=window):
                request = self.request()
                request.input = self.corpus / "vfr-asymmetric.nut"
                request.end_seconds = -1
                request.selection.mode = media.SelectionMode.FPS
                request.selection.fps = 3
                request.sharpness.enabled = True
                request.sharpness.window = window
                files = self.resume_files(self.root / ("resume-vfr-" + str(window)))
                self.interrupt_resume(request, files)
                actual = media.MediaIngest.extract_files(request, files)
                reference = self.resume_files(self.root / ("reference-vfr-" + str(window)))
                reference.resume = False
                expected = media.MediaIngest.extract_files(request, reference)
                self.assertEqual(actual.frames_accepted, expected.frames_accepted)
                for image in reference.files.output_directory.glob("frame_*.png"):
                    self.assertEqual((files.files.output_directory / image.name).read_bytes(), image.read_bytes())
                actual_metadata = json.loads((files.files.output_directory / "extraction_metadata.json").read_text())
                expected_metadata = json.loads((reference.files.output_directory / "extraction_metadata.json").read_text())
                self.assertEqual(actual_metadata["frames"], expected_metadata["frames"])

    def test_resume_changed_input_and_invalid_checkpoint_preserve_output(self):
        request = self.request()
        original_input = request.input
        request.input = self.root / "resume-input.nut"
        shutil.copyfile(original_input, request.input)
        files = self.resume_files(self.root / "resume-input-change")
        self.interrupt_resume(request, files)
        image = next(files.files.output_directory.glob("frame_*.png"))
        before = image.read_bytes()
        with request.input.open("ab") as source:
            source.write(b"changed identity")
        with self.assertRaises(MEDIA_ERROR):
            media.MediaIngest.extract_files(request, files)
        self.assertEqual(image.read_bytes(), before)
        shutil.copyfile(original_input, request.input)
        record = files.files.output_directory / ".media-ingest-checkpoint" / "frame-0.json"
        record.write_text('{"file":"../../outside.png"}', encoding="utf-8")
        with self.assertRaises(MEDIA_ERROR):
            media.MediaIngest.extract_files(request, files)
        self.assertEqual(image.read_bytes(), before)

    def test_callback_exception_preserves_python_error(self):
        def failed(*args):
            raise ValueError("callback sentinel")
        with self.assertRaisesRegex(ValueError, "callback sentinel"):
            media.MediaIngest.extract(self.request(), progress=failed)
        with self.assertRaisesRegex(ValueError, "callback sentinel"):
            media.MediaIngest.extract(self.request(), cancelled=failed)

    def test_float_owned_extraction_and_independent_image_output(self):
        import struct
        request = self.request()
        request.output_format = media.FramePixelFormat.RGBFloat32
        request.input_color.transfer = media.ColorTransfer.Linear
        request.input_color.primaries = media.ColorPrimaries.Bt709
        report, frames = media.MediaIngest.extract(request)
        self.assertEqual(report.frames_accepted, len(frames))
        self.assertEqual(frames[0].layout.row_stride, 64 * 12)
        self.assertEqual(frames[0].layout.color.transfer, media.ColorTransfer.Linear)
        self.assertEqual(frames[0].info.origin, media.FrameOrigin.Decoded)
        files = media.FileExtraction()
        files.files.output_directory = self.root / "float-files"
        files.files.format = media.FrameFileFormat.EXR
        files.files.exr.precision = media.ExrPrecision.Float
        self.assertEqual(media.MediaIngest.extract_files(request, files).frames_accepted, len(frames))
        self.assertTrue((files.files.output_directory / "extraction_metadata.json").is_file())
        color = media.FrameColor()
        color.transfer = media.ColorTransfer.Linear
        color.primaries = media.ColorPrimaries.Bt709
        color.alpha = media.AlphaMode.NoAlpha
        info = media.FrameInfo()
        info.origin = media.FrameOrigin.Rendered
        raw = struct.pack("fff", -.25, 2.0, .1234567)
        surface = media.FrameSurface.from_bytes(1, 1, media.FramePixelFormat.RGBFloat32, raw, color=color, info=info)
        self.assertEqual(surface.pixels, raw)
        self.assertIsNone(surface.info.source_timestamp)
        options = media.ExrOutputOptions()
        options.precision = media.ExrPrecision.Float
        options.compression = media.ExrCompression.Uncompressed
        target = self.root / "renderer é 日本語.exr"
        media.ImageOutput.write_exr(target, surface, options)
        previous = target.read_bytes()
        with self.assertRaises(MEDIA_ERROR):
            media.ImageOutput.write_exr(target, surface, options)
        options.overwrite = True
        def failed():
            raise ValueError("EXR cancellation sentinel")
        with self.assertRaisesRegex(ValueError, "EXR cancellation sentinel"):
            media.ImageOutput.write_exr(target, surface, options, cancelled=failed)
        self.assertEqual(target.read_bytes(), previous)
        self.assertFalse(list(self.root.glob("*.tmp")))

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
