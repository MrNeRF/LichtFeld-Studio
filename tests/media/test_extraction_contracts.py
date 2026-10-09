# SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
# SPDX-License-Identifier: GPL-3.0-or-later
"""CPU regression contracts for production extraction, preview and media probe."""
import argparse
import importlib.util
import json
import os
from pathlib import Path
import subprocess
import struct
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[2]
MODULE_SPEC = importlib.util.spec_from_file_location("media_fixtures", ROOT / "scripts/prepare_media_fixtures.py")
fixtures = importlib.util.module_from_spec(MODULE_SPEC)
MODULE_SPEC.loader.exec_module(fixtures)
FFMPEG = os.environ.get("LFS_MEDIA_TEST_FFMPEG", "ffmpeg")
FFPROBE = os.environ.get("LFS_MEDIA_TEST_FFPROBE", "ffprobe")
RUNNER = None


class ExtractionContracts(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.temp = tempfile.TemporaryDirectory(prefix="lfs-ingest-contracts-")
        cls.addClassCleanup(cls.temp.cleanup)
        cls.root = Path(cls.temp.name)
        cls.corpus = cls.root / "sources with spaces é"
        fixtures.prepare(cls.corpus, FFMPEG, FFPROBE)

    def extract(self, fixture="cfr-asymmetric", **options):
        work = Path(tempfile.mkdtemp(prefix="case-", dir=self.root))
        output = work / "output with spaces é"
        request = {"input": str(self.corpus / (fixture + ".nut")), "output": str(output),
                   "end": 0.35, **options}
        path = work / "request.json"
        path.write_text(json.dumps(request, ensure_ascii=False), encoding="utf-8")
        result = subprocess.run([str(RUNNER), str(path)], capture_output=True, timeout=30)
        self.assertEqual(result.returncode, 0, result.stderr.decode("utf-8", errors="replace"))
        return json.loads(result.stdout), output

    def test_memory_sink_preserves_pixels_pts_and_ownership(self):
        for fixture in ("cfr-asymmetric", "vfr-asymmetric"):
            result, output = self.extract(fixture, sink="memory")
            self.assertTrue(result["success"], result["error"])
            self.assertFalse(output.exists())
            self.assertEqual(result["events"], ["begin"] + ["write"] * len(result["frames"]) + ["complete"])
            source = self.corpus / (fixture + ".nut")
            ref = json.loads(subprocess.check_output([FFPROBE, "-v", "error", "-select_streams", "v:0", "-show_frames", "-show_streams", "-of", "json", str(source)]))
            pixels = subprocess.check_output([FFMPEG, "-v", "error", "-i", str(source), "-f", "rawvideo", "-pix_fmt", "rgb24", "-fps_mode", "passthrough", "-"])
            for index, frame in enumerate(result["frames"]):
                self.assertEqual(frame["decode_index"], index)
                self.assertEqual(frame["delivery_index"], index)
                self.assertEqual(frame["ticks"], int(ref["frames"][index]["best_effort_timestamp"]))
                self.assertEqual(frame["time_base"], [int(v) for v in ref["streams"][0]["time_base"].split("/")])
                self.assertEqual(bytes(frame["pixels"]), pixels[index*6144:(index+1)*6144])
            self.assertEqual(result["payload_bytes"], len(result["frames"])*6144)

    def test_shared_cpu_player_seek_close_and_pixels(self):
        source = self.corpus / "cfr-asymmetric.nut"
        work = Path(tempfile.mkdtemp(prefix="shared-preview-", dir=self.root))
        reference = subprocess.check_output([FFMPEG, "-v", "error", "-i", str(source),
                                            "-f", "rawvideo", "-pix_fmt", "rgb24", "-"])
        for seeks, index in (([], 0), ([0.3], 3), ([0.3, 0.1], 1)):
            request = work / "preview.json"
            request.write_text(json.dumps({"operation": "preview", "input": str(source), "seek": seeks}), encoding="utf-8")
            result = subprocess.run([str(RUNNER), str(request)], capture_output=True, timeout=30)
            self.assertEqual(result.returncode, 0, result.stderr)
            preview = json.loads(result.stdout)
            self.assertTrue(preview["success"], preview["error"])
            self.assertFalse(preview["hardware_decode"])
            self.assertTrue(preview["closed"])
            self.assertAlmostEqual(preview["time"], index / 10)
            self.assertEqual(bytes(preview["pixels"]), reference[index * 6144:(index + 1) * 6144])

    def test_cpu_encode_session_roundtrip_pts_metadata_and_lifecycle(self):
        work = Path(tempfile.mkdtemp(prefix="encode-session-", dir=self.root))
        video = work / "encoded é 日本語.mp4"
        request = work / "encode.json"
        request.write_text(json.dumps({"operation": "encode-session", "output": str(video)}, ensure_ascii=False), encoding="utf-8")
        result = subprocess.run([str(RUNNER), str(request)], capture_output=True, timeout=30)
        self.assertEqual(result.returncode, 0, result.stderr)
        report = json.loads(result.stdout)
        self.assertTrue(report["success"])
        self.assertEqual(report["writer_calls"], 8)
        decoded = json.loads(subprocess.check_output([FFPROBE, "-v", "error", "-show_frames", "-show_streams",
                                                      "-show_format", "-of", "json", str(video)]))
        self.assertEqual(len(decoded["frames"]), 4)
        self.assertEqual(decoded["format"]["tags"]["comment"], report["comment"])
        self.assertEqual((decoded["streams"][0]["width"], decoded["streams"][0]["height"]), (64, 48))
        for index, frame in enumerate(decoded["frames"]):
            self.assertAlmostEqual(float(frame["best_effort_timestamp_time"]), index / 10)
        pixels = subprocess.check_output([FFMPEG, "-v", "error", "-i", str(video), "-pix_fmt", "yuv420p", "-f", "rawvideo", "-"])
        self.assertEqual(len(pixels), 4 * 4608)
        for index in range(4):
            luma = pixels[index * 4608:index * 4608 + 3072]
            self.assertLessEqual(max(abs(value - (32 + index * 32)) for value in luma), 5)

    @unittest.skipUnless(os.environ.get("LFS_MEDIA_TEST_VIDEOTOOLBOX_8K") == "1",
                         "requires an explicit native Mac VideoToolbox 8K run")
    def test_videotoolbox_8k_session_roundtrip_and_recovery(self):
        work = Path(tempfile.mkdtemp(prefix="apple-encode-session-", dir=self.root))
        # Reuse the lifecycle contract: failed open, producer rejection,
        # reentrancy, move ownership and flush also exercise the fallback.
        for width, height in ((8192, 4096), (64, 48)):
            with self.subTest(extent=(width, height)):
                video = work / f"apple-{width}.mp4"
                request = work / "encode.json"
                request.write_text(json.dumps({"operation": "encode-session", "output": str(video),
                                               "width": width, "height": height, "videotoolbox": True}),
                                   encoding="utf-8")
                result = subprocess.run([str(RUNNER), str(request)], capture_output=True, timeout=60)
                self.assertEqual(result.returncode, 0, result.stderr)
                report = json.loads(result.stdout)
                self.assertTrue(report["success"])
                self.assertIn(report["backend"], (0, 2))
                self.assertEqual(report["writer_calls"], 8)
                decoded = json.loads(subprocess.check_output([FFPROBE, "-v", "error", "-show_frames",
                                                              "-show_streams", "-show_format", "-of", "json", str(video)]))
                self.assertEqual(len(decoded["frames"]), 4)
                self.assertEqual(decoded["streams"][0]["codec_name"], "h264")
                self.assertEqual((decoded["streams"][0]["width"], decoded["streams"][0]["height"]), (width, height))
                self.assertEqual(decoded["format"]["tags"]["comment"], report["comment"])
                self.assertEqual(decoded["streams"][0]["color_range"], "tv")
                self.assertEqual(decoded["streams"][0]["color_space"], "smpte170m")
                for index, frame in enumerate(decoded["frames"]):
                    self.assertAlmostEqual(float(frame["best_effort_timestamp_time"]), index / 10)
                pixels = subprocess.check_output([FFMPEG, "-v", "error", "-i", str(video), "-vf", "scale=64:48",
                                                  "-pix_fmt", "yuv420p", "-f", "rawvideo", "-"])
                self.assertEqual(len(pixels), 4 * 4608)
                for index in range(4):
                    luma = pixels[index * 4608:index * 4608 + 3072]
                    # Hardware H.264 can have localized inter-frame artifacts;
                    # mean error still detects stale or incorrectly ranged planes.
                    self.assertLess(sum(abs(value - (32 + index * 32)) for value in luma) / len(luma), 2)

    def test_optional_jpeg_backend_failures_and_cpu_fallback(self):
        reference, files = self.extract(format="jpg")
        self.assertTrue(reference["success"], reference["error"])
        encoded = next(files.glob("*.jpg"))
        for mode, hardware, expected, count, calls in (("short", True, False, 0, 1),
                ("empty", True, False, 0, 1), ("throw-later", True, False, 2, 1),
                ("blocked", True, False, 1, 1), ("short", False, True, 4, 0),
                ("unavailable", True, True, 4, 1), ("memory", True, True, 4, 0)):
            with self.subTest(mode=mode, hardware=hardware):
                work=Path(tempfile.mkdtemp(dir=self.root)); output=work/"output"
                if mode == "blocked":
                    output.mkdir(); (output/"frame_002.jpg").mkdir()
                request={"operation":"jpeg-backend", "input":str(self.corpus/"cfr-asymmetric.nut"),
                         "output":str(output),"backend_mode":mode,"allow_hardware":hardware,"encoded_reference":str(encoded)}
                path=work/"request.json";path.write_text(json.dumps(request),encoding="utf-8")
                result=subprocess.run([str(RUNNER),str(path)],capture_output=True,timeout=30)
                self.assertEqual(result.returncode,0,result.stderr)
                actual=json.loads(result.stdout)
                self.assertEqual(actual["success"],expected,actual)
                self.assertEqual(actual["accepted"],count,actual)
                self.assertEqual(actual["factory_calls"],calls,actual)
                written=[p for p in output.glob("*.jpg") if p.is_file()]
                self.assertEqual(len(written),0 if mode == "memory" else count)
                if expected and mode != "memory":
                    self.assertEqual({p.name:p.read_bytes() for p in written}, {p.name:p.read_bytes() for p in files.glob("*.jpg")})

    def test_memory_sink_matches_file_transform_selection(self):
        for options in ({"rotation":90}, {"scale":0.5}, {"width":40,"height":24},
                        {"start":0.1,"end":0.3}, {"mode":"fps","fps":5},
                        {"sharpness":True,"window":True,"interval":2}):
            with self.subTest(options=options):
                memory, no_files = self.extract(sink="memory", **options)
                legacy, output = self.extract(**options)
                self.assertTrue(memory["success"], memory["error"])
                self.assertTrue(legacy["success"], legacy["error"])
                self.assertFalse(no_files.exists())
                images = sorted(output.glob("*.png"))
                self.assertEqual(len(images), len(memory["frames"]))
                for image, frame in zip(images, memory["frames"]):
                    rgb = subprocess.check_output([FFMPEG,"-v","error","-i",str(image),"-f","rawvideo","-pix_fmt","rgb24","-"])
                    self.assertEqual(bytes(frame["pixels"]),rgb)
                metadata = json.loads((output / "extraction_metadata.json").read_text())
                # Compare source identity even when a window is flushed after later decoding.
                for frame, entry in zip(memory["frames"], metadata["frames"]):
                    self.assertEqual(frame["source_frame"],entry["source_frame"])
                    self.assertEqual(frame["decode_index"],entry["source_frame"]-1)
                    self.assertAlmostEqual(frame["seconds"],entry["timestamp"],places=6)

    def test_sink_failure_and_cancellation_retain_partial_results(self):
        for options, outcome, count, events in (
            ({"fail_after":1},"failed",1,["begin","write","write","abort"]),
            ({"sink_cancel_after":1},"cancelled",1,["begin","write","write","abort"]),
            ({"cancel_after":1},"cancelled",1,["begin","write","abort"]),
            ({"budget":6144},"failed",1,["begin","write","write","abort"]),
            ({"frame_limit":1},"failed",1,["begin","write","write","abort"]),
            ({"fail_begin":True},"failed",0,["begin","abort"]),
            ({"throw_write":True},"failed",0,["begin","write","abort"]),
            ({"throw_unknown":True},"failed",0,["begin","write","abort"]),
            ({"fail_complete":True},"failed",4,["begin"]+["write"]*4+["complete","abort"])):
            with self.subTest(options=options):
                result, output = self.extract(sink="memory", **options)
                self.assertFalse(result["success"])
                self.assertEqual(result["outcome"],outcome)
                self.assertEqual(len(result["frames"]),count)
                self.assertEqual(result["accepted"],count)
                self.assertEqual(result["events"],events)
                self.assertEqual(result["abort_count"],1)
                self.assertFalse(output.exists())

    def test_memory_ignores_filesystem_policy_and_reports_rotated_session(self):
        blocked = self.root / "memory-output-blocked"
        blocked.write_bytes(b"existing file")
        result, _ = self.extract(sink="memory",output=str(blocked),naming="absent/subdir/frame_%d",rotation=90)
        self.assertTrue(result["success"],result["error"])
        self.assertEqual(blocked.read_bytes(),b"existing file")
        self.assertEqual(result["session_size"],[32,64])
        self.assertTrue(all(f["size"] == [32,64] for f in result["frames"]))

    def test_retained_fps_tail_keeps_original_decode_identity(self):
        result, _ = self.extract(sink="memory",mode="fps",fps=100,end=0.35)
        self.assertTrue(result["success"],result["error"])
        frames = result["frames"]
        self.assertEqual([f["delivery_index"] for f in frames],list(range(len(frames))))
        repeated = [f for f in frames if f["decode_index"] == 3]
        self.assertGreater(len(repeated),1)
        self.assertTrue(all(f["ticks"] == repeated[0]["ticks"] and f["pixels"] == repeated[0]["pixels"] for f in repeated))

    def test_invalid_input_never_begins_sink(self):
        for options in ({"interval":0},{"input":str(self.root/"absent.nut")}):
            result, output = self.extract(sink="memory", **options)
            self.assertFalse(result["success"])
            self.assertEqual(result["events"],[])
            self.assertEqual(result["abort_count"],0)
            self.assertFalse(output.exists())

    def test_file_sink_matches_legacy_png_and_jpeg(self):
        for format in ("png","jpg"):
            legacy, old = self.extract(format=format,rotation=90)
            actual, output = self.extract(sink="file",format=format,rotation=90)
            self.assertTrue(actual["success"],actual["error"])
            self.assertTrue(legacy["success"],legacy["error"])
            self.assertFalse((output/"extraction_metadata.json").exists())
            self.assertEqual({p.name:p.read_bytes() for p in old.glob("*."+format)},
                             {p.name:p.read_bytes() for p in output.glob("*."+format)})

    def test_file_writer_failure_is_failed_and_preserves_completed_files(self):
        work = Path(tempfile.mkdtemp(prefix="blocked-",dir=self.root))
        (work/"frame_002.png").mkdir()
        result, output = self.extract(output=str(work))
        self.assertFalse(result["success"])
        self.assertEqual(result["outcome"],"failed")
        self.assertTrue((work/"frame_001.png").is_file())
        self.assertEqual(len(result["progress"]),1)

    def probe(self, source=None, **options):
        work = Path(tempfile.mkdtemp(prefix="probe-", dir=self.root))
        request = {"operation": "probe", "input": str(source or self.corpus / "cfr-asymmetric.nut"), **options}
        path = work / "request.json"
        path.write_text(json.dumps(request, ensure_ascii=False), encoding="utf-8")
        result = subprocess.run([str(RUNNER), str(path)], capture_output=True, timeout=30)
        self.assertEqual(result.returncode, 0, result.stderr.decode("utf-8", errors="replace"))
        return json.loads(result.stdout)

    def test_probe_rgb_dimensions_rationals_and_no_output(self):
        source = self.corpus / "cfr-asymmetric.nut"
        before = {p.name: p.read_bytes() for p in self.corpus.iterdir() if p.is_file()}
        result = self.probe(source)
        self.assertTrue(result["success"], result["error"])
        self.assertTrue(result["stream_info_probed"])
        self.assertEqual(result["selected_video"], 0)
        stream = result["streams"][0]
        self.assertEqual((stream["codec"], stream["width"], stream["height"], stream["depth"]), ("ffv1", 64, 32, 8))
        reference = json.loads(subprocess.check_output([FFPROBE, "-v", "error", "-show_streams", "-of", "json", str(source)]))["streams"][0]
        self.assertEqual(stream["time_base"], [int(x) for x in reference["time_base"].split("/")])
        self.assertEqual(stream["nominal_fps"], [int(x) for x in reference["r_frame_rate"].split("/")])
        self.assertEqual(before, {p.name: p.read_bytes() for p in self.corpus.iterdir() if p.is_file()})
        self.assertIsNone(stream["rotation"])

    def test_probe_headers_and_vfr_keep_declared_rates(self):
        for name in ("cfr-asymmetric", "vfr-asymmetric"):
            source = self.corpus / (name + ".nut")
            result = self.probe(source, headers_only=True)
            self.assertTrue(result["success"], result["error"])
            self.assertFalse(result["stream_info_probed"])
            self.assertEqual(result["streams"][0]["width"], 64)
            self.assertIsNotNone(result["streams"][0]["time_base"])

    def test_probe_audio_and_multiple_video_streams(self):
        source = self.root / "multi stream.nut"
        fixtures.run([FFMPEG, "-v", "error", "-i", str(self.corpus / "cfr-asymmetric.nut"),
                      "-f", "lavfi", "-i", "sine=frequency=400:sample_rate=8000:duration=0.5",
                      "-map", "1:a", "-map", "0:v", "-map", "0:v", "-c:v", "copy", "-c:a", "pcm_s16le", str(source)])
        result = self.probe(source)
        self.assertTrue(result["success"], result["error"])
        self.assertEqual([s["kind"] for s in result["streams"]], [1, 0, 0])
        self.assertEqual(result["selected_video"], 1)
        self.assertIsNone(result["streams"][0]["width"])
        extracted, output = self.extract(input=str(source))
        self.completed(extracted)
        self.assert_frames(output, [0, 1, 2, 3])

    def test_probe_audio_only_is_valid_without_selected_video(self):
        source = self.root / "audio.wav"
        fixtures.run([FFMPEG, "-v", "error", "-f", "lavfi", "-i", "sine=duration=0.2", str(source)])
        result = self.probe(source)
        self.assertTrue(result["success"], result["error"])
        self.assertIsNone(result["selected_video"])
        self.assertEqual(result["streams"][0]["kind"], 1)

    def test_probe_invalid_input_and_options_have_structured_errors(self):
        bad = self.root / "invalid input é.nut"
        bad.write_bytes(b"not a media container")
        for source in (bad, self.root / "missing é.nut"):
            result = self.probe(source)
            self.assertFalse(result["success"])
            self.assertEqual(result["error_code"], "DataLoss" if source == bad else "NotFound")
            self.assertEqual(result["error_domain"], "IO")
            self.assertLess(result["ffmpeg_code"], 0)
        for timeout in (0, -1):
            result = self.probe(timeout_ms=timeout)
            self.assertFalse(result["success"])
            self.assertEqual(result["error_code"], "InvalidArgument")
            self.assertEqual(result["ffmpeg_code"], 0)

    def test_probe_matrix_matches_real_player_and_explicit_extraction(self):
        base = self.root / "rotation-base.mov"
        fixtures.run([FFMPEG, "-v", "error", "-i", str(self.corpus / "cfr-asymmetric.nut"),
                      "-c:v", "mpeg4", "-q:v", "1", str(base)])
        for angle in (90, 180, 270):
            with self.subTest(angle=angle):
                source = self.root / f"rotation-{angle}.mov"
                # Write a known track-header matrix directly. FFmpeg versions
                # differ in whether remuxing a rotate tag creates this side data.
                # Only fixture construction uses this tiny MOV atom walk.
                data = bytearray(base.read_bytes())
                def track_headers(start, end):
                    offset = start
                    while offset + 8 <= end:
                        size, kind = struct.unpack_from(">I4s", data, offset)
                        self.assertGreaterEqual(size, 8)
                        self.assertLessEqual(offset + size, end)
                        if kind == b"tkhd":
                            yield offset
                        elif kind in (b"moov", b"trak"):
                            yield from track_headers(offset + 8, offset + size)
                        offset += size
                headers = list(track_headers(0, len(data)))
                self.assertEqual(len(headers), 1)
                offset = headers[0] + (60 if data[headers[0] + 8] == 1 else 48)
                matrices = {90: (0, 65536, 0, -65536, 0, 0, 0, 0, 1 << 30),
                            180: (-65536, 0, 0, 0, -65536, 0, 0, 0, 1 << 30),
                            270: (0, -65536, 0, 65536, 0, 0, 0, 0, 1 << 30)}
                struct.pack_into(">9i", data, offset, *matrices[angle])
                source.write_bytes(data)
                result = self.probe(source)
                self.assertTrue(result["success"], result["error"])
                stream = result["streams"][0]
                reference = json.loads(subprocess.check_output([FFPROBE, "-v", "error", "-show_streams", "-of", "json", str(source)]))["streams"][0]
                matrix = next(s for s in reference["side_data_list"] if s["side_data_type"] == "Display Matrix")
                clockwise = (-matrix["rotation"]) % 360
                self.assertEqual(stream["legacy_rotation"], clockwise)
                self.assertEqual(stream["rotation_source"], 2)
                self.assertEqual(len(stream["display_matrix"]), 9)
                work = Path(tempfile.mkdtemp(prefix="preview-", dir=self.root))
                request = work / "request.json"
                request.write_text(json.dumps({"operation": "preview", "input": str(source)}), encoding="utf-8")
                preview = subprocess.run([str(RUNNER), str(request)], capture_output=True, timeout=30)
                self.assertEqual(preview.returncode, 0, preview.stderr.decode("utf-8", errors="replace"))
                preview = json.loads(preview.stdout)
                self.assertTrue(preview["success"], preview["error"])
                self.assertEqual(preview["rotation"], clockwise)
                # The SDR player delivers raw pixels; its caller applies rotation.
                self.assertEqual(preview["size"], [64, 32])
                self.assertFalse(preview["gpu_rotation"])
                raw_pixels = fixtures.run([FFMPEG, "-v", "error", "-noautorotate", "-i", str(source),
                                           "-frames:v", "1", "-f", "rawvideo", "-pix_fmt", "rgb24", "pipe:1"])
                self.assertEqual(len(preview["pixels"]), len(raw_pixels))
                self.assertLessEqual(max(abs(a - b) for a, b in zip(preview["pixels"], raw_pixels)), 3)
                filters = {90: "transpose=1", 180: "hflip,vflip", 270: "transpose=2"}
                pixels = fixtures.run([FFMPEG, "-v", "error", "-noautorotate", "-i", str(source),
                                       "-frames:v", "1", "-vf", filters[clockwise],
                                       "-f", "rawvideo", "-pix_fmt", "rgb24", "pipe:1"])
                extracted, output = self.extract(input=str(source), rotation=clockwise)
                self.completed(extracted)
                frame = fixtures.decoded_rgb(output / "frame_001.png", FFMPEG)
                self.assertEqual(len(frame), len(pixels))
                self.assertLessEqual(max(abs(a - b) for a, b in zip(frame, pixels)), 3)

    def completed(self, result):
        self.assertTrue(result["success"], result["error"])
        self.assertEqual(result["outcome"], "completed")
        self.assertEqual(result["error"], "")

    def metadata(self, output):
        return json.loads((output / "extraction_metadata.json").read_text(encoding="utf-8"))

    def reference_frame(self, index, spec=fixtures.SPECS[0]):
        size = spec["width"] * spec["height"] * 3
        return fixtures.source_pixels(spec)[index * size:(index + 1) * size]

    def assert_frames(self, output, indices, rotation=0):
        names = [f"frame_{index + 1:03d}.png" for index in indices]
        self.assertEqual(sorted(path.name for path in output.glob("*.png")), names)
        for name, index in zip(names, indices):
            expected = self.reference_frame(index)
            if rotation:
                # Independent pixel-coordinate mapping, no production helper.
                rows = [[expected[(y * 64 + x) * 3:(y * 64 + x + 1) * 3]
                         for x in range(64)] for y in range(32)]
                if rotation == 90:
                    rows = [list(row) for row in zip(*rows[::-1])]
                elif rotation == 180:
                    rows = [row[::-1] for row in rows[::-1]]
                elif rotation == 270:
                    rows = [list(row) for row in zip(*rows)][::-1]
                expected = b"".join(pixel for row in rows for pixel in row)
            self.assertEqual(fixtures.decoded_rgb(output / name, FFMPEG), expected, name)

    def test_interval_one_exact_pixels_and_schema_two(self):
        result, output = self.extract()
        self.completed(result)
        self.assert_frames(output, [0, 1, 2, 3])
        metadata = self.metadata(output)
        self.assertEqual(metadata["schema_version"], 2)
        self.assertEqual(metadata["source_size"], [64, 32])
        self.assertEqual(metadata["output_size"], [64, 32])
        self.assertEqual(metadata["output"]["size"], [64, 32])
        self.assertEqual(metadata["output"]["bit_depth"], 8)
        self.assertEqual(metadata["processing"]["decoder"]["backend"], "ffmpeg_software")
        self.assertEqual(metadata["processing"]["image_encoder"]["backend"], "cpu")
        self.assertFalse(metadata["processing"]["tone_mapping"]["enabled"])
        self.assertEqual([frame["source_frame"] for frame in metadata["frames"]], [1, 2, 3, 4])
        for index, frame in enumerate(metadata["frames"]):
            self.assertEqual(frame["file"], f"frame_{index + 1:03d}.png")
            self.assertAlmostEqual(frame["timestamp"], index / 10, places=6)
        counts = [entry[0] for entry in result["progress"]]
        self.assertEqual(counts, [1, 2, 3, 4])

    def test_interval_two_uses_source_numbering(self):
        result, output = self.extract(interval=2)
        self.completed(result)
        self.assert_frames(output, [0, 2])

    def test_fps_sampling_selects_requested_times(self):
        result, output = self.extract(mode="fps", fps=5)
        self.completed(result)
        self.assert_frames(output, [0, 2])
        self.assertEqual([round(frame["timestamp"], 6) for frame in self.metadata(output)["frames"]], [0, 0.2])

    def test_trim_keeps_source_numbers_and_excludes_outside_frames(self):
        result, output = self.extract(start=0.1, end=0.25)
        self.completed(result)
        self.assert_frames(output, [1, 2])

    def test_existing_end_boundary_differs_between_interval_and_fps(self):
        # Characterize the current inconsistency explicitly; a later semantic
        # change must deliberately update this contract, not silently alter it.
        interval_result, interval_output = self.extract(end=0.4)
        fps_result, fps_output = self.extract(end=0.4, mode="fps", fps=10)
        self.completed(interval_result)
        self.completed(fps_result)
        self.assert_frames(interval_output, [0, 1, 2, 3, 4])
        self.assert_frames(fps_output, [0, 1, 2, 3])

    def test_vfr_keeps_original_timestamps_and_unique_pixels(self):
        result, output = self.extract(fixture="vfr-asymmetric", end=1.5)
        self.completed(result)
        frames = self.metadata(output)["frames"]
        self.assertEqual([round(frame["timestamp"], 6) for frame in frames], [0, 0.1, 0.4, 0.9])
        self.assertEqual(len({frame["file"] for frame in frames}), 4)
        # Legacy source_frame on VFR is nominal-FPS-derived, not a decoded ordinal.
        for index, frame in enumerate(frames):
            self.assertEqual(fixtures.decoded_rgb(output / frame["file"], FFMPEG), self.reference_frame(index))

    def test_all_explicit_rotations_preserve_full_pixels(self):
        for rotation in (90, 180, 270):
            with self.subTest(rotation=rotation):
                result, output = self.extract(rotation=rotation)
                self.completed(result)
                self.assert_frames(output, [0, 1, 2, 3], rotation)
                size = [32, 64] if rotation in (90, 270) else [64, 32]
                self.assertEqual(self.metadata(output)["output"]["size"], size)

    def test_scale_and_custom_resize_dimensions_and_color(self):
        for options, size in (({"scale": 0.5}, [32, 16]), ({"width": 48, "height": 24}, [48, 24])):
            with self.subTest(options=options):
                result, output = self.extract(**options)
                self.completed(result)
                self.assertEqual(self.metadata(output)["output_size"], size)
                pixels = fixtures.decoded_rgb(output / "frame_001.png", FFMPEG)
                self.assertEqual(len(pixels), size[0] * size[1] * 3)
                center = (size[1] // 2 * size[0] + size[0] // 2) * 3
                self.assertLessEqual(max(abs(pixels[center + c] - (16, 37, 83)[c]) for c in range(3)), 3)
                reference = fixtures.run([FFMPEG, "-v", "error", "-i",
                    str(self.corpus / "cfr-asymmetric.nut"), "-frames:v", "1", "-vf",
                    f"scale={size[0]}:{size[1]}:flags=bilinear", "-f", "rawvideo", "-pix_fmt", "rgb24", "pipe:1"])
                self.assertEqual(len(pixels), len(reference))
                # Independent CLI pipeline, allowing small libswscale-version
                # rounding differences (the runner and CLI may use different FFmpeg).
                self.assertLessEqual(max(abs(a - b) for a, b in zip(pixels, reference)), 3)

    def test_jpeg_independent_decode_and_flat_region_tolerance(self):
        result, output = self.extract(format="jpg", quality=95)
        self.completed(result)
        self.assertEqual(len(list(output.glob("*.jpg"))), 4)
        self.assertEqual(self.metadata(output)["output"]["format"], "jpg")
        for index in range(4):
            pixels = fixtures.decoded_rgb(output / f"frame_{index + 1:03d}.jpg", FFMPEG)
            self.assertEqual(len(pixels), 64 * 32 * 3)
            # JPEG is lossy: compare flat patch centers, away from chroma edges.
            reference = self.reference_frame(index)
            for x, y in ((4, 4), (59, 4), (4, 27), (59, 27), (32, 16)):
                offset = (y * 64 + x) * 3
                self.assertLessEqual(max(abs(pixels[offset + c] - reference[offset + c]) for c in range(3)), 6)

    def test_yuv_limited_range_conversion_matches_independent_decoder(self):
        for matrix in ("bt709", "bt601"):
            with self.subTest(matrix=matrix):
                source = self.root / f"yuv-{matrix}.nut"
                color_tag = "bt709" if matrix == "bt709" else "smpte170m"
                fixtures.run([FFMPEG, "-v", "error", "-f", "rawvideo", "-pixel_format", "rgb24",
                    "-video_size", "64x32", "-framerate", "10", "-i", "pipe:0", "-vf",
                    f"scale=in_range=full:out_range=limited:out_color_matrix={matrix},format=yuv444p",
                    "-c:v", "ffv1", "-colorspace", color_tag, "-color_range", "tv", str(source)],
                    self.reference_frame(0))
                result, output = self.extract(input=str(source), end=0.05)
                self.completed(result)
                pixels = fixtures.decoded_rgb(output / "frame_001.png", FFMPEG)
                reference = fixtures.decoded_rgb(source, FFMPEG)
                self.assertEqual(len(pixels), len(reference))
                self.assertLessEqual(max(abs(a - b) for a, b in zip(pixels, reference)), 3)

    def test_metadata_disabled_and_custom_naming(self):
        result, output = self.extract(metadata=False, naming="capture_%04d")
        self.completed(result)
        self.assertFalse((output / "extraction_metadata.json").exists())
        self.assertEqual(sorted(path.name for path in output.glob("*.png")),
                         [f"capture_{index:04d}.png" for index in range(1, 5)])

    def test_invalid_parameters_fail_without_images(self):
        for options in ({"rotation": 45}, {"interval": 0}, {"mode": "fps", "fps": 0},
                        {"start": 0.3, "end": 0.1}, {"scale": 0}):
            with self.subTest(options=options):
                result, output = self.extract(**options)
                self.assertFalse(result["success"])
                self.assertEqual(result["outcome"], "failed")
                self.assertTrue(result["error"])
                self.assertEqual(list(output.glob("*.png")), [])

    def test_missing_and_corrupt_input_remain_failure_even_if_cancelled(self):
        bad = self.root / "corrupt.nut"
        bad.write_bytes(b"not a video")
        for path in (self.root / "missing.nut", bad):
            with self.subTest(input=str(path)):
                result, output = self.extract(input=str(path), cancel_after=0)
                self.assertFalse(result["success"])
                self.assertEqual(result["outcome"], "failed")
                self.assertTrue(result["error"])
                self.assertEqual(list(output.glob("*.png")), [])

    def test_cancellation_keeps_only_completed_partial_image(self):
        result, output = self.extract(cancel_after=1)
        self.assertFalse(result["success"])
        self.assertEqual(result["outcome"], "cancelled")
        self.assertTrue(result["error"])
        self.assert_frames(output, [0])
        # Baseline currently returns before writing a cancellation manifest.
        self.assertFalse((output / "extraction_metadata.json").exists())

    def test_output_file_instead_of_directory_is_failure(self):
        blocked = self.root / "blocked-output é 日本語"
        blocked.write_text("keep", encoding="utf-8")
        result, _ = self.extract(output=str(blocked))
        self.assertFalse(result["success"])
        self.assertEqual(result["outcome"], "failed")
        self.assertIn(str(blocked), result["error"])
        self.assertNotIn("\ufffd", result["error"])
        self.assertEqual(blocked.read_text(encoding="utf-8"), "keep")

    def test_sharpness_can_reject_all_frames_without_fake_outputs(self):
        result, output = self.extract(sharpness=True, threshold=1e12)
        self.completed(result)
        self.assertEqual(list(output.glob("*.png")), [])
        self.assertFalse((output / "extraction_metadata.json").exists())
        self.assertEqual(result["progress"][-1][2], 4)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--runner", type=Path, required=True)
    args, remaining = parser.parse_known_args()
    RUNNER = args.runner.resolve()
    unittest.main(argv=[__file__, *remaining])
