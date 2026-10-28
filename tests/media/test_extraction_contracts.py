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
from hevc_contracts import assert_hvc1_samples

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
        return json.loads(result.stdout), Path(request["output"])

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

    def test_preview_unsupported_video_preserves_diagnostic(self):
        # An unknown sample-entry codec exercises failure diagnostics
        # independently of the host decoder registry.
        work = Path(tempfile.mkdtemp(prefix="missing-decoder-", dir=self.root))
        source = ROOT / "tests" / "media" / "fixtures" / "av1-cpu-yuv420p.mp4"
        payload = source.read_bytes()
        self.assertEqual(payload.count(b"stsd"), 1)
        codec_offset = payload.index(b"av01", payload.index(b"stsd"))
        unavailable = work / "unavailable.mp4"
        unavailable.write_bytes(payload[:codec_offset] + b"zzzz" + payload[codec_offset + 4:])
        request = work / "preview.json"
        request.write_text(json.dumps({"operation": "preview", "input": str(unavailable)}), encoding="utf-8")
        result = subprocess.run([str(RUNNER), str(request)], capture_output=True, timeout=30)
        self.assertEqual(result.returncode, 0, result.stderr)
        report = json.loads(result.stdout)
        self.assertFalse(report["success"])
        self.assertIn("No usable video stream found", report["error"])

    def test_av1_cpu_extraction_and_preview_without_hardware(self):
        work = Path(tempfile.mkdtemp(prefix="av1-cpu-", dir=self.root))
        for pixel_format in ("yuv420p", "yuv420p10le"):
            source = ROOT / "tests" / "media" / "fixtures" / ("av1-cpu-" + pixel_format + ".mp4")
            self.assertTrue(source.is_file(), "Required generated AV1 fixture is missing")
            request = work / "request.json"
            request.write_text(json.dumps({"input": str(source), "output": str(work / "unused"),
                                           "sink": "memory"}), encoding="utf-8")
            result = subprocess.run([str(RUNNER), str(request)], capture_output=True, timeout=30)
            self.assertEqual(result.returncode, 0, result.stderr)
            report = json.loads(result.stdout)
            self.assertTrue(report["success"], report["error"])
            self.assertEqual(len(report["frames"]), 4)
            self.assertEqual(len({bytes(frame["pixels"]) for frame in report["frames"]}), 4)
            reference = subprocess.check_output([FFMPEG, "-v", "error", "-i", str(source),
                                                "-sws_flags", "fast_bilinear", "-f", "rawvideo", "-pix_fmt", "rgb24", "-"])
            actual = b"".join(bytes(frame["pixels"]) for frame in report["frames"])
            self.assertEqual(len(actual), len(reference))
            # Constant chroma isolates decode/order from version-specific chroma
            # interpolation. External swscale/SIMD rounding allows 3 U8 LSBs.
            self.assertLessEqual(max(abs(a - b) for a, b in zip(actual, reference)), 3)
            request.write_text(json.dumps({"operation": "preview", "input": str(source)}), encoding="utf-8")
            result = subprocess.run([str(RUNNER), str(request)], capture_output=True, timeout=30)
            self.assertEqual(result.returncode, 0, result.stderr)
            preview = json.loads(result.stdout)
            self.assertTrue(preview["success"], preview["error"])
            self.assertFalse(preview["hardware_decode"])
            self.assertEqual(bytes(preview["pixels"]), actual[:64 * 64 * 3])

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

    def test_av1_cpu_session_flush_pts_color_metadata_and_recovery(self):
        # More than the configured lookahead, with distinct luma and repeated
        # ownership/rejection checks in the shared production C++ contract.
        for use_apple, color_range, crf in ((False, 1, 18), (True, 2, 18), (False, 1, 0)):
            with self.subTest(apple_requested=use_apple, color_range=color_range, crf=crf):
                work = Path(tempfile.mkdtemp(prefix="av1-session-", dir=self.root))
                video = work / "AV1 session é 日本語.MP4"
                request = work / "encode.json"
                request.write_text(json.dumps({"operation": "encode-session", "output": str(video),
                                               "av1": True, "videotoolbox": use_apple, "frames": 24, "crf": crf,
                                               "matrix": 1, "range": color_range, "spherical": True}), encoding="utf-8")
                result = subprocess.run([str(RUNNER), str(request)], capture_output=True, timeout=90)
                self.assertEqual(result.returncode, 0, result.stderr)
                report = json.loads(result.stdout)
                self.assertTrue(report["success"])
                self.assertEqual(report["backend"], 0)
                self.assertEqual(report["writer_calls"], 28)
                probe = json.loads(subprocess.check_output([FFPROBE, "-v", "error", "-show_frames",
                                                           "-show_streams", "-show_format", "-of", "json", str(video)]))
                stream = probe["streams"][0]
                self.assertEqual((stream["codec_name"], stream["codec_tag_string"], stream["pix_fmt"]),
                                 ("av1", "av01", "yuv420p"))
                self.assertEqual((stream["width"], stream["height"]), (64, 48))
                self.assertEqual(stream["color_space"], "bt709")
                self.assertEqual(stream["color_range"], "tv" if color_range == 1 else "pc")
                self.assertEqual(probe["format"]["tags"]["comment"], report["comment"])
                spherical = next(item for item in stream["side_data_list"] if item["side_data_type"] == "Spherical Mapping")
                self.assertEqual(spherical["projection"], "equirectangular")
                self.assertEqual(len(probe["frames"]), 24)
                for index, frame in enumerate(probe["frames"]):
                    self.assertAlmostEqual(float(frame["best_effort_timestamp_time"]), index / 10)
                raw = subprocess.check_output([FFMPEG, "-v", "error", "-i", str(video), "-pix_fmt", "yuv420p",
                                               "-f", "rawvideo", "-"], timeout=30)
                self.assertEqual(len(raw), 24 * 4608)
                for index in range(24):
                    luma = raw[index * 4608:index * 4608 + 3072]
                    self.assertLessEqual(max(abs(value - (32 + (index % 4) * 32)) for value in luma), 0 if crf == 0 else 5)

    def test_cpu_encode_session_preserves_explicit_color_options(self):
        work = Path(tempfile.mkdtemp(prefix="encode-color-", dir=self.root))
        video = work / "bt709.mp4"
        request = work / "encode.json"
        request.write_text(json.dumps({"operation": "encode-session", "output": str(video),
                                       "matrix": 1, "range": 1}), encoding="utf-8")
        result = subprocess.run([str(RUNNER), str(request)], capture_output=True, timeout=30)
        self.assertEqual(result.returncode, 0, result.stderr)
        decoded = json.loads(subprocess.check_output([FFPROBE, "-v", "error", "-show_frames", "-show_streams",
                                                      "-of", "json", str(video)]))
        self.assertEqual(len(decoded["frames"]), 4)
        for item in decoded["streams"] + decoded["frames"]:
            self.assertEqual(item["color_space"], "bt709")
            self.assertEqual(item["color_range"], "tv")

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
                                               "width": width, "height": height, "videotoolbox": True,
                                               "verify_apple_recovery": width == 8192}),
                                   encoding="utf-8")
                result = subprocess.run([str(RUNNER), str(request)], capture_output=True, timeout=60)
                self.assertEqual(result.returncode, 0, result.stderr)
                report = json.loads(result.stdout)
                self.assertTrue(report["success"])
                self.assertEqual(report["backend"], 3 if width == 8192 else 2)
                if width == 8192:
                    self.assertEqual(report["recovery_backend"], 2)
                    recovery = json.loads(subprocess.check_output([FFPROBE, "-v", "error", "-show_frames",
                                                                  "-show_streams", "-of", "json", report["recovery_output"]]))
                    self.assertEqual(len(recovery["frames"]), 1)
                    self.assertEqual((recovery["streams"][0]["width"], recovery["streams"][0]["height"]), (64, 48))
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

    @unittest.skipUnless(os.environ.get("LFS_MEDIA_TEST_VIDEOTOOLBOX_8K") == "1",
                         "requires an explicit native Mac VideoToolbox run")
    def test_videotoolbox_hevc_session_codec_color_pts_and_lifecycle(self):
        work = Path(tempfile.mkdtemp(prefix="apple-hevc-session-", dir=self.root))
        for width, height in ((64, 48), (3840, 2160), (7680, 4320)):
            with self.subTest(extent=(width, height)):
                video = work / f"hevc-{width}.mp4"
                request = work / "encode.json"
                request.write_text(json.dumps({"operation": "encode-session", "output": str(video),
                                               "width": width, "height": height, "videotoolbox": True,
                                               "hevc": True, "matrix": 1, "range": 1, "spherical": True}), encoding="utf-8")
                result = subprocess.run([str(RUNNER), str(request)], capture_output=True, timeout=120)
                self.assertEqual(result.returncode, 0, result.stderr)
                report = json.loads(result.stdout)
                self.assertTrue(report["success"])
                self.assertIn(report["backend"], (2, 3))
                self.assertEqual(report["writer_calls"], 8)
                decoded = json.loads(subprocess.check_output([FFPROBE, "-v", "error", "-show_frames",
                                                              "-show_streams", "-show_format", "-of", "json", str(video)]))
                stream = decoded["streams"][0]
                self.assertEqual((stream["codec_name"], stream["codec_tag_string"], stream["pix_fmt"]),
                                 ("hevc", "hvc1", "yuv420p"))
                assert_hvc1_samples(self, FFPROBE, video)
                self.assertEqual((stream["width"], stream["height"]), (width, height))
                spherical = next(item for item in stream["side_data_list"] if item["side_data_type"] == "Spherical Mapping")
                self.assertEqual(spherical["projection"], "equirectangular")
                self.assertEqual(stream["color_space"], "bt709")
                self.assertEqual(stream["color_range"], "tv")
                self.assertEqual(decoded["format"]["tags"]["comment"], report["comment"])
                self.assertEqual(len(decoded["frames"]), 4)
                for index, frame in enumerate(decoded["frames"]):
                    self.assertAlmostEqual(float(frame["best_effort_timestamp_time"]), index / 10)
                pixels = subprocess.check_output([FFMPEG, "-v", "error", "-i", str(video), "-vf", "scale=64:48",
                                                  "-pix_fmt", "yuv420p", "-f", "rawvideo", "-"])
                self.assertEqual(len(pixels), 4 * 4608)
                for index in range(4):
                    luma = pixels[index * 4608:index * 4608 + 3072]
                    self.assertLess(sum(abs(value - (32 + index * 32)) for value in luma) / len(luma), 3)

    @unittest.skipUnless(os.environ.get("LFS_MEDIA_TEST_VIDEOTOOLBOX_8K") == "1",
                         "requires an explicit native Mac VideoToolbox run")
    def test_hevc_apple_software_fallback_and_hardware_recovery(self):
        work = Path(tempfile.mkdtemp(prefix="apple-hevc-fallback-", dir=self.root))
        for width, height in ((64, 48), (7680, 4320)):
            with self.subTest(extent=(width, height)):
                video = work / f"fallback-{width}.mp4"
                request = work / "request.json"
                request.write_text(json.dumps({"output": str(video), "spherical": True, "width": width, "height": height}), encoding="utf-8")
                runner = RUNNER.parent / "media_apple_hevc_fallback_runner"
                result = subprocess.run([str(runner), str(request)], capture_output=True, timeout=60)
                self.assertEqual(result.returncode, 0, result.stderr)
                report = json.loads(result.stdout)
                self.assertEqual(report["backend"], 3)
                self.assertEqual(report["recovery_backend"], 2)
                for path in (video, Path(report["recovery_output"])):
                    assert_hvc1_samples(self, FFPROBE, path)
                    decoded = json.loads(subprocess.check_output([FFPROBE, "-v", "error", "-show_streams",
                                                                  "-show_frames", "-of", "json", str(path)]))
                    stream = decoded["streams"][0]
                    self.assertEqual((stream["codec_name"], stream["codec_tag_string"], stream["pix_fmt"]),
                                     ("hevc", "hvc1", "yuv420p"))
                    self.assertEqual((stream["width"], stream["height"]), (width, height))
                    self.assertEqual(len(decoded["frames"]), 4)
                    spherical = next(item for item in stream["side_data_list"] if item["side_data_type"] == "Spherical Mapping")
                    self.assertEqual(spherical["projection"], "equirectangular")
                    for index, frame in enumerate(decoded["frames"]):
                        self.assertAlmostEqual(float(frame["best_effort_timestamp_time"]), index / 10)
                    pixels = subprocess.check_output([FFMPEG, "-v", "error", "-i", str(path), "-vf", "scale=64:48", "-f", "rawvideo",
                                                      "-pix_fmt", "yuv420p", "-"])
                    self.assertEqual(len(pixels), 4 * 4608)
                    for index in range(4):
                        luma = pixels[index * 4608:index * 4608 + 3072]
                        self.assertLess(sum(abs(value - (32 + index * 32)) for value in luma) / len(luma), 3)

    def test_compatibility_api_embeds_jpeg_source_metadata(self):
        from jpeg_metadata_fixture import read as read_jpeg_metadata
        result, output = self.extract(format="jpg", embed_source_metadata=True, rotation=90)
        self.assertTrue(result["success"], result["error"])
        manifest = json.loads((output / "extraction_metadata.json").read_text(encoding="utf-8"))
        images = list(output.glob("*.jpg"))
        self.assertTrue(images)
        for image in images:
            _, orientation, size, facts, catalog, _ = read_jpeg_metadata(image.read_bytes())
            self.assertEqual((orientation, size), (1, (32, 64)))
            self.assertEqual(facts["AppliedClockwiseRotation"], "90")
            self.assertEqual(catalog, manifest["source_metadata"])

    def test_optional_jpeg_backend_failures_and_cpu_fallback(self):
        reference, files = self.extract(format="jpg")
        self.assertTrue(reference["success"], reference["error"])
        encoded = next(files.glob("*.jpg"))
        for mode, hardware, expected, count, calls in (("short", True, False, 0, 1),
                ("empty", True, False, 0, 1), ("throw-later", True, False, 2, 1),
                ("metadata-bad-app0", True, False, 0, 1),
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

    def test_sdr_manifest_replacement_preserves_legacy_policy(self):
        for image_format in ("png", "jpg"):
            with self.subTest(format=image_format):
                output = Path(tempfile.mkdtemp(prefix="replace-manifest-", dir=self.root))
                manifest = output / "extraction_metadata.json"
                manifest.write_bytes(b'{"previous_run":true}')
                result, _ = self.extract(output=str(output), format=image_format)
                self.completed(result)
                metadata = self.metadata(output)
                self.assertEqual(metadata["schema_version"], 2)
                self.assertEqual(len(metadata["frames"]), 4)
                self.assertTrue(all((output / frame["file"]).is_file() for frame in metadata["frames"]))
                self.assertEqual(list(output.glob("extraction_metadata.json.tmp-*")), [])

    def test_sdr_manifest_cancel_before_publication_preserves_previous_file(self):
        output = Path(tempfile.mkdtemp(prefix="cancel-manifest-", dir=self.root))
        manifest = output / "extraction_metadata.json"
        previous = b'{"previous_run":true}\n'
        manifest.write_bytes(previous)
        result, _ = self.extract(output=str(output), cancel_metadata_commit=True)
        self.assertFalse(result["success"])
        self.assertEqual(result["outcome"], "cancelled")
        self.assertTrue(result["error"])
        self.assertEqual(manifest.read_bytes(), previous)
        self.assert_frames(output, [0, 1, 2, 3])
        self.assertEqual(list(output.glob("extraction_metadata.json.tmp-*")), [])

    def test_sdr_manifest_publication_failure_keeps_committed_images(self):
        output = Path(tempfile.mkdtemp(prefix="failed-manifest-", dir=self.root))
        manifest = output / "extraction_metadata.json"
        manifest.mkdir()
        sentinel = manifest / "keep.txt"
        sentinel.write_bytes(b"unrelated data")
        result, _ = self.extract(output=str(output))
        self.assertFalse(result["success"])
        self.assertEqual(result["outcome"], "failed")
        self.assertTrue(result["error"])
        self.assertEqual(sentinel.read_bytes(), b"unrelated data")
        self.assert_frames(output, [0, 1, 2, 3])
        self.assertEqual(list(output.glob("extraction_metadata.json.tmp-*")), [])

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

    def test_manifest_preserves_exact_source_timing_and_legacy_fields(self):
        offset_source = self.root / "offset-source.nut"
        subprocess.run([FFMPEG, "-v", "error", "-y", "-itsoffset", "2", "-i",
                        str(self.corpus / "cfr-asymmetric.nut"), "-c", "copy", str(offset_source)],
                       check=True, capture_output=True)
        for fixture, options in (("cfr-asymmetric", {"start": 0.1, "end": 0.35}),
                                 ("vfr-asymmetric", {"end": 1.5}),
                                 ("vfr-asymmetric", {"end": 1.5, "sharpness": True, "window": True}),
                                 ("cfr-asymmetric", {"input": str(offset_source)})):
            with self.subTest(fixture=fixture, options=options):
                source = Path(options.get("input", self.corpus / (fixture + ".nut")))
                reference = json.loads(subprocess.check_output([
                    FFPROBE, "-v", "error", "-select_streams", "v:0", "-show_streams",
                    "-show_frames", "-of", "json", str(source)]))
                stream = reference["streams"][0]
                expected_pts = {int(frame["best_effort_timestamp"]) for frame in reference["frames"]}
                result, output = self.extract(fixture, **options)
                self.completed(result)
                manifest = self.metadata(output)
                self.assertEqual(manifest["schema_version"], 2)
                timing = manifest["input"]["video"]
                self.assertEqual(timing["stream_index"], stream["index"])
                self.assertEqual(timing["time_base"], [int(part) for part in stream["time_base"].split("/")])
                self.assertEqual(timing["start_pts"], int(stream["start_pts"]) if "start_pts" in stream else None)
                if source == offset_source:
                    self.assertGreater(timing["start_pts"], 0)
                # The compatibility runner retains the default request even
                # though its CPU-only backend performs software decoding.
                self.assertTrue(manifest["processing"]["decoder"]["hardware_requested"])
                self.assertEqual(manifest["processing"]["decoder"]["backend"], "ffmpeg_software")
                self.assertTrue(manifest["provenance"]["ffmpeg_version"])
                self.assertGreater(manifest["provenance"]["avcodec_version"], 0)
                self.assertGreater(manifest["provenance"]["avformat_version"], 0)
                self.assertTrue(manifest["frames"])
                for frame in manifest["frames"]:
                    self.assertIn(frame["source_pts"], expected_pts)
                    self.assertEqual(frame["timestamp_origin"], "best_effort")
                    seconds = ((frame["source_pts"] - timing["relative_origin_pts"])
                               * timing["time_base"][0] / timing["time_base"][1])
                    self.assertAlmostEqual(frame["timestamp"], seconds, places=6)
                    self.assertNotIn("time_base", frame)
                    self.assertTrue((output / frame["file"]).is_file())
                self.assertEqual(json.loads(json.dumps(manifest)), manifest)

    def test_offset_h264_keeps_first_frame_and_exact_sampling_boundaries(self):
        source = self.root / "offset-h264.mp4"
        subprocess.run([FFMPEG, "-v", "error", "-f", "lavfi", "-i",
                        "testsrc2=size=128x96:rate=30:duration=2", "-vf", "setpts=PTS+2/TB",
                        "-c:v", "libx264", "-bf", "3", "-pix_fmt", "yuv420p", str(source)],
                       check=True, capture_output=True)
        reference = json.loads(subprocess.check_output([
            FFPROBE, "-v", "error", "-select_streams", "v:0", "-show_frames", "-of", "json", str(source)]))
        for options, indices in (({"interval": 1}, list(range(31))),
                                 ({"mode": "fps", "fps": 5}, [0, 6, 12, 18, 24])):
            with self.subTest(options=options):
                result, output = self.extract(input=str(source), end=1, **options)
                self.completed(result)
                frames = self.metadata(output)["frames"]
                self.assertEqual([frame["source_pts"] for frame in frames],
                                 [int(reference["frames"][i]["best_effort_timestamp"]) for i in indices])
                self.assertEqual(frames[0]["timestamp"], 0)

    def test_manifest_identifies_video_after_audio_and_ignores_second_video(self):
        source = self.root / "multiple-streams.mkv"
        subprocess.run([FFMPEG, "-v", "error", "-f", "lavfi", "-i",
                        "sine=frequency=440:duration=1", "-i", str(self.corpus / "cfr-asymmetric.nut"),
                        "-f", "lavfi", "-i", "color=red:size=96x64:rate=10:duration=1",
                        "-map", "0:a", "-map", "1:v", "-map", "2:v", "-c:a", "pcm_s16le",
                        "-c:v", "ffv1", str(source)], check=True, capture_output=True)
        reference = json.loads(subprocess.check_output([
            FFPROBE, "-v", "error", "-select_streams", "v:0", "-show_streams",
            "-show_frames", "-of", "json", str(source)]))
        stream = reference["streams"][0]
        self.assertEqual(stream["index"], 1)
        result, output = self.extract(input=str(source))
        self.completed(result)
        self.assert_frames(output, [0, 1, 2, 3])
        manifest = self.metadata(output)
        self.assertEqual(manifest["source_size"], [64, 32])
        self.assertEqual(manifest["input"]["video"]["stream_index"], 1)
        self.assertEqual(manifest["input"]["video"]["time_base"],
                         [int(part) for part in stream["time_base"].split("/")])
        self.assertEqual([frame["source_pts"] for frame in manifest["frames"]],
                         [int(frame["best_effort_timestamp"]) for frame in reference["frames"][:4]])

    def test_manifest_does_not_invent_missing_source_pts(self):
        elementary = self.root / "missing-pts.m2v"
        source = self.root / "missing-pts.avi"
        subprocess.run([FFMPEG, "-v", "error", "-f", "lavfi", "-i",
                        "testsrc2=size=64x32:rate=10:duration=1", "-c:v", "mpeg2video",
                        "-b:v", "2M", "-f", "mpeg2video", str(elementary)], check=True, capture_output=True)
        # AVI supplies a known duration; the decoder's final delayed frame
        # has neither best-effort nor presentation PTS. Set the container
        # rate explicitly rather than inheriting the raw parser's guess.
        subprocess.run([FFMPEG, "-v", "error", "-i", str(elementary),
                        "-c", "copy", "-r", "10", str(source)], check=True, capture_output=True)
        reference = json.loads(subprocess.check_output([
            FFPROBE, "-v", "error", "-select_streams", "v:0", "-show_frames", "-of", "json", str(source)]))
        self.assertNotIn("best_effort_timestamp", reference["frames"][-1])
        self.assertNotIn("pts", reference["frames"][-1])
        result, output = self.extract(input=str(source), interval=3, end=1)
        self.completed(result)
        frames = self.metadata(output)["frames"]
        self.assertEqual(len(frames), 4)
        for selected, decoded_index in zip(frames[:-1], (0, 3, 6)):
            self.assertEqual(selected["source_pts"], int(reference["frames"][decoded_index]["best_effort_timestamp"]))
            self.assertEqual(selected["timestamp_origin"], "best_effort")
        self.assertIsNone(frames[-1]["source_pts"])
        self.assertEqual(frames[-1]["timestamp_origin"], "missing")
        self.assertAlmostEqual(frames[-1]["timestamp"], .9, places=6)

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

    def test_vfr_file_sink_uses_nominal_rate_when_average_differs(self):
        source = self.root / "variable-rate.mp4"
        subprocess.run([
            FFMPEG, "-v", "error", "-y", "-f", "lavfi", "-i", "testsrc2=size=64x32:rate=30",
            "-frames:v", "12", "-vf", "setpts='if(lt(N,6),N,6+(N-6)*2)/(30*TB)'",
            "-fps_mode", "vfr", "-c:v", "mpeg4", "-q:v", "2", str(source),
        ], check=True, capture_output=True)
        probe = json.loads(subprocess.check_output([
            FFPROBE, "-v", "error", "-select_streams", "v:0", "-show_streams", "-of", "json", str(source),
        ]))["streams"][0]
        self.assertEqual(probe["r_frame_rate"], "30/1")
        self.assertNotEqual(probe["avg_frame_rate"], probe["r_frame_rate"])
        expected = [1, 2, 3, 4, 5, 6, 7, 9, 11, 13, 15, 17]
        for sink in ("legacy", "file"):
            with self.subTest(sink=sink):
                result, output = self.extract(input=str(source), sink=sink, end=-1.0)
                self.completed(result)
                self.assertEqual(sorted(path.name for path in output.glob("*.png")),
                                 [f"frame_{number:03d}.png" for number in expected])
                if sink == "legacy":
                    frames = self.metadata(output)["frames"]
                    self.assertEqual([frame["source_frame"] for frame in frames], expected)
                    self.assertTrue(all((output / frame["file"]).exists() for frame in frames))

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
