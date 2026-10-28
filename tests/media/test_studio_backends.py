# SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
# SPDX-License-Identifier: GPL-3.0-or-later
"""Native Studio backends: real NVDEC/nvJPEG, HDR facade and CUDA diagnostics."""
import argparse
import json
import os
import re
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from hevc_contracts import assert_hvc1_samples
from jpeg_metadata_fixture import read as read_jpeg_metadata

FFMPEG = os.environ.get("LFS_MEDIA_TEST_FFMPEG", "ffmpeg")
FFPROBE = os.environ.get("LFS_MEDIA_TEST_FFPROBE", "ffprobe")
REFERENCE_FFMPEG = os.environ.get("LFS_MEDIA_TEST_REFERENCE_FFMPEG")
RUNNER = None


class StudioBackends(unittest.TestCase):
    @unittest.skipUnless(os.environ.get("LFS_MEDIA_TEST_NVENC_AV1") == "1",
                         "requires explicit qualification on an AV1-capable NVENC GPU")
    def test_av1_provider_quality_on_textured_motion(self):
        # Compare the actual session providers on identical YUV samples. A
        # color-conversion error or silent software fallback invalidates PSNR.
        for name, pattern in (("charts", "testsrc2=size=640x360:rate=30"),
                              ("fractal", "mandelbrot=size=640x360:rate=30:maxiter=100")):
            source = self.root / (name + ".yuv")
            subprocess.run([FFMPEG, "-v", "error", "-f", "lavfi", "-i", pattern,
                            "-frames:v", "48", "-pix_fmt", "yuv420p", "-f", "rawvideo", str(source)],
                           check=True, timeout=30)
            self.assertEqual(source.stat().st_size, 48 * 640 * 360 * 3 // 2)
            for quality in (18, 30, 40):
                scores = {}
                for software in (True, False):
                    with self.subTest(clip=name, quality=quality, software=software):
                        video = self.root / f"quality-{name}-{quality}-{software}.mp4"
                        actual, _ = self.invoke(operation="native-quality", input=str(source), output=str(video),
                                                crf=quality, software=software)
                        self.assertEqual(actual["backend"], "software" if software else "nvenc")
                        self.assertEqual(actual["frames"], 48)
                        info = json.loads(subprocess.check_output(
                            [FFPROBE, "-v", "error", "-count_frames", "-show_streams", "-of", "json", str(video)], timeout=30))["streams"][0]
                        self.assertEqual((info["codec_name"], info["pix_fmt"], info["width"], info["height"],
                                          info["color_space"], info["color_range"], int(info["nb_read_frames"])),
                                         ("av1", "yuv420p", 640, 360, "bt709", "tv", 48))
                        compared = subprocess.run(
                            [FFMPEG, "-hide_banner", "-nostdin", "-f", "rawvideo", "-pixel_format", "yuv420p",
                             "-video_size", "640x360", "-framerate", "30", "-colorspace", "bt709", "-color_range", "tv",
                             "-i", str(source), "-i", str(video), "-lavfi",
                             "[0:v]settb=AVTB,setpts=PTS-STARTPTS[a];[1:v]settb=AVTB,setpts=PTS-STARTPTS[b];[a][b]psnr",
                             "-f", "null", "-"], capture_output=True, text=True, timeout=30)
                        self.assertEqual(compared.returncode, 0, compared.stderr)
                        match = re.search(r"PSNR y:[\d.]+ u:[\d.]+ v:[\d.]+ average:([\d.]+)", compared.stderr)
                        self.assertIsNotNone(match, compared.stderr)
                        scores[software] = float(match[1])
                        self.assertGreater(scores[software], 30)
                self.assertEqual(len(scores), 2, "both actual providers must be measured")
                # CQP and adaptive CRF are content-dependent, not bit-identical.
                # This catches the old 5-7 dB gap without asserting exact PSNR
                # or file sizes across GPU generations and encoder releases.
                self.assertLessEqual(abs(scores[False] - scores[True]), 3.5,
                                     f"{name}, quality {quality}: NVENC={scores[False]}, libaom={scores[True]}")

    @classmethod
    def setUpClass(cls):
        cls.temp = tempfile.TemporaryDirectory(prefix="lfs-studio-media-")
        cls.addClassCleanup(cls.temp.cleanup)
        cls.root = Path(cls.temp.name)
        cls.source = cls.root / "clip.mp4"
        subprocess.run([FFMPEG, "-v", "error", "-f", "lavfi", "-i", "testsrc2=size=128x96:rate=10:duration=0.5",
                        "-c:v", "libx264", "-pix_fmt", "yuv420p",str(cls.source)], check=True, timeout=30)
        cls.hdr = cls.root / "pq.mp4"
        subprocess.run([FFMPEG, "-v", "error", "-f", "lavfi", "-i", "testsrc2=size=128x96:rate=10:duration=0.5",
                        "-pix_fmt", "yuv420p10le", "-color_primaries", "bt2020", "-color_trc", "smpte2084",
                        "-colorspace", "bt2020nc", "-c:v", "libx265", "-x265-params",
                        "pools=1:frame-threads=1:colorprim=bt2020:transfer=smpte2084:colormatrix=bt2020nc",str(cls.hdr)], check=True, timeout=30)

    def invoke(self, runner=None, **options):
        work = Path(tempfile.mkdtemp(dir=self.root))
        output = work / "frames"
        request = {"input": str(self.source), "output": str(output), **options}
        path = work / "request.json"
        path.write_text(json.dumps(request), encoding="utf-8")
        result = subprocess.run([str(runner or RUNNER), str(path)], capture_output=True, timeout=60)
        self.assertEqual(result.returncode,0,result.stderr.decode("utf-8",errors="replace") + result.stdout.decode("utf-8",errors="replace"))
        actual = json.loads(result.stdout)
        self.assertTrue(actual["success"],actual)
        if options.get("operation") in ("native-preview", "native-encode", "native-encode-session", "native-conversion", "native-yuv-color", "native-quality"):
            return actual, output
        self.assertTrue(actual["hardware"])
        self.assertTrue(actual["hdr"])
        self.assertTrue(actual["cuda_diagnostics"])
        return actual,output

    def test_p010_raw_precision_pitch_range_matrix_and_stream(self):
        result, _ = self.invoke(operation="native-yuv-color")
        self.assertEqual(result["checked_components"], 33 * 9 * 3 * 6)
        self.assertEqual(result["checked_chroma_locations"], 14)
        self.assertGreaterEqual(result["legacy_matrix_fallbacks"], 10)
        self.assertTrue(result["nondefault_stream"])

    def test_semiplanar_extraction_matches_host_color_without_frame_transfer(self):
        for bits, matrix, range_name in ((10, "smpte170m", "tv"), (10, "smpte170m", "pc"),
                                        (10, "bt709", "tv"), (10, "bt709", "pc"),
                                        (10, "bt2020nc", "tv"), (10, "bt2020nc", "pc"),
                                        (10, "smpte240m", "tv"), (10, "fcc", "pc"),
                                        (8, "bt709", "tv"), (8, "bt2020nc", "pc")):
            with self.subTest(bits=bits, matrix=matrix, range=range_name):
                source = self.root / f"sdr-{bits}-{matrix}-{range_name}.mp4"
                command = [FFMPEG, "-v", "error", "-f", "lavfi", "-i", "testsrc2=size=256x144:rate=10:duration=0.5",
                           "-pix_fmt", "yuv420p10le" if bits == 10 else "yuv420p", "-colorspace", matrix,
                           "-color_range", range_name, "-color_trc", "bt709", "-color_primaries", "bt709",
                           "-c:v", "libx265" if bits == 10 else "libx264"]
                if bits == 10:
                    command += ["-x265-params", "pools=1:frame-threads=1"]
                subprocess.run(command + [str(source)], check=True, timeout=30, capture_output=True)
                _, actual = self.invoke(input=str(source), jpeg=True)
                _, reference = self.invoke(input=str(source), jpeg=True, cpu_color=True)
                metadata = json.loads((actual / "extraction_metadata.json").read_text(encoding="utf-8"))
                baseline = json.loads((reference / "extraction_metadata.json").read_text(encoding="utf-8"))
                color = metadata["processing"]["color_conversion"]
                self.assertEqual(color["p010_frames" if bits == 10 else "nv12_frames"], metadata["performance"]["selected_frames"])
                self.assertEqual(color["backend"], "cuda")
                self.assertEqual(color["gpu_frames"], metadata["performance"]["selected_frames"])
                self.assertEqual(color["hardware_readback_frames"], 0)
                self.assertEqual(color["rgb_readback_frames"], 0)
                self.assertEqual(metadata["performance"]["timing_seconds"]["cuda_upload"], 0)
                self.assertEqual(baseline["processing"]["color_conversion"]["backend"], "host")
                self.assertEqual(baseline["processing"]["color_conversion"]["p010_frames"], 0)
                self.assertEqual(baseline["processing"]["color_conversion"]["nv12_frames"], 0)
                self.assertEqual(metadata["processing"]["decoder"]["backend"], "nvdec")
                self.assertEqual(metadata["processing"]["image_encoder"]["backend"], "nvimagecodec_cuda")
                files = sorted(actual.glob("*.jpg"))
                self.assertEqual([p.name for p in files], [p.name for p in sorted(reference.glob("*.jpg"))])
                for image in files:
                    def pixels(path):
                        return subprocess.check_output([FFMPEG, "-v", "error", "-i", str(path),
                                                        "-pix_fmt", "rgb24", "-f", "rawvideo", "-"])
                    a, b = pixels(image), pixels(reference / image.name)
                    self.assertEqual(len(a), len(b))
                    errors = [abs(x-y) for x, y in zip(a, b)]
                    # P010 swscale dithers during RGB8 conversion; subsequent JPEG
                    # quantization can amplify local differences near sharp edges.
                    self.assertLess(sum(errors) / len(errors), 3)
                    self.assertLessEqual(max(errors), 40)

    def test_nv12_chroma_and_default_matrices_keep_gpu_and_legacy_pixels(self):
        source = self.root / "nv12-review.mp4"
        subprocess.run([FFMPEG, "-v", "error", "-f", "lavfi", "-i",
                        "testsrc2=size=256x144:rate=30:duration=0.2", "-c:v", "libx264",
                        "-pix_fmt", "yuv420p", "-colorspace", "smpte170m", str(source)],
                       check=True, timeout=30, capture_output=True)
        baseline_runner = os.environ.get("LFS_MEDIA_TEST_NV12_BASELINE_RUNNER")
        legacy_pixels = None
        bt709_pixels = None
        for matrix, chroma in ((6, 0), (6, 2), (7, 1), (4, 2), (1, 0), (1, 2)):
            with self.subTest(matrix=matrix, chroma=chroma):
                variant = self.root / f"nv12-{matrix}-{chroma}.mp4"
                subprocess.run([FFMPEG, "-v", "error", "-i", str(source), "-c:v", "copy",
                                "-bsf:v", f"h264_metadata=matrix_coefficients={matrix}:chroma_sample_loc_type={chroma}",
                                str(variant)], check=True, timeout=30, capture_output=True)
                frames = json.loads(subprocess.check_output([FFPROBE, "-v", "error", "-select_streams", "v:0",
                                                            "-show_frames", "-of", "json", str(variant)]))["frames"]
                self.assertEqual(frames[0]["chroma_location"], ("left", "center", "topleft")[chroma])
                self.assertEqual(frames[0]["color_space"], {1: "bt709", 4: "fcc", 6: "smpte170m", 7: "smpte240m"}[matrix])
                _, output = self.invoke(input=str(variant), jpeg=True)
                meta = json.loads((output / "extraction_metadata.json").read_text(encoding="utf-8"))
                color = meta["processing"]["color_conversion"]
                self.assertEqual(meta["processing"]["decoder"]["backend"], "nvdec")
                self.assertEqual(color["backend"], "cuda")
                self.assertEqual(color["nv12_frames"], len(meta["frames"]))
                self.assertEqual(color["hardware_readback_frames"], 0)
                pixels = [p.read_bytes() for p in sorted(output.glob("*.jpg"))]
                self.assertTrue(pixels)
                if matrix == 6 and chroma == 0:
                    legacy_pixels = pixels
                if matrix in (4, 6, 7):
                    self.assertEqual(pixels, legacy_pixels)
                    if baseline_runner:
                        _, control = self.invoke(input=str(variant), jpeg=True, runner=baseline_runner)
                        self.assertEqual(pixels, [p.read_bytes() for p in sorted(control.glob("*.jpg"))])
                if matrix == 1:
                    if chroma == 0:
                        bt709_pixels = pixels
                    else:
                        self.assertEqual(pixels, bt709_pixels)

    def test_p010_top_left_and_mpeg1_center_keep_gpu_conversion(self):
        for codec, bits, location, suffix in (("libx265", 10, "topleft", ".mp4"), ("mpeg1video", 8, "center", ".mpg")):
            with self.subTest(codec=codec):
                source = self.root / (codec + suffix)
                command = [FFMPEG, "-v", "error", "-f", "lavfi", "-i",
                           "testsrc2=size=256x144:rate=30:duration=0.2", "-c:v", codec,
                           "-pix_fmt", "yuv420p10le" if bits == 10 else "yuv420p"]
                if bits == 10:
                    command += ["-colorspace", "bt709", "-chroma_sample_location", "topleft",
                                "-x265-params", "pools=1:frame-threads=1:chromaloc=2"]
                subprocess.run(command + [str(source)], check=True, timeout=30, capture_output=True)
                frames = json.loads(subprocess.check_output([FFPROBE, "-v", "error", "-select_streams", "v:0",
                                                            "-show_frames", "-of", "json", str(source)]))["frames"]
                self.assertEqual(frames[0]["chroma_location"], location)
                _, output = self.invoke(input=str(source), jpeg=True)
                meta = json.loads((output / "extraction_metadata.json").read_text(encoding="utf-8"))
                color = meta["processing"]["color_conversion"]
                self.assertEqual(meta["processing"]["decoder"]["backend"], "nvdec")
                self.assertEqual(color["backend"], "cuda")
                self.assertEqual(color["p010_frames" if bits == 10 else "nv12_frames"], len(meta["frames"]))
                self.assertGreater(color["gpu_frames"], 0)
                self.assertEqual(color["hardware_readback_frames"], 0)
                baseline_runner = os.environ.get("LFS_MEDIA_TEST_NV12_BASELINE_RUNNER")
                if bits == 8 and baseline_runner:
                    _, control = self.invoke(input=str(source), jpeg=True, runner=baseline_runner)
                    self.assertEqual([p.read_bytes() for p in sorted(output.glob("*.jpg"))],
                                     [p.read_bytes() for p in sorted(control.glob("*.jpg"))])

    def test_p010_rotation_scoring_and_resize_preserve_output(self):
        source = self.root / "p010-geometry.mp4"
        subprocess.run([FFMPEG, "-v", "error", "-f", "lavfi", "-i",
                        "testsrc2=size=256x144:rate=10:duration=0.5", "-pix_fmt", "yuv420p10le",
                        "-colorspace", "bt709", "-color_range", "tv", "-c:v", "libx265",
                        "-x265-params", "pools=1:frame-threads=1", str(source)],
                       check=True, timeout=30, capture_output=True)
        for options in ({}, {"rotation": 90}, {"rotation": 180}, {"rotation": 270},
                        {"sharpness": True}, {"sharpness": True, "window": True}, {"scale": .5}):
            with self.subTest(options=options):
                _, output = self.invoke(input=str(source), jpeg=True, **options)
                metadata = json.loads((output / "extraction_metadata.json").read_text(encoding="utf-8"))
                size = [144, 256] if options.get("rotation") in (90, 270) else [256, 144]
                if options.get("scale"):
                    size = [128, 72]
                self.assertEqual(metadata["output_size"], size)
                self.assertGreater(len(metadata["frames"]), 0)
                color = metadata["processing"]["color_conversion"]
                if not options.get("scale") and not options.get("window"):
                    self.assertEqual(color["backend"], "cuda")
                    self.assertGreater(color["p010_frames"], 0)
                    self.assertEqual(color["hardware_readback_frames"], 0)
                for image in output.glob("*.jpg"):
                    pixels = subprocess.check_output([FFMPEG, "-v", "error", "-i", str(image),
                                                      "-pix_fmt", "rgb24", "-f", "rawvideo", "-"])
                    self.assertEqual(len(pixels), size[0] * size[1] * 3)
                    self.assertGreater(max(pixels) - min(pixels), 20)

    def test_native_manifest_preserves_offset_reordered_pts(self):
        source = self.root / "offset-reordered.mkv"
        subprocess.run([FFMPEG, "-v", "error", "-f", "lavfi", "-i",
                        "sine=frequency=440:duration=4", "-itsoffset", "2", "-f", "lavfi", "-i",
                        "testsrc2=size=128x96:rate=30:duration=4", "-map", "0:a", "-map", "1:v",
                        "-c:a", "pcm_s16le", "-c:v", "libx264", "-bf", "3", "-g", "120",
                        "-pix_fmt", "yuv420p", str(source)], check=True, timeout=30)
        reference = json.loads(subprocess.check_output([
            FFPROBE, "-v", "error", "-select_streams", "v:0", "-show_streams",
            "-show_frames", "-of", "json", str(source)]))
        stream = reference["streams"][0]
        self.assertEqual(stream["index"], 1)
        self.assertTrue(any(frame["pict_type"] == "B" for frame in reference["frames"]))
        for options in ({}, {"fps": 2}):
            with self.subTest(options=options):
                _, output = self.invoke(input=str(source), end_seconds=3, **options)
                manifest = json.loads((output / "extraction_metadata.json").read_text())
                self.assertIn("nvdec", json.dumps(manifest))
                timing = manifest["input"]["video"]
                self.assertEqual(timing["stream_index"], 1)
                self.assertGreater(timing["start_pts"], 0)
                self.assertEqual(timing["time_base"], [int(p) for p in stream["time_base"].split("/")])
                expected = {int(frame["best_effort_timestamp"]) for frame in reference["frames"]}
                self.assertTrue(manifest["frames"])
                self.assertEqual(manifest["frames"][0]["source_pts"],
                                 int(reference["frames"][0]["best_effort_timestamp"]))
                self.assertEqual(manifest["frames"][0]["timestamp"], 0)
                if "fps" in options:
                    self.assertEqual([frame["timestamp"] for frame in manifest["frames"]],
                                     [0, .5, 1, 1.5, 2, 2.5])
                for frame in manifest["frames"]:
                    self.assertIn(frame["source_pts"], expected)
                    self.assertEqual(frame["timestamp_origin"], "best_effort")
                    seconds = ((frame["source_pts"] - timing["relative_origin_pts"])
                               * timing["time_base"][0] / timing["time_base"][1])
                    self.assertAlmostEqual(frame["timestamp"], seconds, places=6)

    def test_shared_player_uses_native_decode_and_seeks(self):
        # Spatial/temporal luma identifies the frame and layout. Constant chroma
        # makes the RGB oracle independent of chroma upsampling changes between
        # the fixture FFmpeg executable and the linked player FFmpeg version.
        source = self.root / "native-preview-gradient.mp4"
        subprocess.run([FFMPEG, "-v", "error", "-f", "lavfi", "-i",
                        "nullsrc=size=128x96:rate=10:duration=0.5,format=yuv420p,"
                        "geq=lum='32+N*32+X/8+Y/12':cb='96+X/4':cr='160-Y/3'",
                        "-c:v", "libx264", "-pix_fmt", "yuv420p", str(source)], check=True, timeout=30)
        actual, _ = self.invoke(operation="native-preview", input=str(source))
        self.assertTrue(actual["hardware_decode"])
        self.assertEqual(actual["size"], [128, 96])
        self.assertAlmostEqual(actual["time"], .1, places=6)
        reference = subprocess.check_output([FFMPEG, "-v", "error", "-i", str(source),
                                              "-vf", "format=nv12,select=eq(n\\,1),"
                                              "scale=flags=bilinear:in_color_matrix=bt601:"
                                              "out_color_matrix=bt709:out_range=full", "-frames:v", "1",
                                              "-f", "rawvideo", "-pix_fmt", "rgb24", "-"])
        pixels = bytes(actual["pixels"])
        self.assertEqual(len(pixels), len(reference))
        # NVDEC supplies NV12; the player uses SWS_BILINEAR (FAST_BILINEAR
        # biases neutral limited-range samples). Compare that same conversion,
        # rather than planar YUV420P with FFmpeg's default scaler. Independent swscale
        # builds can round this path differently. These bounds also pass the
        # unchanged dev player (the migrated player is byte-identical to it).
        errors = [abs(a - b) for a, b in zip(pixels, reference)]
        self.assertLess(sum(errors) / len(errors), 2)
        self.assertLessEqual(max(errors), 8)
        # This fixture must distinguish the requested frame, matrix, range and UV order;
        # otherwise avoiding chroma edges would weaken the seek/color contract.
        for wrong in ("select=eq(n\\,0)", "in_color_matrix=bt709", "in_range=full", "swapuv"):
            filter_ = ("format=nv12,select=eq(n\\,1),scale=flags=bilinear:"
                       "in_color_matrix=bt601:out_color_matrix=bt709:out_range=full")
            if wrong.startswith("select"):
                filter_ = filter_.replace("select=eq(n\\,1)", wrong)
            elif wrong.startswith("in_color_matrix"):
                filter_ = filter_.replace("in_color_matrix=bt601", wrong)
            elif wrong == "swapuv":
                filter_ = filter_.replace("format=nv12,", "swapuv,format=nv12,")
            else:
                filter_ += ":" + wrong
            incorrect = subprocess.check_output([FFMPEG, "-v", "error", "-i", str(source), "-vf", filter_,
                                                "-frames:v", "1", "-f", "rawvideo", "-pix_fmt", "rgb24", "-"])
            self.assertGreater(sum(abs(a-b) for a,b in zip(reference, incorrect)) / len(reference), 2, wrong)

    def test_cuda_conversion_matches_scalar_color_contract(self):
        actual, _ = self.invoke(operation="native-conversion")
        self.assertEqual(actual["extents"], 3)
        self.assertEqual(actual["checked_bytes"], (2 * 2 + 34 * 18 + 320 * 240) * 3 // 2)
        self.assertTrue(actual["nondefault_stream"])

    @unittest.skipUnless(REFERENCE_FFMPEG, "Matched FFmpeg reference executable is not configured")
    def test_native_preview_chroma_edges_with_matched_reference(self):
        # Keep the complex chroma-edge check in addition to the portable seek
        # oracle. Fixture encoding can use system FFmpeg; RGB conversion must
        # use the qualified executable built from the player's FFmpeg version.
        actual, _ = self.invoke(operation="native-preview")
        self.assertTrue(actual["hardware_decode"])
        self.assertAlmostEqual(actual["time"], .1, places=6)
        reference = subprocess.check_output([REFERENCE_FFMPEG, "-v", "error", "-i", str(self.source),
                                              "-vf", "format=nv12,select=eq(n\\,1),"
                                              "scale=flags=bilinear:in_color_matrix=bt601:"
                                              "out_color_matrix=bt709:out_range=full", "-frames:v", "1",
                                              "-f", "rawvideo", "-pix_fmt", "rgb24", "-"])
        pixels = bytes(actual["pixels"])
        self.assertEqual(len(pixels), len(reference))
        errors = [abs(a-b) for a,b in zip(pixels, reference)]
        self.assertLess(sum(errors) / len(errors), 2)
        self.assertLessEqual(max(errors), 8)

    def test_shared_encoder_keeps_native_backend_and_studio_producers(self):
        for codec in ("h264", "hevc"):
            with self.subTest(codec=codec):
                path = self.root / "native-é-日本語.mp4"
                actual, _ = self.invoke(operation="native-encode", output=str(path), hevc=codec == "hevc")
                self.assertEqual(actual["backend"], "nvenc")
                probe = json.loads(subprocess.check_output([FFPROBE, "-v", "error", "-show_streams",
                                                            "-show_format", "-show_frames", "-of", "json", str(path)]))
                self.assertEqual(probe["streams"][0]["codec_name"], codec)
                self.assertEqual((probe["streams"][0]["width"], probe["streams"][0]["height"]), (320, 240))
                self.assertEqual(probe["streams"][0].get("color_space"), "smpte170m")
                self.assertEqual(probe["streams"][0].get("color_range"), "tv")
                self.assertEqual(actual["frames"], 12)
                self.assertEqual(len(probe["frames"]), 12)
                # The Studio adapter, not the leaf module, prepares application provenance.
                self.assertIsInstance(json.loads(probe["format"]["tags"]["comment"]), dict)
                for index, frame in enumerate(probe["frames"]):
                    self.assertAlmostEqual(float(frame["best_effort_timestamp_time"]), index / 10, places=6)
                pixels = subprocess.check_output([FFMPEG, "-v", "error", "-i", str(path),
                                                   "-f", "rawvideo", "-pix_fmt", "rgb24", "-"])
                frame_bytes = 320 * 240 * 3
                self.assertEqual(len(pixels), 12 * frame_bytes)
                for index in range(12):
                    frame = pixels[index * frame_bytes:(index + 1) * frame_bytes]
                    error = sum(abs(value - (64 + index * 12)) for value in frame) / len(frame)
                    self.assertLess(error, 6, f"Studio producer {index} color error {error}")

    def test_cuda_handoff_waits_on_streams_and_exception_unwind(self):
        result = subprocess.run([str(RUNNER), "--handoff-unit"], capture_output=True, timeout=60)
        log = result.stdout.decode("utf-8", errors="replace") + result.stderr.decode("utf-8", errors="replace")
        self.assertEqual(result.returncode, 0, log)
        self.assertIn("[  PASSED  ] 2 tests.", log)
        self.assertNotIn("[  SKIPPED ]", log)

    def test_nvenc_accepts_upper_quality_values_without_software_fallback(self):
        for codec in ("h264", "hevc"):
            for crf in (48, 49, 51):
                with self.subTest(codec=codec, crf=crf):
                    path = self.root / f"nvenc-{codec}-{crf}.mp4"
                    self.invoke(operation="native-encode-session", output=str(path),
                                hevc=codec == "hevc", crf=crf)
                    probe = json.loads(subprocess.check_output(
                        [FFPROBE, "-v", "error", "-show_streams", "-show_frames", "-of", "json", str(path)]))
                    self.assertEqual(probe["streams"][0]["codec_name"], codec)
                    self.assertEqual(len(probe["frames"]), 12)

    @unittest.skipUnless(os.environ.get("LFS_MEDIA_TEST_NVENC_AV1") == "1",
                         "requires explicit qualification on an AV1-capable NVENC GPU")
    def test_native_av1_8k_extent(self):
        path = self.root / "native-av1-8k.mp4"
        actual, _ = self.invoke(operation="native-encode-session", output=str(path), av1=True,
                                width=7680, height=4320, frames=2)
        self.assertEqual(actual["backend"], "nvenc")
        probe = json.loads(subprocess.check_output([FFPROBE, "-v", "error", "-show_streams", "-show_frames", "-of", "json", str(path)]))
        stream = probe["streams"][0]
        self.assertEqual((stream["codec_name"], stream["width"], stream["height"]), ("av1", 7680, 4320))
        self.assertEqual(len(probe["frames"]), 2)
        raw = subprocess.check_output([FFMPEG, "-v", "error", "-i", str(path), "-vf", "scale=64:36",
                                       "-pix_fmt", "yuv420p", "-f", "rawvideo", "-"], timeout=60)
        size = 64 * 36 * 3 // 2
        self.assertEqual(len(raw), 2 * size)
        for index in range(2):
            luma = raw[index * size:index * size + 64 * 36]
            self.assertLess(sum(abs(value - (32 + index * 16)) for value in luma) / len(luma), 3)

    @unittest.skipUnless(os.environ.get("LFS_MEDIA_TEST_NVENC_AV1") == "1",
                         "requires explicit qualification on an AV1-capable NVENC GPU")
    def test_native_av1_quality_boundaries(self):
        for crf in (1, 18, 51):
            path = self.root / f"av1-quality-{crf}.mp4"
            actual, _ = self.invoke(operation="native-encode-session", output=str(path), av1=True, crf=crf)
            self.assertEqual(actual["backend"], "nvenc")
            probe = json.loads(subprocess.check_output([FFPROBE, "-v", "error", "-show_streams", "-show_frames", "-of", "json", str(path)]))
            self.assertEqual(probe["streams"][0]["codec_name"], "av1")
            self.assertEqual(len(probe["frames"]), 12)

    def test_av1_zero_quality_preserves_codec_on_native_or_cpu_fallback(self):
        path = self.root / "av1-zero.mp4"
        actual, _ = self.invoke(operation="native-encode", output=str(path), av1=True, crf=0, allow_software=True)
        self.assertIn(actual["backend"], ("nvenc", "software"))
        probe = json.loads(subprocess.check_output([FFPROBE, "-v", "error", "-show_streams", "-show_frames", "-of", "json", str(path)]))
        self.assertEqual(probe["streams"][0]["codec_name"], "av1")
        self.assertEqual(len(probe["frames"]), 12)
        pixels = subprocess.check_output([FFMPEG, "-v", "error", "-i", str(path), "-pix_fmt", "rgb24", "-f", "rawvideo", "-"])
        size = 320 * 240 * 3
        self.assertEqual(len(pixels), 12 * size)
        for index in range(12):
            frame = pixels[index * size:(index + 1) * size]
            self.assertLess(sum(abs(value - (64 + index * 12)) for value in frame) / len(frame), 3)

    @unittest.skipUnless(os.environ.get("LFS_MEDIA_TEST_NVENC_AV1") == "1",
                         "requires explicit qualification on an AV1-capable NVENC GPU")
    def test_native_av1_session_and_tensor_producer(self):
        for operation in ("native-encode-session", "native-encode"):
            with self.subTest(operation=operation):
                path = self.root / (operation + "-av1.mp4")
                actual, _ = self.invoke(operation=operation, output=str(path), av1=True, spherical=True)
                if operation == "native-encode-session":
                    self.assertEqual(actual["backend"], "nvenc")
                probe = json.loads(subprocess.check_output([FFPROBE, "-v", "error", "-show_streams", "-show_frames", "-of", "json", str(path)]))
                stream = probe["streams"][0]
                self.assertEqual((stream["codec_name"], stream["codec_tag_string"], stream["pix_fmt"]), ("av1", "av01", "yuv420p"))
                self.assertEqual(len(probe["frames"]), 12)
                for index, frame in enumerate(probe["frames"]):
                    self.assertAlmostEqual(float(frame["best_effort_timestamp_time"]), index / 10)
                if operation == "native-encode-session":
                    raw = subprocess.check_output([FFMPEG, "-v", "error", "-i", str(path), "-pix_fmt", "yuv420p", "-f", "rawvideo", "-"])
                    size = 320 * 240 * 3 // 2
                    self.assertEqual(len(raw), 12 * size)
                    for index in range(12):
                        luma = raw[index * size:index * size + 320 * 240]
                        self.assertLess(sum(abs(value - (32 + index * 16)) for value in luma) / len(luma), 3)

    def test_native_session_preserves_distinct_queued_cuda_frames(self):
        path = self.root / "cuda-session.mp4"
        actual, _ = self.invoke(operation="native-encode-session", output=str(path))
        self.assertEqual(actual["backend"], "nvenc")
        pixels = subprocess.check_output([FFMPEG, "-v", "error", "-i", str(path),
                                           "-f", "rawvideo", "-pix_fmt", "yuv420p", "-"])
        frame_bytes = 320 * 240 * 3 // 2
        self.assertEqual(len(pixels), 12 * frame_bytes)
        for index in range(12):
            luma = pixels[index * frame_bytes:index * frame_bytes + 320 * 240]
            error = sum(abs(value - (32 + index * 16)) for value in luma) / len(luma)
            self.assertLess(error, 3, f"NVENC retained frame {index} was overwritten: {error}")

    def test_native_hevc_session_preserves_codec_pts_and_frames(self):
        path = self.root / "cuda-hevc.mp4"
        self.invoke(operation="native-encode-session", output=str(path), hevc=True, spherical=True)
        assert_hvc1_samples(self, FFPROBE, path)
        probe = json.loads(subprocess.check_output([FFPROBE, "-v", "error", "-show_streams", "-show_frames", "-of", "json", str(path)]))
        stream = probe["streams"][0]
        self.assertEqual(stream["codec_name"], "hevc")
        self.assertEqual(stream["codec_tag_string"], "hvc1")
        self.assertEqual(stream["pix_fmt"], "yuv420p")
        spherical = next(item for item in stream["side_data_list"] if item["side_data_type"] == "Spherical Mapping")
        self.assertEqual(spherical["projection"], "equirectangular")
        self.assertEqual(len(probe["frames"]), 12)
        pixels = subprocess.check_output([FFMPEG, "-v", "error", "-i", str(path), "-f", "rawvideo", "-pix_fmt", "yuv420p", "-"])
        size = 320 * 240 * 3 // 2
        self.assertEqual(len(pixels), 12 * size)
        for index, frame in enumerate(probe["frames"]):
            self.assertAlmostEqual(float(frame["best_effort_timestamp_time"]), index / 10)
            luma = pixels[index * size:index * size + 320 * 240]
            self.assertLess(sum(abs(value - (32 + index * 16)) for value in luma) / len(luma), 3)

    def test_gpu_jpeg_embedded_metadata_and_verified_replay(self):
        for resume in (False, True):
            with self.subTest(resume=resume):
                _, plain = self.invoke(jpeg=True, resume=resume, rotation=90)
                actual, output = self.invoke(jpeg=True, resume=resume, rotation=90, embed_source_metadata=True)
                manifest = json.loads((output / "extraction_metadata.json").read_text(encoding="utf-8"))
                self.assertEqual(actual["accepted"], len(manifest["frames"]))
                self.assertIn("nvimagecodec", json.dumps(manifest).lower())
                self.assertTrue(list(output.glob("*.jpg")))
                by_name = {frame["file"]: frame for frame in manifest["frames"]}
                for image in output.glob("*.jpg"):
                    original, orientation, size, facts, catalog, _ = read_jpeg_metadata(image.read_bytes())
                    self.assertEqual(original, (plain / image.name).read_bytes())
                    self.assertEqual((orientation, size), (1, (96, 128)))
                    self.assertEqual(facts["AppliedClockwiseRotation"], "90")
                    self.assertEqual(catalog, manifest["source_metadata"])
                    self.assertEqual(int(facts["SourceTimestampTicks"]), by_name[image.name]["source_pts"])
                if resume:
                    before = {p.name: (p.read_bytes(), p.stat().st_mtime_ns) for p in output.glob("*.jpg")}
                    request = output.parent / "request.json"
                    repeated = subprocess.run([str(RUNNER), str(request)], capture_output=True, timeout=60)
                    self.assertEqual(repeated.returncode, 0, repeated.stderr.decode(errors="replace"))
                    self.assertEqual(before, {p.name: (p.read_bytes(), p.stat().st_mtime_ns) for p in output.glob("*.jpg")})
                else:
                    blocked = output.parent / "no-replace"
                    blocked.mkdir()
                    first = sorted(output.glob("*.jpg"))[0]
                    target = blocked / first.name
                    target.write_bytes(first.read_bytes())
                    before = (target.read_bytes(), target.stat().st_mtime_ns)
                    request = output.parent / "no-replace.json"
                    request.write_text(json.dumps({"input": str(self.source), "output": str(blocked),
                                                  "jpeg": True, "rotation": 90, "embed_source_metadata": True,
                                                  "overwrite": False}), encoding="utf-8")
                    rejected = subprocess.run([str(RUNNER), str(request)], capture_output=True, timeout=60)
                    self.assertNotEqual(rejected.returncode, 0)
                    self.assertEqual(before, (target.read_bytes(), target.stat().st_mtime_ns))
                    self.assertEqual(list(blocked.iterdir()), [target])

    def test_native_decode_jpeg_rotation_and_scoring(self):
        reference = json.loads(subprocess.check_output([
            FFPROBE, "-v", "error", "-select_streams", "v:0", "-show_frames",
            "-show_streams", "-of", "json", str(self.source)]))
        expected_pts = {int(frame["best_effort_timestamp"]) for frame in reference["frames"]}
        expected_time_base = [int(part) for part in reference["streams"][0]["time_base"].split("/")]
        for options in ({}, {"jpeg":True}, {"jpeg":True,"rotation":90},
                        {"jpeg":True,"rotation":180}, {"jpeg":True,"rotation":270},
                        {"jpeg":True,"scale":0.5},
                        {"jpeg":True,"sharpness":True}, {"jpeg":True,"sharpness":True,"window":True}):
            with self.subTest(options=options):
                actual, output = self.invoke(**options)
                metadata = json.loads((output/"extraction_metadata.json").read_text(encoding="utf-8"))
                self.assertEqual(actual["accepted"],len(metadata["frames"]))
                self.assertGreater(actual["accepted"],0)
                self.assertIn("nvdec",json.dumps(metadata))
                self.assertTrue(metadata["processing"]["decoder"]["hardware_requested"])
                self.assertEqual(metadata["input"]["video"]["time_base"], expected_time_base)
                for frame in metadata["frames"]:
                    self.assertIn(frame["source_pts"], expected_pts)
                    self.assertEqual(frame["timestamp_origin"], "best_effort")
                expected_size = [96,128] if options.get("rotation") in (90,270) else [128,96]
                if options.get("scale"):
                    expected_size = [64,48]
                self.assertEqual(metadata["output_size"],expected_size)
                if options.get("jpeg") and not options.get("window"):
                    self.assertIn("nvimagecodec",json.dumps(metadata).lower())
                if options.get("window"):
                    self.assertEqual(metadata["processing"]["image_encoder"]["backend"],"cpu")
                for image in output.glob("*.jpg" if options.get("jpeg") else "*.png"):
                    pixels = subprocess.check_output([FFMPEG, "-v", "error", "-i",str(image),"-f", "rawvideo", "-pix_fmt", "rgb24", "-"])
                    self.assertEqual(len(pixels),expected_size[0]*expected_size[1]*3)
                    self.assertGreater(max(pixels)-min(pixels),20)

    def test_gpu_rotation_matches_independent_transform(self):
        _, original = self.invoke(jpeg=True)
        first = sorted(original.glob("*.jpg"))[0]
        for rotation, transform in ((90,"transpose=clock"), (180,"hflip,vflip"), (270,"transpose=cclock")):
            with self.subTest(rotation=rotation):
                _, rotated = self.invoke(jpeg=True, rotation=rotation)
                expected = subprocess.check_output([FFMPEG, "-v", "error", "-i", str(first),
                                                    "-vf", transform, "-f", "rawvideo", "-pix_fmt", "rgb24", "-"])
                actual = subprocess.check_output([FFMPEG, "-v", "error", "-i", str(sorted(rotated.glob("*.jpg"))[0]),
                                                  "-f", "rawvideo", "-pix_fmt", "rgb24", "-"])
                self.assertEqual(len(actual), len(expected))
                # JPEG quantization/chroma differ after rotation; the image geometry must agree.
                mean_error = sum(abs(a - b) for a, b in zip(actual, expected)) / len(actual)
                self.assertLess(mean_error, 10, f"GPU rotation disagrees with FFmpeg: MAE {mean_error}")

    def test_sparse_seek_preserves_targets_without_repeated_long_gop_decode(self):
        original_source = self.source
        self.addCleanup(setattr, self, "source", original_source)
        for label, gop, variable in (("long", 120, False), ("short", 6, False),
                                     ("variable", 120, True), ("one-second", 30, False),
                                     ("two-second", 60, False)):
            with self.subTest(clip=label):
                self.source = self.root / f"sparse-{label}.mp4"
                command = [FFMPEG, "-v", "error", "-f", "lavfi", "-i",
                           "testsrc2=size=128x96:rate=30:duration=4"]
                if variable:
                    command += ["-vf", "select=not(eq(mod(n\\,5)\\,1))", "-fps_mode", "vfr"]
                subprocess.run(command + ["-c:v", "libx264", "-g", str(gop),
                                          "-keyint_min", str(gop), "-sc_threshold", "0",
                                          "-pix_fmt", "yuv420p", str(self.source)], check=True, timeout=30)
                # The short clip samples one keyframe in five, so whole keyframe intervals lie between targets.
                fps = 1 if label in ("two-second", "short") else 2 if variable or label == "one-second" else 5
                _, output = self.invoke(fps=fps, end_seconds=4)
                metadata = json.loads((output / "extraction_metadata.json").read_text())
                self.assertEqual(metadata["performance"]["selected_frames"], 4 * fps)
                probe = json.loads(subprocess.check_output([
                    FFPROBE, "-v", "error", "-select_streams", "v:0", "-show_frames",
                    "-show_streams", "-show_entries", "frame=best_effort_timestamp_time,best_effort_timestamp,key_frame:stream=index,time_base,start_pts",
                    "-of", "json", str(self.source)]))
                timestamps = [float(frame["best_effort_timestamp_time"]) for frame in probe["frames"]]
                source_pts = {float(frame["best_effort_timestamp_time"]): int(frame["best_effort_timestamp"])
                              for frame in probe["frames"]}
                timing = metadata["input"]["video"]
                self.assertEqual(timing["stream_index"], probe["streams"][0]["index"])
                self.assertEqual(timing["time_base"], [int(part) for part in probe["streams"][0]["time_base"].split("/")])
                for index, frame in enumerate(metadata["frames"]):
                    expected = next(value for value in timestamps if value + 1e-7 >= index / fps)
                    self.assertAlmostEqual(frame["timestamp"], expected, places=6)
                    self.assertEqual(frame["source_pts"], source_pts[expected])
                    self.assertEqual(frame["timestamp_origin"], "best_effort")
                decoded = metadata["performance"]["decoded_frames"]
                seeks = metadata["processing"]["frame_selection"]["keyframe_seek_count"]
                last_target = (4 * fps - 1) / fps
                last = timestamps.index(next(value for value in timestamps if value + 1e-7 >= last_target))
                # A seek restarts at the keyframe before its target and only moves forward, so no frame is
                # decoded twice and nothing past the last target is read.
                self.assertLessEqual(decoded, last + 1)
                if label in ("one-second", "two-second"):
                    # Targets one keyframe interval apart decode on instead of seeking again.
                    self.assertLessEqual(seeks, 1)
                if label == "short":
                    # Targets several keyframe intervals apart still seek.
                    self.assertGreater(seeks, 2)

    def test_verified_gpu_jpeg_resume(self):
        work = Path(tempfile.mkdtemp(dir=self.root))
        output = work / "frames"
        path = work / "request.json"
        def run(**options):
            request = {"input": str(self.source), "output": str(output),
                       "jpeg": True, "resume": True, **options}
            path.write_text(json.dumps(request), encoding="utf-8")
            return subprocess.run([str(RUNNER), str(path)], capture_output=True, timeout=60)

        interrupted = run(exit_after=1)
        self.assertEqual(interrupted.returncode, 73, interrupted.stderr.decode(errors="replace"))
        first = next(output.glob("*.jpg"))
        original = (first.read_bytes(), first.stat().st_mtime_ns)
        complete = run()
        self.assertEqual(complete.returncode, 0, complete.stderr.decode(errors="replace"))
        self.assertEqual(json.loads(complete.stdout)["accepted"], 4)
        self.assertEqual((first.read_bytes(), first.stat().st_mtime_ns), original)
        identity = json.loads((output/".media-ingest-checkpoint"/"identity.json").read_text())
        self.assertEqual(identity["image_encoder"], "nvimagecodec_cuda")
        metadata = json.loads((output/"extraction_metadata.json").read_text())
        self.assertEqual(metadata["processing"]["image_encoder"], {"backend": "nvimagecodec_cuda", "batch_size": 1})
        images = {image.name: (image.read_bytes(), image.stat().st_mtime_ns) for image in output.glob("*.jpg")}
        replay = run()
        self.assertEqual(replay.returncode, 0, replay.stderr.decode(errors="replace"))
        self.assertEqual(images, {image.name: (image.read_bytes(), image.stat().st_mtime_ns) for image in output.glob("*.jpg")})
        first.unlink()
        restored = run()
        self.assertEqual(restored.returncode, 0, restored.stderr.decode(errors="replace"))
        self.assertEqual(first.read_bytes(), original[0])
        unavailable = run(disable_jpeg_provider=True)
        self.assertNotEqual(unavailable.returncode, 0)
        self.assertIn("provider is unavailable", unavailable.stdout.decode(errors="replace"))
        first.write_bytes(b"corrupt")
        corrupt = run()
        self.assertNotEqual(corrupt.returncode, 0)
        self.assertEqual(first.read_bytes(), b"corrupt")

        # A CPU checkpoint keeps its provider even when the GPU becomes available.
        output = work / "cpu"
        cpu = run(disable_jpeg_provider=True)
        self.assertEqual(cpu.returncode, 0, cpu.stderr.decode(errors="replace"))
        resumed_cpu = run()
        self.assertEqual(resumed_cpu.returncode, 0, resumed_cpu.stderr.decode(errors="replace"))
        self.assertEqual(json.loads((output/".media-ingest-checkpoint"/"identity.json").read_text())["image_encoder"], "cpu")
        for options in ({"rotation": 90}, {"scale": .5}, {"sharpness": True, "window": True}):
            with self.subTest(options=options):
                output = Path(tempfile.mkdtemp(dir=work)) / "frames"
                fresh = run(**options)
                self.assertEqual(fresh.returncode, 0, fresh.stderr.decode(errors="replace"))
                before = {image.name: image.read_bytes() for image in output.glob("*.jpg")}
                self.assertTrue(before)
                replay = run(**options)
                self.assertEqual(replay.returncode, 0, replay.stderr.decode(errors="replace"))
                self.assertEqual(before, {image.name: image.read_bytes() for image in output.glob("*.jpg")})
                gpu_output = output
                output = Path(tempfile.mkdtemp(dir=work)) / "cpu-reference"
                reference = run(disable_jpeg_provider=True, **options)
                self.assertEqual(reference.returncode, 0, reference.stderr.decode(errors="replace"))
                self.assertEqual(set(before), {image.name for image in output.glob("*.jpg")})
                name = sorted(before)[0]
                def decoded(image):
                    return subprocess.check_output([FFMPEG, "-v", "error", "-i", str(image),
                                                    "-f", "rawvideo", "-pix_fmt", "rgb24", "-"])
                actual, expected = decoded(gpu_output/name), decoded(output/name)
                self.assertEqual(len(actual), len(expected))
                # Different JPEG providers quantize/chroma-sample differently;
                # the transformed content must agree with the CPU writer.
                self.assertLess(sum(abs(a-b) for a,b in zip(actual,expected))/len(actual), 10)

    def test_memory_delivery_keeps_software_pixels(self):
        actual, output = self.invoke(memory=True)
        reference = subprocess.check_output([FFMPEG, "-v", "error", "-i",str(self.source),"-f", "rawvideo", "-pix_fmt", "rgb24", "-"])
        delivered = b"".join(bytes(frame) for frame in actual["frames"])
        self.assertEqual(delivered,reference[:len(delivered)])
        self.assertFalse(output.exists())

    def test_hdr_host_backend_remains_available(self):
        actual, output = self.invoke(input=str(self.hdr),hdr=True,hardware=False)
        self.assertEqual(actual["accepted"],4)
        self.assertEqual(len(list(output.glob("*.png"))),4)
        metadata = json.loads((output/"extraction_metadata.json").read_text(encoding="utf-8"))
        self.assertIn("libplacebo",json.dumps(metadata))
        color = metadata["processing"]["color_conversion"]
        self.assertEqual(color["backend"], "host")
        self.assertEqual(color["gpu_frames"], 0)
        self.assertEqual(color["host_frames"], 4)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--runner", type=Path, required=True)
    args, remaining = parser.parse_known_args()
    RUNNER = args.runner.resolve()
    availability = subprocess.run([str(RUNNER), "--check-gpu"], timeout=30)
    if availability.returncode == 77:
        sys.exit(77)
    if availability.returncode:
        raise RuntimeError("Studio runner could not query GPU availability")
    unittest.main(argv=[__file__,*remaining])
