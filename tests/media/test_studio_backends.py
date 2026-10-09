# SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
# SPDX-License-Identifier: GPL-3.0-or-later
"""Native Studio backends: real NVDEC/nvJPEG, HDR facade and CUDA diagnostics."""
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

FFMPEG = os.environ.get("LFS_MEDIA_TEST_FFMPEG", "ffmpeg")
FFPROBE = os.environ.get("LFS_MEDIA_TEST_FFPROBE", "ffprobe")
RUNNER = None


class StudioBackends(unittest.TestCase):
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

    def invoke(self, **options):
        work = Path(tempfile.mkdtemp(dir=self.root))
        output = work / "frames"
        request = {"input": str(self.source), "output": str(output), **options}
        path = work / "request.json"
        path.write_text(json.dumps(request), encoding="utf-8")
        result = subprocess.run([str(RUNNER), str(path)], capture_output=True, timeout=60)
        self.assertEqual(result.returncode,0,result.stderr.decode("utf-8",errors="replace") + result.stdout.decode("utf-8",errors="replace"))
        actual = json.loads(result.stdout)
        self.assertTrue(actual["success"],actual)
        if options.get("operation") in ("native-preview", "native-encode", "native-encode-session", "native-conversion"):
            return actual, output
        self.assertTrue(actual["hardware"])
        self.assertTrue(actual["hdr"])
        self.assertTrue(actual["cuda_diagnostics"])
        return actual,output

    def test_shared_player_uses_native_decode_and_seeks(self):
        actual, _ = self.invoke(operation="native-preview")
        self.assertTrue(actual["hardware_decode"])
        self.assertEqual(actual["size"], [128, 96])
        self.assertAlmostEqual(actual["time"], .1, places=6)
        reference = subprocess.check_output([FFMPEG, "-v", "error", "-i", str(self.source),
                                              "-vf", "format=nv12,select=eq(n\\,1),"
                                              "scale=flags=fast_bilinear:in_color_matrix=bt601:"
                                              "out_color_matrix=bt709:out_range=full", "-frames:v", "1",
                                              "-f", "rawvideo", "-pix_fmt", "rgb24", "-"])
        pixels = bytes(actual["pixels"])
        self.assertEqual(len(pixels), len(reference))
        # NVDEC supplies NV12; compare the matching conversion, rather than
        # planar YUV420P with FFmpeg's default scaler. Independent swscale
        # builds can round this path differently. These bounds also pass the
        # unchanged dev player (the migrated player is byte-identical to it).
        errors = [abs(a - b) for a, b in zip(pixels, reference)]
        self.assertLess(sum(errors) / len(errors), 2)
        self.assertLessEqual(max(errors), 8)

    def test_cuda_conversion_matches_scalar_color_contract(self):
        actual, _ = self.invoke(operation="native-conversion")
        self.assertEqual(actual["extents"], 3)
        self.assertEqual(actual["checked_bytes"], (2 * 2 + 34 * 18 + 320 * 240) * 3 // 2)
        self.assertTrue(actual["nondefault_stream"])

    def test_shared_encoder_keeps_native_backend_and_studio_producers(self):
        path = self.root / "native-é-日本語.mp4"
        actual, _ = self.invoke(operation="native-encode", output=str(path))
        self.assertEqual(actual["backend"], "nvenc")
        probe = json.loads(subprocess.check_output([FFPROBE, "-v", "error", "-show_streams",
                                                    "-show_format", "-show_frames", "-of", "json", str(path)]))
        self.assertEqual((probe["streams"][0]["width"], probe["streams"][0]["height"]), (320, 240))
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

    def test_native_decode_jpeg_rotation_and_scoring(self):
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
