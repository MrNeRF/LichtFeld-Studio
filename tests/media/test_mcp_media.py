# SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
# SPDX-License-Identifier: GPL-3.0-or-later
"""Production MCP media tools, asynchronous job state and runtime event routing."""
import argparse
import json
import os
from pathlib import Path
import subprocess
import tempfile
import unittest

RUNNER = None
FFMPEG = os.environ.get("LFS_MEDIA_TEST_FFMPEG", "ffmpeg")


class McpMedia(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.temp = tempfile.TemporaryDirectory(prefix="lfs-mcp-media-")
        cls.addClassCleanup(cls.temp.cleanup)
        cls.root = Path(cls.temp.name)
        cls.source = cls.root / "clip é 日本語.nut"
        # Keep the MCP output/rotation reference lossless RGB. The fixture tool
        # can use a newer FFmpeg than Studio; YUV conversion rounding differs.
        subprocess.run([FFMPEG, "-v", "error", "-f", "lavfi", "-i",
                        "testsrc2=size=64x48:rate=10:duration=0.5", "-c:v", "ffv1",
                        "-pix_fmt", "bgr0",
                        str(cls.source)], check=True, timeout=30)

    def invoke(self, cancel=False, **options):
        directory = Path(tempfile.mkdtemp(dir=self.root))
        output = directory / "frames é 日本語"
        request = {"input": str(self.source), "output_directory": str(output),
                   "selection": {"mode": "interval", "interval": 1},
                   "end_seconds": 0.35, "allow_hardware_decode": False,
                   "write_metadata": True, **options}
        path = directory / "request.json"
        path.write_text(json.dumps({"request": request, "cancel": cancel}), encoding="utf-8")
        result = subprocess.run([str(RUNNER), str(path)], capture_output=True, timeout=90)
        self.assertEqual(result.returncode, 0, result.stderr.decode("utf-8", errors="replace"))
        return json.loads(result.stdout), output

    def test_cpu_color_matches_frozen_dev_with_strides_and_nonfinite_values(self):
        path = self.root / "cpu-color.json"
        path.write_text(json.dumps({"operation": "cpu-color"}), encoding="utf-8")
        result = subprocess.run([str(RUNNER), str(path)], capture_output=True, timeout=30)
        self.assertEqual(result.returncode, 0, result.stderr.decode("utf-8", errors="replace"))
        self.assertEqual(json.loads(result.stdout), {"success": True, "checked_bytes": 768, "backend": "cpu"})

    def test_probe_extract_options_and_single_active_job(self):
        result, output = self.invoke(geometry={"clockwise_rotation": 90})
        self.assertTrue(result["probe"]["success"])
        self.assertTrue(result["capabilities"]["software_decode"])
        self.assertTrue(result["capabilities"]["ffmpeg_license"])
        self.assertEqual(result["probe"]["media"]["streams"][0]["width"], 64)
        self.assertTrue(result["start"]["success"])
        self.assertEqual(result["duplicate"]["error"]["code"], "FailedPrecondition")
        self.assertEqual(result["job"]["status"], "finished")
        self.assertFalse(result["job"]["active"])
        self.assertEqual(result["job"]["details"]["frames_accepted"], 4)
        images = sorted(output.glob("*.png"))
        self.assertEqual(len(images), 4)
        actual = subprocess.check_output([FFMPEG, "-v", "error", "-i", str(images[0]),
                                          "-f", "rawvideo", "-pix_fmt", "rgb24", "-"])
        expected = subprocess.check_output([FFMPEG, "-v", "error", "-i", str(self.source),
                                            "-vf", "transpose=clock", "-frames:v", "1",
                                            "-f", "rawvideo", "-pix_fmt", "rgb24", "-"])
        self.assertEqual(actual, expected)
        types = [event["type"] for event in result["events"]]
        self.assertEqual(types[0], "media.extract.started")
        self.assertIn("media.extract.progress", types)
        self.assertEqual(types[-1], "media.extract.completed")
        self.assertTrue(all(event["data"]["job_id"] == "media.extract" for event in result["events"]))

    def test_exr_job_and_schema_validation(self):
        result, output = self.invoke(format="exr", exr_precision="float", exr_compression="none",
                                     input_transfer="linear", input_primaries="bt709")
        self.assertTrue(result["capabilities"]["exr"])
        self.assertIn("YUV444", result["capabilities"]["float_sdr_profile"])
        self.assertTrue(result["start"]["success"])
        self.assertEqual(result["job"]["status"], "finished")
        self.assertEqual(len(list(output.glob("*.exr"))), 4)
        metadata = json.loads((output / "extraction_metadata.json").read_text(encoding="utf-8"))
        self.assertEqual(metadata["output"]["color_profile"], "linear-sdr")
        for options in [{"format": "exr", "jpeg_quality": 95}, {"format": "png", "exr_precision": "half"},
                        {"format": "exr", "input_transfer": "pq"}, {"format": "exr", "convert_hdr_to_sdr": True}]:
            result, output = self.invoke(**options)
            self.assertFalse(result["start"].get("success", False))
            self.assertEqual(result["start"]["error"]["code"], "InvalidArgument")
            self.assertFalse(output.exists())

    def test_cancel_retains_typed_error_and_terminal_event(self):
        result, output = self.invoke(cancel=True)
        self.assertEqual(result["job"]["status"], "cancelled")
        self.assertEqual(result["job"]["error_info"]["code"], "Cancelled")
        self.assertFalse(result["job"]["cancel_supported"])
        self.assertEqual(result["events"][-1]["type"], "media.extract.cancelled")
        self.assertEqual(len(list(output.glob("*.png"))), result["job"]["details"]["frames_accepted"])

    def test_failure_is_visible_without_changing_error_code(self):
        result, output = self.invoke(input=str(self.root / "missing.nut"))
        self.assertEqual(result["probe"]["error"]["code"], "NotFound")
        self.assertEqual(result["job"]["status"], "failed")
        self.assertEqual(result["job"]["error_info"]["code"], "Unavailable")
        self.assertEqual(result["events"][-1]["type"], "media.extract.failed")
        self.assertFalse(output.exists())

    def test_nested_schema_rejects_bad_types_enums_ranges_and_unknown_fields(self):
        for options in ({"selection": {"mode": "invalid"}}, {"geometry": {"width": 2**40}},
                        {"sharpness": {"enabled": "bad"}}, {"geometry": None},
                        {"format": "tiff"}, {"unexpected": True}):
            with self.subTest(options=options):
                result, output = self.invoke(**options)
                self.assertEqual(result["start"]["error"]["code"], "InvalidArgument")
                self.assertFalse(result["events"])
                self.assertFalse(output.exists())


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--runner", type=Path, required=True)
    args, remaining = parser.parse_known_args()
    RUNNER = args.runner.resolve()
    unittest.main(argv=[__file__, *remaining])
