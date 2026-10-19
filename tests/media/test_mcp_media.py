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
import importlib.util

RUNNER = None
FFMPEG = os.environ.get("LFS_MEDIA_TEST_FFMPEG", "ffmpeg")
_exif_spec = importlib.util.spec_from_file_location("source_exif_fixture", Path(__file__).resolve().parent / "source_exif_fixture.py")
source_exif_fixture = importlib.util.module_from_spec(_exif_spec)
_exif_spec.loader.exec_module(source_exif_fixture)


class McpMedia(unittest.TestCase):
    def test_tiff_decoded_tags_and_original_source_ranges(self):
        path = self.root / "TIFF metadata é 日本語.bin"
        for big in (False, True):
            for little in (False, True):
                with self.subTest(big=big, little=little):
                    payload = source_exif_fixture.write_tiff(path, big=big, little=little, pages=2)
                    request = self.root / "tiff-request.json"
                    request.write_text(json.dumps({"operation": "source-exif", "arguments": {"input": str(path)}}), encoding="utf-8")
                    response = subprocess.run([str(RUNNER), str(request)], capture_output=True, check=True, timeout=30)
                    self.assertEqual(response.stderr, b"")
                    value = json.loads(response.stdout)["exif"]
                    source_exif_fixture.check_tiff(self, value, path, payload, big=big, little=little, pages=2)
                    maker = source_exif_fixture.write_tiff_graph(path, big=big, little=little)
                    request = self.root / "tiff-request.json"
                    request.write_text(json.dumps({"operation": "source-exif", "arguments": {"input": str(path)}}), encoding="utf-8")
                    response = subprocess.run([str(RUNNER), str(request)], capture_output=True, check=True, timeout=30)
                    self.assertEqual(response.stderr, b"")
                    value = json.loads(response.stdout)["exif"]
                    source_exif_fixture.check_tiff_graph(self, value, path, maker)

    def test_png_idat_segmentation_and_ancillary_crc_recovery(self):
        path = self.root / "review PNG.png"
        request = self.root / "review-png.json"
        for options, count, status in (({"idat_chunks": 5000}, 1, "collected"),
                                        ({"idat_chunks": 5000, "include_exif": False}, 0, "absent"),
                                        ({"damaged_ancillary": True}, 1, "malformed")):
            with self.subTest(options=options):
                payload = source_exif_fixture.write_png(path, **options)
                request.write_text(json.dumps({"operation": "source-exif", "arguments": {"input": str(path)}}), encoding="utf-8")
                result = subprocess.run([str(RUNNER), str(request)], capture_output=True, check=True, timeout=30)
                value = json.loads(result.stdout)
                self.assertTrue(value["success"])
                source_exif_fixture.check_png(self, value["exif"], payload, count=count, status=status)

    def test_png_source_exif_shared_representation(self):
        path = self.root / "PNG EXIF é 日本語.png"
        payload = source_exif_fixture.write_png(path)
        request = self.root / "png-source-exif.json"
        request.write_text(json.dumps({"operation": "source-exif", "arguments": {"input": str(path)}}), encoding="utf-8")
        result = subprocess.run([str(RUNNER), str(request)], capture_output=True, check=True, timeout=30)
        value = json.loads(result.stdout)
        self.assertTrue(value["success"])
        source_exif_fixture.check_png(self, value["exif"], payload)

    def test_source_exif_invalid_utf8_diagnostic_remains_malformed(self):
        path = self.root / "invalid-diagnostic.avif"
        source_exif_fixture.write_invalid_heif(path)
        request = self.root / "source-exif-invalid.json"
        request.write_text(json.dumps({"operation": "source-exif", "arguments": {"input": str(path)}}), encoding="utf-8")
        result = subprocess.run([str(RUNNER), str(request)], capture_output=True, check=True, timeout=30)
        value = json.loads(result.stdout)
        self.assertTrue(value["success"])
        self.assertEqual(value["exif"]["status"], "malformed")
        self.assertIn("\ufffd", value["exif"]["diagnostic"])

    def test_source_exif_schema_originals_and_unicode(self):
        path = self.root / "original EXIF é 日本語.jpg"
        payload = source_exif_fixture.write_jpeg(path)
        request = self.root / "source-exif-request.json"
        for args in ({"input": str(path)}, {"input": {}}, {"input": str(path), "unexpected": True}, {}):
            request.write_text(json.dumps({"operation": "source-exif", "arguments": args}), encoding="utf-8")
            result = subprocess.run([str(RUNNER), str(request)], capture_output=True, check=True, timeout=30)
            value = json.loads(result.stdout)
            if args == {"input": str(path)}:
                self.assertTrue(value["success"])
                source_exif_fixture.check(self, value["exif"], payload)
            else:
                self.assertEqual(value["error"]["code"], "InvalidArgument")

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
        self.assertIn("subsampled chroma", result["capabilities"]["float_sdr_profile"])
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

    def test_avif_job_and_schema_validation(self):
        for options in ({}, {"avif_quality": 70, "avif_speed": 1}):
            result, output = self.invoke(format="avif", input_transfer="linear", input_primaries="bt709", **options)
            self.assertTrue(result["capabilities"]["avif"])
            self.assertTrue(result["start"]["success"])
            self.assertEqual(result["job"]["status"], "finished")
            self.assertEqual(len(list(output.glob("*.avif"))), 4)
            metadata = json.loads((output / "extraction_metadata.json").read_text(encoding="utf-8"))
            self.assertEqual(metadata["output"]["format"], "avif")
            self.assertEqual(metadata["output"]["bit_depth"], 8)
            self.assertEqual(metadata["output"]["compression"], "lossy" if options else "lossless-integer-codes")
        for options in ({"format": "avif", "jpeg_quality": 95},
                        {"format": "avif", "exr_precision": "half"},
                        {"format": "png", "avif_speed": 1},
                        {"format": "avif", "avif_quality": 101},
                        {"format": "avif", "avif_speed": 10},
                        {"format": "avif", "avif_quality": 4294967396},
                        {"format": "avif", "avif_speed": 4294967304},
                        {"format": "avif", "avif_depth": 4294967304},
                        {"format": "avif", "avif_speed": True},
                        {"format": "avif", "avif_depth": 8.5}):
            result, output = self.invoke(**options)
            self.assertFalse(result["start"].get("success", False))
            self.assertEqual(result["start"]["error"]["code"], "InvalidArgument")
            self.assertFalse(output.exists())

    def test_jxl_uint16_job_and_schema_validation(self):
        for transfer in ("linear", "srgb"):
            result, output = self.invoke(format="jxl", input_transfer="linear", input_primaries="bt709",
                                         jxl_precision="uint16", jxl_transfer=transfer, jxl_effort=1)
            self.assertEqual(len(list(output.glob("*.jxl"))), 4)
            metadata = json.loads((output / "extraction_metadata.json").read_text())
            self.assertEqual(metadata["output"]["bit_depth"], 16)
            self.assertEqual(metadata["output"]["transfer"], transfer)
        for options in ({"format": "png", "jxl_precision": "uint16"},
                        {"format": "jxl", "jxl_precision": "half"},
                        {"format": "jxl", "jxl_transfer": "srgb"},
                        {"format": "jxl", "jxl_precision": "uint16", "jxl_transfer": "bt709"}):
            result, output = self.invoke(**options)
            self.assertEqual(result["start"]["error"]["code"], "InvalidArgument")
            self.assertFalse(output.exists())

    def test_jxl_job_and_schema_validation(self):
        for options in ({}, {"jxl_distance": 1., "jxl_effort": 1}):
            result, output = self.invoke(format="jxl", input_transfer="linear", input_primaries="bt709", **options)
            self.assertTrue(result["capabilities"]["jxl"])
            self.assertTrue(result["start"]["success"])
            self.assertEqual(result["job"]["status"], "finished")
            self.assertEqual(len(list(output.glob("*.jxl"))), 4)
            metadata = json.loads((output / "extraction_metadata.json").read_text(encoding="utf-8"))
            self.assertEqual(metadata["output"]["format"], "jxl")
            self.assertEqual(metadata["output"]["bit_depth"], 32)
            self.assertEqual(metadata["output"]["compression"], "lossy" if options else "lossless")
        for options in ({"format": "jxl", "jpeg_quality": 95},
                        {"format": "jxl", "exr_precision": "half"},
                        {"format": "png", "jxl_effort": 1},
                        {"format": "jxl", "jxl_distance": 0.},
                        {"format": "jxl", "jxl_effort": 10},
                        {"format": "jxl", "jxl_effort": 1},
                        {"format": "jxl", "jxl_effort": 2},
                        {"format": "jxl", "jxl_effort": 3},
                        {"format": "jxl", "jxl_effort": 4.5},
                        {"format": "jxl", "jxl_effort": True},
                        {"format": "jxl", "jxl_effort": 4294967300}):
            result, output = self.invoke(**options)
            self.assertFalse(result["start"].get("success", False))
            self.assertEqual(result["start"]["error"]["code"], "InvalidArgument")
            self.assertFalse(output.exists())

    def test_tiff16_job_and_schema_validation(self):
        result, output = self.invoke(format="tiff16", tiff16_transfer="bt709",
                                     input_transfer="linear", input_primaries="bt709")
        self.assertTrue(result["capabilities"]["tiff16"])
        self.assertTrue(result["start"]["success"])
        self.assertEqual(result["job"]["status"], "finished")
        self.assertEqual(len(list(output.glob("*.tif"))), 4)
        metadata = json.loads((output / "extraction_metadata.json").read_text(encoding="utf-8"))
        self.assertEqual(metadata["output"]["bit_depth"], 16)
        self.assertEqual(metadata["output"]["transfer"], "bt709")
        for options in ({"format": "tiff16", "jpeg_quality": 95},
                        {"format": "tiff16", "exr_precision": "half"},
                        {"format": "png", "tiff16_transfer": "linear"},
                        {"format": "tiff16", "tiff16_transfer": "pq"}):
            result, output = self.invoke(**options)
            self.assertEqual(result["start"]["error"]["code"], "InvalidArgument")
            self.assertFalse(output.exists())

    def test_hdr_profile_job_and_explicit_units(self):
        for hlg,profile,transfer in [(False,"pq-nits","smpte2084"),(True,"hlg-scene","arib-std-b67")]:
            source=self.root/("hlg.mkv" if hlg else "pq.mkv")
            subprocess.run([FFMPEG,"-v","error","-f","lavfi","-i",
                            "testsrc2=size=64x48:rate=10:duration=0.5",
                            "-vf",f"format=yuv444p10le,setparams=color_primaries=bt2020:color_trc={transfer}:colorspace=bt2020nc:range=limited",
                            "-c:v","ffv1","-level","3",str(source)],check=True,timeout=30)
            result,output=self.invoke(input=str(source),format="exr",float_profile=profile,exr_precision="float")
            self.assertTrue(result["capabilities"]["rgb_float_hlg" if hlg else "rgb_float_pq"])
            self.assertTrue(result["start"]["success"])
            self.assertEqual(result["job"]["status"],"finished")
            self.assertEqual(len(list(output.glob("*.exr"))),4)
            metadata=json.loads((output/"extraction_metadata.json").read_text())
            self.assertEqual(metadata["output"]["units"],"scene-relative" if hlg else "cd/m2")
            self.assertEqual(metadata["output"]["color_profile"],"linear-hlg-scene" if hlg else "linear-pq-nits")
            result,output=self.invoke(float_profile=profile)
            self.assertEqual(result["start"]["error"]["code"],"InvalidArgument")
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
