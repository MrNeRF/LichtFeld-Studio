# SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
# SPDX-License-Identifier: GPL-3.0-or-later
"""Exact FPS boundary selection against FFprobe integer timestamps."""
import argparse
from fractions import Fraction
import json
import os
from pathlib import Path
import subprocess
import tempfile
import unittest

FFMPEG = os.environ.get("LFS_MEDIA_TEST_FFMPEG", "ffmpeg")
FFPROBE = os.environ.get("LFS_MEDIA_TEST_FFPROBE", "ffprobe")
RUNNER = None
STUDIO_RUNNER = None


def run(command):
    return subprocess.run(command, check=True, capture_output=True, timeout=90).stdout


def create_fixture(path, offset):
    command = [FFMPEG, "-v", "error", "-nostdin"]
    if offset:
        command += ["-f", "lavfi", "-i", "sine=frequency=440:duration=12"]
        command += ["-f", "lavfi", "-i", "testsrc2=size=128x96:rate=30:duration=10"]
        command += ["-map", "0:a", "-map", "1:v"]
    else:
        command += ["-f", "lavfi", "-i", "testsrc2=size=128x96:rate=30:duration=10",
                    "-map", "0:v"]
    command += ["-vf", "select=not(eq(mod(n\\,7)\\,1))", "-fps_mode", "vfr",
                "-c:v", "libx264", "-g", "60", "-keyint_min", "60",
                "-sc_threshold", "0", "-pix_fmt", "yuv420p",
                "-enc_time_base", "1:15360", "-video_track_timescale", "15360"]
    if offset:
        command += ["-c:a", "aac", "-output_ts_offset", "2.0"]
    command += [str(path)]
    run(command)


def oracle(path, fps=5):
    probe = json.loads(run([FFPROBE, "-v", "error", "-select_streams", "v:0",
                           "-show_streams", "-show_frames", "-show_entries",
                           "stream=time_base,start_pts,duration_ts:frame=best_effort_timestamp",
                           "-of", "json", str(path)]))
    stream = probe["streams"][0]
    time_base = Fraction(stream["time_base"])
    frames = [int(frame["best_effort_timestamp"]) for frame in probe["frames"]]
    origin = Fraction(int(stream.get("start_pts") or frames[0])) * time_base
    end = origin + Fraction(stream.get("duration_ts", 0)) * time_base
    count = int((end - origin) * fps)
    selected = []
    for index in range(count):
        target = origin + Fraction(index, fps)
        frame = next((pts for pts in frames if pts * time_base >= target), None)
        if frame is None:
            break
        selected.append(frame)
    return time_base, selected


class ExactFrameBoundaries(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.temp = tempfile.TemporaryDirectory(prefix="lfs-exact-frame-boundary-")
        cls.addClassCleanup(cls.temp.cleanup)
        cls.root = Path(cls.temp.name)
        cls.sources = {}
        for name, offset in (("vfr", False), ("offset", True)):
            source = cls.root / f"{name}.mp4"
            create_fixture(source, offset)
            cls.sources[name] = source

    def invoke(self, runner, source, output, hardware=False):
        request = {"input": str(source), "output": str(output), "mode": "fps",
                   "fps": 5, "metadata": False}
        if hardware:
            request.update(end_seconds=10, hardware=True)
        else:
            request.update(end=10, sink="memory", include_pixels=False, hardware=False)
        request_path = output.parent / "request.json"
        request_path.write_text(json.dumps(request), encoding="utf-8")
        result = subprocess.run([str(runner), str(request_path)], capture_output=True, timeout=120)
        self.assertEqual(result.returncode, 0, result.stderr.decode(errors="replace"))
        actual = json.loads(result.stdout)
        self.assertTrue(actual["success"], actual)
        if hardware:
            manifest = json.loads((output / "extraction_metadata.json").read_text(encoding="utf-8"))
            self.assertIn("nvdec", json.dumps(manifest).lower())
            return manifest["frames"]
        return actual["frames"]

    def assert_oracle(self, runner, hardware):
        for name, source in self.sources.items():
            with self.subTest(path=name):
                time_base, expected = oracle(source)
                self.assertEqual(len(expected), 50)
                stream = json.loads(run([FFPROBE, "-v", "error", "-select_streams", "v:0",
                                         "-show_streams", "-of", "json", str(source)]))["streams"][0]
                origin = Fraction(int(stream["start_pts"]) * time_base)
                self.assertEqual(Fraction(expected[37]) * time_base,
                                 origin + Fraction(37, 5))
                output = self.root / f"{name}-{hardware}"
                output.mkdir()
                frames = self.invoke(runner, source, output, hardware)
                observed = [int(frame["source_pts"] if hardware else frame["ticks"])
                            for frame in frames]
                self.assertEqual([Fraction(pts) * time_base for pts in observed],
                                 [Fraction(pts) * time_base for pts in expected])

    def test_software_selection_matches_rational_oracle(self):
        if not RUNNER:
            self.skipTest("software runner is not configured")
        self.assert_oracle(RUNNER, False)

    def test_nvdec_selection_matches_rational_oracle(self):
        if not STUDIO_RUNNER:
            self.skipTest("NVDEC runner is not configured")
        self.assert_oracle(STUDIO_RUNNER, True)


def main():
    global RUNNER, STUDIO_RUNNER
    parser = argparse.ArgumentParser()
    parser.add_argument("--runner")
    parser.add_argument("--studio-runner")
    args = parser.parse_args()
    RUNNER, STUDIO_RUNNER = args.runner, args.studio_runner
    if STUDIO_RUNNER:
        availability = subprocess.run([str(STUDIO_RUNNER), "--check-gpu"],
                                      capture_output=True, timeout=30)
        if availability.returncode == 77:
            raise SystemExit(77)
        if availability.returncode:
            raise RuntimeError("Studio runner could not query GPU availability")
    unittest.main(argv=[__file__])


if __name__ == "__main__":
    main()
