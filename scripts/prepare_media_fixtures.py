#!/usr/bin/env python3
# SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
# SPDX-License-Identifier: GPL-3.0-or-later
"""Prepare a small offline media corpus without CUDA or the LichtFeld encoder.

Container bytes can vary with FFmpeg. Decoded pixels and timestamps are the
reference contracts; file checksums identify this particular generated corpus.
No network requests are made. An existing destination is never overwritten.
"""

import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import tempfile


SPECS = (
    {"id": "cfr-asymmetric", "width": 64, "height": 32,
     "ticks": [0, 1, 2, 3, 4], "projection": "rectilinear"},
    {"id": "vfr-asymmetric", "width": 64, "height": 32,
     "ticks": [0, 1, 4, 9, 16], "projection": "rectilinear"},
    {"id": "panorama-layout", "width": 128, "height": 64,
     "ticks": [0, 1, 2, 3, 4], "projection": "equirectangular"},
)


def dji_ac204(directory):
    """Original packet payloads in an independent, synthetic single-track MP4.

    The 30,000/1,001 clock is a test input, not original device timing.
    No video codec, decoder or network is involved.
    """
    import struct
    corpus = Path(__file__).resolve().parents[1] / "tests/media/fixtures/dji-ac204/ac204_djmd_first300.bin"
    data = corpus.read_bytes()
    samples, offset = [], 0
    while offset < len(data):
        size, = struct.unpack_from(">I", data, offset)
        offset += 4
        if size > len(data) - offset:
            raise ValueError("truncated DJI packet fixture")
        samples.append(data[offset:offset + size])
        offset += size
    def u32(*values):
        return struct.pack(">" + "I" * len(values), *values)
    def box(kind, payload):
        return u32(8 + len(payload)) + kind + payload
    def full(kind, payload, flags=0):
        return box(kind, u32(flags) + payload)
    identity = u32(0x10000, 0, 0, 0, 0x10000, 0, 0, 0, 0x40000000)
    duration = 1001 * len(samples)
    ftyp = box(b"ftyp", b"isom" + u32(0) + b"isommp42")
    mdat = box(b"mdat", b"".join(samples))
    mvhd = full(b"mvhd", u32(0, 0, 30000, duration, 0x10000) + b"\x01\x00" + bytes(10) + identity + bytes(24) + u32(2))
    tkhd = full(b"tkhd", u32(0, 0, 1, 0, duration) + bytes(16) + identity + bytes(8), flags=3)
    mdhd = full(b"mdhd", u32(0, 0, 30000, duration) + bytes(4))
    hdlr = full(b"hdlr", u32(0) + b"meta" + bytes(12) + b"DJI fixture\0")
    dref = full(b"dref", u32(1) + full(b"url ", b"", flags=1))
    stsd = full(b"stsd", u32(1) + box(b"djmd", bytes(6) + b"\x00\x01"))
    stts = full(b"stts", u32(1, len(samples), 1001))
    stsc = full(b"stsc", u32(1, 1, len(samples), 1))
    stsz = full(b"stsz", u32(0, len(samples), *map(len, samples)))
    stco = full(b"stco", u32(1, len(ftyp) + 8))
    stbl = box(b"stbl", stsd + stts + stsc + stsz + stco)
    minf = box(b"minf", full(b"nmhd", b"") + box(b"dinf", dref) + stbl)
    trak = box(b"trak", tkhd + box(b"mdia", mdhd + hdlr + minf))
    output = directory / "dji-ac204 é 日本語.mp4"
    output.write_bytes(ftyp + mdat + box(b"moov", mvhd + trak))
    return output


def source_pixels(spec):
    """Asymmetric corner colors plus a unique center patch for every frame.

    The panorama is only a layout fixture, not a calibrated lens/stitching or
    photogrammetry reference. Projection is declared in the manifest.
    """
    width, height = spec["width"], spec["height"]
    frames = bytearray()
    corners = ((255, 0, 0), (0, 255, 0), (0, 0, 255), (255, 255, 0))
    for index in range(len(spec["ticks"])):
        for y in range(height):
            for x in range(width):
                if width // 3 <= x < 2 * width // 3 and height // 3 <= y < 2 * height // 3:
                    color = (16 + index * 32, 37, 83)
                else:
                    color = corners[(2 if y >= height // 2 else 0) + (x >= width // 2)]
                frames.extend(color)
    return bytes(frames)


def run(command, data=None):
    try:
        result = subprocess.run(command, input=data, stdout=subprocess.PIPE,
                                stderr=subprocess.PIPE, check=True, timeout=60)
    except subprocess.CalledProcessError as error:
        raise RuntimeError(error.stderr.decode("utf-8", errors="replace")) from error
    return result.stdout


def digest(data):
    return hashlib.sha256(data).hexdigest()


def inspect(path, ffprobe):
    return json.loads(run([ffprobe, "-v", "error", "-select_streams", "v:0",
                           "-show_streams", "-show_frames", "-of", "json", str(path)]))


def decoded_rgb(path, ffmpeg):
    return run([ffmpeg, "-v", "error", "-i", str(path), "-map", "0:v:0",
                "-fps_mode", "passthrough", "-f", "rawvideo", "-pix_fmt", "rgb24", "pipe:1"])


def verify_fixture(path, spec, ffmpeg, ffprobe):
    actual = inspect(path, ffprobe)
    streams = actual.get("streams", [])
    if len(streams) != 1:
        raise RuntimeError(f"{spec['id']}: expected one video stream")
    stream = streams[0]
    if (stream["width"], stream["height"]) != (spec["width"], spec["height"]):
        raise RuntimeError(f"{spec['id']}: unexpected dimensions")
    frames = actual.get("frames", [])
    expected_times = [tick / 10 for tick in spec["ticks"]]
    if len(frames) != len(expected_times):
        raise RuntimeError(f"{spec['id']}: unexpected frame count")
    times = [float(frame["best_effort_timestamp_time"]) for frame in frames]
    if any(abs(actual_time - expected_time) > 0.000001
           for actual_time, expected_time in zip(times, expected_times)):
        raise RuntimeError(f"{spec['id']}: unexpected timestamps {times}")
    pixels = decoded_rgb(path, ffmpeg)
    if pixels != source_pixels(spec):
        raise RuntimeError(f"{spec['id']}: lossless RGB round-trip failed")
    return {"frame_count": len(frames), "timestamps_seconds": times,
            "time_base": stream["time_base"], "decoded_rgb_sha256": digest(pixels)}


def prepare(destination, ffmpeg="ffmpeg", ffprobe="ffprobe"):
    destination = Path(destination).absolute()
    if destination.exists():
        raise FileExistsError(f"Refusing to overwrite existing destination: {destination}")
    versions = {"ffmpeg": run([ffmpeg, "-version"]).decode().splitlines()[0],
                "ffprobe": run([ffprobe, "-version"]).decode().splitlines()[0]}
    destination.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix=".media-fixtures-", dir=destination.parent) as staging:
        root = Path(staging)
        corpus = root / "corpus"
        corpus.mkdir()
        fixtures = []
        for spec in SPECS:
            path = corpus / (spec["id"] + ".nut")
            command = [ffmpeg, "-v", "error", "-nostdin", "-f", "rawvideo",
                       "-pixel_format", "rgb24", "-video_size", f"{spec['width']}x{spec['height']}",
                       "-framerate", "10", "-i", "pipe:0", "-map", "0:v:0"]
            if spec["id"].startswith("vfr"):
                command += ["-vf", "setpts=N*N", "-fps_mode", "vfr"]
            command += ["-c:v", "ffv1", "-level", "3", "-pix_fmt", "bgr0",
                        "-threads", "1", "-enc_time_base", "1:10", "-map_metadata", "-1",
                        "-fflags", "+bitexact", "-flags:v", "+bitexact", str(path)]
            run(command, source_pixels(spec))
            verified = verify_fixture(path, spec, ffmpeg, ffprobe)
            fixtures.append({**spec, **verified, "file": path.name,
                             "sha256": digest(path.read_bytes()), "size_bytes": path.stat().st_size,
                             "origin": "generated offline from prepare_media_fixtures.py",
                             "generator_version": 1, "pixel_format": "rgb24 reference"})
        manifest = {"schema_version": 1, "tools": versions, "fixtures": fixtures,
                    "scope": "fixture integrity only; not a LichtFeld extraction regression run"}
        (corpus / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
        # The destination is published only once the entire corpus has passed.
        if destination.exists():
            raise FileExistsError(f"Destination appeared during preparation: {destination}")
        corpus.rename(destination)
    return manifest


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--ffmpeg", default="ffmpeg")
    parser.add_argument("--ffprobe", default="ffprobe")
    args = parser.parse_args()
    try:
        manifest = prepare(args.output, args.ffmpeg, args.ffprobe)
    except (OSError, RuntimeError, ValueError, KeyError, subprocess.TimeoutExpired) as error:
        parser.exit(1, f"Media fixture preparation failed: {error}\n")
    print(json.dumps({"output": str(args.output), "fixtures": len(manifest["fixtures"])}))


if __name__ == "__main__":
    main()
