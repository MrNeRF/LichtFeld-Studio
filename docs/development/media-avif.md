# AVIF image output

AVIF reading continues to use the existing libheif/dav1d path. Writing enables
only libheif's built-in AOM AV1 encoder through the `avif-encode` feature in the
existing libheif overlay. It does not enable HEVC, other encoders, AOM decoding
or dynamic plugin loading. No second AVIF container library is introduced.

The existing scoped Microsoft registry pins libaom 3.15.1 (source commit
`44d0a57786f432d933ff64b653347c66f4d0fa1d`) alongside libjxl; the general
baseline and other dependency versions stay unchanged. libheif remains 1.21.2. The existing overlay also releases the previous AOM
codec before reinitializing it for an auxiliary alpha image; otherwise the
provider overwrites live allocations. RGBA and repeated/cancelled writes run
under Linux ASan/UBSan/LeakSanitizer to cover this path.
The newer AOM port avoids requiring x86 NASM on Apple Silicon. The version
reported at runtime includes the actual provider, and forms part of resume
identity. Encoding uses CPU SIMD and explicitly bounded per-call threads; no
GPU AVIF encoder or aggregate process memory limit is promised.

## Pixel and color contracts

The shared writer accepts interleaved gray, gray-alpha, RGB and RGBA, including
padded/unaligned rows. Eight-bit input contains UInt8 samples. Ten/twelve-bit
input contains native-endian UInt16 **code values**, in [0, 1023] or [0, 4095],
not values normalized across the full UInt16 range. The existing application
reader still returns gray as RGB, gray-alpha as RGBA and UInt16 normalized to
[0, 65535]. This difference is explicit and tested against direct dav1d decode.

Input is straight alpha, with sRGB transfer and declared BT.709 or BT.2020
primaries. Full-range NCLX is written; RGB uses an identity matrix and 4:4:4,
so lossless RGB is not subjected to a lossy chroma conversion. Alpha is encoded
losslessly even when color uses lossy quality. All provider configuration calls
are checked. The encoder is explicitly AOM rather than whichever provider has
the highest plugin priority.

Defaults are lossless, quality 90 (ignored for lossless), speed 8, 8-bit output
and one thread. Higher speed is faster; supported speed is 0..9, lossy quality
0..100 and depth 8/10/12. Core limits bound pixels and serialized output bytes,
not libaom/libheif working allocations or their internal encoded representation.
Serialization uses the existing atomic file service and does not build a second
whole-file vector. Memory output and previous destination files are committed
only after success. Cancellation cannot preempt an individual provider call.

`ImageOutput::writeAvif` accepts relative sRGB BT.709 RGB8 at depth 8, or relative
linear BT.709/BT.2020 Float32 RGB/RGBA. For float, the existing TIFF SDR conversion
is shared: validate finite values and alpha, unassociate if declared, clamp RGB
to [0, 1], apply sRGB transfer, then round half up to the chosen integer codes.
Alpha is rounded without a transfer curve. Lossless preserves **those codes**,
not the original floats. Nits/scene-relative masters require an explicit prior
conversion; existing EXR workflows remain available. No implicit tone map.

## Integration

FileFrameSink writes `.avif` sequences through the same public writer. Resume
identity includes provider/version, quantization policy, depth, speed and
effective compression settings. Ignored lossless quality does not invalidate a
checkpoint. Existing files retain bytes and mtimes; missing files must reproduce
the stored checksum. Generated-name matching handles the five-character suffix.

CLI: `--format avif`, `--avif-quality` (enables lossy), `--avif-speed` and
`--avif-depth 8|10|12`. Video extraction takes the declared linear Float32 SDR
path before the explicit integer conversion. Python exposes AvifOutputOptions,
ImageOutput.write_avif and FileFrameSinkOptions.avif. MCP exposes avif_quality,
avif_speed and avif_depth. Capability reporting reflects the AOM provider.
The media stub is generated with CMake/nanobind, never edited manually.

Generic display-byte writing and Tensor save_image route `.avif` through the
shared lossless sRGB UInt8 writer. JPEG quality does not select AVIF lossiness.
Nonempty generic metadata comments are rejected, not discarded. No source
EXIF/XMP, auxiliary/depth map, arbitrary ICC or animation preservation is added.
Existing prepared-cache formats and policies are retained. Render/timeline
sequence UI and AV1 video encoding are separate work.

## Licenses and distribution

libheif retains its LGPL terms. libaom has a BSD main license, an additional
patent grant and bundled third-party notices. Source license compatibility is
separate from patent obligations. The selected vcpkg port includes LICENSE, PATENTS and the fastfeat, vector and
x86inc notices in its installed copyright text. The existing install-time
license generator consumes that text. No supplemental copies, generator or
AOM overlay are added. The x86inc copyright refers to the x264 project but grants
ISC-style permissions; it does not introduce an x264 encoder.

Provider sources: https://github.com/strukturag/libheif/tree/v1.21.2 and
https://aomedia.googlesource.com/aom/+/44d0a57786f432d933ff64b653347c66f4d0fa1d/ .

## Validation boundaries

Export tests cover all four channel layouts and integer precisions, independent
sample/alpha decode, output budgets, concurrent calls, publication failures,
cancellation and recovery. Public API tests additionally cover BT.2020 NCLX,
premultiplied/straight equivalence, zero alpha, invalid declarations and
non-finite values. AVIF-specific resume tests reject truncated or modified files,
wrong checksums and malformed records while preserving existing files; restoring
the original bytes/record allows the extraction to finish.

Focused production bindings and a C++ consumer using only public media headers
are tested on Windows, Linux and macOS. Linux ASan/UBSan/LeakSanitizer exercises
the encoder; provider release libraries are not fully instrumented. Photographic
4K/8K workload measurements are separate from unit tests and from the uniform
color smoke test. Encoding large images and concurrent calls can require
substantial RAM even with one provider thread per call. No aggregate concurrency
or provider working-memory bound is implied by pixel/output-byte limits.

Allocation failures use ResourceExhausted, including provider allocation errors
and callback-copy failures during publication setup. Standard exception details
and native stdio errors are retained. No callback exception crosses the C writer
ABI; it is captured and handled after returning from the provider.
