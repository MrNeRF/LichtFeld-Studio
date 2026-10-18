# JPEG XL image import

JPEG XL reading uses libjxl 0.12.0 through the existing shared image-codec
library. The standard dependency exposes a common read/write library; this
integration adds no application JXL encoder or export format. CPU SIMD is
selected by libjxl/Highway. No GPU JPEG XL decoding is promised.

## Pixel contracts

- Bare codestreams and JXL containers are recognized. File reads stream through
  a bounded sliding window; memory inputs are borrowed synchronously.
- Integer samples up to 8 bits use UInt8 storage; wider samples through 16 bits
  use UInt16 spanning the full range. Native libjxl normalization can differ
  from a mathematical integer rescale by one UInt16 unit for 10/12-bit samples;
  tests require exact recovery of their original source codes.
- Floating-point samples are returned as Float32, including negative and
  above-one values. Nonfinite results and unsupported precision are rejected.
- Orientation is applied by libjxl once; probe dimensions describe the oriented
  pixels. Alpha is returned straight, including associated-alpha input.
- Raw decoding preserves the original color encoding. Display image paths
  request SDR sRGB conversion through libjxl's CMS. PQ/HLG and declared HDR
  display conversion require an explicit future tone-mapping policy; raw reads
  remain available. This does not change existing training precision policy.
- Animation, additional image frames and auxiliary channels require an
  explicit future import policy and are rejected instead of silently dropped.

## Limits and ownership

Defaults are per call: 64 Mi pixels, 1 GiB owning decoded output, 1 GiB tracked
libjxl allocation payload, 1 MiB ICC profile, 4 MiB buffered file input and
2 GiB encoded input. These are separate limits, not a whole-process RAM budget;
CMS/library overhead and caller storage also exist. The decoder currently uses
its serial runner to avoid nested thread pools in loader workers.

Owning output is replaced only after the complete image and container have
been validated. Decode targets stage validation before allocating or writing
caller storage. Conversion to a requested integer target clamps and rounds;
request Float32 for unbounded samples. Decoding is synchronous and has no new
cooperative cancellation API.

## Integration and verification

`.jxl` participates in existing dataset discovery and image lookup without
changing the preference for legacy images over same-stem companions. Loader
and prepared-cache policies are reused. The photo picker recognizes `.jxl`.
No Python signatures change, so no stub edits are required.

The headless codec tests include original fixed fixtures with generator source,
integer/float precision, orientation, profiles, alpha, resource limits,
corruption, ownership and concurrent calls. Build using the project's vcpkg
dependencies, then configure `tests/image_codecs` as for the other codecs.
The full project additionally exercises discovery and pipelined loading.

The official Microsoft vcpkg Git registry is pinned at
`b3ae22aef2b857af6e80d756c13f15db12be4e8a` for `libjxl` alone (0.12.0).
Other packages retain the existing builtin baseline and overrides. This avoids
copying an overlay port or requiring the local vcpkg checkout's builtin version
database to contain 0.12.0. A first configure may fetch registry metadata.

libjxl uses BSD-3-Clause with its separate additional IP rights grant. Both
texts are included under `resources/licenses`; dependency notices come from
the existing install/package generator. Source licensing does not constitute
a blanket patent guarantee.
