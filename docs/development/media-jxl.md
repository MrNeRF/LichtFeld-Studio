# JPEG XL image import

JPEG XL reading uses libjxl 0.12.0 through the existing shared image-codec
library. The standard dependency exposes a common read/write library; this
integration adds no application JXL encoder or export format. CPU SIMD is
selected by libjxl/Highway. No GPU JPEG XL decoding is promised.

## Pixel contracts

- Bare codestreams and JXL containers are recognized. File reads stream through
  a bounded sliding window; memory inputs are borrowed synchronously.
- Integer samples up to 8 bits use UInt8 storage; wider samples through 16 bits
  use UInt16 spanning the full range. With explicit Native policy, libjxl normalization can differ
  from a mathematical integer rescale by one UInt16 unit for 10/12-bit samples;
  tests require exact recovery of their original source codes.
- Floating-point samples are returned as Float32, including negative and
  above-one values. Nonfinite results and unsupported precision are rejected.
- Orientation is applied by libjxl once; probe dimensions describe the oriented
  pixels. Alpha is returned straight, including associated-alpha input.
- Default decoding requests SDR sRGB through libjxl's CMS at every sample
  precision, including Float32. Training and display therefore use the same
  output curve regardless of structured versus ICC-only original profiles.
  Source/HDR sample access requires explicit
  `JxlDecodeLimits::color_policy = JxlColorPolicy::Native`; this uses libjxl's
  DATA encoding, which can differ from the original ICC for XYB input.
  `display=true` always requests SDR sRGB, overriding Native policy.
- SDR conversion rejects structured PQ/HLG, declared HDR intensity, ICC cicp
  PQ/HLG and ICC-only RGB/gray TRCs matching PQ/HLG. No implicit tone mapping
  is performed. This gate is not a universal classifier of arbitrary opaque
  LUT profiles. Native access remains available for an explicit future HDR
  processing policy.
- Static internal layers are coalesced by libjxl into one displayed image.
  Animation, additional displayed frames and auxiliary channels require an
  explicit future import policy and are rejected instead of silently dropped.
- Float32 targets decode directly into private Float32 staging, including for
  integer-depth codestreams. This preserves out-of-range reconstructed samples
  and avoids prior integer clipping/quantization. Native integer image output
  still uses its declared bounded integer storage.

## Limits and ownership

Defaults are per call: 64 Mi pixels, 1 GiB owning decoded output, 1 GiB tracked
libjxl allocation payload, 1 MiB ICC profile, 4 MiB buffered file input and
2 GiB encoded input. These are separate limits, not a whole-process RAM budget;
CMS/library overhead, runner storage and caller storage also exist. The
resizable libjxl runner uses at most eight additional workers by default,
bounded by its image-size suggestion and a shared active worker budget based
on hardware concurrency. Concurrent loader calls share this additional-worker
budget; caller threads are not counted in it. `max_threads=1` selects serial
decoding. Decoder allocation accounting is atomic for parallel callbacks.

Owning output is replaced only after the complete image and container have
been validated. The owning Image path decodes directly into a private candidate
vector, then moves it to the result without a full-image copy. A failed decode
leaves the previous result intact. Decode targets stage validation before
allocating or writing caller storage. libjxl emits the requested type and channels into uninitialized
private staging; the validated result reaches caller storage with one copy,
without an intermediate scalar per-sample conversion. Integer targets are bounded;
request Float32 for unbounded samples. Decoding is synchronous and has no new
cooperative cancellation API.

## Integration and verification

`.jxl` participates in existing dataset discovery and image lookup without
changing the preference for legacy images over same-stem companions. Loader
and prepared-cache policies are reused. The photo picker recognizes `.jxl`.
COLMAP models must reference the actual `.jxl` filenames: a model that still
names `.JPG` files does not resolve a same-stem `.jxl` companion automatically,
as with EXR and WebP. Re-encoding images requires updating those model names.
Lossless JXL can cost more to decode than other encodings of the same pixels.
Prepared-cache reuse avoids repeated source decoding during training, but cold
loads and evaluation paths that read source images still pay that decode cost.
No Python signatures change, so no stub edits are required.
Display selection is passed through the shared generic file/memory dispatch
for JXL and HEIF. Existing generic overloads remain available; other codec
display behavior is unchanged. Missing/unopenable, empty and corrupt inputs
have distinct diagnostics. Existing dataset discovery expectations remain exact.

The headless codec tests include original fixed fixtures with generator source,
integer/float precision, orientation, profiles, alpha, resource limits,
corruption, ownership and concurrent calls. Build using the project's vcpkg
dependencies, then configure `tests/image_codecs` as for the other codecs.
The full project additionally exercises discovery and pipelined loading.

An additional audit uses the official `libjxl/conformance` corpus at revision
`b1d0f990b03e57bf6d137c365cd5dc8b470b9191`. All 33 non-animated cases without
auxiliary channels decode through explicit Native file and memory Float32 targets within the
published per-case reference tolerances, after accounting for decoded-data ICC
and straight versus associated alpha. An independent direct-libjxl decoder
also passes these references. The five animation and two auxiliary-channel
cases are deliberately rejected. Display acceptance was checked for all 33;
this is not a full metadata/JPEG-reconstruction conformance certification.

Three small official bitstreams and their license/provenance are retained as
offline regressions for ICC profiles, patches, coalesced static layers and
Float32 reconstruction beyond the integer range. Float32 output limits are
checked before caller allocation. The focused JXL suite has 26 tests, including
ICC-only PQ/HLG rejection, canonical sRGB consistency across precisions and
profile representations, serial/parallel equality, simultaneous parallel
readers, allocation-limit recovery, distinct input diagnostics and preservation
of owning output when trailing input invalidates an otherwise decoded image.

The official Microsoft vcpkg Git registry is pinned at
`b3ae22aef2b857af6e80d756c13f15db12be4e8a` for `libjxl` alone (0.12.0).
Other packages retain the existing builtin baseline and overrides. This avoids
copying an overlay port or requiring the local vcpkg checkout's builtin version
database to contain 0.12.0. A first configure may fetch registry metadata.
The parallel runner uses `libjxl_threads` from that same standard package;
no additional overlay is needed.

libjxl uses BSD-3-Clause with its separate additional IP rights grant. Both
texts are included under `resources/licenses`; dependency notices come from
the existing install/package generator. Source licensing does not constitute
a blanket patent guarantee.
