# JPEG XL image I/O

JPEG XL reading uses libjxl 0.12.0 through the existing shared image-codec
library. Reading and writing share the existing dependency and codec boundary.
CPU SIMD is
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
allocating or writing caller storage. libjxl emits the requested type and
channels into uninitialized private staging; the validated result reaches
caller storage with one copy, without an intermediate scalar per-sample
conversion. Integer targets are bounded; request Float32 for unbounded samples.
Decoding is synchronous and has no new cooperative cancellation API.

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
checked before caller allocation. Reader regressions include
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

## Image and sequence output

`encode_jxl_memory` and `write_jxl` use one shared libjxl encoder. Borrowed,
interleaved UInt8, native-endian UInt16 and Float32 gray/gray-alpha/RGB/RGBA
samples are accepted with an explicit byte stride. Padded rows are packed only
when needed; contiguous input is borrowed without another full-image copy.
Internal packing uses uninitialized owned storage, combining float validation,
unassociation and packing in one row pass instead of zero-filling and then
overwriting a full-image vector.
Memory output owns its bytes only on success, including aliased input/output.
The file writer drains 64 KiB chunks directly into the existing atomic file
service instead of constructing a complete encoded-file vector first.

Lossless is the default. Lossy mode explicitly selects perceptual distance
in `(0,25]`; effort `[1,9]` controls the speed/compression tradeoff (default 4).
Distance must remain valid but has no encoding effect in lossless mode. Alpha
is encoded losslessly even when RGB is lossy. Lossy RGB, including invisible
RGB, is not an exact-sample preservation contract.

Color is declared as sRGB, linear sRGB or linear BT.2020. No tone mapping or
implicit integer reduction is performed. Float32 inputs must be finite; alpha
must be in `[0,1]`. Explicit premultiplied float input is unassociated, with
RGB zeroed at alpha zero. Lossless Float32 uses left modular prediction:
libjxl 0.12's adaptive prediction rejects some valid signed IEEE32 patterns.
The qualified path retains negative/above-one values, subnormals and signed
zero without a lossy fallback. Float32 requires JPEG XL level 10 and may have
more limited interoperability than integer level-5 images. Application reads
still default to canonical sRGB; explicit Native reads are needed to compare
lossless source samples. Lossy DATA color can differ from ORIGINAL color.

Lossless Float32 requires effort 4..9 with libjxl 0.12, including positive-only
input. Efforts 1..3 are rejected before provider allocation or file publication;
they remain available for integer input and lossy Float32. Video-extraction
preflight rejects the unsupported combination before creating an output
folder. Requested effort is never changed silently. With alpha present, both
color and extra-channel resampling are fixed at 1 so alpha remains exact even
at lossy distance 10..25; high-distance files can therefore be larger than
libjxl's automatically resampled output.

Encoder limits are per call: 64 Mi pixels and 1 GiB tracked provider allocation
payload. `max_output_bytes = 0` uses a 256 MiB memory-output cap; streamed file
output instead uses `max(256 MiB, 2 * packed_raw_bytes + 1 MiB)`, with checked
arithmetic. A nonzero value is an explicit hard cap for either API. This lets
large lossless Float32 files exceed 256 MiB without raising the encoder RAM
budget or staging a whole encoded file in memory. They are configurable
through `JxlEncodeOptions`, and are separate from borrowed input, optional
packing, the memory-result vector and library/CMS/runner overhead. They do not
constitute a whole-process RAM bound or guarantee every allowed image can be
encoded within the working-memory budget. The existing shared JXL additional
worker budget also covers encoding. Cancellation is checked around row work,
provider submission/output and atomic publication; an individual synchronous
provider call is not preemptible. Failure or cancellation leaves previous
memory output and existing destination files intact.

`ImageOutput::writeJxl` accepts declared relative sRGB RGB8 or relative linear
BT.709/BT.2020 Float32 RGB/RGBA. `FileFrameSink` writes `.jxl` sequences through
that same writer. Video extraction deliberately uses the existing declared
linear Float32 SDR conversion path; legacy RGB8 extraction does not provide a
complete sRGB color contract. Integer writer access remains available through
the codec API and explicitly declared RGB8 through `ImageOutput`. PQ/HLG nits
or scene-HDR master export remains an explicit separate policy; EXR is retained
for those existing workflows.

CLI: `--format jxl` defaults to lossless; `--jxl-distance N` selects lossy and
`--jxl-effort N` selects effort. Existing float-source declarations/overrides
apply. Python exposes `JxlOutputOptions`, `ImageOutput.write_jxl` and
`FileFrameSinkOptions.jxl`; MCP uses `format: "jxl"`, optional `jxl_distance`
and `jxl_effort`. Shared preflight rejects incompatible options before output.
Extraction manifests report JXL mode, nominal Float32 precision and effort,
without EXR compression or JPEG quality labels. Resume identities include the
libjxl version, encoding policy and effective settings; verified files are
retained and missing files must reproduce their recorded encoded checksum.

This adds backend writers and sequence sinks, not a new timeline export GUI or
a single animated JXL file. Source EXIF/XMP/auxiliary-channel preservation stays
with the separately planned metadata/channel work; arbitrary source metadata
is not silently copied into newly rendered or converted images.

The existing generic display-byte writer and `save_image` also route `.jxl`
through the shared lossless sRGB writer. Their existing UInt8 display-image
policy remains in force; use `ImageOutput::writeJxl` for Float32 masters.
JPEG quality does not select JXL lossiness in this compatibility path.
Nonempty explicit generic metadata comments are rejected for JXL pending the
common metadata policy. The Python convenience binding omits its automatically
generated provenance stamp for JXL in both full and minimal-provenance modes;
this format-specific behavior is documented in its generated stub. Other
formats retain their existing stamp policy.
