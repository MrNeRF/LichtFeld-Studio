# Shared Media Ingest module and optional CLI

## Original image EXIF

`media-ingest source-exif INPUT` explicitly reads JPEG APP1 EXIF before the first
SOS marker, PNG eXIf chunks through IEND, or the Exif items associated with a HEIF
container's primary image.
It does not decode pixels, require an HEVC pixel decoder, change training
exposure/orientation/color, write files, or inspect auxiliary HEIF images.
`MediaIngest::sourceExif(path)` provides the owning C++ result;
`lichtfeld.media.MediaIngest.source_exif_json(path)` returns the shared JSON
representation, also used by the `media.source_exif` MCP query.

Collection and interpretation have separate statuses. `collected` means original
blocks were retained, not that their EXIF structure is valid. `parsed` describes
supported classic TIFF directories and values only; it does not mean every
vendor-specific structure has been interpreted. Unsupported, malformed and
limited interpretations retain the original blocks. An `io_error` collection
status is distinct from absent EXIF. A successful inspection response can thus
contain an unsuccessful collection or interpretation status; consumers must
inspect those fields.

JSON binary fields use hexadecimal strings (`binary_encoding: "hex"`), including
the complete JPEG `Exif\0\0` envelope or HEIF offset prefix. PNG uses the original
bare TIFF chunk payload and the explicit `png_exif` envelope; no synthetic prefix
is added. Ordered entries
retain original ID/type/count, IFD offset, entry ordinal and directory path
(root ordinal followed by parent entry ordinals). Duplicate IDs do not overwrite
one another. ASCII retains raw bytes, including NUL and invalid UTF-8. Unsigned
RATIONAL and signed SRATIONAL retain exact numerator/denominator, including zero
denominators. Nonfinite floating values use `nan`, `+inf`, or `-inf`; their
original bit patterns remain in the retained block. No orientation, GPS or date
normalization is applied.

MakerNotes remain opaque. A private parse copy masks their tag ID while FFmpeg
decodes their bytes, preventing vendor-IFD guessing; the adapter restores the
original ID by ordinal. Root IFDs are parsed separately to avoid FFmpeg synthetic
extra-IFD IDs overwriting real tags. Originals are never rewritten. A structural
guard checks ranges, overlapping directory/value storage, directory visits,
depth and aggregate value allocation before FFmpeg. BigTIFF, repeated directory
offsets, chained nested IFDs, zero-count values and unsupported directory-pointer
representations are retained without a typed interpretation.

Defaults: JPEG header window 1 MiB, original block 1 MiB, original bytes 4 MiB,
64 blocks; PNG additionally examines at most 64 MiB and 4096 non-IDAT chunks.
All chunk headers and CRC bytes, including those of IDAT, consume the byte
budget. Non-IDAT payloads are examined for CRC validation; IDAT pixel payloads
are discarded through the stream buffer up to 8 KiB; larger spans use seeking.
Neither path decodes pixels or validates their CRC. Scan limits are
explicit and preserve completed original blocks on later errors. PNG eXIf
duplicates or placement after IDAT are retained but reported as malformed under
the PNG third-edition ordering rules; collection does not validate pixel data.
A bad ancillary CRC marks collection malformed but scanning continues at the
next declared chunk boundary to retain later intact EXIF. Bad critical CRCs,
invalid framing and truncated chunks remain terminal; no byte-search recovery
is attempted. Repeated identical CRC diagnostics are reported once.
Typed interpretation across all blocks allows 4096 entries, 64 directories
and an 8 MiB decoded-value storage charge. Per-directory FFmpeg limits and depth
are also respected. Collection limits do not bound libheif's container parsing
allocations or promise a hard process/time ceiling. No complete-file metadata
archive, vendor MakerNote relocation, export metadata writing or sidecar support
is implied by this API.

## Production ownership

`src/media/CMakeLists.txt` defines the shared C++23 `lfs_media` target in the
application's existing build and dependency graph. It owns the probe, RGB8/float frame
sinks, ingest facade, extractor and HDR facade. `lfs_video` owns Studio's native
GPU adapters, preview and scene-video encoding. The dialog calls
`MediaIngest::extractFiles`; embedded Python exposes `lichtfeld.media` next to
`lichtfeld.io`. The legacy extractor remains a compatibility adapter to the same
implementation.

There is one shared implementation of each stateful core responsibility:

| Target | Ownership | Dependencies |
|---|---|---|
| `lfs_diagnostics` | Profiler state and CPU accounting | Standard library; optional native operations registered by Studio |
| `lfs_logger` | Existing logger and environment settings | spdlog, `lfs_diagnostics` |
| `lfs_error` | Errors, memory domains, failure reports, reporters, bus, envelopes, latches and guarded tasks | Logger, nlohmann-json; GNU stacktrace support where needed |
| `lfs_image_codecs` | Image decoding and encoding, including stream-based EXR | Shared errors; existing PNG/JPEG/ZLIB/TIFF/WebP/OpenEXRCore packages |
| `lfs_core` | Tensor/application core and native diagnostics adapter | Public shared error and codec targets; existing GPU dependencies |
| `lfs_media` | Probe, frame ownership/sinks, extraction and facade | Error, codecs, logger, FFmpeg |
| `media-ingest` | CLI arguments and JSON protocol | `lfs_media` |

Shared libraries use distinct export macros and the application's runtime/install
locations. Hidden-symbol builds must resolve the same symbols as Windows DLLs.
Portable installation includes the new libraries and optional CLI; Unix leaf
libraries resolve siblings with `$ORIGIN` or `@loader_path`, including the app's
`../lib` layout. Windows binaries receive the same product version resources.

`failure_report` owns one dedup table and invokes an optional writer hook.
`crash_handler` registers Studio's existing diagnostic writer. A headless consumer
uses the real logging/error/contract implementation without a crash-handler or
tensor dependency. `MemoryDomain` and `AllocationFailure` live in
`core/memory_domain.hpp`; allocation coordination remains in `memory_pressure.hpp`.

CUDA sampling, NVML and native event operations now live in `lfs_core`, behind an
immutable process-lifetime diagnostics backend. The CPU diagnostics DLL keeps
one profiler state and does not import CUDA. Registration rejects a different
replacement backend so live event handles remain valid. The CLI consequently
does not acquire a tensor/CUDA/Vulkan dependency through the logger.

The old nested `media/` project, duplicate manifests, SDK export/install logic,
private core copies and diagnostic shims have been removed. A separate distributable
SDK/package is deferred until the shared module's contracts and dependency
qualification are stable; there is no `LichtFeldMedia::media` package export in
this revision. Media contracts are registered by `tests/CMakeLists.txt` in the
root build and consume its production targets; no second production configuration
or standalone test project discovers dependencies.

## Capabilities and host backends

Studio registers its built HDR and CUDA-JPEG adapters before application media
work and embedded Python initialization. GUI workers also ensure registration.
The `lfs_media` leaf contains no tensor/CUDA/Vulkan implementation. Host adapters
preserve the existing NVDEC, nvImageCodec/nvJPEG, libplacebo and tensor/Metal paths.
A registered backend indicates build availability; device, codec and allocation
availability are checked when it is used. It is not a guarantee that every input
can use hardware acceleration.

`IngestRequest::allow_hardware_decode` controls optional hardware decode and JPEG
acceleration. File extraction can use the existing accelerated pipeline. Custom
frame sinks keep software RGB8 delivery and do not receive native hardware
surfaces. An unavailable optional JPEG adapter falls back to CPU; invalid batches,
writer failures and cancellation propagate failure with actual accepted-frame
counts. No failed or empty encoded batch is reported as successful extraction.

`MediaIngest::capabilities()` reports registered host HDR/CUDA support and compiled
VideoToolbox availability on Apple. The headless CLI has no Studio HDR adapter;
`--hdr-to-sdr` returns a structured Unsupported error before output is created.
Studio GUI and Python retain HDR conversion through their registered adapter.
`codecBuildInfo()` reports the actual linked FFmpeg version, license and build
configuration. Runtime fallback and supported input formats remain properties of
that FFmpeg/device build.

## Build and CLI

Use the root toolchain, manifest and existing configuration. Add
`-DLFS_BUILD_MEDIA_CLI=ON` to build the optional executable, then build the
`media-ingest` target. The option defaults to OFF. Building that target does not
compile the application/tensor targets, but root configuration still discovers
Studio dependencies; this revision does not claim a CPU-only root configuration.

```sh
cmake --build build --target media-ingest --config Release
media-ingest version
media-ingest capabilities
media-ingest probe "clip.nut" --timeout-ms 10000
media-ingest probe "clip.nut" --headers-only
media-ingest extract "clip.nut" --output "frames" --interval 2 --metadata
media-ingest extract "clip.nut" --output "frames" --fps 5 --start 1 --end 10 \
  --scale 0.5 --rotate 90 --format jpeg --quality 95 --name 'frame_%05d' --quiet
```

The CLI uses the application's generated Git version. Help includes
`--hdr-to-sdr`, selection, geometry, filename, metadata and sharpness options.
Unknown/conflicting options and invalid numeric values produce structured errors.
Output defaults to PNG; PNG/JPEG metadata is opt-in. EXR requires the bounded
linear SDR profile and always records metadata. See [linear SDR/EXR contracts](media-linear-exr.md)
for precision, input limits, independent image output and UI/API options.
Rotation remains explicit.

Except help, stdout contains one final JSON object (`schema_version: 1`, `success`).
Probe includes container/stream inventory, optional color/orientation metadata,
rationals `[numerator, denominator]` and signed-tick timestamps. Extraction reports
`frames_accepted` and `discarded`. Failures include code/domain/message, context
operations/fields and native status where present. Partial counts are retained in
error context. Progress JSON lines and diagnostic logs use stderr; `--quiet`
suppresses progress.

SIGINT and SIGTERM request cooperative cancellation. Checks occur at decode,
processing and writing boundaries; a signal does not forcibly interrupt every
blocked operation. Completed images remain available. POSIX signal tests exercise
SIGTERM after extraction starts; Windows needs separate console delivery
qualification because `terminate()` bypasses CRT signal handlers.

| Exit code | Meaning |
|---|---|
| 0 | Success/help |
| 2 | InvalidArgument |
| 3 | Unsupported, Unavailable, NotFound, PermissionDenied, DataLoss |
| 4 | Other structured failure |
| 130 | Cancelled |

## Python

```python
from pathlib import Path
import lichtfeld as lf
request = lf.media.IngestRequest()
request.input = Path("clip.nut")
request.selection.mode = lf.media.SelectionMode.Interval
request.selection.interval = 2
report, frames = lf.media.MediaIngest.extract(request, payload_budget=64 * 1024 * 1024)
files = lf.media.FileExtraction()
files.files.output_directory = Path("frames")
lf.media.MediaIngest.extract_files(request, files)
```

Probe, extraction and native processing release the GIL. Progress/cancellation
callbacks reacquire it and preserve Python exceptions. Native frame surfaces retain
ownership; `.pixels` creates a Python bytes copy only when requested. Memory
extraction defaults to a 256 MiB payload budget and 100,000-frame limit. Native
structured errors use the existing Studio exception translation. The synchronous
memory convenience call raises on failed extraction; C++ custom sinks can retain
partial frames and inspect the structured result.

## Extraction manifest publication

The extractor publishes `extraction_metadata.json` through the existing shared
atomic-file service for both SDR and float output. SDR retains its legacy
replacement policy; float output retains its explicit metadata overwrite option.
The existing schemas and frame naming/timestamps are unchanged. This publication
does not request durable filesystem synchronization.

A write, close or publication failure is an extraction failure. Cancellation
observed before publication is reported as cancellation. Either leaves the
previous manifest intact and removes the owned temporary file. Images already
committed remain available; a failure to publish the manifest does not roll them
back. Cancellation earlier in extraction still does not publish a partial
manifest. A pre-existing manifest therefore describes its previous run, not the
newly interrupted one: callers must use the terminal job/result state rather
than infer completion from manifest existence or progress alone.

This is per-file publication, not an atomic transaction of a whole sequence,
a checkpoint/resume protocol or a power-loss durability guarantee. Once a
manifest is published, later cancellation cannot undo that commit.

## Extraction manifest timing and provenance

Extraction manifests retain `schema_version: 2` and all legacy fields. Optional
additive fields describe the actual extraction without introducing a separate
metadata API or a second manifest. Consumers should tolerate additional keys.

- `input.video.stream_index` identifies the selected stream;
  `input.video.time_base` is its `[numerator, denominator]` in seconds per tick.
- `input.video.start_pts` is the declared stream start, or `null` when unknown.
  `relative_origin_pts` records the effective origin used by the existing
  relative-seconds calculation, which uses zero when the stream start is absent.
- Each frame's `source_pts` is the original integer timestamp, or `null` when
  absent. `timestamp_origin` is `best_effort`, `presentation`, or `missing`.
  Relative seconds are calculated by subtracting the stream origin in ticks
  before scaling. This preserves the first frame and FPS sampling boundaries
  when the stream starts at a nonzero timestamp, including native decoding.
  Missing source timing is not replaced with a synthetic integer PTS. The legacy
  `timestamp` remains the extraction's relative seconds; `source_frame` remains
  its legacy nominal-rate-derived number, not a decoded ordinal. The timebase is
  stored once at stream level rather than repeated in every frame.
- `provenance` records `ffmpeg_version`, `avcodec_version` and `avformat_version`
  once per manifest. Numeric library versions use FFmpeg's packed version integer.
- `processing.decoder.hardware_requested` records the request independently of
  the existing `backend` field, which describes the observed execution path.

Existing color, EXR precision/overrides, encoder backend and performance fields
remain authoritative. No projection, vendor metadata or unknown calibration is
inferred. These additions use the shared extractor for legacy, CLI, Python and
host file extraction; custom in-memory sinks retain their existing `FrameInfo`.
Manifest publication and overwrite policies are unchanged.
When metadata is requested, publication failure is terminal rather than a
best-effort warning; already published images are not rolled back. PNG/JPEG
metadata is opt-in, while EXR retains its mandatory manifest and explicit
overwrite policy. Verified replay and missing-image reconstruction retain the
same selected source timestamps as uninterrupted extraction.

## Verified image-sequence resume

`media-ingest extract ... --resume` and `FileExtraction.resume = true` opt into
checkpointed file extraction. Defaults and ordinary extraction remain unchanged.
PNG, JPEG and SDR linear EXR are supported through the existing file writers.
This option does not resume an interrupted encoded video or a custom memory sink.

The `.media-ingest-checkpoint` directory contains a versioned input/request
identity and one bounded immutable JSON record per selected image. Input identity
uses a streaming SHA-256 of the complete input; stream, selection, geometry,
sharpness, color, output encoding and pipeline/build identity must match. An
existing extraction without a checkpoint is not adopted. Changed requests,
malformed checkpoints and modified/truncated output files fail without replacing
those files. Missing recorded images can be reconstructed only when their
regenerated encoded bytes match the recorded hash.

Checkpoint hashing uses the project's existing OpenSSL SHA-256 provider through
a private media dependency. The algorithm, input bytes and checkpoint digest
format are unchanged; input, selected pixel rows and encoded files are still
verified in full.

The extractor replays decoding and selection from the original start to preserve
VFR timestamps, naming, deduplication and sharpness windows. It verifies retained
file bytes and regenerated selected pixels. Verified images are not re-encoded
or rewritten. JPEG resume reuses the registered Studio GPU encoder when hardware
is requested and it can allocate one frame; otherwise a new checkpoint selects
the existing CPU writer. The standalone CLI has no registered Studio GPU encoder.
The selected encoder is recorded in the checkpoint identity. CPU checkpoints
stay CPU when a GPU later becomes available; GPU checkpoints fail explicitly if
their provider is unavailable. Encoding failures never silently switch providers.
GPU JPEG bytes pass through the same staging, record-before-image and no-replace
publication as CPU output. Only missing images are encoded, one frame at a time;
decoded pixels still pass through host memory for checkpoint verification.
The existing hardware decode option remains explicit. Input, pixel and retained-output verification still
cost CPU time and I/O. A verified replay can be slower than a fresh ordinary
extraction, particularly for JPEG; resume promises verified reuse, not a speedup.
The extractor's detailed upload/JPEG/file-write timing counters cover its native
writer, not checkpoint sink work; use total elapsed time for resume comparisons.
Accepted-frame and manifest counts include verified retained images as well as
newly published images; progress describes the replayed selection.

Sparse hardware FPS extraction can continue decoding between nearby targets
instead of seeking repeatedly into the same GOP. The demuxer index determines
whether the target keyframe is beyond the retained decoder position; only then
does another seek help. Without a usable index, observed seek preroll is the
fallback and is refreshed when forward decoding encounters a keyframe. This applies to ordinary extraction and replay,
preserves the first-frame-at-or-after-target selection, and does not enable
sparse seeking for sharpness, interval, software or Dolby Vision paths.

For a new image, the existing writer encodes into the checkpoint directory,
then an immutable record of the expected encoded hash is atomically published
before the image is published with no-replace. The shared core publication
primitive renames the staged image on the same filesystem, avoiding a second
image write. Its older-filesystem hard-link/reservation fallbacks remain shared.
Empty reservation files left by process death on exFAT are treated as unpublished:
an initial identity is republished only before any frame record/image exists;
an empty last record is regenerated; an empty recorded image is reconstructed
only if its regenerated encoded digest matches. Nonempty corrupt files remain
errors. Completion checks run before the previous manifest can be replaced.
Staging cleanup cannot turn a committed image into a failed/unaccounted commit.
Abandoned staged copies can remain even after a successful replay: retained
final images return without re-encoding or removing an older staged copy.

Checkpoint frequency is one record per image. Record reads are capped at 16 KiB;
hashing uses fixed-size buffers. File count grows linearly with selected
images; decoding, input/pixel hashing, retained-file hashing and staging I/O have a
cost. The final manifest retains its existing schema and overwrite policy,
including EXR's explicit overwrite option. Replaying an already completed EXR
sequence therefore requires `files.exr.overwrite = true` to replace its manifest
(`--overwrite` in the CLI); retained images still are not overwritten.
Application build/version participates in checkpoint identity, covering changes
to the image encoder implementations as well as FFmpeg. Checkpoint format 1
from the initial implementation is incompatible with format 2; no silent adoption.
Each retained output is read/hashed once during replay, after pixel verification.
Checkpoints remain available after
completion. Resume cannot be combined with stale-output removal. Use one active
extraction per output directory; concurrent extraction is not a supported mode.
This assumes trusted local input/output directories: protection against another
writer swapping files between verification and publication/open is not promised.
No durable synchronization, global transaction or power-loss guarantee is added.

## Verification

Configure the application with its existing preset/toolchain and enable
`BUILD_TESTS`, `BUILD_FORMAT_TESTS` or `BUILD_VISUALIZER_TESTS`. The root
`media_contracts` target builds the production consumers and media test runners.
Existing Release jobs build and run these targets; no CI job/workflow is added.
The CPU reference preview runner disables hardware decode for its own player
translation unit. Studio's `lfs_video` and native GPU tests keep their configured
backends. Shared leaf modules are the same targets used by the application.

```sh
cmake --build build --target media_contracts --config Release --parallel 2
ctest --test-dir build -C Release -L media --output-on-failure --no-tests=error
```

Contracts compare independent RGB/PTS references, legacy output bytes/names/
metadata, probe/preview behavior, sink ownership/budgets/lifecycle, actual writer
failures, CLI version/help, DLL-shared error state and diagnostic registration.
Injected JPEG adapters verify incomplete batches, mid-batch failure, retained
counts, disabled hardware, CPU fallback and memory sink isolation. These injected
contracts do not validate GPU processing.

The actual Python binding group is tested for nested options, native ownership,
file output, GIL/callback behavior, cancellation and exception preservation.
`MediaStudioBackendContracts`, registered in existing root test configurations,
uses a real CUDA device for NVDEC/JPEG, rotation/sharpness, host HDR and native GPU
diagnostics; a missing device gives the explicit CTest skip code 77. Hardware
qualification is distinct from the offline CPU results. Tests use generated media
and do not download a public corpus.

## Dependencies

The root package baseline/overlays and existing FFmpeg/image codec stack are reused.
OpenImageIO is not required or reintroduced. FFmpeg is discovered once in the root
configuration; product targets link FFmpeg only inside `lfs_media`, with PRIVATE
includes and links. Package `optimized`/`debug` qualifiers and transitive
dependencies are preserved. Playback, codec/muxer ownership and decoded-frame
metadata interpretation belong to media. HDR/libplacebo/tensor adapters consume
borrowed media plane/component/colour descriptors, including mastering display,
HDR10+, Dolby Vision, ICC and film-grain data; they do not interpret FFmpeg objects.

The decoded descriptor's hardware handle identifies borrowed storage only.
Hardware views expose no CPU plane pointers; VideoToolbox's pixel-buffer identity
comes from its dedicated slot. Current HDR adapters require a downloaded software
frame. Direct decoded GPU import needs a separate device, ownership and
synchronization contract; the presence of a handle does not enable that path.

The provider exports the media C++ API only. ELF/Mach-O visibility lists hide
third-party static symbols and prevent assembly constants from becoming
interposable. Windows uses the package's private import libraries without DLL
forwarders. No public `av_*`, codec or resampler functions escape media.

Root CLI contracts inspect provider exports, product consumer imports/definitions
and final link commands, including stripped Release binaries. Tests that construct
or independently inspect FFmpeg fixtures have explicit PRIVATE links and includes;
they are the only allowed exceptions to the product ownership rule.

RGB-to-YUV420p conversion belongs to the tensor library. A single Slang program
serves CUDA, Vulkan and Metal; its scalar quantization/BT.601 formulas are shared
with the CPU implementation. Producers can allocate result planes or reuse
validated storage. Byte rounding, nonfinite values, strided inputs and tensor
execution timelines retain explicit contracts.

Encoder writers borrow planes only until the callback returns. Reentrant open,
write, close and ownership transfer are rejected before session state changes;
normal serialized callers incur no extra synchronization or pixel copies.

License obligations follow the repository GPL-3.0-or-later distribution and exact
installed package notices. FFmpeg's effective license is build dependent; the
runtime reports it alongside configuration flags. There is no separate manifest
or copied SDK license inventory in this revision. A replacement codec build or
future dependency must undergo its own version/transitive/license review.

## CUDA extraction color conversion

Studio's JPEG extraction pipeline can consume decoded CUDA NV12 and P010
frames directly when resizing and HDR tone mapping are not requested. P010
samples retain their ten meaningful bits through color conversion before the
final RGB8/JPEG quantization. Supported matrices are BT.601, BT.709 and
BT.2020 nonconstant luminance, with limited or full range. Other matrix tags
use the same BT.601 default as the host RGB8 converter; NV12 retains the
original integer kernel for that default. BT.2020 constant-luminance and
unsupported layouts retain the host path. Chroma-location tags do not gate
GPU conversion: sampling follows the existing host swscale policy, which
does not configure chroma position from those tags either.
The existing NV12 BT.601 conversion keeps its original integer rounding.

This path uses the existing decoder context, stream handoff and JPEG encoder;
it does not install another codec. Rotation remains on the GPU. Sharpness
may require an RGB readback; window selection, resizing, resumable extraction
and requested HDR processing retain their existing routing contracts.

Extraction metadata records `processing.color_conversion.backend` and counts
for GPU/host conversion, actual NV12/P010 frames, hardware-frame readbacks
and RGB readbacks. These describe work performed, rather than claiming a
fully GPU pipeline merely because hardware decoding was requested. The input
probe's pixel format can differ from the decoder's actual CUDA surface layout.
Here `gpu_frames`, `nv12_frames` and `p010_frames` count only direct CUDA JPEG
color conversions; the latter two are GPU-converted layout counts, not input
inventory. A P010 source converted through the host route reports zero
`p010_frames`. `host_frames` and backend `host` mean host-dispatched conversion,
which can still invoke libplacebo/Vulkan HDR tone mapping or a tensor GPU linear
renderer. They do not claim that all of that work executes on the CPU.

## MCP integration

`media.capabilities` exposes the registered backends and codec build. `media.probe` is a synchronous read-only tool using the same inventory JSON as the CLI. `media.extract` accepts the facade's selection, geometry, sharpness, time range and backend options and starts the `media.extract` runtime job. It returns promptly; decoding and file writing run in a guarded worker. Only one extraction can run at a time.

Use `runtime.job.describe`, `runtime.job.wait` and `runtime.job.control` (`cancel`) for state and control, or `lichtfeld://runtime/jobs/media.extract`. Media job queries/control bypass the GUI work queue. `runtime.events.tail` and subscriptions expose `media.extract.started`, `.progress`, `.completed`, `.failed` and `.cancelled`, with generation, progress and accepted counts. Progress state is updated on every callback; event delivery is throttled to 10 Hz plus the final update to keep the journal bounded. Native typed error codes are retained in the wire envelope. Application shutdown stops MCP requests, cancels and joins the worker before releasing the viewer/GPU runtime.

The `MediaMcpContracts` test uses the production tool registry and event routing, verifies file pixels against FFmpeg, overlapping-job rejection, cancellation, retained failures and nested schema validation. It is registered with existing root tests without adding a CI workflow.
On macOS, its native runner loads the complete static CPython embedding archive,
following Studio's existing test-host convention. Shared Studio libraries resolve
Python symbols from their host; an ordinary archive link can discard symbols
needed at runtime before the runner enters `main`.


### TIFF and BigTIFF source inspection

The same explicit `source-exif` / `source_exif_json` / `media.source_exif`
operation also recognizes classic TIFF and BigTIFF by signature. Its TIFF result
has scope `tiff-libtiff-observed-metadata` and a separate `tiff` object; `blocks`
is empty because TIFF IFDs are not standalone EXIF envelopes. Image directories,
SubIFDs, EXIF, GPS and interoperability directories are visited with libtiff,
without requesting pixel strips/tiles, applying orientation or changing training
exposure. A collected result means this metadata inspection succeeded, not that
an EXIF IFD necessarily exists or that the image pixels are valid.

`source_ranges` contains exact, owned bytes at their original 64-bit file offsets
as requested by libtiff. Ranges may overlap and remain relative to the source;
they cannot be transplanted as EXIF or used to reconstruct a complete TIFF.
`complete_metadata_archive` and `relocatable_exif` are explicitly false. Ignored
fields and unvisited vendor structures are not guaranteed to have their referenced
payloads present. No atomic snapshot of a concurrently changing file is promised;
repeated reads of already retained spans are checked for disagreement.

`directories` separately exposes library-decoded tag values, parent offsets and
IFD kind. Tag `type` is libtiff's field type; `count` and
`decoded_storage_bytes` describe its decoded representation, not necessarily the
original on-disk type/count. Defaults may be supplied by the library, strings may
be terminated, duplicate tags may be collapsed, and rationals are already
converted to float/double. Exact source bytes are never replaced by these values.
Byte/string values use hex; non-finite floats use `nan`, `-inf` or `+inf` strings.
MakerNotes remain opaque, not recursively interpreted as vendor IFDs.

Defaults bound libtiff read callbacks to 4096 calls / 64 MiB cumulatively
(the initial signature probe reads at most 12 additional bytes), directory visits
to 64, copied decoded entries to 4096 and widened value storage to 8 MiB.
Retained read spans also obey the shared 1 MiB per-span / 4 MiB aggregate limits;
repeated fully covered spans are not copied again. JPEG/PNG scan budgets and
standalone EXIF block counts do not govern TIFF inspection. Before callbacks read
or retain data, offsets/lengths are checked against the opened source extent.
Libtiff uses separate 8 MiB single-allocation / 32 MiB outstanding-allocation
ceilings, with mapping and strip chopping disabled and strip/tile arrays deferred.
These bounds are not a hard limit on total process memory or execution time.
Library errors and non-auto-registration warnings are surfaced as malformed,
including library allocation failures; explicitly unconfigured compression is
unsupported, and application budget exhaustion is limited.
Complete observed spans and decoded prefixes survive later failures.

This does not add metadata writing, complete archival preservation, vendor
normalization, GUI controls or a new TIFF decoder. Custom-directory next-IFD
chains and vendor-specific directory schemas are not universally traversed;
libtiff's public directory/getter contracts determine the decoded coverage.
