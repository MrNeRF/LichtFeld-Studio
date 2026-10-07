# CPU frame delivery and sinks

## Public contracts

`media/frame_sink.hpp` defines synchronous CPU delivery in RGB8. The public
contract uses owned C++ types and core `lfs::Result<void>`; no FFmpeg, CUDA,
renderer or JSON types appear in it. It belongs to the shared `lfs_media` target
used by Studio and root tests. Separate installed SDK packaging is deferred.

`FrameLayout` contains positive width/height, byte stride and pixel format.
`FrameView` borrows read-only pixels during `FrameSink::write`. `requiredBytes()`
returns `lfs::Result<std::size_t>` and validates dimensions, format, stride, overflow
and buffer size before copying. Invalid layouts return `InvalidArgument`.
The minimum span is `(height - 1) * stride + width * 3`; trailing padding after
the last row is unnecessary. The view does not extend its owner's lifetime.

`FrameSurface::copyOf(view)` returns `lfs::Result<FrameSurface>` and creates an immutable owning snapshot, copies only
visible row bytes and zeroes internal padding. Copying a surface shares its
immutable pixel ownership. Retained pixels stay valid after decoder reuse or
extraction teardown. A default/moved-from surface has an empty view, which does
not pass frame validation.

| FrameInfo property | Meaning |
|---|---|
| source_timestamp | Original signed FFmpeg best-effort timestamp, or presentation timestamp if unavailable, with source rational timebase; absent if both are missing |
| timestamp_origin | Missing, BestEffort or Presentation; relative fallback timing is not presented as original PTS |
| decode_index | Zero-based ordinal within this decode run; counts decoded frames including skipped/trimmed frames; after a seek this is not an absolute source-file frame index |
| delivery_index | Zero-based accepted delivery order, independent of nominal source naming |
| relative_seconds | Existing extractor's normalized source-relative time, including its fallback when source timing is absent |
| legacy_source_frame | Existing nominal-FPS-derived, one-based frame number for file naming/schema-2 compatibility; not a VFR decode ordinal |
| sharpness_score | Selected frame score when filtering is enabled; otherwise zero |

Window candidates and retained FPS-tail frames retain their own source timestamp
and decode ordinal. Flushing a candidate after a later frame is decoded does not
replace its identity with the current decoder frame. Repeated delivery of a
retained frame has a new delivery index and the same original source identity.

## Lifecycle and ownership

`begin(SinkSession)` receives source metadata and the final post-resize,
post-rotation RGB8 layout. Source inventory is described from the already-open
FFmpeg context. All callbacks run synchronously on the extraction caller's
thread, in order, with no concurrent callback. A slow callback naturally blocks
further decoding; no hidden worker, unbounded queue or background task is added.

The extraction calls begin after input, decoder and request validation. Failures
before begin produce no sink callbacks. An attempted begin is followed by ordered
writes and either successful complete or abort. A failing/throwing begin or
complete also receives abort. `abort` is noexcept and must release sink resources
without throwing. A successful complete is not followed by abort.

A successful write commits one accepted delivery and advances delivery_index.
`ErrorCode::Cancelled` requests cancellation; any other failed result ends the
extraction as Failed. Callback exceptions, including unknown exceptions, are
translated by the compatibility extractor boundary and call abort. The terminal
summary contains accepted count, outcome and error text. A sink must preserve
its own accepted output if partial results are needed; an attempted write which
returns failure is not counted. Sink result failures use the core IO error domain.

The extractor's existing `cancel_requested` callback remains active before and
between frames. Progress follows the existing selection/discard convention;
frame acceptance is additionally visible in the sink summary. Cancellation cannot
interrupt arbitrary blocking work inside a callback. Sink objects and getters
must not be read from another thread while extraction is mutating them.

## Built-in sinks

`MemoryFrameSink(payload_budget, frame_limit)` owns a vector of FrameSurface
snapshots. Defaults are 256 MiB of pixel payload and 100,000 surfaces. Both limits
are checked before pixel allocation/copy. Resource exhaustion returns the core
ResourceExhausted category, retaining previously accepted frames. `payloadBytes()`
counts allocated pixel bytes including internal row padding, excluding vector,
shared ownership and metadata overhead. The frame-count limit bounds that separate
overhead; the pixel budget is not a total process-RAM limit. A new successful begin
clears prior results. Abort preserves them and records Failed/Cancelled.

`FileFrameSink` uses the production PNG/JPEG writers: RGB8 PNG compression 6;
JPEG quality 1..100, full chroma above 90. It accepts an output directory, naming
pattern and format/quality options. Directory creation occurs at begin. Packed
RGB views require no additional pixel copy. Padded views are repacked once.
Filenames use legacy_source_frame; duplicate names return AlreadyExists rather
than silently overwriting. Other invalid configuration, lifecycle and codec
failures receive InvalidArgument, FailedPrecondition or Unavailable as appropriate.
Directory creation failures retain native filesystem status and UTF-8 path detail;
permission denial receives PermissionDenied. Invalid views passed to either
built-in write return InvalidArgument. FrameSurface validation itself throws.
The sink writes images only; it does not produce extractor schema-2 metadata.
Completed files survive abort. This implementation does not add atomic image
publication or clean up a partly written image after an underlying writer fails.

## Existing extractor integration

`VideoFrameExtractor::extract(params, error)` remains the compatibility API.
Its CPU RGB paths deliver to FileFrameSink. Existing selection, naming, pixels,
progress and schema-2 metadata remain unchanged on successful extraction.
Legacy duplicate source filenames are still skipped before delivery. CPU writer
failure now returns Failed immediately instead of warning and returning Completed;
completed earlier files remain available and failed writes do not advance progress.

`extractToSink(params, sink, error)` delivers through a supplied sink, with software
FFmpeg decode and CPU RGB output. CUDA JPEG encoding and hardware decoding are
not selected for this path. Output directory/naming/metadata flags do not create
files in the extractor; the supplied sink owns output policy. Selection, explicit
rotation, resizing, trim and sharpness are reused. Distinct selection deliveries
are not removed because a hypothetical legacy filename would collide: memory
receives repeated retained FPS-tail frames, while FileFrameSink rejects name
collisions. Choose file naming/selection appropriately for that sink.

The existing hardware decode/encoded CUDA JPEG file path remains in the legacy
adapter, including its batching and sparse-seek policy. Its pre-encoded writes do
not pass through the CPU FrameView API. HDR tone mapping remains the existing
renderer path; RGB8 delivery does not introduce HDR float output. Legacy metadata
write failures retain their existing warning behavior. These paths need their own
hardware/platform qualification and are not proved by CPU SDR tests.

## Tests and reproducibility

The root media CTest targets consume the production probe, extractor, sinks,
shared errors and image writers. A CPU reference preview translation unit is
compiled only in the test runner. The existing Release CI steps build
`media_contracts` and run the root media label; no workflow/job or dependency is added. See
[media test instructions](../../tests/media/README.md) for toolchain commands.

Eight additional Python methods exercise retained memory pixels against independent
FFmpeg decoding, CFR/VFR rational PTS against ffprobe, post-teardown ownership,
rotation/resize/trim/FPS/window selection against real PNG output, source identity,
no filesystem effects, repeated FPS tail, sink begin/write/complete failures,
callback exceptions, cancellation, memory/frame limits, invalid pre-begin input,
PNG/JPEG byte equivalence, and real second-image writer failure. Tests use the
existing deterministic offline corpus; no public media download is needed.

`MediaFrameSinkUnitContracts` checks padded snapshots, zeroed padding, shared
ownership, signed timestamps, invalid layouts/short buffers/overflow, lifecycle
reuse, budget error category and retained partial results. It also writes and
decodes a padded PNG and checks duplicate-name rejection. Assertions run in Release.

Benchmark qualification must compare the same compiler, dependency versions,
input and options, separating file output from retained memory. File/memory are
different workloads. Local results and exact revision bookkeeping belong in the
implementation plan/PR description, not in this durable contract document.

JPEG quality preserves the legacy writer policy: zero selects 90; other values
are clamped to 1–100. PNG ignores JPEG quality.
