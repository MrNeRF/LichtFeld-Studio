# Source metadata catalog

The metadata catalog is an explicit, metadata-only operation. It does not run
when decoding images, doing a default video probe, extracting frames or training a dataset.
It does not change exposure, orientation, camera pose or image pixels.

```cpp
auto metadata = lfs::media::MediaIngest::sourceMetadata(path);
// Or reuse an existing description without another file read or parse:
auto catalog = lfs::media::catalogSourceMetadata(std::move(description));
```

The CLI exposes `media-ingest source-metadata INPUT`. Its version 1 JSON result
contains a `static_image_exif` scope, an index status, the existing EXIF JSON
representation and occurrence references. `source-exif` keeps its existing
contract. Successful transport does not imply complete or valid metadata:
consumers must inspect collection, interpretation and index statuses separately.

## Ownership and identity

The catalog owns an immutable shared snapshot containing the existing `ImageExifDescription`. Original envelopes,
observed TIFF source ranges and decoded values are retained in that owner.
The occurrence index stores only ordinal references, not another copy of values.
Duplicate IDs and unknown tags remain distinct and ordered. References select
an EXIF block/entry or TIFF directory/tag. They are meaningful only within the
catalog they came from; they are not globally stable source identifiers.

Checked C++ lookup returns null for invalid indices. The public API exposes only
const payload/index accessors. Copies share the same immutable snapshot without
copying large metadata buffers; moves transfer snapshot ownership. Returned
pointers remain valid while an owning catalog or bound handle is alive. No public
payload mutation or index reordering API can silently retarget an existing handle.
As with ownership transfer generally, do not retain mutable aliases into vectors
moved into a snapshot. The catalog is a description, not a relocatable metadata archive.
TIFF offsets and opaque MakerNotes remain tied to their original source.

No rational division, UTF-8 coercion, unit inference, clock conversion or tag
normalization is performed by the catalog. The existing serializer preserves
binary values as hex and uses its existing diagnostic-text policy.

## Limits and partial observations

The default limit is 4096 indexed occurrences. It bounds additional index length,
not provider RAM, collected payloads, serialization memory or allocator capacity.
Existing collector/parser budgets remain independent. At the first omitted
occurrence, indexing stops and reports `Limited`; all already collected originals
and decoded values remain in the owner. An exact fit, or an empty collection with
a zero index budget, reports `Complete` for the index only. It does not upgrade a
Limited, Malformed, Unsupported, Absent or NotExamined source collection.

A caller-supplied description with mismatched block/interpretation counts is
rejected before indexing. Allocation failures during collection or indexing are
reported through the existing resource-exhaustion error contract. Argument-copy
allocations before entry to the by-value adapter are the caller's responsibility;
move an existing description to avoid them. There is no asynchronous work or
new provider operation to cancel in the indexing adapter.

## Supported sources

The catalog indexes the existing JPEG/PNG/HEIF EXIF block descriptions
and observed TIFF tags. It adds no new image carrier reader, vendor decoder, Python
binding or metadata writer. Container/stream observations use the existing video probe as described below. HEIF remains scoped to the existing primary-image
collector. Unsupported carriers retain the existing collector status.

The common observation and timing contracts below build on retained source data.
They do not decode vendor telemetry or claim to enumerate every container box.

## Container and stream catalog

`MediaIngest::sourceContainerMetadata(path, probe_options, index_limits)` uses the
existing `MediaProbe::inspect` operation. `catalogContainerMetadata` can consume
an already collected probe description without opening it again. The CLI command
is `media-ingest source-container-metadata INPUT`. Its scope is `container_stream`.

`ProbeOptions::collect_source_metadata` defaults to false. Explicit collection
retains the container dictionary, each stream dictionary and codec coded side-data
blocks exposed by FFmpeg. It observes video, audio, subtitle, data and attachment
stream metadata without decoding proprietary telemetry or scanning data packets.
Caller-built container descriptions must retain a metadata node per described
stream unless collection explicitly reports `limited`. Retained nodes must match
the corresponding native stream identities; incomplete inventories cannot claim
a complete observation.

Existing stream descriptions retain native stream indices, rational timebases,
start/duration, orientation and color information. These remain static demuxer
observations and may include values derived during probing. No sample PTS is
inferred from a stream start time or nominal frame rate.

The default collection budget is 1024 stream-metadata nodes, 4096 total dictionary
entries/side-data blocks and 8 MiB of copied key/value/payload bytes. Budgets are
aggregate and checked before copying each record. Empty values consume a record.
A partial observation reports `limited` independently from the occurrence-index
status. Collection stops at the first omitted record and preserves the retained
prefix. These budgets do not constrain existing probe descriptions, FFmpeg
allocations, dictionary creation during demuxing, vector/string capacity or JSON
serialization. Probe deadlines remain cooperative I/O deadlines, not a preemptive
limit on every CPU operation.

Text key/value bytes are owned and serialized both as text presentation and exact
hex bytes, so invalid UTF-8 is not silently lost. Codec side data is owned binary
with its numeric FFmpeg type; its layout is provider/version specific and is not
promised as a portable metadata carrier. The observation records the provider
name, FFmpeg release string and numeric libavcodec API version alongside these
blocks. Caller-constructed descriptions may leave this identification absent;
the serializer reports null instead of inventing a provider. Neither dictionary values nor side-data
copies are a lossless snapshot of original container boxes. File relocation,
source file identity, portable schema normalization and sidecar writing require
additional contracts.

Index order is container tags, stream descriptions, then each retained stream's
tags and side data. Stream-description references use the description vector;
stream-tag/side-data references use the source-metadata stream vector. Each source
node explicitly retains its native stream index, including when collection stops
early. Ordinals are catalog-local, not interchangeable stream IDs.

## Observation identity and provenance

`bind(reference)` creates an owning handle tied to the immutable catalog snapshot.
`resolve(handle)` rejects a handle from a different observation even when file
names and ordinals are identical. Copies share the exact same snapshot; moves
transfer it. Handles retain the snapshot so its payload cannot disappear while
a fact or prepared series refers to it. A default/moved-from catalog has no
snapshot and cannot resolve old handles. No separate identity token allocation
or second copy of the payload is needed.

Collection records a display locator separately from the original native locator.
`source_locator_bytes` contains hex-encoded POSIX filename bytes on Linux/macOS,
or explicit little-endian UTF-16 code units on Windows. The corresponding
`source_locator_encoding` is `posix_bytes` or `utf16le`. Display sanitization never
changes these bytes. The POSIX FFmpeg I/O boundary also uses native filename bytes
rather than a sanitized UI string. In-memory adapters leave the locator absent
unless one is explicitly supplied by the caller.

Neither the path nor the in-memory identity is a content digest, stale-file
detector, persistent cross-process ID or cache/resume key. No extra whole-file
hashing is introduced and existing checkpoint hashing is not duplicated. Durable
export/resume identity remains a separate requirement before using metadata
across changed/reopened sources.

Original payloads, library-interpreted EXIF/TIFF values, demuxer observations and
explicitly derived numeric values remain distinct. No exact on-disk byte claim is
made for a demuxer dictionary or decoded field. Every projected fact and numeric
series has a source handle. Derived numeric series require a derivation label;
non-derived series cannot silently carry one. That label is a declaration by the
producer, not an automatically verified transformation or a multi-source lineage
graph.

## Typed facts and concrete consumer

`summarizeSourceMetadata` projects existing stream width/height, nominal/average
frame rate, clockwise orientation, primaries and transfer into static facts.
The container CLI returns this summary alongside the full catalog. It traverses
the occurrence index, retains duplicate stream occurrences separately and does
not bypass an index limit. The defaults are 4096 facts and 64 KiB of fact text,
checked before copying. This is a supported-property summary, not a statement
that all metadata is normalized. Image-only catalogs return Unsupported for this
stream-summary operation; their original EXIF/TIFF descriptions remain available.

Pixel counts, degrees and rational frames per second have explicit units. Missing
properties stay absent. Invalid supplied dimensions/rationals/nonfinite angles
are marked Invalid, not replaced with usable defaults. Rational values are not
divided. Each fact points back to the source stream occurrence and is marked a
static demuxer observation, because FFmpeg may derive properties while probing.
It has no invented frame timestamp, GPS datum or camera calibration. Original
storage retains data outside this deliberately small standardized projection.

## Per-frame values and high-frequency numeric series

`MetadataNumericSeries` stores an interleaved numeric array, channel definitions,
original sample times, explicit discontinuity segments and sample-wide states.
It does not allocate a vector for every sample. Units, optional declared
resolution, coordinate axes and datum are supplied by the producer; unknown
values stay unspecified. Resolution is not inferred sensor accuracy. This numeric
view is an interpretation; arbitrary-width integers, original rationals and
vendor binary representations remain in source storage rather than being
advertised as losslessly converted to double.

`validateMetadataSeries` checks source handles, clock/stream association, valid
timebases, array shape with overflow protection, channel/sample/value/text limits,
known enum values, explicit derived state and finite Present samples. Invalid,
Absent and Unsupported samples can retain their supplied placeholder/invalid
values without presenting them as measurements. The state applies to the entire
sample; channels with different validity or clocks require separate series.
Default limits are 64 channels, one million samples, 64 MiB of numeric values and
64 KiB of text. Timing/state storage is separately bounded by sample count;
allocator capacity, input buffers and provider RAM are not included in the value
byte budget. Validation does not undo allocations a caller already made.

Negative timestamps and duplicate sample times are allowed. Missing FFmpeg
sentinel timestamps and invalid timebases are rejected. Segment ordinals must
increase across clock discontinuities; times are nondecreasing only within each
segment. PerFrame values additionally require strictly increasing explicit frame
indices within each segment. SampleSeries values may preserve explicitly supplied frame associations, including
multiple sensor samples for the same frame; no association is inferred from FPS. Untimed/undecoded observations remain in source storage instead of being
promoted to a timed series with fabricated timestamps.

The timing/series and typed-summary declarations live in the internal
`src/media/metadata_observation.hpp` header. They are not installed as a public SDK
contract. The CLI consumes the static summary. Numeric series are internal
contracts; no production telemetry reader is supplied by this module.

`prepareMetadataSeries` validates once and takes ownership of the numeric arrays
in an immutable snapshot. Its view retains source ownership, exposes const data
and supports repeated half-open window searches in O(log samples), apart from
bounded clock-label comparison. It neither copies sample arrays on each query
nor revalidates their contents. Moved-from views reject queries and expose empty data; a valid
zero-sample prepared series returns an empty range. Pass an rvalue to avoid caller-side argument
copies; mutable aliases must not be retained into the transferred vectors.

Window lookup returns original samples from one explicit clock and segment,
including equal-time samples. It uses existing FFmpeg rational timestamp
comparison, avoiding double conversion and tick subtraction overflow. It does not
map clocks, infer drift/offset, interpolate, sort source data, extrapolate or bridge
segments. An absent segment yields an empty range. The internal one-off
`metadataSampleRange` helper remains available but validates once per call in
O(samples × channels); repeated queries should use the prepared view.

The probe prepares its native filename before allocating the FFmpeg input
context. A path-allocation failure returns the existing `ResourceExhausted` error
without leaving an unowned input context. This also removes an unnecessary display
path conversion at the I/O boundary; native filename handling is unchanged.
The cooperative deadline compares elapsed whole milliseconds with the positive
millisecond timeout, avoiding overflow from widening an extreme timeout to
nanoseconds. The maximum signed 64-bit timeout is covered by the CLI regression.
