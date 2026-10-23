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
representation, explicit XMP collection and occurrence references. `source-exif` keeps its existing
contract. Successful transport does not imply complete or valid metadata:
consumers must inspect collection, interpretation and index statuses separately.

## Ownership and identity

The catalog owns an immutable shared snapshot containing `ImageExifDescription` and `SourceXmp`. Original envelopes,
observed TIFF source ranges and decoded values are retained in that owner.
The occurrence index stores only ordinal references, not another copy of values.
Duplicate IDs and unknown tags remain distinct and ordered. References select
an EXIF block/entry, TIFF directory/tag or XMP block. They are meaningful only within the
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
and observed TIFF tags, plus opaque JPEG/PNG XMP envelopes and references to
observed TIFF XMP byte values and primary-image HEIF/AVIF XMP MIME items. It reuses existing carrier scans
and adds no vendor decoder, Python binding or metadata writer. Container/stream observations use the existing video probe as described below. HEIF remains scoped to the existing primary-image
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
a fact refers to it. A default/moved-from catalog has no
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
made for a demuxer dictionary or decoded field. Every projected fact has a source
handle. The static stream projection does not claim sensor measurements or an
automatically verified transformation graph.

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

The typed-summary declarations live in the internal
`src/media/metadata_observation.hpp` header. They are not installed as a public SDK
contract. The CLI consumes the static summary.

## Application use and I/O boundaries

The first planned application consumer is image-dataset preparation: camera and
lens metadata can supply explicit initial-intrinsics candidates, and GPS can
supply georeferencing observations. The camera-field interpretation is supplied
by the follow-up; catalog collection and the CLI are the implemented consumers
here. No dataset, training, intrinsics or georeferencing integration is enabled
by this catalog API. A focal length alone is not a calibrated camera model.

The probe prepares its native filename before allocating the FFmpeg input
context. A path-allocation failure returns the existing `ResourceExhausted` error
without leaving an unowned input context. This also removes an unnecessary display
path conversion at the I/O boundary. The shared `path_to_native_utf8` helper
preserves native POSIX filename bytes for probe, extraction, playback and encoding;
it no longer sanitizes these I/O filenames as display text. Windows keeps its
native-path and UTF-8 conversion. UI and JSON display paths still use their
separate sanitized representation.
The cooperative deadline compares elapsed whole milliseconds with the positive
millisecond timeout, avoiding overflow from widening an extreme timeout to
nanoseconds. The maximum signed 64-bit timeout is covered by the CLI regression.

## Image camera and location summary

`source-metadata INPUT` adds a `summary` with scope `static_image_properties`.
The original `metadata` object and the `source-exif` command remain unchanged.
Internally, `summarizeImageMetadata` consumes the existing immutable catalog;
it does not reopen the source, decode pixels, apply image transforms or affect
the training exposure reader. This interface remains internal alongside the
stream summary until the metadata contracts stabilize.

| Directory | Interpreted fields |
|---|---|
| Primary image IFD | Camera make/model, EXIF orientation code, modification civil time |
| Exif IFD | Exposure duration, f-number, photographic sensitivity and explicit extended sensitivity fields, focal length, 35 mm equivalent focal length, lens make/model, original/digitized civil time, subseconds and UTC offsets |
| GPS IFD | Signed latitude/longitude, altitude with reference 0/1, map datum, UTC date and seconds of day |

Orientation is an EXIF code, including mirror cases, not a clockwise angle.
Focal lengths are in millimetres, exposure in seconds, coordinates in degrees
and altitude in metres. Photographic sensitivity is not automatically relabelled
ISO: its extended marker is unsupported in that fact, while separately present
standard-output, recommended-exposure and ISO-speed fields retain their own
semantics. Zero equivalent focal length is unknown, not a zero-length lens.

EXIF groups use the envelope ordinal and full directory path; only IFD0 and its
direct Exif/GPS children supply the image summary. TIFF uses the first main image
directory and its own direct Exif/GPS children, identified by the reader-supplied
kind and parent offset. Thumbnails, later pages, SubIFDs and their metadata remain
in the raw catalog and do not produce primary-image facts. Numeric tag IDs in
unrelated directories are not
interpreted as GPS or Exif fields. A limited catalog cannot borrow missing
operands from retained originals outside its reference index.

Each fact owns its value and a source-bound handle. Compound facts additionally
retain handles to their operands and name their derivation. Duplicate tag
occurrences remain separate and carry `ambiguous: true`; duplicate hemisphere,
altitude or time-component references prevent arbitrary selection. Callers must
not resolve different image directories or duplicate facts by taking the first
semantic match. The source references identify their original groups.
Uniqueness refers to occurrences retained by the collector: the observed TIFF
inventory is not an additional validator of duplicate tags in the original file.

An incomplete reference index cannot establish uniqueness of compound operands,
and an interrupted TIFF collection may omit fields within an observed directory.
Coordinate, altitude and assembled capture-time facts then report `unsupported`
with `Directory metadata is incomplete`, rather than declaring operands absent
or selecting an apparently unique retained reference. A fully parsed EXIF block
remains self-contained if collection stopped before another envelope.

Exact EXIF rationals remain rational values for scalar fields. TIFF rationals
are already floating-point values from libtiff and are not reconstructed into
invented exact fractions. DMS coordinates and GPS seconds of day explicitly
derive double values; original components remain in the catalog. GPS dates and
times are separate observations from the same GPS directory, not a clock mapping
to the video. No datum, geoid, accuracy, timezone or UTC timestamp is inferred.

Original and digitized dates are validated as civil dates and assembled with
subseconds and an offset only from the same Exif directory. Fractional digits
are retained without conversion through floating point. An absent or explicitly
unknown offset yields a local civil timestamp with `Timezone unspecified`, not
UTC. A malformed offset also leaves a valid civil date and subseconds usable as
local time; its separate offset fact remains Invalid and its original record is
retained among the compound inputs. Duplicate offsets, incomplete directories,
invalid dates and invalid subseconds do not acquire this fallback. Modification
time remains a civil field; it is not joined across IFDs.
Partial dates, leap-second values and non-ASCII text are not normalized into
valid timestamps. Unknown, invalid, ambiguous and unsupported observations are
explicit; unrecognized tags remain available in the original catalog.

The image summary has independent fact and aggregate text limits, defaulting to
4096 facts and 64 KiB. Text accounting includes semantic names, stored strings,
diagnostics and derivations. Grouping scratch storage scales with the retained
catalog index and directory paths; it is not a provider-wide memory ceiling.
Allocation failure returns `ResourceExhausted`. Newer altitude-reference modes,
MakerNotes, XMP, vendor telemetry, calibration, metadata writing and automatic
pose or exposure changes are outside this interpretation table.

The ASCII projection accepts trailing NUL padding only when all bytes after the
first terminator are zero; it does not hide another string or payload. This
tolerance does not suppress provider warnings: TIFF diagnosed as malformed
remains malformed, and its compound summary values remain unsupported.
Seven-bit ASCII text is transported with JSON string escaping. Subsecond trailing spaces
are formatting padding and are omitted from the significant fractional digits.
Unknown dates accept all-space and colon-preserving blank forms; subsecond
padding follows [CIPA Exif](https://www.cipa.jp/std/documents/e/DC-008-2012_E.pdf).
Unknown offsets are retained as unknown, including colon-preserving blanks.
Zero f-number and focal length mean an unknown manual-lens value, with the
original rational retained; a zero denominator remains Invalid. The GPS
hemisphere projection accepts exactly one ASCII N/S or E/W byte without a NUL
terminator. This narrow compatibility case does not relax other ASCII fields
or override provider diagnostics.
Offset validation checks numeric hour/minute syntax, not membership in a
geographic timezone registry. Leap-second representations are retained as
unsupported interpretations, not asserted to be corrupt. The internal summary
scope is a closed enum; JSON identifiers remain unchanged.

## Opaque XMP collection

`MediaIngest::sourceMetadata(path, index_limits, xmp_limits)` explicitly collects JPEG APP1, PNG iTXt, observed TIFF tag 700 and primary-image HEIF/AVIF MIME XMP while using the existing EXIF carrier scan. The CLI `source-metadata INPUT` exposes the result as `metadata.image_xmp`; `source-exif`, default probing, image decoding and training do not request this additional collection. `image_exif` and the camera summary retain their existing meaning. The historical image catalog scope identifier remains unchanged.

`core::image_codecs::read_source_metadata` returns the collected EXIF and XMP snapshots from that one scan. The existing `read_source_exif` still takes the EXIF-only path. `catalogSourceMetadata` can consume an already collected EXIF description and XMP snapshot without rereading the source. `imageXmp`, `xmpBlock`, `XmpBlockReference` and source-bound handles expose immutable, owning occurrences through the same catalog/index contracts as other metadata.

| Carrier | Preserved observation | Payload view |
| --- | --- | --- |
| JPEG standard XMP APP1 | Exact APP1 data, including the NUL-terminated Adobe identifier | Remaining packet bytes |
| JPEG Extended XMP APP1 | Each exact segment, identifier, GUID, declared full length and segment offset, in source order | Individual segment data; no reassembly or completeness claim |
| PNG XMP iTXt | Exact chunk data including keyword, compression fields, language and translated keyword; CRC validity reported separately | Original uncompressed text or bounded zlib-inflated text |
| TIFF tag 700 | Reference to the existing library-decoded byte tag in its observed directory | The byte value from that tag; original on-disk representations remain in the TIFF source ranges |
| HEIF/HEIC/AVIF primary-image MIME | Owning copy of the libheif-decoded item value and item provenance | The decoded value; source MIME type and content encoding are reported separately |

JPEG/PNG collection preserves packet bytes, including unknown namespaces, malformed XML, non-UTF-8 bytes, duplicates and conflicting properties. TIFF exposes the provider-decoded tag inventory: libtiff can normalize noncanonical types or collapse duplicate fields, while the separately retained source ranges preserve the observed on-disk representations. JSON transports these representations and payloads as hex. Collection status and per-block status describe carrier framing and limits, **not XML or RDF validity**. XML is not parsed, external entities are never resolved, and no XMP field is used to alter camera calibration, GPS, orientation, color or timestamps. Extended JPEG GUID/range framing is checked, but segments are not sorted, linked to a standard packet, hashed, merged or declared complete. Those operations require their own interpretation contract and corpus.

The collector accepts compressed PNG iTXt and nonempty language fields as PNG representations, preserving their originals. This is broader than Adobe's recommended uncompressed XMP iTXt convention. Invalid compression flags/methods, unterminated carrier fields, invalid/trailing compressed streams and CRC damage remain visible as malformed observations. A malformed or limited block has no usable payload view; retained envelope bytes remain available. Intact later blocks can still be collected. No pixel decoding is performed; PNG IDAT payload CRCs and pixel validity are outside this operation.

`XmpLimits` defaults to 64 blocks, 1 MiB per original envelope, 4 MiB aggregate retained original plus inflated bytes, and 1 MiB inflated text per compressed block. Inflate uses at most the remaining aggregate budget, with one extra scratch byte to distinguish an exact-boundary result from overflow. Scratch capacity is not retained per packet. Original blocks are kept when inflation is limited; no partial XML payload is returned. Hard parameter ceilings are 4096 blocks, 16 MiB per envelope/inflation and 64 MiB aggregate. EXIF retention/parsing, JPEG header window, PNG scan/chunk and reference-index limits remain separate. These are retained-byte/work bounds, not a process-wide allocator ceiling. Allocation failures caught by the public media collection API return ResourceExhausted. TIFF C callbacks contain exceptions at the C ABI and return a Limited snapshot with a diagnostic, preserving the established TIFF reader contract.

Compressed iTXt is inflated incrementally with bounded scratch space. Exact output limits, empty packets, truncated streams and trailing compressed bytes are checked before exposing any inflated payload. For uncompressed iTXt, the compression-method byte is ignored as required by the PNG decoder contract.

Exhausting EXIF retention freezes the retained EXIF prefix, while the combined PNG scan continues looking for XMP within its existing scan-byte and chunk limits. The EXIF-only reader retains its early-stop behavior.

The enclosing scan's limited, malformed or I/O status is preserved conservatively, including partial observations and cases where a TIFF packet representation is unsupported. This can reflect a framing/EXIF problem even when an individual XMP block is intact. An index limit does not discard already collected XMP originals. A valid opaque packet does not establish its truth, uniqueness or precedence over EXIF; camera interpretation continues using EXIF/TIFF only.

This increment does not collect XMP from JPEG XL, WebP, EXR, videos or external sidecars. Those carriers report unsupported rather than absent for this XMP collector. TIFF XMP uses the existing observed tag inventory and traversal order, including main image directories and visited child directories. `XmpBlock::tiff_tag` stores directory/tag ordinals into the paired EXIF snapshot; `source_xmp_bytes` resolves a checked borrowed byte view in core, and `SourceMetadataCatalog::xmpBytes` resolves it within an immutable catalog. Keep an owning snapshot/catalog or handle alive while using the view. Catalog construction rejects invalid associations. TIFF blocks retain no second packet vector.

TIFF JSON blocks report `representation: libtiff_decoded_bytes` and `tiff_tag` with directory/tag ordinals and the original directory offset. Library-decoded bytes and sparse original source ranges remain distinct: libtiff can normalize or ignore nonconforming tags and collapse duplicate fields within one directory. Collection reports the observed inventory and enclosing warnings rather than claiming a complete raw TIFF archive. In main image directories, libtiff narrows SHORT/LONG tag 700 values that fit in a byte to BYTE. It ignores FLOAT, DOUBLE and out-of-range SHORT values with a "tag ignored" diagnostic, which the enclosing scan reports as Malformed. Child-directory inventories can retain numeric representations; the byte-view resolver reports Unsupported for those rather than interpreting numeric storage as packet bytes. Observed BYTE and UNDEFINED byte vectors preserve embedded NULs, arbitrary bytes and padding without an additional packet allocation. Zero-count TIFF tags remain observable with a Malformed block and `payload_bytes: null`. XML remains opaque.

HEIF/HEIC/AVIF collection opens one libheif context shared by the EXIF and XMP paths. It enumerates MIME metadata attached to the primary image, through libheif's existing item associations. Secondary images, thumbnails and unassociated items do not become primary-image XMP observations. The MIME essence `application/rdf+xml` is recognized case-insensitively, with optional parameters; the declared content type is retained unchanged. This recognition does not validate MIME parameters or XML. Enumeration retains the provider's order and item IDs, without a claim of physical box ordering. A hard cap of 4096 primary-image MIME candidates bounds the additional ID list and enumeration, including non-XMP MIME types. The pinned provider defaults to 1000 total container items, including images and EXIF; it rejects larger containers before collection. The 4096-candidate cap is defensive and does not imply provider support for that many items.

HEIF JSON blocks use `envelope: heif_mime`, `representation: libheif_decoded_bytes` and `heif_item` with `image_item`, `metadata_item`, `content_type` and `content_encoding`, plus hexadecimal `content_type_bytes` and `content_encoding_bytes` preserving their original bytes when display text requires UTF-8 replacement. Libheif prepares metadata when opening the context and may decompress MIME items before collection quotas are applied. `bytes` and usable `payload_bytes` therefore contain the provider-decoded value, not a compressed on-disk archive. The `compressed` field remains false for this retained decoded representation; the source content encoding identifies provider decompression. `XmpLimits::inflated_bytes` governs PNG only. HEIF per-block and aggregate quotas charge the decoded payload plus the retained MIME type/encoding strings. Oversized items are skipped with Limited status, allowing later smaller items to be retained. After the block-count limit, the first additional matching XMP item sets Limited status and ends enumeration; unrelated MIME items alone do not mark a full collection Limited.

These HEIF quotas bound retained collector data, not libheif's initial parse/decompression work or process-wide allocation. Existing provider limits remain separate. The qualified minimal libheif builds currently reject zlib/deflate metadata with an unsupported-compression diagnostic; this increment does not enable new compression features. A provider that already supports those encodings can expose decoded values through the same collector. An invalid container, unsupported content encoding or corrupt compressed item can prevent libheif from opening the context, yielding the existing EXIF failure snapshot and a corresponding XMP failure; the collector cannot recover later packets from a context that was never opened. The EXIF-only API keeps its existing collection semantics and does not request XMP enumeration. `source-metadata` may continue collecting XMP after EXIF retention is limited, without changing the EXIF snapshot. Image decoding and training do not request this collection.

XMP byte quotas charge the logical size of referenced TIFF packets, even though their storage is shared. The existing TIFF read, directory, entry, decoded-value and source-range quotas still apply independently; exhausting them can limit the observed XMP inventory. XMP quota exhaustion keeps the original TIFF tag observations available. JPEG inspection stops at the first scan header, as does existing EXIF collection. No metadata writing or sidecar discovery is performed.

Storage references: [Adobe XMP specifications, Part 3](https://developer.adobe.com/xmp/docs/xmp-specifications/) and [PNG iTXt specification](https://www.w3.org/TR/png-3/#11iTXt). A future increment should add qualified carrier support and namespace-aware interpretation when it has a concrete consumer, then vendor adapters, export/sidecar preservation and dataset integration. It must not infer broad vendor telemetry support from opaque packet retention.

When the shared reference index omits only XMP blocks, already fully indexed EXIF/TIFF camera operands remain usable. The overall catalog/summary still reports its index limit. Actual missing EXIF/TIFF references retain the conservative incomplete-directory behavior. A regression case covers both sides of this boundary.

`SourceMetadataCatalog::imageExifIndexStatus()` records whether every collected EXIF/TIFF value was indexed. It is independent of the overall index status and of collection/parsing status. Omitting only trailing XMP references keeps this sub-index complete; camera summaries use that recorded state directly.

When combined PNG collection continues after its first EXIF retention omission, the EXIF snapshot freezes both its retained prefix and its status/diagnostic at the same point as `source-exif`. Later carrier errors or scan exhaustion remain visible in the XMP scan status, while intact XMP packets can still be retained.
