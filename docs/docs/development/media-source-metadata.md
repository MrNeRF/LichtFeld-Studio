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

The catalog indexes JPEG/PNG/HEIF/JPEG XL EXIF block descriptions
and observed TIFF tags, plus opaque JPEG/PNG XMP envelopes and references to
observed TIFF XMP byte values, primary-image HEIF/AVIF XMP MIME items and JPEG XL XML boxes. It reuses existing carrier scans
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

`MediaIngest::sourceMetadata(path, index_limits, xmp_limits)` explicitly collects JPEG APP1, PNG iTXt, observed TIFF tag 700, primary-image HEIF/AVIF MIME XMP and JPEG XL XML boxes while using the existing EXIF carrier scan. The CLI `source-metadata INPUT` exposes the result as `metadata.image_xmp`; `source-exif`, default probing, image decoding and training do not request this additional collection. `image_exif` and the camera summary retain their existing meaning. The historical image catalog scope identifier remains unchanged.

`core::image_codecs::read_source_metadata` returns the collected EXIF and XMP snapshots from that one scan. The existing `read_source_exif` still takes the EXIF-only path. `catalogSourceMetadata` can consume an already collected EXIF description and XMP snapshot without rereading the source. `imageXmp`, `xmpBlock`, `XmpBlockReference` and source-bound handles expose immutable, owning occurrences through the same catalog/index contracts as other metadata.

| Carrier | Preserved observation | Payload view |
| --- | --- | --- |
| JPEG standard XMP APP1 | Exact APP1 data, including the NUL-terminated Adobe identifier | Remaining packet bytes |
| JPEG Extended XMP APP1 | Each exact segment, identifier, GUID, declared full length and segment offset, in source order | Individual segment data; no reassembly or completeness claim |
| PNG XMP iTXt | Exact chunk data including keyword, compression fields, language and translated keyword; CRC validity reported separately | Original uncompressed text or bounded zlib-inflated text |
| TIFF tag 700 | Reference to the existing library-decoded byte tag in its observed directory | The byte value from that tag; original on-disk representations remain in the TIFF source ranges |
| JPEG XL XML box | Owning copy of the libjxl-decoded box value and box-event provenance | The decoded value; Brotli box provenance is reported separately |
| HEIF/HEIC/AVIF primary-image MIME | Owning copy of the libheif-decoded item value and item provenance | The decoded value; source MIME type and content encoding are reported separately |

JPEG/PNG collection preserves packet bytes, including unknown namespaces, malformed XML, non-UTF-8 bytes, duplicates and conflicting properties. TIFF exposes the provider-decoded tag inventory: libtiff can normalize noncanonical types or collapse duplicate fields, while the separately retained source ranges preserve the observed on-disk representations. JSON transports these representations and payloads as hex. Collection status and per-block status describe carrier framing and limits, **not XML or RDF validity**. XML is not parsed, external entities are never resolved, and no XMP field is used to alter camera calibration, GPS, orientation, color or timestamps. Extended JPEG GUID/range framing is checked, but segments are not sorted, linked to a standard packet, hashed, merged or declared complete. The image summary performs bounded association and reassembly under the interpretation contract below; collection still retains the original segments separately.

The collector accepts compressed PNG iTXt and nonempty language fields as PNG representations, preserving their originals. This is broader than Adobe's recommended uncompressed XMP iTXt convention. Invalid compression flags/methods, unterminated carrier fields, invalid/trailing compressed streams and CRC damage remain visible as malformed observations. A malformed or limited block has no usable payload view; retained envelope bytes remain available. Intact later blocks can still be collected. No pixel decoding is performed; PNG IDAT payload CRCs and pixel validity are outside this operation.

`XmpLimits` defaults to 64 blocks, 1 MiB per original envelope, 4 MiB aggregate retained original plus inflated bytes, and 1 MiB inflated text per compressed block. Inflate uses at most the remaining aggregate budget, with one extra scratch byte to distinguish an exact-boundary result from overflow. Scratch capacity is not retained per packet. Original blocks are kept when inflation is limited; no partial XML payload is returned. Hard parameter ceilings are 4096 blocks, 16 MiB per envelope/inflation and 64 MiB aggregate. EXIF retention/parsing, JPEG header window, PNG scan/chunk and reference-index limits remain separate. These are retained-byte/work bounds, not a process-wide allocator ceiling. Allocation failures caught by the public media collection API return ResourceExhausted. TIFF C callbacks contain exceptions at the C ABI and return a Limited snapshot with a diagnostic, preserving the established TIFF reader contract.

Compressed iTXt is inflated incrementally with bounded scratch space. Exact output limits, empty packets, truncated streams and trailing compressed bytes are checked before exposing any inflated payload. For uncompressed iTXt, the compression-method byte is ignored as required by the PNG decoder contract.

Exhausting EXIF retention freezes the retained EXIF prefix, while the combined PNG scan continues looking for XMP within its existing scan-byte and chunk limits. The EXIF-only reader retains its early-stop behavior.

The enclosing scan's limited, malformed or I/O status is preserved conservatively, including partial observations and cases where a TIFF packet representation is unsupported. This can reflect a framing/EXIF problem even when an individual XMP block is intact. An index limit does not discard already collected XMP originals. A valid opaque packet does not establish its truth, uniqueness or precedence over EXIF; camera interpretation continues using EXIF/TIFF only.

This increment does not collect XMP from WebP, EXR, videos or external sidecars. Those carriers report unsupported rather than absent for this XMP collector. TIFF XMP uses the existing observed tag inventory and traversal order, including main image directories and visited child directories. `XmpBlock::tiff_tag` stores directory/tag ordinals into the paired EXIF snapshot; `source_xmp_bytes` resolves a checked borrowed byte view in core, and `SourceMetadataCatalog::xmpBytes` resolves it within an immutable catalog. Keep an owning snapshot/catalog or handle alive while using the view. Catalog construction rejects invalid associations. TIFF blocks retain no second packet vector.

TIFF JSON blocks report `representation: libtiff_decoded_bytes` and `tiff_tag` with directory/tag ordinals and the original directory offset. Library-decoded bytes and sparse original source ranges remain distinct: libtiff can normalize or ignore nonconforming tags and collapse duplicate fields within one directory. Collection reports the observed inventory and enclosing warnings rather than claiming a complete raw TIFF archive. In main image directories, libtiff narrows SHORT/LONG tag 700 values that fit in a byte to BYTE. It ignores FLOAT, DOUBLE and out-of-range SHORT values with a "tag ignored" diagnostic, which the enclosing scan reports as Malformed. Child-directory inventories can retain numeric representations; the byte-view resolver reports Unsupported for those rather than interpreting numeric storage as packet bytes. Observed BYTE and UNDEFINED byte vectors preserve embedded NULs, arbitrary bytes and padding without an additional packet allocation. Zero-count TIFF tags remain observable with a Malformed block and `payload_bytes: null`. XML remains opaque.

HEIF/HEIC/AVIF collection opens one libheif context shared by the EXIF and XMP paths. It enumerates MIME metadata attached to the primary image, through libheif's existing item associations. Secondary images, thumbnails and unassociated items do not become primary-image XMP observations. The MIME essence `application/rdf+xml` is recognized case-insensitively, with optional parameters; the declared content type is retained unchanged. This recognition does not validate MIME parameters or XML. Enumeration retains the provider's order and item IDs, without a claim of physical box ordering. A hard cap of 4096 primary-image MIME candidates bounds the additional ID list and enumeration, including non-XMP MIME types. The pinned provider defaults to 1000 total container items, including images and EXIF; it rejects larger containers before collection. The 4096-candidate cap is defensive and does not imply provider support for that many items.

HEIF JSON blocks use `envelope: heif_mime`, `representation: libheif_decoded_bytes` and `heif_item` with `image_item`, `metadata_item`, `content_type` and `content_encoding`, plus hexadecimal `content_type_bytes` and `content_encoding_bytes` preserving their original bytes when display text requires UTF-8 replacement. Libheif prepares metadata when opening the context and may decompress MIME items before collection quotas are applied. `bytes` and usable `payload_bytes` therefore contain the provider-decoded value, not a compressed on-disk archive. The `compressed` field remains false for this retained decoded representation; the source content encoding identifies provider decompression. `XmpLimits::inflated_bytes` does not apply to HEIF; it bounds compressed PNG and JPEG XL output. HEIF per-block and aggregate quotas charge the decoded payload plus the retained MIME type/encoding strings. Oversized items are skipped with Limited status, allowing later smaller items to be retained. After the block-count limit, the first additional matching XMP item sets Limited status and ends enumeration; unrelated MIME items alone do not mark a full collection Limited.

These HEIF quotas bound retained collector data, not libheif's initial parse/decompression work or process-wide allocation. Existing provider limits remain separate. The libheif overlay enables its existing zlib and raw-deflate header compression support, using the existing zlib dependency. Brotli header compression remains disabled. Valid `compress_zlib` and `deflate` MIME items no longer prevent context open, so EXIF and subsequent uncompressed packets remain available. A provider patch bounds each decompressed value to 16 MiB, or a smaller nonzero `max_memory_block_size`; it checks the output extent before appending, rejects invalid/truncated streams and trailing bytes, and releases zlib state on both normal and exceptional exits. The provider also caps retained zlib/deflate metadata values at 64 MiB per context, including secondary and unassociated items. The aggregate is checked after the bounded item decode and before its context-owned copy, so at most one additional bounded decoded item can exist at rejection. These bounds precede collector quotas; they do not cover unrelated uncompressed metadata, provider structures or all process allocations. Values accepted by the provider may still exceed collection quotas, in which case the collector skips them and keeps later smaller packets. An invalid container, unsupported content encoding or corrupt compressed item can prevent libheif from opening the context, yielding the existing EXIF failure snapshot and a corresponding XMP failure; the collector cannot recover later packets from a context that was never opened. The EXIF-only API keeps its existing collection semantics and does not request XMP enumeration. `source-metadata` may continue collecting XMP after EXIF retention is limited, without changing the EXIF snapshot. Image decoding and training do not request this collection.

JPEG XL collection uses the existing libjxl box API, with a metadata-only decoder and the existing sliding-window input adapter. It collects `xml ` boxes before and after codestream boxes, including Brotli-compressed `brob` boxes whose underlying type is `xml `. All XML box values remain opaque: an XML box can carry XMP or other XML, and collection does not validate either syntax or namespaces. Bare codestreams cannot contain container boxes and report absent XML metadata. The same metadata-only scan collects `Exif` boxes, including Brotli-compressed values. `source-exif` requests only the EXIF carrier; pixel decoding and training do not request metadata collection.

JPEG XL EXIF JSON uses `envelope: jxl_offset`, `representation: libjxl_decoded_bytes` and the same `jxl_box` provenance. Each retained value includes its four-byte big-endian offset to the TIFF header, measured from the end of that offset field. The existing bounded classic-TIFF interpreter handles the decoded value without modifying it. Collection status describes the container scan; malformed or unsupported TIFF interpretation is reported separately per block. Camera, GPS and time summaries consume the same existing EXIF entries. EXIF orientation is retained as an observation and does not override the image decoder's codestream orientation.

EXIF and XML have independent block-count and retained-byte quotas. EXIF collection accepts up to 16 MiB per block and 64 MiB in aggregate, with at most 4096 blocks; the existing EXIF interpreter applies its own smaller parsing limits. An exhausted carrier retains its earlier complete values while the shared scan can continue for the other carrier. A compressed value exceeding its output budget is discarded rather than partially exposed. Values are retained at libjxl box-completion events, before parsing the next header; a malformed or truncated following header therefore does not discard the completed value. A later container error is reported for carriers still being scanned. A carrier already stopped at its quota retains that earlier snapshot. Bare codestreams report absent EXIF and XML.

If libjxl reports a decoder error while collecting a compressed EXIF or XML box and its memory manager has not exhausted the decoder budget, that carrier retains its completed values and reports Malformed. When the other carrier is still active, a fresh decoder scans it alone, skipping the damaged carrier without decompressing it. This recovery rewinds the same open file and replaces the other carrier's partial observation rather than appending duplicate values. It happens at most once and uses the original carrier quotas and the same per-pass input, box-event and decoder-memory bounds. The first decoder is destroyed and superseded payloads are released before the second scan. A structural framing error or a second damaged carrier still yields a failure; recovery does not treat an invalid enclosing container as valid. Normal inputs use one scan. Encoded bytes and box events can be examined twice only on this recovery path; the input window is reused and decoder memory budgets are not concurrent.

JPEG XL JSON uses `envelope: jxl_xml`, `representation: libjxl_decoded_bytes` and `jxl_box` with the provider box-event `ordinal` (including structural boxes) and `brotli_compressed`. The owning `bytes` and usable `payload_bytes` contain the box value returned by libjxl, after Brotli decompression when applicable. The envelope header and compressed source stream are not retained as an archive. `compressed` stays false for the retained decoded representation. Catalog byte views and bound handles reuse the existing immutable storage contract.

JPEG XL uses the XMP block-count, per-block and aggregate retained-byte quotas. For `brob` XML, `inflated_bytes` additionally bounds decoded output. The output buffer grows incrementally, with at most one extra logical scratch byte to distinguish exact quota boundaries from overflow. An oversized uncompressed box can be skipped so later smaller boxes remain observable. Exceeding decoded output while decompressing stops that carrier with Limited status and preserves earlier complete packets; the partial packet is discarded and later boxes of that carrier are not claimed as examined. The first additional XML box after the block-count limit also stops the XML carrier. Errors preserve earlier complete packets with a diagnostic. XML text is never interpreted.

The metadata scanner separately caps encoded file size at 2 GiB, enumerated box events at 4096, sliding-window input at 4 MiB and allocations routed through the libjxl memory manager at 64 MiB. The pinned provider creates its internal Brotli decoder with the default allocator, outside that memory manager; the output quota bounds bytes delivered to the collector, not all Brotli allocations. Provider memory, input buffering, retained payload capacity and JSON output remain separate allocations; these limits are not a total-process memory ceiling. Collection validates the provider-observed container/box stream, not the image pixels or full codestream. No metadata decoder threads or pixel output buffers are created. Existing pixel readers retain their own validation and limits.

JPEG XL collection uses libjxl from the pinned vcpkg registry with its standard build options. Decoder-construction allocation fixes should be tracked upstream rather than requiring an application-specific port for metadata collection.

XMP byte quotas charge the logical size of referenced TIFF packets, even though their storage is shared. The existing TIFF read, directory, entry, decoded-value and source-range quotas still apply independently; exhausting them can limit the observed XMP inventory. XMP quota exhaustion keeps the original TIFF tag observations available. JPEG inspection stops at the first scan header, as does existing EXIF collection. No metadata writing or sidecar discovery is performed.

Storage references: [Adobe XMP specifications, Part 3](https://developer.adobe.com/xmp/docs/xmp-specifications/) and [PNG iTXt specification](https://www.w3.org/TR/png-3/#11iTXt). A future increment should add qualified carrier support and namespace-aware interpretation when it has a concrete consumer, then vendor adapters, export/sidecar preservation and dataset integration. It must not infer broad vendor telemetry support from opaque packet retention.

When the shared reference index omits only XMP blocks, already fully indexed EXIF/TIFF camera operands remain usable. The overall catalog/summary still reports its index limit. Actual missing EXIF/TIFF references retain the conservative incomplete-directory behavior. A regression case covers both sides of this boundary.

`SourceMetadataCatalog::imageExifIndexStatus()` records whether every collected EXIF/TIFF value was indexed. It is independent of the overall index status and of collection/parsing status. Omitting only trailing XMP references keeps this sub-index complete; camera summaries use that recorded state directly.

When combined PNG collection continues after its first EXIF retention omission, the EXIF snapshot freezes both its retained prefix and its status/diagnostic at the same point as `source-exif`. Later carrier errors or scan exhaustion remain visible in the XMP scan status, while intact XMP packets can still be retained.


The regression corpus checks exact decoded Brotli bytes and aggregate statuses, high-expansion JXL output, near-`UINT64_MAX` box lengths, and a Linux sparse-file encoded-input ceiling. Opaque external-entity markup is preserved; on Linux an inotify watch verifies that the referenced local file is neither opened nor read. Zlib/deflate fixtures cover both HEIC and AVIF, EXIF and later-packet preservation, malformed/truncated/trailing streams, empty output and the exact 16 MiB provider boundary.
## Standard XMP property interpretation

`summarizeImageMetadata` projects selected standard XMP properties from packets already retained in the catalog. The existing `source-metadata` CLI calls this summary. Collection and `MediaIngest::sourceMetadata` preserve original packets independently; normal probing, pixel decoding and training do not request interpretation. Python callers can request the same image catalog and summary through `lichtfeld.media.MediaIngest.source_metadata_json(path)`.

Expat recognizes expanded namespace URIs, not prefix spelling, in abbreviated attributes and simple RDF property elements. Supported fields are TIFF Make/Model/Orientation; EXIF ExposureTime, FNumber, FocalLength, GPSLatitude/Longitude, GPSAltitude/AltitudeRef, GPSMapDatum, DateTimeOriginal/Digitized; CIPA LensMake/LensModel; auxiliary Lens; XMP CreateDate/ModifyDate. Unknown properties remain in the packet. Structured or qualified forms of recognized properties yield Unsupported facts. This is a scalar projection rather than a complete RDF graph implementation; selected DJI source properties use the separate interpretation contract below.

Each fact carries its packet reference and `xmp_property` namespace, local name, description ordinal and property occurrence ordinal. Namespace and name refer to a shared static field descriptor rather than per-fact string copies. Ordinals count direct description attributes/properties, including unknown ones, excluding `rdf:about` and namespace declarations. Altitude requires one valid reference in the same description and packet, or in the uniquely associated Standard/Extended portions described below; `xmp_inputs` identifies that operand. Repeated observations remain separate with their original provenance. Concordant Present values with matching units do not create a conflict, including aliases within one RDF description; differing or non-comparable values are marked ambiguous. Numeric comparison uses exact reduced rationals or equality at the floating carrier precision, without an epsilon; large rational operands that cannot be converted exactly remain conservatively non-comparable. Pre-existing EXIF ambiguities remain unresolved. Collection or interpretation limits set `summary.limited` independently and do not mark unrelated valid facts ambiguous. No preferred source is selected. GPS coordinates require the standard degrees/minutes or degrees/minutes/seconds form and hemisphere. Datum is retained without conversion or a WGS84 assumption. Dates retain declared precision, fractions and timezone, including no timezone. Leap-second interpretation is Unsupported. Invalid values retain their text instead of becoming numeric zero.

`summary.xmp_packets` reports parsed, malformed, unsupported or limited with packet provenance. No facts are published from a packet failing XML parsing, supported subject/root checks or parsing budgets. A later independent packet can still be interpreted. A supported root is `rdf:RDF`, optionally inside `x:xmpmeta`; standard image descriptions have an empty or omitted `rdf:about` and no subject/description qualifiers. The observed DJI subject is handled only as described below. Extended JPEG XMP fragments use the bounded assembly contract below. Non-primary TIFF directory packets remain preserved without promotion to primary camera properties.

DTDs are rejected before processing their internal/external subsets; external resolution and parameter entities are disabled. The parser never opens files or URLs. Predefined XML entities, numeric references, CDATA and supported XML encodings use Expat XML validation. Interpretation limits: 1 MiB per packet, 4 MiB aggregate input, 64 packet results, depth 32, 16384 elements, 128 ordinary attributes per element (namespace declarations are excluded from this count and remain subject to packet/name bounds), 1024-byte expanded element/attribute/namespace names, 1024 recognized properties and at most 64 KiB decoded property text per packet. Shared summary fact/text budgets also apply; EXIF projection precedes XMP. These controls are not a total-process allocation ceiling. Collector limits remain independent.

Schema references: [Adobe TIFF](https://developer.adobe.com/xmp/docs/xmp-namespaces/tiff/), [Adobe EXIF](https://developer.adobe.com/xmp/docs/xmp-namespaces/exif/), [Expat API](https://libexpat.github.io/doc/api/latest/). Expat is an explicit media dependency using the existing 2.9.0 version override and standard vcpkg port. Its registry selection follows the existing security pins; no separate registry routing is added.

## DJI photo source properties

The image summary also interprets selected properties in the exact
`http://www.dji.com/drone-dji/1.0/` namespace. They remain source declarations
under `vendor.dji.*` semantics, independently of standard EXIF/XMP GPS facts.
The parser does not select a decoder from the camera's manufacturer name.
The observed DJI `rdf:about="DJI Meta Data"` label is accepted specifically for
these vendor properties and retained as `xmp_property.rdf_about`. Standard
properties in that named description are not promoted to unqualified image
facts. Other nonempty RDF subjects and description qualifiers retain the
existing Unsupported behavior. A named description cannot supply an Extended
XMP association or participate in cross-portion altitude derivation. Its
`HasExtendedXMP` declaration is still counted as invalid association evidence,
including the element form, rather than hidden by the selective photo
projection. A completely parsed DJI description without that declaration does
not block an independently valid association.

| Properties | Summary semantics and units |
| --- | --- |
| `GpsLatitude`, `GpsLongitude`, historical `GpsLongtitude` | `vendor.dji.position.latitude/longitude`, signed decimal degrees within physical bounds |
| `AbsoluteAltitude`, `RelativeAltitude` | `vendor.dji.altitude.absolute/relative`, finite signed metres; zero is a valid value |
| `FlightRollDegree`, `FlightPitchDegree`, `FlightYawDegree` | `vendor.dji.flight.roll/pitch/yaw`, finite source angles in degrees |
| `GimbalRollDegree`, `GimbalPitchDegree`, `GimbalYawDegree` | `vendor.dji.gimbal.roll/pitch/yaw`, finite source angles in degrees |
| `GpsStatus`, `AltitudeType`, `Version` | `vendor.dji.gps.status`, `vendor.dji.altitude.type`, `vendor.dji.version`, source text without status normalization |
| `SurveyingMode`, `CamReverse` | `vendor.dji.surveying_mode`, `vendor.dji.camera.reverse`, unitless integer flags 0/1; other nonnegative modes remain Unsupported with their original text |

Present means the declared scalar passed its syntax and range checks. It does
not establish a valid GNSS fix, a vertical datum, a camera optical center, an
exposure epoch or a usable photogrammetric pose. Those qualifications depend on
the model, metadata version and accompanying source properties. The summary
does not convert heights, normalize angles or apply camera/gimbal reversal.
Flight and gimbal observations remain separate; unknown status strings are
preserved rather than interpreted as RTK or a successful fix.

Equal longitude aliases agree; differing or invalid observations flag that
vendor semantic using the existing conflict policy. Different standard and
vendor coordinates do not conflict automatically, since their measured subject
and epoch may differ. Source handles and namespace/property/description/ordinal
provenance retain each occurrence. Existing parser, output and text budgets
apply to these properties too. Unselected DJI properties remain available in
the original XMP bytes. Their presence does not qualify every DJI device,
MakerNote, video telemetry format, RTK profile or calibration representation.

Format reference: DJI Media File Metadata Whitepaper, Part 2, sections 3.3.1–3.3.17,
linked from the [official DJI announcement](https://github.com/dji-sdk/Payload-SDK-Doc/blob/ca93b6470fe8e157550f70ed16ffd8c4d2b5af3a/source/introduction/announcement.md).

## JPEG Extended XMP interpretation

The existing image summary and `source-metadata` CLI associate JPEG Extended XMP with its Standard XMP packet using `xmpNote:HasExtendedXMP` in the `http://ns.adobe.com/xmp/note/` namespace. The link is one simple attribute or property element, captured with a separate 64-byte bound and validated as a 32-character uppercase hexadecimal identifier after XML whitespace trimming. The collector and interpreter share `source_extended_xmp_segment`, a checked borrowed APP1 framing view; the interpreter adds no JPEG file scanner.

Segments can arrive in any marker order and can split XML tokens or UTF-8 characters. They must have one consistent declared length and cover the serialization exactly, with no gaps, empty segments, duplicate ranges or overlaps. The MD5 of the original joined bytes must match the GUID in both the segments and their unique standard reference. MD5 is the format-defined association check; catalog observation identity remains independent of that digest. A policy/provider that cannot perform MD5 yields Unsupported for the extension while keeping the standard observations. Crypto policy is left to the existing OpenSSL configuration.

The extension is allocated only after framing, association, range and coverage checks. Its reconstructed value is limited to 1 MiB and counts toward the existing 4 MiB aggregate XML input budget. Interpretation visits at most 64 indexed XMP input blocks, in addition to the existing 64 packet-result limit and shared output budgets. A collection not confirmed complete (including a malformed or failed carrier scan), omitted references or a capped input set cannot prove a complete extension. Any Standard JPEG packet that does not parse completely (Limited, Malformed or Unsupported, including unavailable payloads) makes the association incomplete: its unparsed remainder may contain a duplicate reference, even when another Standard packet supplies a valid link. Facts from independently parsed Standard packets are retained. Unknown malformed segment association or invalid/ambiguous standard declarations prevents speculative assembly. Completed groups with independent valid associations can still be interpreted when a different known group fails its digest or XML validation. The original catalog/collection bytes and statuses remain separate.

Standard and Extended portions use the same bounded Expat parser and scalar projection. A reconstruction contributes no facts if integrity, supported XML/RDF shape or parser budgets fail. Their supported scalar facts join the existing image summary; repeated observations remain explicit, with the same concordance/conflict rules as ordinary packets. A GPS altitude operand can cross the associated portions only when each has one unqualified image description and the reference is unique across both. Complex/multiple-description operand reconciliation retains the existing conservative per-description behavior. An extension's own `HasExtendedXMP` property does not trigger recursive association.

`summary.xmp_assemblies` records each assembly once: GUID, declared byte extent, digest-verification state, the standard-packet reference and ordered fragment inputs with their offsets and byte lengths. `summary.xmp_packets[].extended_xmp` refers to that GUID. Each assembled fact has `xmp_packet_kind: jpeg_extended_assembly` and an `xmp_assembly` reference; a standard fact using an extended operand reports `retained_packet` with the association reference. `source` remains a real catalog occurrence; the first payload-ordered fragment anchors an assembled packet. Cross-portion operand locations use the existing fact `inputs` handles and include a source in `xmp_inputs`. Full input provenance remains available even when output quotas omit other facts or packet results. Shared immutable ownership keeps input handles alive without retaining another copy of the assembled XML.

Ordinary packets without extended declarations use the existing streaming projection. Extended inputs are prepared as a bounded set before projection so later duplicate/invalid associations cannot retroactively invalidate published facts. Parsers are released after each parse; owned parsed scalar text is bounded by aggregate input and per-packet text limits. One validated assembly buffer is used at a time and released after parsing. A standard declaration without retained fragments is reported explicitly; detecting this declaration can cause a second bounded interpretation pass over that catalog. These limits do not establish a whole-process memory ceiling or complete RDF graph support.

Format reference: [Adobe XMP Specification Part 3, section 1.1.3.1](https://github.com/adobe/XMP-Toolkit-SDK/blob/main/docs/XMPSpecificationPart3.pdf).

## Python image metadata access

`lichtfeld.media.MediaIngest.source_metadata_json(path)` returns an owned JSON
string with the same `schema_version`, `success`, `metadata` and `summary`
fields as `media-ingest source-metadata INPUT`. It reuses the catalog reader,
image summary and JSON serializers. Binary originals remain hex encoded;
partial collection and interpretation retain their original states and
provenance. The call uses the same default collection and summary budgets as
the CLI. `source_exif_json` keeps its existing result format.

The explicit call accepts the existing Python filesystem path conversion and
releases the GIL during collection, interpretation and serialization. An
empty source argument raises the existing LichtFeld error; filesystem argument
conversion failures (including embedded NUL) raise TypeError. A missing file or an
unsupported/malformed carrier can instead return catalog diagnostics, just as
in the CLI. A successful call does not imply that every packet was parsed.
Returned strings do not borrow catalog, parser or source-file storage.

For example, a plugin can inspect the preserved declarations in-process:

```python
import json
import lichtfeld

result = json.loads(lichtfeld.media.MediaIngest.source_metadata_json(image_path))
for fact in result["summary"]["facts"]:
    if fact["state"] == "present" and not fact["ambiguous"]:
        print(fact["semantic"], fact["value"], fact["unit"])
```

Validity and ambiguity are properties of the interpreted declaration, not a
GNSS or calibration quality guarantee. Dataset/training loading does not
request this operation automatically.

## Preserving static source observations during extraction

File extraction accepts `FileExtraction.preserve_source_metadata` in C++/Python
and `--preserve-source-metadata` in the CLI. It is false by default. Enabling it
also requests `extraction_metadata.json`, using the existing atomic manifest
writer. A single `source_metadata` object contains the version-1 catalog and
static stream summary; `source_metadata_policy` identifies
`preserve_static_container_v1`. All listed output frames belong to that input;
their existing `source_pts`, input stream index/timebase and output transform
fields identify the extraction. The observation is stored once, rather than
copied into every frame record.

Collection uses the same opened FFmpeg context as extraction, before frame
delivery, and the existing container collection/index/summary limits. It
preserves demuxer-exposed tags and coded side data, with exact retained bytes in
hex and explicit limited states. This is a bounded static observation, not an
archive of all original container bytes or a scan of timed telemetry packets.
The selected stream and other observed streams retain their identities.

The stream description retains at most 4096 bytes of rotation-tag text.
Larger text is omitted with `rotate_tag_limited: true`; numeric rotation is
still interpreted from the demuxer value without copying it. The summary
reports this omission as limited. Raw tags keep their separate collection
budget, so an omitted description string can still have exact retained bytes
in the catalog. Normal orientation JSON is unchanged.

This opt-in includes any GPS, identifiers or other private data contained in
tags or opaque payloads. Turning it off omits this snapshot from newly generated manifests; it does
not redact files from earlier runs. There is no field-level privacy filter: a payload may contain information outside its
visible tag name. Existing manifest filenames are not an anonymization feature.
Original source declarations remain separate from output dimensions, rotation,
resampling and color processing. They are not injected as current EXIF/XMP
properties of the transformed images or asserted as per-frame GPS fixes.

Resume includes the preservation policy in its existing input-digest and
pipeline/provider identity checks. Switching it in either direction rejects a
checkpoint before adopting/replacing recorded images. The final manifest is
regenerated from a verified resumed run; completed images can survive an
interruption before manifest publication. Images and the manifest are not a
single filesystem transaction. Cancellation retains the existing extraction
semantics and atomic publication checks.

When every candidate is rejected, explicit source preservation still publishes
the observation with an empty `frames` array. A resumed empty run regenerates
that manifest. With source preservation disabled, the historical zero-frame
behavior is unchanged.

The CLI and Python bindings use this same file-extraction path. Rendered image
sequences have no video input to observe; their metadata needs an explicitly
identified source/consumer rather than an invented camera declaration.


### Embedded image extraction metadata

JPEG, PNG and TIFF16 video extraction accept `FileFrameSinkOptions.embed_source_metadata` in C++ and
Python, or `--embed-source-metadata` in the CLI. The default is false. Enabling it
also preserves the static source catalog in `extraction_metadata.json`, using the
same observation collected from the open demuxer. Other output formats reject
this option before creating output. It is a required-write policy: preparation,
encoding or publication failures fail extraction instead of quietly dropping
metadata. Existing completed images can remain after a later extraction failure.
Direct C++ `FileFrameSink` callers must supply an explicit owned container catalog in
`SinkSession.source_metadata`; the writer does not open another source file.
Rendered sequences do not acquire a source observation automatically.

```python
output = lichtfeld.media.FileExtraction()
output.files.format = lichtfeld.media.FrameFileFormat.JPEG
output.files.output_directory = destination
output.files.embed_source_metadata = True
lichtfeld.media.MediaIngest.extract_files(request, output)
```

Each JPEG contains EXIF Orientation=1 and PixelXDimension/PixelYDimension for the
encoded, resized and physically rotated output. EXIF ColorSpace is uncalibrated;
this writer does not infer an ICC profile from the source declarations. Original
orientation, dimensions, calibration, timestamps and private payloads remain in
the source observation, separately from those output properties. Source camera
EXIF, thumbnails and opaque MakerNote offsets are not transplanted into the new
image's EXIF directory.

XMP uses the application namespace `https://lichtfeld.io/ns/media/1.0/`:

- `MetadataPolicy`: `jpeg-static-source-xmp-v1` or `png-static-source-xmp-v1`.
- `PixelWidth`, `PixelHeight`, `Orientation`: output pixel geometry.
- `AppliedClockwiseRotation`: the physical extraction rotation in degrees.
- `SourceFrame`: the existing one-based selected source-frame number, when known.
- `SourceTimestampTicks`, `SourceTimeBaseNumerator`, `SourceTimeBaseDenominator`
  and `TimestampOrigin`: the original integer video clock and its provenance,
  when available. These fields do not claim an absolute exposure date.
- `SourceMetadata`: the version-1 catalog/summary JSON envelope also present in
  the manifest. It retains raw byte fields and the catalog's completeness flags.

The source observation is static. A container GPS tag does not establish a GPS
measurement at every frame; it is kept inside the source envelope, without
creating EXIF GPS fixes or replacing temporal telemetry. The opt-in includes
retained GPS, identifiers, private payloads and source locator bytes together.
It does not provide selective redaction.

For JPEG, small source envelopes share the Standard XMP tree with frame properties, reserving
space for the largest supported frame fields within APP1 capacity. They do not
require MD5. Larger envelopes use Adobe Extended XMP: one RDF tree split into APP1
chunks, an uppercase
MD5 identifier, and the Standard XMP `xmpNote:HasExtendedXMP` link. MD5 is used for
that format identifier, not authentication. A provider without MD5 rejects that
large-packet write. Serialized source XML is bounded to 32 MiB; existing input
collector limits still apply separately. Metadata readers have their own scan,
packet and interpretation budgets and can report partial observations for large
exports. The CPU opt-in uses the existing bounded JPEG memory encoder; its encoded
image limit is 64 MiB, separate from metadata and encoder working memory.

CPU and GPU encoders retain their pixel/quality/chroma policy. JPEG metadata
insertion copies encoded bytes into the existing atomic file service without
recompression. PNG streams the XMP fragments through its encoder into the same
atomic file service.
The metadata option and a SHA-256 of the prepared static source packet participate
in checkpoint identity. Changing either rejects resume before adopting images.
This includes a different source locator even when its file bytes are identical:
retained image provenance must agree with the final manifest. Embedded frame records
also bind the timestamp origin alongside its integer clock. The manifest records
`embedded_metadata_policy`.
Image publication is atomic per file, while the sequence and its final manifest
remain separate commits governed by the existing extraction/recovery contract.

For ordinary CPU output the sink prepares source XML on its first frame and reuses
it thereafter. An all-GPU batch job therefore keeps no unused CPU packet. Invalid
or oversized metadata can fail that first write after directory creation, before
any image publication. Verified resume prepares its shared packet at session start.

#### PNG carrier and writer contract

For PNG, `embed_source_metadata` writes one uncompressed `iTXt` chunk before
`IDAT`, with keyword `XML:com.adobe.xmp`, empty language/translated keyword,
and a read-only XMP packet (`xpacket end="r"`). Each chunk retains a valid PNG
CRC. The packet uses the same source envelope and output/frame properties as JPEG;
PNG pixel dimensions remain represented by `IHDR`. Physically applied rotation
is recorded as output provenance. Original source EXIF stays in the catalog.

The existing serial libpng and parallel stripe encoders stream the extra chunk
during encoding. Their pixel compression, comment handling, color/alpha policy
and default output bytes are unchanged. The extraction producer remains RGB8;
the existing independent UINT16 PNG codec retains its previous contract. There is
no intermediate full encoded-file copy for embedding. The static source XML is
prepared once per session. A small frame prefix, the borrowed source XML and the
read-only footer are streamed into one chunk without copying the full packet for
each frame.
PNG uses a single packet within the 32 MiB XML budget and requires no MD5 provider.
This budget is separate from encoder working memory and metadata-reader limits.

The core packed-byte writer has an explicit PNG-XMP overload. It borrows packet
fragments for the synchronous call and accepts a caller-supplied UTF-8 XML packet;
NUL-containing packets, oversized chunk payloads and non-PNG destinations reject
before publication. Existing overloads remain available. The extraction facade
constructs its ASCII-safe XML through the shared image metadata writer.
Atomic publication observes the combined request/file cancellation callback.
Verified resume uses the same prepared packet, PNG-specific policy and source
identity, plus per-frame clock/origin checks. Unsupported embedding formats fail
before output creation. Image and manifest publication retain their existing
separate recovery boundaries.

Carrier reference: [PNG specification, textual information](https://www.w3.org/TR/png/#11textinfo).


### TIFF16 embedded source provenance

The existing `embed_source_metadata` opt-in also accepts TIFF16 file extraction.
The producer still supplies relative linear float samples to the existing SDR
quantizer. The TIFF retains its UINT16 samples, LZW/strip policy, orientation,
chromaticities, transfer tables and `lfsEncoding` description used by image loaders.
Source provenance is stored separately in standard TIFF tag 700 (`XMLPacket`),
using BYTE values and the `tiff16-static-source-xmp-v1` policy.

The packet contains the same owned static source catalog and frame/output facts
as PNG. It does not turn static video GPS into per-frame measurements or overwrite
output geometry with source camera geometry. Raw fields, completeness flags and
private source information follow the existing opt-in retention policy.

The source properties are prepared once per session. libtiff requires contiguous
XMP and copies the tag value: TIFF therefore assembles one bounded packet per
write and libtiff retains its own copy until that writer closes. This is separate
from quantized pixels, scanline storage and encoder working memory; the 32 MiB
packet budget is not a global RAM ceiling. PNG continues to stream borrowed
fragments, and JPEG retains its existing Standard/Extended packet rules.

The core and `ImageOutput::writeTiff16` explicit-XMP overloads consume a UTF-8
packet synchronously. They reject embedded NUL and packets above 32 MiB; XML
well-formedness is the caller's responsibility. Existing overloads pass no XMP.
The extraction consumer supplies escaped, read-only XMP. Failed/cancelled writes
preserve the previous destination through the existing atomic file service.
Resume binds the policy, source packet digest and frame timestamp origin before
adopting existing images. Source metadata remains static and is observed once
from the open source context.
