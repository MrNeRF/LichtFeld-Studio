# LichtFeld `.licht` Container Byte Grammar, Version 1.0

Status: published and frozen grammar for format 1.0. This document is the
self-contained on-disk contract; no repository plan, source file, or C/C++
object layout is required to implement it.

## 1. Conformance language and primitive encodings

The words **MUST**, **MUST NOT**, **SHOULD**, and **MAY** are normative.

All integers are unsigned little-endian unless a field says otherwise. Readers
MUST perform checked arithmetic before adding an offset and a size. UUID fields
are the 16 octets of an RFC 4122 UUID in network byte order (the byte sequence
returned by Python's `uuid.UUID.bytes`); UUIDs are not integer-swapped. A UUID
whose 16 octets are zero is the null UUID.

Times are unsigned nanoseconds since 1970-01-01T00:00:00Z. Fourcc values are
exactly four bytes in `[A-Z0-9]`. All reserved fields MUST be written as zero
and MUST be rejected when nonzero. Alignment padding MUST be written as zero;
the reader validates padding that lies inside a row's described chunk span.

CRC fields use CRC32c Castagnoli, reflected polynomial `0x82f63b78`, initial
state `0xffffffff`, and final XOR `0xffffffff`. The check value for the nine
ASCII bytes `123456789` is `0xe3069283`. A table below saying `CRC32c [0,N)`
means the CRC covers exactly bytes at relative offsets zero through `N-1`; the
CRC field itself follows that range and is not treated as zero-filled input.
CRCs detect accidental corruption. They are not authentication, tamper
evidence, or rollback protection.

Version 1.0 uses these common enum values:

| Name | Value | Meaning |
|---|---:|---|
| `STORED` | 0 | Payload bytes are uncompressed. |
| `ZSTD` | 1 | One standard zstd frame, no dictionary or skippable frames, with a declared content size and no trailing frame. |
| `CRC_NONE` | 0 | No per-block table. |
| `CRC32C_BLOCKS_V1` | 1 | The per-block table in section 7. |

## 2. Physical address space

| File range | Size | Contents |
|---|---:|---|
| `[0,256)` | 256 B | Immutable superblock. |
| `[256,4096)` | 3,840 B | Bootstrap gap, zero on creation and ignored by 1.x readers. |
| `[4096,8192)` | 4,096 B | Head slot A. |
| `[8192,12288)` | 4,096 B | Head slot B. |
| `[12288,65536)` | 53,248 B | Bootstrap gap, zero on creation and ignored by 1.x readers. |
| `[65536,...)` | variable | Append generations: chunk spans, index blob, alignment, and a commit record. |

The append region begins at exactly 65,536. Chunk headers, index blobs, and
commit records begin on 64-byte boundaries. Each generation ends immediately
after its 256-byte commit record. Its authoritative end is
`committed_file_end`, not physical EOF.

Physical EOF MAY be greater than `committed_file_end`; the suffix is an orphan
tail and is invisible to normal readers. Physical EOF MUST be at least the
selected head's `committed_file_end`.

## 3. Superblock: 256 bytes at file offset 0

The superblock is written once. Compaction and Save As create a new
`file_uuid`. An atomically replaced autosave sidecar is a new file incarnation
and also receives a new `file_uuid`.

| Offset | Size | Encoding | Field | Required value or rule |
|---:|---:|---|---|---|
| 0 | 8 | bytes | `magic` | `89 4c 46 53 0d 0a 1a 0a` (`\x89LFS\r\n\x1a\n`). |
| 8 | 2 | u16 | `format_major` | `1`. |
| 10 | 2 | u16 | `format_minor` | `0` for this grammar. See section 13. |
| 12 | 4 | u32 | `byte_order_tag` | `0x01020304`; its byte sequence is `04 03 02 01`. |
| 16 | 4 | u32 | `superblock_bytes` | `256`. |
| 20 | 4 | u32 enum | `container_role` | `0` master, `1` autosave sidecar. |
| 24 | 16 | UUID | `project_uuid` | Non-null and stable across master, sidecars, Save As, and compaction. |
| 40 | 16 | UUID | `file_uuid` | Non-null file-incarnation identity. |
| 56 | 8 | u64 | `head_a_offset` | `4096`. |
| 64 | 8 | u64 | `head_b_offset` | `8192`. |
| 72 | 4 | u32 | `head_slot_bytes` | `4096`. |
| 76 | 4 | u32 | `head_slot_count` | `2`. |
| 80 | 8 | u64 | `append_region_offset` | `65536`. |
| 88 | 8 | u64 | `creation_time_unix_ns` | Creation time; deterministic fixtures use the constants listed by their generator. |
| 96 | 16 | UUID | `base_explicit_commit_uuid` | Sidecar binding. Non-null only for role 1. |
| 112 | 8 | u64 | `autosave_sequence` | Sidecar candidate ordering; at least 1 for role 1. |
| 120 | 8 | zero | `reserved_0` | Zero. |
| 128 | 16 | UUID | `sidecar_snapshot_uuid` | Non-null only for role 1 and equal to the sidecar commit's snapshot UUID. |
| 144 | 108 | zero | `reserved_1` | Zero. |
| 252 | 4 | u32 | `superblock_crc32c` | CRC32c `[0,252)`. |

For a master (role 0), offsets 96 through 143 MUST all be zero. For a sidecar
(role 1), the three binding values MUST be nonzero, and `project_uuid` is the
master project identity. Other role values are invalid.

## 4. Head slot: 4,096 bytes at offsets 4,096 and 8,192

The slot at 4,096 has `slot_id=0`; the slot at 8,192 has `slot_id=1`. An
all-zero slot is uninitialized, not corrupt. Any other slot is valid only if
every rule below and all transitively referenced metadata validate.

| Offset | Size | Encoding | Field | Required value or rule |
|---:|---:|---|---|---|
| 0 | 8 | bytes | `magic` | ASCII `LFSHEAD` followed by `00`. |
| 8 | 4 | u32 | `slot_id` | Physical slot number, 0 or 1. |
| 12 | 4 | u32 | `head_bytes` | `4096`. |
| 16 | 8 | u64 | `head_sequence` | At least 1; sole head comparison key. |
| 24 | 8 | u64 | `generation` | Selected commit generation; at least 1. Never a comparison key. |
| 32 | 16 | UUID | `project_uuid` | Equal to superblock. |
| 48 | 16 | UUID | `file_uuid` | Equal to superblock. |
| 64 | 16 | UUID | `commit_uuid` | Non-null and equal to commit record. |
| 80 | 8 | u64 | `commit_offset` | 64-byte aligned, in append region. |
| 88 | 8 | u64 | `commit_bytes` | `256`. |
| 96 | 8 | u64 | `committed_file_end` | Equal to commit record and `commit_offset + 256`. |
| 104 | 4 | u32 | `commit_crc32c_echo` | Equal to the referenced commit's CRC field. |
| 108 | 4 | u32 | `head_flags` | Zero in version 1.0. |
| 112 | 8 | u64 | `preview_offset` | Zero, or the absolute offset of the selected generation's preview payload. See §4.1. |
| 120 | 4 | u32 | `preview_bytes` | Zero iff `preview_offset` is zero; otherwise the preview payload length, at most 16,777,216. |
| 124 | 4 | u32 | `preview_format` | Zero iff `preview_offset` is zero; `1` = PNG. Other values invalidate the slot in version 1.0. |
| 128 | 3,964 | zero | `reserved` | Zero. |
| 4092 | 4 | u32 | `head_crc32c` | CRC32c `[0,4092)`. |

### 4.1 Preview locator and the foreign-reader contract

The preview locator is an **AUDIT-ONLY accelerator**, never semantic authority.
It exists so that a thumbnailer or asset browser can extract a
project preview without implementing the container: read the superblock, read
both 4,096-byte head slots at their fixed offsets, CRC32c-validate each over
`[0,4092)`, select the valid slot with the higher `head_sequence`, and read
`preview_bytes` bytes at `preview_offset`. No index parsing and no
decompression are required.

Rules, all slot-invalidating on violation:

1. An all-zero locator triple means the selected generation has no preview.
2. When non-zero, `preview_offset` MUST be 64-byte aligned, and
   `preview_offset + preview_bytes` MUST NOT exceed `committed_file_end`.
3. The locator MUST reference exactly the stored payload span of a `THMB`
   chunk row present in the selected generation's index, and that row MUST use
   `compression = none`. The payload is the preview image verbatim.
4. Full readers resolve `THMB` through the index like any chunk; the locator
   is an accelerator for foreign readers, never an alternative authority.
5. Writers that publish a generation without regenerating a preview MUST
   either carry the prior `THMB` chunk forward and point the locator at it, or
   publish a zero locator. A locator pointing at a span not owned by the
   selected generation's `THMB` row is invalid.

Within one file incarnation, each successful publication increments
`head_sequence` by one and writes the inactive slot. It also increments
`generation` by one for an ordinary same-file append. Sequences reset to one in
a new file incarnation. Deliberately writing the same authority tuple to both
slots is not normal, but is recognized as the duplicate-slot case in section
11.

## 5. Commit record: 256 bytes

The commit record is the last 256 bytes of a generation and starts at a
64-byte boundary.

| Offset | Size | Encoding | Field | Required value or rule |
|---:|---:|---|---|---|
| 0 | 8 | bytes | `magic` | ASCII `LFSCOMIT`. |
| 8 | 2 | u16 | `record_bytes` | `256`. |
| 10 | 2 | u16 | `record_version` | `1`. |
| 12 | 4 | u32 enum | `commit_kind` | See table below. |
| 16 | 16 | UUID | `project_uuid` | Equal to superblock. |
| 32 | 16 | UUID | `file_uuid` | Equal to superblock. |
| 48 | 16 | UUID | `commit_uuid` | Non-null and equal to head. |
| 64 | 8 | u64 | `generation` | Equal to head. |
| 72 | 16 | UUID | `parent_commit_uuid` | Same-file parent, or null for a root. |
| 88 | 8 | u64 | `parent_commit_offset` | Same-file parent offset, or zero for a root. |
| 96 | 16 | UUID | `explicit_ancestor_commit_uuid` | See lineage rules below. |
| 112 | 16 | UUID | `snapshot_uuid` | Non-null consistency identity. |
| 128 | 8 | u64 | `wallclock_unix_ns` | Commit wallclock. |
| 136 | 8 | u64 | `index_offset` | 64-byte aligned and before this record. |
| 144 | 8 | u64 | `index_stored_bytes` | Exact stored index-blob length; nonzero. |
| 152 | 8 | u64 | `index_uncompressed_bytes` | Exact decoded index length; nonzero. |
| 160 | 4 | u32 | `index_stored_crc32c` | CRC32c of exactly the stored blob. |
| 164 | 4 | u32 | `index_uncompressed_crc32c` | CRC32c of exactly the decoded index. |
| 168 | 4 | u32 enum | `index_compression` | `STORED` or `ZSTD`. |
| 172 | 4 | u32 | `commit_flags` | Zero in version 1.0. |
| 176 | 8 | u64 | `committed_file_end` | Exactly `this_record_offset + 256`. |
| 184 | 2 | u16 | `min_reader_major` | Minimum reader version tuple. |
| 186 | 2 | u16 | `min_reader_minor` | Minimum reader version tuple. |
| 188 | 2 | u16 | `min_safe_writer_major` | Minimum safe writer version tuple. |
| 190 | 2 | u16 | `min_safe_writer_minor` | Minimum safe writer version tuple. |
| 192 | 16 | 128-bit LE bitmap | `required_reader_capabilities` | Bit 0 is the least significant bit of byte 192. |
| 208 | 16 | 128-bit LE bitmap | `required_writer_capabilities` | Same allocation as reader bitmap. |
| 224 | 28 | zero | `reserved` | Zero. |
| 252 | 4 | u32 | `commit_crc32c` | CRC32c `[0,252)`. |

Commit kinds are:

| Value | Name | Valid location |
|---:|---|---|
| 1 | `EXPLICIT` | Master only. |
| 2 | `AUTOSAVE` | Sidecar only. A master containing it is corrupt. |
| 3 | `RECOVERED` | Master only; a recovery made durable as an explicit head. |
| 4 | `COMPACTION` | Master only; root of a new file incarnation. |

Lineage rules are exact:

* A root has generation 1, null parent UUID, and parent offset zero.
* A non-root parent UUID and offset are both nonzero. The parent record MUST be
  earlier in the same file, CRC-valid, have the same project/file UUIDs, and
  have generation exactly one less. Readers walk this lineage to the root and
  reject repeated commit UUIDs, cycles, or broken links; ancestor indexes need
  not be replayed. The child's generation bytes, including its index and commit,
  begin no earlier than the parent's `committed_file_end`.
* `EXPLICIT`, `RECOVERED`, and `COMPACTION` records set
  `explicit_ancestor_commit_uuid` equal to their own commit UUID.
* A sidecar is a one-generation file. Its `AUTOSAVE` record is a root and sets
  `explicit_ancestor_commit_uuid` to the superblock's
  `base_explicit_commit_uuid`.
* A `COMPACTION` root has generation 1 because generations are scoped to a file
  incarnation. Project-level history, if desired, belongs in `PROJ`.

The bytes between the end of the stored index blob and the aligned commit
record MUST be zero. The index range and the commit record MUST be disjoint.

## 6. Chunk header and physical chunk span

Every physical chunk has one 64-byte header. Fourcc is both the logical type
and the first four header bytes; there is no additional chunk magic.

| Offset | Size | Encoding | Field | Required value or rule |
|---:|---:|---|---|---|
| 0 | 4 | fourcc | `fourcc` | `[A-Z0-9]{4}`. |
| 4 | 2 | u16 | `chunk_version` | At least 1. |
| 6 | 2 | u16 | `header_bytes` | `64`. |
| 8 | 16 | UUID | `instance_uuid` | Non-null logical key component. |
| 24 | 4 | u32 bitmap | `chunk_flags` | Bits below; all other bits zero. |
| 28 | 2 | u16 enum | `compression` | `STORED` (0), `ZSTD` (1), or `BYTESHUFFLE_ZSTD` (2). |
| 30 | 2 | u16 enum | `block_crc_kind` | `CRC_NONE` or `CRC32C_BLOCKS_V1`. |
| 32 | 8 | u64 | `stored_bytes` | Exact stored payload size. |
| 40 | 8 | u64 | `uncompressed_bytes` | Exact decoded payload size. |
| 48 | 8 | u64 | `block_crc_table_offset` | Absolute offset, or zero when no table. |
| 56 | 4 | u32 | `payload_crc32c` | CRC32c of stored payload bytes. |
| 60 | 4 | u32 | `header_crc32c` | CRC32c `[0,60)`. |

Chunk flag assignments are:

| Bit | Name | Rule |
|---:|---|---|
| 0 | `TENSOR_PAYLOAD` | Payload offset MUST be 4,096-byte aligned. Otherwise it is 64-byte aligned. |
| 1 | `HAS_BLOCK_CRCS` | `block_crc_kind` MUST be 1 and the table MUST be present. |

When bit 1 is clear, `block_crc_kind` and `block_crc_table_offset` MUST both be
zero. When it is set, the table begins immediately after the chunk header. The
payload begins at the first applicable alignment boundary after the complete
table. Without a table, it begins at the first applicable alignment boundary
after the header. All bytes between header/table and payload are zero.

For `STORED`, stored and uncompressed sizes MUST be equal. For `ZSTD`, the
stored bytes are one frame as specified in section 1, and decoding MUST produce
exactly `uncompressed_bytes`.

An index live row describes the occupied span
`[header_offset, payload_offset + stored_bytes)`. Occupied spans of distinct
rows MUST NOT overlap. The whole span MUST end no later than the current
commit's `index_offset`. Per-generation alignment between described spans is
not part of a row and is ignored.

Tensor-bearing payloads are aligned to 4 KiB for portable page IO. On Windows,
a mapped view MUST start at
`floor(payload_offset / 65536) * 65536`; consumers add
`payload_offset - mapped_view_offset` to the returned base pointer. Mapping
directly from a merely 4-KiB-aligned payload is invalid on Windows. The mapping
length uses checked arithmetic and includes that pointer adjustment.

## 7. Per-block CRC table

Version 1 uses 4 MiB (`4,194,304`) stored-byte blocks. This bounds a lazy read's
extra integrity work while costing only 1 KiB of CRC entries per GiB of
payload. A table is mandatory when `stored_bytes >= 1 GiB`, and optional below
that threshold. A present table always covers the entire stored payload,
including a shorter final block.

The table header is 64 bytes, immediately followed by `block_count` u32 CRCs in
payload order.

| Offset | Size | Encoding | Field | Required value or rule |
|---:|---:|---|---|---|
| 0 | 8 | bytes | `magic` | ASCII `LFSBCRC` followed by `00`. |
| 8 | 2 | u16 | `table_version` | `1`. |
| 10 | 2 | u16 | `header_bytes` | `64`. |
| 12 | 2 | u16 | `entry_bytes` | `4`. |
| 14 | 2 | u16 enum | `crc_algorithm` | `1` (CRC32c). |
| 16 | 4 | u32 | `block_size` | `4,194,304`. |
| 20 | 4 | zero | `reserved_0` | Zero. |
| 24 | 8 | u64 | `payload_offset` | Equal to the owning row. |
| 32 | 8 | u64 | `stored_bytes` | Equal to the owning row. |
| 40 | 8 | u64 | `block_count` | `ceil(stored_bytes / block_size)` and at least 1. |
| 48 | 4 | u32 | `entries_crc32c` | CRC32c over exactly `4 * block_count` entry bytes. |
| 52 | 8 | zero | `reserved_1` | Zero. |
| 60 | 4 | u32 | `table_header_crc32c` | CRC32c `[0,60)`. |

Entry `i` is the CRC32c of stored payload interval
`[i*block_size, min(stored_bytes, (i+1)*block_size))`. The table header and
entries are metadata and are fully validated during open. Consumed blocks are
validated before use. `verify` and a production background sweep validate all
entries and the whole-payload CRC.

## 8. Complete index

The commit points directly to a stored index blob. `STORED` means the blob is
the uncompressed byte sequence below. `ZSTD` means decoding the one frame gives
that sequence. Both commit CRCs are mandatory even in stored mode (and then
usually equal). Production writers MUST use `ZSTD` for the index. `STORED` is a
conformance/test encoding retained so deterministic golden fixtures can run in
a stdlib-only environment; readers MUST accept both encodings.

### 8.1 Uncompressed index header: 64 bytes

| Offset | Size | Encoding | Field | Required value or rule |
|---:|---:|---|---|---|
| 0 | 8 | bytes | `magic` | ASCII `LFSINDEX`. |
| 8 | 2 | u16 | `index_version` | `1`. |
| 10 | 2 | u16 | `header_bytes` | `64`. |
| 12 | 2 | u16 | `row_bytes` | `96`. |
| 14 | 2 | u16 enum | `sort_order` | `1`, unsigned bytewise `{fourcc, instance_uuid}`. |
| 16 | 8 | u64 | `row_count` | Number of following rows. |
| 24 | 8 | u64 | `generation` | Equal to commit. |
| 32 | 16 | UUID | `commit_uuid` | Equal to commit. |
| 48 | 4 | u32 bitmap | `index_flags` | Bit 0: tombstone present; bit 1: base ref present. Exactly derived from rows. |
| 52 | 12 | zero | `reserved` | Zero. |

The decoded length MUST equal `64 + row_count * 96` exactly.

### 8.2 Index row: 96 bytes

| Offset | Size | Encoding | Field | Required value or rule |
|---:|---:|---|---|---|
| 0 | 4 | fourcc | `fourcc` | First key component. |
| 4 | 2 | u16 | `chunk_version` | Echo or row-kind sentinel. |
| 6 | 1 | u8 enum | `row_kind` | 0 live, 1 tombstone, 2 sidecar base reference. |
| 7 | 1 | u8 enum | `compression` | Echo or zero sentinel. |
| 8 | 4 | u32 | `chunk_flags` | Echo or zero sentinel. |
| 12 | 4 | zero | `reserved_0` | Zero. |
| 16 | 16 | UUID | `instance_uuid` | Second key component; non-null. |
| 32 | 8 | u64 | `header_offset` | Live physical offset; zero otherwise. |
| 40 | 8 | u64 | `payload_offset` | Live physical offset; zero otherwise. |
| 48 | 8 | u64 | `stored_bytes` | Header echo, base echo, or zero sentinel. |
| 56 | 8 | u64 | `uncompressed_bytes` | Header echo, base echo, or zero sentinel. |
| 64 | 8 | u64 | `source_generation` | Physical source generation, or deletion generation. |
| 72 | 4 | u32 | `payload_crc32c` | Header/base echo, or zero sentinel. |
| 76 | 4 | u32 | `header_crc32c` | Header/base echo, or zero sentinel. |
| 80 | 16 | zero | `reserved_1` | Zero. |

Rows MUST be in strictly increasing unsigned bytewise order by the 20-byte key
`fourcc || instance_uuid`. Therefore duplicate keys are invalid regardless of
row kind. Fourcc and UUID are never locale- or text-sorted.

A master index is the complete logical table for that commit: one row for each
live chunk plus any retained tombstones. Unchanged live rows carry their prior
same-file physical offsets and `source_generation`; changed rows point to the
new generation. There is at most one live row for a key.

Live rows (`row_kind=0`) require `chunk_version>=1`, a known compression enum,
allowed chunk flags, nonzero aligned offsets, and
`1 <= source_generation <= commit.generation`. Every row field MUST echo its
physical chunk header. Open validates the header CRC, header fields, table
metadata, bounds, and overlap. Its occupied span MUST lie in that source
generation's payload region: from the prior generation's committed end (or
65,536 for generation 1) through, but not including, the source generation's
index. Whole-payload CRC validation may remain lazy.

A tombstone (`row_kind=1`) is encoded exactly as follows: `chunk_version=0`,
`compression=0`, `chunk_flags=0`, both offsets and both sizes zero, both CRCs
zero, and `1 <= source_generation <= commit.generation`. The key identifies the
deleted object and `source_generation` is the generation that recorded the
deletion. Tombstones have no physical span. Compaction MAY discard tombstones
that no longer serve an external merge/recovery policy.

A base reference (`row_kind=2`) is valid only in a sidecar. It identifies an
unchanged live row in the bound master commit by logical key, never by physical
offset. Both offset fields are zero. Version, compression, flags, sizes,
source generation, and CRC echoes MUST exactly match the bound master's live
row during recovery validation. A master containing a base reference is
corrupt.

### 8.3 Sidecar completeness

The sidecar index is a complete logical overlay on its one explicit base:

1. Every live key in the bound master index appears exactly once in the
   sidecar as a live replacement, exact base reference, or tombstone.
2. A sidecar may add new keys only as live rows.
3. Every base reference resolves to a live base key and all echoes match.
4. A tombstone for a key absent from the base is invalid.
5. Sidecar live spans are wholly inside the sidecar; master offsets are never
   copied into them.

This representation makes completeness mechanically checkable and preserves
the plan's logical-key rule without introducing cross-file physical pointers.

## 9. Autosave sidecar binding and recovery predicate

An autosave sidecar uses the same superblock, heads, chunks, index, and commit
grammar. It is built as a complete one-generation temporary file and atomically
replaces `project.licht.autosave`. Its binding lives in superblock offsets
96-143 and is echoed by the autosave commit as specified above.

The recovery predicate is exact and admits no discretion:

> Offer recovery iff `sidecar.project_uuid == master.project_uuid` and
> `sidecar.base_explicit_commit_uuid == master.head.commit_uuid` and the
> sidecar is complete and CRC-valid.

All pathname candidates (sidecar, temp, and backup) are parsed and validated.
Among candidates satisfying the predicate, recovery chooses the unique highest
`autosave_sequence`. Two distinct valid candidates tied at the highest sequence
are an ambiguity and recovery MUST stop for explicit user choice. Candidates
whose base differs from the current master head are stale: ignore and delete
them, never offer them. No two sidecars are merged.

An explicit save deletes the sidecar only after the master head is durable.
Compaction merges or discards the sidecar before making its base UUID
unreachable.

### 9.1 Recovery decline policy (owner decision 2026-07-31)

Declining recovery never deletes a valid bound sidecar. The application opens the master head,
suppresses re-prompting for that same sidecar within the session, and leaves the file in place;
it is deleted only when it becomes stale under the standard predicate — that is, after the next
durable explicit master commit. Interactive decline and a headless open without recovery are
therefore identical in end state.

## 10. Writer commit state machine

```text
UNLOCKED
   | acquire exclusive sibling lock and validate current head
   v
LOCKED_OLD_AUTHORITY -- assign IDs --> PLANNED -- preflight --> READY
   |                                                        |
   |                                              append/finalize chunks
   |                                                        v
   |                                                  CHUNKS_WRITTEN
   |                                                        |
   |                                                   append index
   |                                                        v
   |                                                   INDEX_WRITTEN
   |                                                        |
   |                                                   append commit
   |                                                        v
   |                                                   COMMIT_WRITTEN
   |                                                        |
   |                                                flush append ranges
   |                                                        v
   |                                                   DATA_DURABLE
   |                                                        |
   |                                     one aligned inactive-head write
   |                                                        v
   |                                                    HEAD_WRITTEN
   |                                                        |
   |                                                  flush head/file
   |                                                        v
   +----------------------------------------------- NEW_HEAD_DURABLE
                                                            |
                                                     publish/report success
                                                            v
                                                        COMMITTED
```

The ten normative steps and every crash boundary are:

| Boundary | Last completed action | Crash-visible result |
|---:|---|---|
| 0 | Before/across lock acquisition | Existing valid head only. No write is allowed without the held OS lock. |
| 1 | Current head fully validated | Existing valid head only. |
| 2 | Generation, commit UUID, and snapshot UUID assigned in memory | Existing valid head only. IDs need not be reused. |
| 3 | Disk preflight completed | Existing valid head only. Preflight is not a durability promise. |
| 4 | During or after dirty chunk appends | Existing valid head only. Partial/final chunks are orphan tail; a row is formed only after its stored CRC is final. |
| 5 | Complete index appended | Existing valid head only. Index is orphan tail. |
| 6 | Commit record appended | Existing valid head only. Commit and prior bytes are still unpublished orphan tail. |
| 7 | First `fdatasync`/`FlushFileBuffers` completed | Existing valid head only; a durable but unauthoritative generation exists. Repair may find it but never auto-promotes it. |
| 8 | Inactive head write issued/completed, second flush not complete | Old head, or new head if the platform actually made the complete slot durable. A torn/failed new slot is invalid, so open falls back with warning. |
| 9 | Second flush completed | New head is authoritative and must open. The old slot remains a valid fallback. |
| 10 | Success reported | New head. User-visible success is forbidden before boundary 9. |

The durability total order is:

```text
all indexed payload bytes durable with final CRCs
    < complete index durable
    < commit record durable
    < first flush complete
    < inactive head write
    < second flush complete
    < publication / success report
```

No head may reference an index whose payload bytes are not durable.

An index row may reuse a prior physical span only with a positive clean proof
bound to that exact content's snapshot or mutation epoch. No proof means the
payload is rewritten. Explicit save treats every matrix-owned chapter as dirty
unless an audited API supplies the proof. Clearing a dirty flag without such a
proof MUST NOT cause reuse.

The write cursor starts exactly at the active head's
`committed_file_end`—never at physical EOF. Under the writer lock, an orphan
tail is either truncated to that cursor before any append begins or left
untouched until compaction. It MUST NOT be truncated mid-append, used as an
authority, or promoted by normal open/repair without explicit user approval.

## 11. Reader open state machine

```text
START
  -> read exactly 256 bytes; validate superblock grammar and CRC
  -> read slots A and B from superblock geometry
  -> for each nonblank slot independently:
       head CRC and fixed fields
       project/file UUID binding and committed bounds
       commit CRC, head echo, kind, identity, and version/capability fields
       recursive same-file parent linkage
       stored index bounds and CRC
       decode index; decoded size and CRC
       index header, strict key ordering, duplicate rejection
       every live-row header CRC and exact echoes
       block-table metadata CRCs
       every described span subset of [65536, committed_file_end)
       every span ends before current index; no overlap
  -> resolve structurally valid heads by head_sequence only
  -> apply selected-head reader-version and required-reader-capability gates
       supported -> OPEN
       selected unsupported and no supported rollback conflict
         -> UNSUPPORTED_NEWER (inspection only)
       greater-sequence selected unsupported while an older head is supported
         -> HARD_FAIL (never roll back)
  -> pin selected committed_file_end; never read append data beyond it
  -> optionally verify payload CRCs lazily/background
```

Validation order is security-relevant: enclosing CRC and bounds checks happen
before interpreting variable counts or allocating decoded sizes. A parser MUST
cap/decode to `index_uncompressed_bytes`, checked against row-count arithmetic.
Index, block-CRC tables, and non-payload (JSON/small) materialized chunks are
implementation-capped at 512 MiB; `TENSOR_PAYLOAD` and lazy-binary (`CKPT`/`PPIS`,
including `ByteShuffleZstd`) payload-class chunks raise that materialize bound
to 16 GiB, remain `size_t`-addressable, and MUST still cross-check the declared
`uncompressed_bytes` against the zstd frame content size before allocate.
It MUST NOT scan EOF for a replacement head.

Head resolution is exact. In this list, “structurally valid” means that all
grammar, identity, CRC, lineage, index, row-header, bounds, duplicate, and
overlap checks above passed; reader compatibility is a separate gate:

* No structurally valid heads because the heads are blank or corrupt: enter
  explicit repair-only state. Do not scan automatically.
* One structurally valid head: select it. A nonblank invalid peer produces a visible
  recovery warning; an all-zero uninitialized peer does not.
* Two structurally valid heads with different sequences: select the greater sequence.
  `generation`, offsets, wallclock, and UUID lexical order are not tiebreakers.
* Equal sequences and different commit UUIDs: split-brain, hard fail loudly.
* Equal sequences and the same commit UUID but different commit CRC echoes:
  corruption, hard fail loudly.
* Equal sequences, same commit UUID/CRC echo, and identical authority tuple
  `{generation, commit_offset, commit_bytes, committed_file_end,
  project_uuid, file_uuid}`: duplicate-slot write; accept either and emit a
  diagnostic. Slot IDs and therefore head CRC values naturally differ.
* Equal sequences and same commit UUID/CRC echo but any other authority field
  differs: corruption, hard fail loudly.
* After structural authority is selected, if it requires a
  `min_reader_version` above the implementation or any unknown required-reader
  capability, and no older structurally valid head is supported, terminate as
  `unsupported_newer`. This is a valid future authority, not corruption and
  not repairable. Read-only inspection of its already validated superblock,
  heads, commit, and index is permitted; semantic open is refused.
* If the greater-`head_sequence` authority is `unsupported_newer` while an
  older structurally valid head is reader-supported, hard fail loudly. A
  reader MUST NOT silently or explicitly roll back to that older generation.

If the greater-sequence slot is invalid and an older slot validates, open the
older generation and surface a recovery warning that names the rejected slot
and byte-precise reason. “Invalid” here means structural corruption or an
incomplete/torn publication; it never includes a structurally valid future
version or required-reader capability. This fallback is never silent. A payload corruption
found later by lazy verification leaves metadata inspectable but marks that
chunk unusable; production UI hydration must not install partial state.

### 11.1 Chapter and cross-chapter validation

Semantic hydration is transactional: a truncated payload, a JSON type mismatch,
an out-of-range enum, an excessive count, or a non-finite value produces a
typed error before any partial project state is installed. The following 1.0
decisions are normative:

* Every `SELM` node UUID must exist in `SCNG`, and every selected Gaussian index
  must be smaller than the bound splat count; otherwise hydration refuses the
  selection before slicing.
* A version-1 `SELM` slice uses encoding `1` (`RawU8`): the slice data is exactly
  `element_count` bytes, with one selection-group ID per element, and
  `data_bytes == element_count`. Production writers store the enclosing `SELM`
  chunk with `ZSTD` compression. Encoding value `2` was assigned to the
  experimental `DeltaBitpack` codec and withdrawn before publication; it is
  permanently reserved. Writers MUST NOT emit it, and readers MUST reject it
  with a typed data-loss error rather than interpreting its bytes.
* A `REFS` content hash is authoritative. A matching hash accepts and refreshes
  changed size/mtime cache hints. A mismatch leaves the reference unresolved
  and requests an explicit relink. A missing file retains a placeholder row.
* For a resumed training node, `CKPT.iteration` is the active trainer iteration
  even when it differs from a `PRMS` strategy limit. `PRMS` still retains every
  inactive strategy session. An unknown active strategy is refused. Any `CKPT`
  legacy-payload window whose base plus length exceeds its stored payload is a
  bounds error.
* A `GUIL` area tree must have exactly one root, no missing parent, and no cycle.
  An unknown space type without retained opaque state is invalid.
* A `SEQR` keyframe naming a camera absent from `SCNG`/`VIEW` is refused with a
  typed diagnostic. A missing `VIEW.active_camera_uuid` instead opens in the
  documented `MissingActiveCamera` degraded state and selects a default camera.
* `SCNG` rejects duplicate node UUIDs and parent cycles. A training node bound
  to `CKPT` must not also have a live `SPLT` authority. Standalone `PPIS` is for
  non-checkpoint ownership; a damaged standalone payload refuses hydration.
* `METR` iterations are monotonic and all persisted values are finite. A
  violation rejects the chapter without sorting or dropping samples.
* A `THMB` locator must agree with the indexed `THMB` stored span and that row
  must use stored compression. Disagreement invalidates the publishing head
  slot; an older valid slot may be selected only with a visible warning.
* A known fourcc at an unsupported chunk version is retained as opaque. Any
  save that cannot declare capability bit 5 refuses rather than dropping it.
  Unknown JSON members and UUID arrays likewise require capability bit 6.
* Undeclared project/vendor capability bits 112–127 are inspectable but
  write-unsafe. A sidecar base-reference key absent from its master is invalid.
  A tombstone and live row sharing one key is duplicate-key corruption.

## 12. Inspect, extract, and verify behavior

The independent reference CLI performs the complete metadata open sequence for
all commands.

* `inspect` prints the superblock, both head states, selected commit, index, and
  row/header metadata. For `unsupported_newer`, it prints those structurally
  validated fields plus the compatibility diagnostic and returns that terminal
  state; it does not semantically hydrate chunks. It never reads orphan-tail
  contents.
* `extract` selects exactly one live row by fourcc and UUID, validates its
  stored payload CRC and all block CRCs, then writes decoded bytes by default
  (or stored bytes when explicitly requested). Tombstones and base references
  have no extractable bytes in that file.
* `verify` sweeps every selected-index live payload, validates every per-block
  entry, whole stored CRC, and zstd decoded size. It reports the selected
  generation and recovery warnings.

Every reference-parser diagnostic names the absolute file offset, field,
expected value/range, and observed value. The terminal taxonomy is:

* `open_gen_N`: generation N is structurally valid and reader-compatible.
* `unsupported_newer`: all structurally valid heads require a newer reader
  version or an unknown required-reader capability. The selected authority
  remains inspectable, is not corruption, and repair cannot change it.
* `repair_only`: no structural authority can be selected; explicit recovery
  tooling may scan, but normal open does not.
* `hard_fail`: bootstrap corruption or contradictory authorities, including a
  greater-sequence unsupported authority paired with an older supported one.

`extract` and `verify` require `open_gen_N`; they refuse
`unsupported_newer` because interpreting or validating future payload
semantics is outside this reader's declared capability set.

For known JSON chapter payloads, duplicate object keys are invalid rather than
first- or last-wins. A semantic reader MUST reject them consistently at every
nesting level. A decoded JSON chapter greater than 67,108,864 bytes, or nesting
that exceeds an implementation's bounded parser stack, MUST fail with a typed
resource-exhaustion or data-loss result; it MUST NOT be silently dropped.

For `METR`, sample iterations are non-negative and monotonically
non-decreasing, and every persisted floating-point value is finite. A semantic
reader rejects the entire `METR` chapter on violation; it never sorts, filters,
or silently removes hostile samples.

## 13. Versioning and capability policy

`format_major` changes only when the bootstrap/fixed metadata grammar becomes
incompatible. A reader rejects an unknown major. Within major 1, fixed offsets
and the zero-reserved rule do not change; `format_minor` announces additive
semantics expressed through chunks and capabilities. A 1.x reader may inspect
a greater minor when the selected commit's minimum reader tuple and required
reader bitmap are supported.

`min_reader_version` is the oldest implementation that can interpret the
selected generation. `min_safe_writer_version` is the oldest implementation
that may modify it without loss. A reader below the former refuses that head.
A writer below the latter opens read-only. Unknown required-reader bits refuse
semantic open; unknown required-writer bits permit read-only open but forbid
write/compaction. A future head that is structurally valid but unsupported must
not be silently replaced by an older supported head.

Reader and writer compatibility are independent classifications. Once
`open_gen_N` is selected, an implementation compares its declared writer
version and writer-capability bitmap with the selected commit. If either
`min_safe_writer_version` is greater or any required-writer bit is missing,
inspection remains valid but the file is `write-unsafe`; every save,
compaction, or other declared-safe mutation MUST be refused before writing any
byte. This is not `unsupported_newer`, because the reader gates passed.

Capability bits share this 128-bit registry:

| Bit | Name | Reader/writer meaning |
|---:|---|---|
| 0 | `INDEX_ZSTD_V1` | Reader can decode the exact zstd index framing. |
| 1 | `CHUNK_ZSTD_V1` | Reader/semantic writer handles zstd chunk payloads. |
| 2 | `BLOCK_CRC32C_V1` | Reader validates section 7 tables; writer preserves/regenerates them. |
| 3 | `INDEX_TOMBSTONES_V1` | Reader/writer understands row kind 1. |
| 4 | `SIDECAR_OVERLAY_V1` | Recovery implementation understands binding and row kind 2. |
| 5 | `OPAQUE_CHUNK_PRESERVATION` | Writer carries unknown live chunks byte-for-byte when declared safe. |
| 6 | `RETAINED_JSON_FIELDS` | Semantic writer preserves unknown JSON object and UUID-addressed array fields. |
| 7 | `CLEAN_PROOF_REUSE` | Writer enforces snapshot/mutation-epoch proof before reusing spans. |
| 8 | `CHUNK_BYTESHUFFLE_ZSTD_V1` | Reader/writer handles f32-word byte-plane prefilter + zstd (`Compression::ByteShuffleZstd` = 2). Size-preserving filter; payload size must be a multiple of 4. Writers fall back to plain `ZSTD` when size `% 4 != 0`. |
| 9-31 | Core container | Reserved for assignment by a later published format revision. |
| 32-63 | Built-in chapters | Cross-chapter semantic features. |
| 64-95 | Embedded protocols | LFKP/LFSP/LFAD and successors. |
| 96-111 | Experimental | MUST be zero in released files. |
| 112-127 | Vendor/private | Requires a namespaced declaration in `PROJ`; unknown bits force read-only. |

A commit sets only capabilities actually required by that generation. For
example, a stored index does not set bit 0; a generation containing a block
table sets bit 2; an autosave sidecar sets bit 4.

`chunk_version` starts at 1 independently for each fourcc. Any semantic change
to that chunk's payload increments it. Unknown fourcc values and supported
container metadata remain opaque. A known fourcc with a newer unsupported
chunk version is also opaque, never partially decoded and reserialized. New
state normally receives a new fourcc. A cross-cutting semantic change also
receives a capability bit. Embedded formats retain and bump their own internal
magic/version; their bytes are opaque to the container.

## 14. Spec decisions log

These decisions freeze details on which the plan was intentionally silent:

1. UUIDs use RFC 4122 network-order octets; all scalar integers are LE.
2. Superblock role and all sidecar binding fields occupy offsets 20 and 96-143;
   masters zero the binding range.
3. Fixed-header CRC fields are the final four bytes and cover the exact prefix;
   stored and decoded index CRCs live in the commit.
4. Head duplicate detection compares commit CRC echoes and a defined authority
   tuple because slot IDs make the two head CRC values inherently different.
5. Generations are scoped to `file_uuid`; roots after compaction and sidecar
   replacement are generation 1. Head sequence is likewise incarnation-local.
6. Parent linkage is same-file, recursively CRC-validated, and exactly
   generation-minus-one. Sidecar-to-master ancestry uses the explicit-ancestor
   UUID, never a parent offset.
7. Index header and rows are fixed at 64 and 96 bytes. Sort is raw unsigned
   `{fourcc, UUID-byte-sequence}` and duplicate keys are always corruption.
8. Tombstones are explicit row kind 1 with no physical span and a deletion
   generation. Sidecar-only base references are row kind 2 with zero physical
   offsets and exact base metadata echoes.
9. The sidecar enumerates every base live key as replacement, base reference,
   or tombstone, making "complete relative to base" testable without inherited
   physical offsets.
10. Chunk flags assign tensor alignment and block-table presence. Non-tensor
    payloads are 64-byte aligned; tensor payloads are 4,096-byte aligned.
11. Block tables cover stored bytes in 4 MiB blocks, are mandatory at 1 GiB,
    and cost 1 KiB/GiB. Their header and entry array have separate CRCs.
12. The whole stored chunk payload owns the canonical payload CRC. Zstd decode
    size is checked separately.
13. Index and chunk zstd encodings are one standard frame with content size,
    no dictionary, no skippable/multiple frames, and no trailing bytes.
14. Append/index/commit starts are 64-byte aligned; padding inside described
    structures is zero. Bootstrap gaps are zeroed by writers but ignored by
    readers to leave bootstrap expansion space.
15. Current format version is 1.0. Fixed major-1 offsets never consume reserved
    bytes; additive changes use chunks, versions, and capability bits.
16. The capability registry allocates core, chapter, embedded, experimental,
    and vendor ranges and assigns bits 0 through 7 above.
17. Missing optional Python `zstandard` support does not alter the grammar.
18. (2026-07-30, pre-release amendment — part of the 1.0 grammar, so item 15 is
    not violated) The head slot carries a 16-byte preview locator at offset 112
    (§4.1) so foreign readers can thumbnail a project from fixed offsets alone;
    an all-zero locator is valid, so fixtures produced before the amendment
    remain byte-identical. Evidence: Krita's `mergedimage.png` foreign-reader
    surface — a post-release fourcc reservation could never deliver this
    property precisely because of item 15.
    Golden fixtures use `STORED` indexes for deterministic byte comparisons;
    production indexes use `ZSTD`, and the production C++ reader validates both
    encodings in the project-container tests.
18. The reference outcome taxonomy is `open_gen_N`, `unsupported_newer`,
    `hard_fail`, or `repair_only`; `unsupported_newer` is the inspectable,
    non-corrupt, non-repairable terminal for a structurally valid authority
    beyond the reader's declared version/capabilities. Sidecar recovery
    additionally reports offer, stale-delete, invalid, or ambiguous.
19. Fixed marker bytes are `LFSHEAD\0`, `LFSCOMIT`, `LFSINDEX`, and
    `LFSBCRC\0`. Enum values start at zero for storage/CRC absence, one for the
    first active encoding, one through four for commit kinds, and zero through
    two for index row kinds.
20. Version components are u16; roles, flags, and fixed enum slots use the
    tabled u8/u16/u32 widths; offsets, sizes, generations, sequences, and times
    are u64. Times are unsigned Unix nanoseconds.
21. Fourcc bytes are uppercase ASCII letters/digits, instance UUIDs are
    non-null, and the fourcc itself is the chunk-header discriminator—there is
    no separate chunk magic.
22. A block-table reference is an absolute u64 file offset and, when present,
    points exactly to the byte after its owning 64-byte chunk header. Its end is
    followed only by zero alignment padding before the payload.
23. An index is a directly referenced stored blob, not a chunk and not a
    footer. Every live span for the selected commit ends before that commit's
    index; the zero-padded aligned commit follows the index blob.

There are no unresolved byte-grammar questions in version 1.0. Policy choices
such as the production zstd compression level and compaction scheduling do not
change bytes and remain implementation decisions.
