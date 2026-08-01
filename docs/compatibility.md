# `.licht` compatibility register

This register is normative for released `.licht` grammars. A row is immutable after
publication: locked production fixtures are append-only, and bytes already named by a row
may not be regenerated under that row's writer SHA.

P8 authority/base writer SHA: `8ca8028e6214b1f424c373b24d479cd90ff2e918`.
Current candidate producer base SHA: `bcb183ad2d9ee92cfcecb3a6b41a37da4fdaf6fb`.

## Published-grammar register

| Spec | Fixture manifest root SHA-256 | Reader capabilities | Minimum reader | Minimum safe writer | Writer SHA | Status |
|---|---|---|---|---|---|---|
| 1.0 | `43ef54b7c1a254905ac924a83f0f4026d0fde27e7d24de276591c0372d263fdb` | bits 0–7 (`INDEX_ZSTD_V1`, `CHUNK_ZSTD_V1`, `BLOCK_CRC32C_V1`, `INDEX_TOMBSTONES_V1`, `SIDECAR_OVERLAY_V1`, `OPAQUE_CHUNK_PRESERVATION`, `RETAINED_JSON_FIELDS`, `CLEAN_PROOF_REUSE`) | 1.0 | 1.0 | `bcb183ad2d9ee92cfcecb3a6b41a37da4fdaf6fb` | Published grammar candidate; content-locked by P8 and the pre-release SELM withdrawal |

Published-grammar register line: `licht/1.0 manifest=43ef54b7c1a254905ac924a83f0f4026d0fde27e7d24de276591c0372d263fdb reader=1.0 writer=1.0 caps=0-7 writer_sha=bcb183ad2d9ee92cfcecb3a6b41a37da4fdaf6fb`.

The candidate becomes **Published** only after all three events occur: the implementation,
C++ compatibility tests, and locked fixtures merge to the main branch; a release tag names
this exact manifest root; and the release notes link that tag back to this immutable row. P8
does not create that tag. HEAD identity is provenance only and never substitutes for the
fixture hashes.

The shipped verification surface is the production C++ reader/writer and its gtests.
`P8CompatibilityTest.ReleaseCorpusCppReaderOpensEveryLockedArtifact` opens every locked
production fixture; `P8CompatibilityTest.ReleaseCorpusProductionPathsReproduceManifestSha256`
re-emits every deterministic production path and compares its SHA-256; the container golden
test reproduces the lower-level fixtures byte-for-byte. Semantic chapter validation is pinned
by the `ProjectDocument` and chapter tests. A validation change that alters the accepted 1.0
grammar requires a new register row and new locked fixture; it may not silently rewrite this
row.

Fixture update note: P8 created the initial production-writer fixture set from the six required
producer paths plus the foreign-preview audit case. Existing container goldens remain
byte-identical. The fixtures live under `tests/fixtures/licht/` because only C++ tests consume
them.

Fixture update note: P8 Round 2 relocks `recovered-commit.licht` through the real bound-
sidecar recovery materialization path. The other reproducible production rows remain
byte-identical, and the explicitly non-reproducible headless-training row remains unchanged.

Fixture update note: before 1.0 publication, selection encoding value 2
(`DeltaBitpack`) was withdrawn and permanently reserved. The seven existing production
fixtures remain byte-identical because their `SELM` chapters contain no mask slices. The
append-only `selection-raw-u8.licht` row records a populated production `SELM` chapter using
encoding 1 (`RawU8`) and updates the candidate manifest root.

## Minimum-safe-writer and capability bump checklist

Every item below is mandatory before a change to `min_safe_writer_version`,
`required_reader_capabilities`, or `required_writer_capabilities` merges. The review must show
the cited C++ compatibility-test changes in the same diff.

1. Add the old-reader/new-writer classification cell to
   `P8CompatibilityTest.V10CorpusAndSyntheticNewerClassificationRefusePayloadAccess`.
2. Add the prohibited-writer producer cells to
   `P8CompatibilityTest.ProhibitedOldWriterMatrixHasNoWriteEffectForEveryProducer`, including
   byte identity, generation identity, and lock-release assertions.
3. Append a production emission and SHA-locked manifest row under
   `tests/fixtures/licht/release_corpus/`; keep
   `P8CompatibilityTest.ReleaseCorpusProductionPathsReproduceManifestSha256` green.
4. Assign the semantic change to an existing capability bit or a new `chunk_version` in
   `docs/licht_format_spec.md` §13. A new registry bit requires a post-1.0 owner decision;
   P8 itself assigns none.
5. Append a new immutable row to this register and keep
   `P8CompatibilityTest.OpaqueAndRetainedJsonSurviveSafeAppendAndCompaction` green when the
   change affects opaque or retained state.
6. Classify any validation change as semantic-only or structural. Semantic-only checks must
   preserve `P8CompatibilityTest.ReleaseCorpusProductionPathsReproduceManifestSha256`;
   structural changes require a new locked fixture, manifest root, compatibility-register row,
   and a corresponding C++ classification case.

An undeclarable semantic change—one that cannot be represented by a capability requirement,
minimum version, or chunk version—is a release blocker. It must not be smuggled into unchanged
1.0 bytes.

## Locked-fixture procedure

Existing manifest rows and fixture SHA-256 values are immutable; a release only appends new
names. A fixture update must add its compatibility note here, record its producing SHA in
`tests/fixtures/licht/release_corpus/manifest.json`, and extend the deterministic C++ emission
and open tests named above. No fixture-generation utility ships with the product tree.

## LTS converter escape hatch

If a future 2.0 reader intentionally drops direct 1.x hydration, users install or run the
last 1.x LTS converter, open the original file read-only, verify it, and write a new 2.0 file
to a different path. The converter never overwrites the only source, never asks an old writer
to mutate a newer file, and emits a conversion report containing source/destination SHA-256,
the selected source commit UUID, dropped/degraded fields, and the target grammar row. The 1.x
locked fixtures and the corresponding tagged product source remain available from the release
so conversion behavior can be reproduced.

## Legacy import precedence

Legacy `checkpoint.resume`, standalone `.ppisp`, and `layout.json` files are import-only inputs;
the application never rewrites them. A checkpoint becomes the authoritative `CKPT` payload and
its pending parameters become `PRMS`; training-owned PPISP state stays inside `CKPT`, while a
standalone non-checkpoint PPISP import uses `PPIS`. Live `.rad` sources remain external `REFS`
rows and retain their relink fingerprint. If a directory contains both `layout.json` and an
explicitly opened `.licht` project, the project's `GUIL` chapter is applied last and wins; the
legacy layout file remains unchanged.

## Platform recovery register

The Linux C++ tests execute all four named cells. Windows is explicitly deferred here; these
are obligations, not passing stubs, and no Windows test reports them as passing until native
`ReplaceFileW`, `FlushFileBuffers`, volume-exhaustion, and `LockFileEx` coverage exists.

| Windows test id | Status | Reason | Ownership pointer |
|---|---|---|---|
| `WIN-P8-D1-AUTOSAVE-KILLPOINTS` | DEFERRED | POSIX process-stop injection does not exercise `ReplaceFileW` durability. | Windows project-container test work |
| `WIN-P8-D1-APPEND-PUBLISH` | DEFERRED | Linux `fdatasync`/signal boundaries do not cover `FlushFileBuffers`. | Windows project-container test work |
| `WIN-P8-D1-ENOSPC-PARENT-CHILD` | DEFERRED | The isolated Linux tmpfs case has no Windows volume-quota harness yet. | Windows volume-exhaustion test work |
| `WIN-P8-D1-COMPACTION-KILLPOINTS` | DEFERRED | Windows mapped-reader replacement must be exercised with native handles. | Windows project-container test work |
| `WIN-P8-D2-TWO-PROCESS-LOCK` | DEFERRED | POSIX `flock` coverage is not evidence for `LockFileEx`. | Windows two-process lock test work |
