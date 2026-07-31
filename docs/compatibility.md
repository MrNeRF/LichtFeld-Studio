# `.licht` compatibility register

This register is normative for released `.licht` grammars. A row is immutable after
publication: release-corpus entries are append-only, and bytes already named by a row may
not be regenerated under that row's writer SHA.

P8 authority/base writer SHA: `8ca8028e6214b1f424c373b24d479cd90ff2e918`.

## Published-grammar register

| Spec | Release manifest root SHA-256 | Tagged parser tree SHA-256 | Reader capabilities | Minimum reader | Minimum safe writer | Writer SHA | Status |
|---|---|---|---|---|---|---|---|
| 1.0 | `690a820b806dc8e633e8aaacd7b381a4a9f25963160e5e92e3b490217bc63326` | `9c951ac2aac079dbc7569242d26432de0f747c8510aad0f3e4088b2034e1f96d` | bits 0–7 (`INDEX_ZSTD_V1`, `CHUNK_ZSTD_V1`, `BLOCK_CRC32C_V1`, `INDEX_TOMBSTONES_V1`, `SIDECAR_OVERLAY_V1`, `OPAQUE_CHUNK_PRESERVATION`, `RETAINED_JSON_FIELDS`, `CLEAN_PROOF_REUSE`) | 1.0 | 1.0 | `8ca8028e6214b1f424c373b24d479cd90ff2e918` | Published grammar candidate; content-locked by P8 |

Published-grammar register line: `licht/1.0 manifest=690a820b806dc8e633e8aaacd7b381a4a9f25963160e5e92e3b490217bc63326 tagged_tree=9c951ac2aac079dbc7569242d26432de0f747c8510aad0f3e4088b2034e1f96d reader=1.0 writer=1.0 caps=0-7 writer_sha=8ca8028e6214b1f424c373b24d479cd90ff2e918`.

The candidate becomes **Published** only after all three events occur: the machinery and
locked artifacts merge to the main branch; a release tag names this exact manifest root and
tagged-parser tree hash; and the release notes link that tag back to this immutable row. P8
does not create that tag. Descendant commits run the gates by content, so HEAD identity is
informational and never substitutes for either hash.

The tagged 1.0 parser is scoped to structural container validation. The live parser may add
semantic JSON-chapter checks without invalidating that structural snapshot. Canonical
dual-parser comparison excludes only `Container.path`, `Container.warnings`,
`WriteCompatibility.reasons`, and `HeadAttempt.error` diagnostic text; superblock, selected
head, commit, index rows, normalized offsets, CRC fields, and per-chunk verification results
must agree exactly. A live validation change that alters container structure or the accepted
1.0 structural grammar requires a new tagged snapshot and a new register row; it may not
silently rewrite this freeze.

Fixture update note: P8 creates the initial production-writer release corpus from the six
required producer paths plus the foreign-preview audit case. Existing container goldens
remain byte-identical.

Fixture update note: P8 Round 2 relocks `recovered-commit.licht` through the real bound-
sidecar recovery materialization path. The other reproducible production rows remain
byte-identical, and the explicitly non-reproducible headless-training row remains unchanged.

## Minimum-safe-writer and capability bump checklist

Every item below is mandatory before a change to `min_safe_writer_version`,
`required_reader_capabilities`, or `required_writer_capabilities` merges. The
`scripts/licht_compat_policy.py` diff ratchet rejects a declaration change unless the same
diff changes a cited compatibility test symbol.

1. Add the old-reader/new-writer classification cell to
   `P8CompatibilityTest.V10CorpusAndSyntheticNewerClassificationRefusePayloadAccess` and
   the tagged-parser job `scripts/licht_compat_matrix.py`.
2. Add the prohibited-writer producer cells to
   `P8CompatibilityTest.ProhibitedOldWriterMatrixHasNoWriteEffectForEveryProducer`, including
   byte identity, generation identity, and lock-release assertions.
3. Append a production emission and SHA-locked manifest row under
   `tools/licht_inspect/fixtures/release_corpus/`; keep
   `P8CompatibilityTest.ProductionReleaseCorpusEmissionPathsOpenAtFormatOneZero` green.
4. Assign the semantic change to an existing capability bit or a new `chunk_version` in
   `docs/licht_format_spec.md` §13. A new registry bit requires a post-1.0 owner decision;
   P8 itself assigns none.
5. Append a new immutable row to this register and keep
   `P8CompatibilityTest.OpaqueAndRetainedJsonSurviveSafeAppendAndCompaction` green when the
   change affects opaque or retained state.
6. Classify any live parser validation change as semantic-only or structural. Semantic-only
   checks keep the 1.0 structural freeze and must preserve
   `P8CompatibilityTest.ReleaseCorpusProductionPathsReproduceManifestSha256`; structural
   changes require a new tagged snapshot, tree hash, corpus row, and compatibility-register
   row, with `scripts/licht_compat_matrix.py` proving the two scopes explicitly.

An undeclarable semantic change—one that cannot be represented by a capability requirement,
minimum version, or chunk version—is a release blocker. It must not be smuggled into unchanged
1.0 bytes.

## Release-fixture procedure

`python tools/licht_inspect/make_fixtures.py --check` is the normal operation and performs no
writes. The exceptional `--write` path requires both `RELEASE_FIXTURE_UPDATE=1` and an exact,
non-empty `--compatibility-note` already present in this document. Existing manifest rows and
fixture SHA-256 values are immutable; a release only appends new names.

## LTS converter escape hatch

If a future 2.0 reader intentionally drops direct 1.x hydration, users install or run the
last 1.x LTS converter, open the original file read-only, verify it, and write a new 2.0 file
to a different path. The converter never overwrites the only source, never asks an old writer
to mutate a newer file, and emits a conversion report containing source/destination SHA-256,
the selected source commit UUID, dropped/degraded fields, and the target grammar row. The 1.x
release corpus and frozen parser remain downloadable so that conversion is reproducible.

## Legacy import precedence

Legacy `checkpoint.resume`, standalone `.ppisp`, and `layout.json` files are import-only inputs;
the application never rewrites them. A checkpoint becomes the authoritative `CKPT` payload and
its pending parameters become `PRMS`; training-owned PPISP state stays inside `CKPT`, while a
standalone non-checkpoint PPISP import uses `PPIS`. Live `.rad` sources remain external `REFS`
rows and retain their relink fingerprint. If a directory contains both `layout.json` and an
explicitly opened `.licht` project, the project's `GUIL` chapter is applied last and wins; the
legacy layout file remains unchanged.

## Platform recovery register

The S' release run executes all four named Linux cells. Windows is explicitly deferred here;
these are obligations, not passing stubs, and the Windows runner must not report success until
the P0d `ReplaceFileW`/`LockFileEx` recovery prototype is available.

| Windows test id | Status | Reason | Ownership pointer |
|---|---|---|---|
| `WIN-P8-D1-AUTOSAVE-KILLPOINTS` | DEFERRED | POSIX process-stop injection does not exercise `ReplaceFileW` durability. | P0d Windows append/replace prototype, `PROJECT_FORMAT_PLAN.md` §7 P0 |
| `WIN-P8-D1-APPEND-PUBLISH` | DEFERRED | Linux `fdatasync`/signal boundaries do not cover `FlushFileBuffers`. | P0d Windows append/replace prototype, `PROJECT_FORMAT_PLAN.md` §7 P0 |
| `WIN-P8-D1-ENOSPC-PARENT-CHILD` | DEFERRED | The isolated Linux tmpfs case has no Windows volume-quota harness yet. | P0d Windows disk-full prototype, `PROJECT_FORMAT_PLAN.md` §7 P0 |
| `WIN-P8-D1-COMPACTION-KILLPOINTS` | DEFERRED | Windows mapped-reader replacement must be exercised with native handles. | P0d `ReplaceFileW` 1175–1177 matrix, `PROJECT_FORMAT_PLAN.md` §7 P0 |
| `WIN-P8-D2-TWO-PROCESS-LOCK` | DEFERRED | POSIX `flock` coverage is not evidence for `LockFileEx`. | P0d Windows lock/mapped-reader prototype, `PROJECT_FORMAT_PLAN.md` §7 P0 |
