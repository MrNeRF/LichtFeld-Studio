# The `.licht` Project Format

`.licht` is LichtFeld Studio's project file. **One file is your whole session** — like a `.blend`.
Open `project.licht` and you are exactly where you left off: the trained model at its iteration,
the full scene graph, your selections, the panel layout, the split view, unsaved code-editor
buffers, the sequencer timeline, and the camera you were flying. Close and reopen — or copy the
file to another machine — and continue.

It replaces the old scattered files (`checkpoint.resume`, `.ppisp` sidecars, `layout.json`, and
runtime-only state) and is the **only** project format the app writes. Exports like `.ply`,
`.rad`, `.spz`, `.sog`, `.usdz`, and `.html` are separate one-way bakes, never project state.

## The shape of the file

```
project.licht
├─ superblock     magic bytes, project id, and the fixed offsets of the
│                 two head slots below
│
├─ head slot A    selected generation metadata + thumbnail locator (CRC32c)
├─ head slot B    alternate generation metadata + thumbnail locator (CRC32c)
│                 the 4 KiB slots are overwritten in place when publishing
│
├─ generation 1   [scene][selection][settings][checkpoint 9 GB][layout]
│                 + a table of contents
├─ generation 2   [scene][selection]            ← only what changed
│                 + a table of contents that points back at generation 1
│                   for every part it reused
└─ generation 3   ...            append-only chunk and index records
```

Large tensor and lazy-binary payloads use framed compression. The payload bytes begin with the
eight-byte `LFSZFRM\0` magic, a little-endian version/reserved pair, and a `u32` record count. A
fixed table then stores one little-endian `{stored_bytes, uncompressed_bytes}` `u64` pair per
record, followed by the concatenated independent Zstandard frames. Records split the serialized
byte stream at approximately 64 MiB boundaries. Readers validate each covering block before
dispatching that record to a worker, while the payload CRC is finalized over the complete stored
byte stream. ByteShuffle payloads frame the shuffled byte stream and unshuffle only after all records decode.

SPLT chapters use the `LFSPLT2\0` raw-tensor payload inside that framed stream.
The little-endian header is `magic[8]`, `version u16` (2), `reserved u16` (0),
`active_sh_degree u32`, `max_sh_degree u32`, `scene_scale f32`, `tensor_count u32`,
four reserved bytes, and `manifest_bytes u64`. It is followed by 64-byte descriptors
(`id u32`, `dtype u8`, `rank u8`, four `u64` dimensions, `offset u64`, `length u64`,
and reserved bytes), then a frozen-range count and `{start u64,count u64}` entries,
then contiguous tensor bytes. The manifest must cover the data exactly. CKPT/LFKP
payloads retain their separate format.

Moving a camera and pressing save writes a few kilobytes: a clean 9 GB checkpoint is
not copied again, because generation 2's table of contents just points back at
the bytes generation 1 already wrote. The preview thumbnail is a stored `THMB`
chunk; the head stores only its locator, which updates atomically on publish.

## Capability bits

Each generation declares required reader and writer capabilities; a reader refuses
requirements it does not implement. The canonical bit assignments are the
[`CapabilityBit` enum](../src/io/include/io/project_container.hpp), rather than a
second registry in this guide. Declaring an enum value does not enable support:
bit 10 (`TILESET_REFERENCES`) remains reserved and unsupported, while collision
volumes require bit 11 (`COLLISION_VOLUMES_V1`).

## Training and Edit Mode

A `CKPT` chapter is resumable project state only while `SCNG` identifies the
training-model node and binds that node to the checkpoint instance. Ordinary
scene splats do not retain optimizer or training-session state.

`PROJ.training_initialization` is optional. It stores `data_origin` (three finite
floats), optional `scene_center` (three finite floats), and optional
`init_reference_uuid` bound to a `REFS` row with key/kind `training_init`.
The path and fingerprint live only in `REFS`. Untrained GUI reopen restores these
values; a missing init file warns and falls back to the restored point cloud.
Headless dataset-project adoption uses the stored init unless `--init` overrides
it, and recomputes the origin and center on import. Older projects may omit the
object; the format version is unchanged.

Switching a completed or paused training session to Edit Mode is an explicit
conversion: the current trained model becomes an editable splat, the training
binding is removed from `SCNG`, and the formerly resumable `CKPT` is retired by
the next full project synchronization. Before releasing the trainer, the
application adopts any completed final training generation; if its writer is
still active, the Edit Mode transition is deferred instead of allowing stale
append authority into the editable session. Manual saves and non-training
autosaves therefore persist the editable scene without an orphan checkpoint.
Lightweight autosaves made while training is still active preserve the binding
and checkpoint so reopening the project can restore the paused resumable state.

## How it works

A custom chunked binary container, built around three facts: training checkpoints are huge,
crashes happen, and files outlive programs.

- **Append-only saves.** A save appends only the parts that changed as a new *generation*, then
  publishes it. Unchanged multi-GB payloads are never rewritten — Ctrl+S after a small tweak is a
  tiny append, not a 10 GB rewrite. Only the idle 4 KiB head slot is overwritten to publish.
- **Crash-safe.** Each publish writes validated head and commit metadata. On open, CRC and
  generation checks select a complete head; a half-written tail from an interrupted save is ignored.
- **Checksummed throughout.** Every record and payload carries a CRC32c to catch corruption.
- **Organized into chapters.** State is split into typed chapters (model, scene graph, parameters,
  layout, sequencer, camera, …); each chapter is the single source of truth for its fields.
- The `PROJ` chapter may optionally carry a `license` object with a non-empty `identifier` and an
  optional `notice`; omitted `license` means that no project license is declared.
- Deprecated: the `GUIL` panel registry's `active_tabs.scene_panel` (the former Scene/History
  tab). LichtFeld Studio no longer writes it and ignores it when reading; projects saved by
  earlier versions may still carry it.
- **Autosave & recovery.** A periodic autosave writes to a separate `<project>.licht.autosave`
  sidecar. Recovery validates that sidecar against the master head and can materialize it into a
  retained recovery session before the next durable save. Compaction and recovery publication
  may also replace the master file.
- **Compaction.** Because saves append, dead bytes build up over time; compaction rewrites the
  live generations into a fresh file and atomically swaps it in (run it yourself, or accept the
  suggestion around ~50% waste).

## Command-line opening and recovery

Use `-v project.licht` to open a project in the GUI. Headless training resumes
use `--headless --resume project.licht`; a complete autosave newer than the
master head is recovered automatically. Ambiguous recovery candidates remain
an error.

A project saved before any training carries no checkpoint and is a dataset
source instead: `--headless --data-path project.licht --output-path <dir>` trains
it from scratch with the dataset options stored in `PRMS`, reading images from
the dataset folder recorded in `REFS` or, when that folder is not reachable, from
the embedded dataset copy extracted to the per-user cache. A project that already
holds a checkpoint is rejected there and must be continued with `--resume`.

The current grammar is **1.1** on this development branch. Version 1.1 makes CKPT history
explicit: SCNG binds exactly one resumable checkpoint when a training node exists, while
additional live CKPT chapters are historical and are copied by compaction. The existing
Version major/minor fields and commit minimum-reader fields are the compatibility mechanism;
old 1.0 readers reject a 1.1 multi-checkpoint commit before validation with an unsupported
version error rather than reporting data loss. The framed payload layout is guarded by
reader/writer tests; no compatibility promise is made for pre-framed development files.

## User shader state

`SHDR` is an optional singleton JSON chapter keyed by the project UUID. It uses
the existing JSON chunk encoding and this payload (UUIDs shown are examples):

```json
{
  "version": 1, "api_version": 1, "contains_embedded_source": true,
  "assets": [{
    "uuid": "11111111-1111-4111-8111-111111111111",
    "name": "Identity", "language": "slang",
    "source": "import lfs; LfsElement effect(LfsElement e, LfsContext c) { return e; }",
    "parameters": {}
  }],
  "bindings": [{"target": "global", "shader_uuid": "11111111-1111-4111-8111-111111111111"}]
}
```

Node bindings additionally use `target:"node"` and `node_uuid`. There is no
SHDR reveal binding: `SEQR.timeline.reveal_clip` owns `effect:12` and `shader_id`.
All referenced assets belong to SHDR. Empty libraries/bindings omit the chapter;
removing the last entry tombstones it. Programs, diagnostics and process revisions
are never persisted. Unknown root/asset/binding fields are retained by identity;
explicit deletion removes the matching entry.

Limits are 256 assets, 1 MiB UTF-8 source per asset, 16 MiB aggregate source,
32 MiB logical chapter, 65,536 bindings, 256-byte names and parameter names,
64-byte language tags, and 256 named overrides per asset. Overrides are finite
float32-representable scalars or vectors of 2–4 components. Duplicate identities
or slots reject every ambiguous entry. Bad parameters are skipped individually;
invalid assets/bindings are skipped with warnings. Scene-aware restore checks splat
node identities and the eight-distinct-bound-assets limit before applying bindings.
Pending overrides retain their representation when saving before reflection.

Unknown languages and invalid source remain available for editing; compilation
failure never removes a clip. A missing/rejected Custom asset removes only its
clip before timeline restore. Malformed JSON, oversized chapters and future
payload/API versions fail opening under the existing JSON-session policy. An
unsupported SHDR chunk encoding follows the existing document read-only guard
so a later save cannot erase it.

Capability bit 12 (`USER_SHADERS_V1`) is writer-required for SHDR, with bit 5
for opaque preservation. It is reader- and writer-required for Custom SEQR,
so old readers reject that feature before parsing its timeline. Effective document
saves recompute only bit 12; unrelated inherited requirements survive. Copy and
compaction preserve requirements. Removing shader state clears bit 12 on save.

EDTR remains version 2. A Shader entry adds `kind:"shader"`, `target`
(`kind:global|node|reveal`, plus `node_uuid` for a node), and optional `shader_id`.
`embedded_buffer` and `share_warning` hold text only if it differs from the asset
source, or has no associated asset. Legacy `modified` describes presence of that
draft; optional `file_modified` preserves the filesystem dirty state when it
differs. This keeps old EDTR parsers valid without duplicating associated source.
These additive fields require retained-JSON writer bit 6. A draft without an
asset does not require bit 12. Python entries retain their existing representation.
SEQR remains version 1; switching a Custom clip to a built-in removes `shader_id`.

Normal saves, autosaves/recovery, Save As and training snapshots carry SHDR.
Portable and gallery publication omit SHDR, Shader drafts and Custom clips,
preserve other timeline tracks, and disclose the omission.
