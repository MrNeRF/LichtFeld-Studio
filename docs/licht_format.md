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

The current grammar is **1.0** on this development branch. The framed payload layout is guarded by
reader/writer tests; no compatibility promise is made for pre-framed development files.
