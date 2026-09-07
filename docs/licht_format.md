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

## Training and Edit Mode

A `CKPT` chapter is resumable project state only while `SCNG` identifies the
training-model node and binds that node to the checkpoint instance. Ordinary
scene splats do not retain optimizer or training-session state.

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
- **Autosave & recovery.** A periodic autosave writes to a separate `<project>.licht.autosave`
  sidecar. Recovery validates that sidecar against the master head and can materialize it into a
  retained recovery session before the next durable save. Compaction and recovery publication
  may also replace the master file.
- **Compaction.** Because saves append, dead bytes build up over time; compaction rewrites the
  live generations into a fresh file and atomically swaps it in (run it yourself, or accept the
  suggestion around ~50% waste).

## GT Compare 1:1 state and compatibility

GT Compare adds optional camera calibration to the existing `SCNG` chapter.
It does not change the container grammar, chapter versions, or minimum-reader
version. The usual container-version checks still apply; accepting these fields
is not a promise that every older release can open every `.licht` file.

### Temporary view state

Projects always open with GT Compare in Fit. The 1:1 request, crop position, pending
work, failures, decoded images, and GPU tiles are runtime state and are not saved.

### Camera calibration

`SCNG.nodes[].camera.undistortion` is optional. A writer includes it when the camera
has precomputed undistortion parameters. The existing distortion coefficients and
camera model remain in the camera record.

| Field within `undistortion` | Meaning and requirements |
| --- | --- |
| `source` | Required calibration object for the distorted source image. |
| `destination` | Required calibration object for the undistorted output. |
| `prepared` | Required Boolean. Whether the camera's runtime intrinsics use the destination calibration. |
| `crop_solve_failed` | Optional Boolean, default `false`. Preserves the crop solver's fallback status; written only when `true`. |

Each calibration object has six required fields: `focal_x`, `focal_y`, `center_x`,
`center_y`, `width`, and `height`. Focal lengths must be finite and positive;
principal-point coordinates must be finite; width and height must be positive
32-bit integers. Intrinsics are in pixel units at the corresponding calibration
resolution. Native comparison scales the saved calibration to the decoded source's
resolution and uses the undistorted output grid for 1:1 sampling.

When the record is present, the camera must have distortion and use a supported
internal model: `PINHOLE` (0), `FISHEYE` (2), or `THIN_PRISM_FISHEYE` (4). These are
LFS model identifiers, not COLMAP model identifiers. Writers store the source
calibration in the camera's outer `focal_x`, `focal_y`, `center_x`, `center_y`,
`camera_width`, and `camera_height`, including when `prepared` is true. These outer values
must match `source`. Dimensions match exactly; float comparisons allow a relative
tolerance of `1e-5`, with a minimum scale of 1.

A missing `undistortion` member is accepted as a legacy camera. A null, partial,
invalid, or inconsistent record is rejected as `DataLoss`; it is not silently
replaced by a newly solved calibration. Restoring a valid record reconstructs the
undistortion parameters without repeating the crop solve. Camera transform copies
preserve both calibrations, the prepared state, and the crop-solver fallback status.

### Supported transitions

| Save/open sequence | Behavior |
| --- | --- |
| Older project without these fields → this version | Opens with Fit as the default. Supported pinhole cameras with an image source can use 1:1. Distorted cameras recover undistortion from stored source calibration when available; otherwise dataset reimport and resave may be required. |
| This version → this version | Opens in Fit and preserves valid calibration. The native image is prepared when 1:1 is requested; transient loading and crop state are not restored. |
| This version → upstream reader without 1:1 | The tested upstream chapter reader accepts the additive records. It has no native-size comparison behavior and does not interpret the new calibration. |
| This version → older resave → this version | Not a lossless calibration round trip. Rebuilding a camera record in the older writer drops `undistortion`. Usable source calibration allows reconstruction, but the exact destination calibration and crop-solver state are lost. |

The calibration compatibility check on 2026-09-06 used the actual `SCNG` reader/writer
sources from upstream commit `1466bb107f317a4a3222261333392dd2c3fbcd93`.
It exercised calibrated records for all three supported models, both prepared
states, both crop-solver statuses, and legacy pinhole/distorted records. At that
revision, an opaque chapter byte round trip preserves the calibration. Materializing
`SCNG` from typed camera records drops the unrecognized calibration.
This verifies the chapter paths, not a complete GUI save/open cycle for every
older release, and does not promise metadata preservation on an older resave.

Legacy distorted cameras without an `undistortion` record recompute undistortion
from the stored source calibration, reusing identical calibrations within a load.
There is no automatic migration when usable source calibration cannot be recovered. Reimport the
original dataset with this version and resave to establish usable calibration.
Reopening and resaving the old project alone is not guaranteed to do so.

See [GT Compare: Fit and 1:1](docs/features/gt-compare.md) for controls, loading,
retry, and memory behavior.

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
