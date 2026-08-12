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
├─ head slot A    "live version = generation 2"   + its preview   (CRC32c)
├─ head slot B    "live version = generation 1"   + its preview   (CRC32c)
│                 two slots: a save writes the idle one, never the live,
│                 then flips which slot is live
│
├─ generation 1   [scene][selection][settings][checkpoint 9 GB][layout]
│                 + a table of contents
├─ generation 2   [scene][selection]            ← only what changed
│                 + a table of contents that points back at generation 1
│                   for every part it reused
└─ generation 3   ...            the file only grows downward; nothing is
                                 ever overwritten in place
```

Moving a camera and pressing save writes a few kilobytes: the 9 GB checkpoint is
not copied again, because generation 2's table of contents just points back at
the bytes generation 1 already wrote. The preview thumbnail travels with the
head record, so it updates atomically the instant a save becomes live.

## How it works

A custom chunked binary container, built around three facts: training checkpoints are huge,
crashes happen, and files outlive programs.

- **Append-only saves.** A save appends only the parts that changed as a new *generation*, then
  publishes it. Unchanged multi-GB payloads are never rewritten — Ctrl+S after a small tweak is a
  tiny append, not a 10 GB rewrite.
- **Crash-safe.** Each publish writes a small validated "head" record into one of two alternating
  slots, never touching the live one. On open, a checksum picks the valid head, so after any crash
  you get exactly the previous or the new generation — never a mix. A half-written tail from an
  interrupted save is ignored, never used.
- **Checksummed throughout.** Every record and payload carries a CRC32c to catch corruption.
- **Organized into chapters.** State is split into typed chapters (model, scene graph, parameters,
  layout, sequencer, camera, …); each chapter is the single source of truth for its fields.
- **Autosave & recovery.** A periodic autosave writes to a separate `<project>.licht.autosave`
  sidecar — never the original file — so a crash offers to recover your last session. The `.licht`
  itself is written only when you explicitly save.
- **Compaction.** Because saves append, dead bytes build up over time; compaction rewrites the
  live generations into a fresh file and atomically swaps it in (run it yourself, or accept the
  suggestion around ~50% waste).

## Version

## Command-line opening and recovery

Use `-v project.licht` to open a project in the GUI. Headless training resumes
use `--headless --resume project.licht`; a complete autosave newer than the
master head is recovered automatically. Ambiguous recovery candidates remain
an error.

The current grammar is **1.0**. The container can gain new capabilities without breaking older
readers, and its byte layout is frozen and guarded by reader/writer tests.
