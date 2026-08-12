# The `.licht` Project Format

`.licht` is LichtFeld Studio's project file. **One file is your whole session** — like a `.blend`.
Open `project.licht` and you are exactly where you left off: the trained model at its iteration,
the full scene graph, your selections, the panel layout, the split view, unsaved code-editor
buffers, the sequencer timeline, and the camera you were flying. Close and reopen — or copy the
file to another machine — and continue.

It replaces the old scattered files (`checkpoint.resume`, `.ppisp` sidecars, `layout.json`, and
runtime-only state) and is the **only** project format the app writes. Exports like `.ply`,
`.rad`, `.spz`, `.sog`, `.usdz`, and `.html` are separate one-way bakes, never project state.

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

The current grammar is **1.0**. The container can gain new capabilities without breaking older
readers, and its byte layout is frozen and guarded by reader/writer tests.
