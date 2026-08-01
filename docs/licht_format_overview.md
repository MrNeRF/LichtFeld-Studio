# The `.licht` Project Format — Vision & Overview

*Non-normative companion to the frozen plan (`PROJECT_FORMAT_PLAN.md`), the byte grammar
(`docs/licht_format_spec.md`), the field ownership matrix (`docs/licht_ownership_matrix.md`) and
the UUID semantics note (`docs/licht_uuid_semantics.md`). Where this document and those disagree,
those win.*

## What we are building

One file that **is** the session. Like a `.blend`: open `project.licht` and you are exactly where
you left off — the trained model at its iteration, the scene graph with every node, your
selections, the panel layout, the split view, the code-editor buffers you never saved, the
sequencer timeline, the camera you were flying. Close the app, reopen, continue. Move the file to
another machine, continue there.

Today that session is scattered across `checkpoint.resume` (LFKP), `.ppisp` sidecars,
`layout.json`, and a pile of runtime state that simply dies with the process. The `.licht` format
replaces all of it.

**One format.** `.licht` is the *only* thing the app writes. The legacy writers are deleted;
their readers are kept forever as importers. There is no dual-write transition and no precedence
dance between old and new files. Interop exports — `.ply`, `.rad`, `.spz`, `.sog`, `.usdz`,
`.html` — are unaffected: one-way bakes, never project state.

## The container in one page

A custom 64-bit chunked container, designed around three facts: training checkpoints are huge,
crashes are real, and files outlive programs.

- **Append-only generations.** A normal save appends the changed chapters as a new *generation*
  and then publishes it. Unchanged multi-GB payloads are never rewritten — Ctrl+S after a layout
  tweak is a metadata-sized append, not a 10 GiB rewrite.
- **Two alternating head slots.** Publication is a small validated head record written to the
  inactive slot of an A/B pair (never rewriting the active one), CRC-validated on open. The head
  CRC is *the* atomicity mechanism — measured on ext4, a 4096-byte slot write is not
  reader-atomic, so validation, not the filesystem, decides which head is live.
- **Old-or-new, never mixed.** After any crash, opening the file resolves to exactly the previous
  or the new generation. The write cursor is `committed_file_end`; nothing past it is ever
  parsed. An orphan tail from a crashed append is reclaimed, never auto-promoted.
- **Everything checksummed.** Every row and payload carries CRC32c. CRCs catch corruption, not
  malice.
- **Compaction** rewrites live generations into a fresh file and atomically replaces — explicit
  or idle maintenance, auto-suggested around ~50 % dead bytes.

The format was chosen over Zip64 and SQLite in an adversarial three-design review, then the
original footer-authority grammar was itself refuted and replaced by the head-slot design in a
second review round. The read-side contract is frozen and defended in the shipped tree by C++
reader/writer tests: locked binary fixtures, byte-for-byte writer reproduction, malformed-file
classification, bounded reads, publication crash matrices, and document-level chapter
validation. Earlier independent development tooling was removed once those guarantees were
covered by the product implementation and tests.

## Chapters: single authority for every field

Project state is organized into typed chapters, each the *single* authority for its fields — the
109-row ownership matrix is normative, and duplicate authorities were deleted in code rather than
reconciled at load time:

| Chapter | Owns |
|---|---|
| `PROJ` | Project identity, UUID lineage, commit metadata |
| `REFS` | External references: dataset root, live `.rad` sources, background/env images, PLY-sequence dirs — relative-preferred paths + fingerprints |
| `SCNG` | Scene graph: nodes, transforms, visibility, groups, camera enablement — keyed by 128-bit node UUIDs |
| `SELM` | Selection groups and per-node mask slices |
| `SPLT` | Embedded splat payloads (LFSP verbatim, page-aligned) for non-training nodes |
| `PCLD` / `MESH` | Point-cloud / mesh payloads — distinct chunks with their own versioning (owner decision 2026-07-30) |
| `CKPT` | The training checkpoint (LFKP embedded byte-verbatim via bounded windows): model, optimizer, strategy, iteration |
| `PPIS` | `.ppisp` bytes for non-checkpoint sessions only |
| `PRMS` | Strategy presets and pending next-run parameter edits (never mutates an active trainer on load) |
| `GUIL` | Panel layout, dock/splitter dims, window geometry — framework-agnostic, zero ImGui |
| `VIEW` | Render settings, per-viewport cameras, nav mode, split state, bookmarks |
| `EDTR` | Code-editor session incl. unsaved buffers (flagged — a shared project can leak them, the UI warns) |
| `SEQR` | Sequencer timeline, clips, playhead |
| `METR` | Training metrics history |

JSON chapters go through a retained DOM: unknown fields written by a newer version survive a
round-trip through an older one *semantically*, not just byte-wise. Unknown binary chunks are
carried forward verbatim by declared-safe writers and by compaction.

**What stays out, deliberately:** theme, language, UI scale, HUD toggles (user-global, not
project); ImGui anything (the framework left; the format never knew it); datasets and live `.rad`
payloads (referenced, fingerprinted, relinked when moved — never embedded, never silently
re-resolved by filename).

## Source-backed nodes (owner decision 2026-07-30)

- **Imported splats (PLY/SPZ/SOG) are always embedded** in `SPLT`. A saved project is
  self-contained for them; no relink flow, at the price of fatter files.
- **Live `.rad` nodes stay external and read-only.** `.rad` is a paged LOD source; embedding it
  would defeat its purpose. Destructive edits on a live-RAD node are refused until the user
  explicitly bakes it into an embedded resident node.

## Saving during training

The hard problem: a checkpoint mid-training is ~10 GiB of GPU tensors that must be captured
consistently without stalling the optimizer or the viewer.

The measured design: at a safe point the optimizer pauses once (one defined clock, including all
stream syncs and the cold path), tensors leave the GPU in bands through a ≤512 MiB pinned ring
into pageable staging, training resumes, and disk IO happens afterwards in the background. A
single snapshot UUID proves consistency across every tensor and the CPU-side state. Measured on
the dev rig: pinned peak exactly 512 MiB, zero extra host RAM beyond staging, step-time
regression after resume −0.07 %, D2H efficiency 84 %.

The pause SLA is **bandwidth-scaled** (owner decision 2026-07-30):
`gate = snapshot_bytes / measured_pinned_D2H × 1.12`. Sub-second for 10 GiB on reference PCIe 4
hardware, proportionally and honestly longer on slower buses — physics is stated, not hidden.

**Autosave** is a single bounded sidecar (`project.licht.autosave`), never the master file — a
training session dirties ~100 % of the checkpoint every cycle, which would grow an append-only
master by ~120 GB/h. The sidecar is validated against the master's head commit UUID; a stale
sidecar is ignored and deleted, a fresh one triggers the recovery prompt. Explicit save merges it
back. Disk ceiling: master + one autosave steady-state, + one more transiently during
replacement.

## Opening

Footerless, head-slot open: validate both heads, pick the live generation, hand the GUI its shell
in <100 ms, then hydrate in the background. Restore is 1:1 for everything in the ownership
matrix, with a defined order (checkpoint → scene → per-node enablement → event-driven GUI restore
after plugin registration). Training resume is *statistical* — same trajectory family, not
bit-exact RNG replay; no RNG state exists to capture, and the format does not pretend otherwise.

## Compatibility promises

1. Released `.licht` files remain readable by all future versions — forever.
2. Older apps open newer files read-only by default; they may write only when the file's
   `min_safe_writer_version` and capability bits declare it safe. No silent data loss by an old
   writer, ever.
3. Unknown chunks and unknown JSON fields survive every declared-safe write and compaction.
4. Second instance of the app: read-only or Save As. Concurrent writers are unsupported and
   enforced via an OS lock on a held fd, not existence checks.

## Where it stands (2026-07-30)

- **P0** (invariants, byte grammar, ownership matrix, snapshot and OS-semantics studies): done,
  exited with a formal verdict; its development-only executables are retained in branch history,
  not the shipped tree.
- **P1** (node UUID identity through scene/undo/selection/sequencer, serializer gap fixes,
  field-wise config serialization, retained-DOM layer): landed.
- **P2** (the production C++ container core, `src/io/project/`): in build — the v1 prototype it
  replaces implemented the refuted footer grammar and serves as input only.
- **P3–P8**: chapters, training snapshot productionization, GUI session chapters, lifecycle
  (File menu / `--project` / restore-last-session / save-on-close), autosave & recovery,
  compatibility hardening — in that order, each gated on builds, targeted C++ gtests, and GUI
  validation.
