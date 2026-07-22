# LichtFeld Project Format (`.licht`) — Design & Implementation Plan v2

One master file, like Blender's `.blend`: training checkpoints, scene graph, GUI layout,
code-editor session, split-screen/viewport state — everything (plus, during training, exactly one
disposable autosave sidecar — never more). Reopening the application restores the last session
per the ownership matrix. Exports (`.rad`, `.spz`, `.sog`, `.ply`, `.usdz`, `.html`)
remain one-way bakes from the project and are unchanged.

**Provenance:** v1 plan (2026-07-18) from a 6-way state-surface audit + 3-design container study.
v2 (2026-07-18) after a three-way adversarial review: Codex gpt-5.6-sol max-effort (51 findings,
2 rounds, session `019f744a-b3ba-71b0-80b9-632308e9c89a`) + independent Grok high-effort review +
Claude synthesis. Round logs in session scratchpad. v1's container grammar (mutable header, EOF
footer, `prev_index_offset`, full-rewrite saves, O(dirty)-autosave claim) was **refuted** and is
replaced wholesale by the append-generation design in §3.

---

## 1. Decision log (settled; reopen only with new evidence)

1. **Container**: custom little-endian **append-only** chunked container, published byte grammar.
   Zip64 rejected (dual local-header/central-directory metadata is a parser-differential bug farm;
   foreign-tool *writability* is a liability — a re-zip destroys alignment and commit heads;
   conceded by its own proposer). SQLite rejected as tensor store (2 GiB value limit, BLOB reads
   copy, mmap exposes DB pages not tensor spans). Directory/CAS rejected as the user-visible
   artifact (partial-copy desync); its ideas live on internally. We adopt zip's *discipline*
   instead of its format: exact byte-offset grammar doc, golden fixtures, an independent read-only
   parser in CI, shared fuzz corpus, an inspect/extract CLI tool.
2. **One format (owner decision)**: `.licht` is the only format the app **writes**. Standalone
   `checkpoint.resume`/LFKP files, `.ppisp` sidecars, `layout.json` become **import-only**:
   readers kept forever, writer code paths deleted. No dual-write transition, no precedence
   rules. Headless CLI training output is a `.licht` project too. Interop exports unaffected.
3. **Normal save = append one generation**: append dirty chunks + a complete live index + a
   commit record, flush, then publish via one of two alternating validated head slots. Unchanged
   payloads are never rewritten — metadata-only Ctrl+S is milliseconds, not a 10 GB rewrite.
4. **Compaction** (atomic temp + replace) only for: garbage collection, Save As, migration,
   explicit repair, idle maintenance. Never the normal save path. Auto-triggers at idle once
   dead bytes exceed ~50 % of live size (or file > 2× live), surfaced in the UI — append-only
   masters must not bloat unboundedly because nobody ever presses Save As.
5. **Autosave = one bounded sidecar** (`project.licht.autosave`, atomically replaced each cycle),
   *not* appended to the master. Rationale: an active-training checkpoint is ~100 % dirty, so
   in-master autosave grows ~10 GB per cycle and hands the master to the highest-frequency,
   highest-risk writer. The sidecar bounds steady-state disk to `master + one autosave`, confines
   appender bugs to a disposable artifact, and keeps the master off cloud-sync hot paths.
6. **Identity**: persistent 128-bit UUIDs for project, file incarnation, commit, snapshot, and
   every scene node. Names are display labels. v1's `instance_id = hash(name)` is deleted.
7. **Directory model**: every commit's index is complete (no delta-chain replay on open);
   unchanged rows carry their prior physical offsets forward. At most one live row per
   `{fourcc, instance_uuid}` — duplicates are corruption. Tombstones, not a SUPERSEDED flag.
8. **Commit authority**: the two head slots are the *only* automatic authority. No EOF/footer
   scanning on open; scanning lives in an explicit ask-the-user repair path.
9. **Ownership matrix** (P0 artifact) is normative: exactly one authoritative chapter per field.
   Training model + optimizer: `CKPT` only (no duplicate `SPLT` for the training node).
   Non-training splats: `SPLT`. Exact resume params: `CKPT`; project/UI overrides: `PRMS` with an
   explicit precedence rule. Training PPISP: `CKPT`; `PPIS` only for non-checkpoint sessions.
   Camera enablement: `SCNG`, reconciled after checkpoint load.
10. **Compatibility contract**: new LichtFeld reads every released `.licht` forever (hard
    promise). Old apps get read-only inspection of newer files, and may write only when the
    file's `min_safe_writer_version` + capability bitmaps declare it safe — never a blanket
    "old app edits and resaves safely" (Blender itself doesn't promise that). Unknown chunks are
    preserved as opaque index handles; a chapter whose version exceeds the app's support is
    treated as opaque, never parse-and-dump.
11. **JSON chapters**: retained-DOM is the single source of truth. Typed accessors read/write
    through the DOM in place; serialization dumps the DOM; unknown keys and unknown array-element
    fields survive because nothing is rebuilt from a parallel typed model. Array elements are
    addressed by UUID. Semantic preservation, not byte-exact formatting.
12. **Training snapshot**: honest promise — render/UI threads never wait on persistence; the
    *optimizer* pauses at a safe point for a **measured, bounded** D2H capture; disk IO happens
    after training resumes. Consistency + bounded pause + bounded RAM cannot all be uncapped:
    v1 picks consistency + bounded pause, with a RAM preflight that defers autosave loudly
    rather than stalling the trainer at disk speed.
13. **Concurrency**: one exclusive writer per project via a stable sibling lockfile — an OS lock
    *held on the open fd/handle* (POSIX `flock`/`fcntl`; Windows `LockFileEx`), not
    existence-checking, so a killed process releases it and stale locks cannot deny writes;
    creation is atomic (`O_EXCL`/`CREATE_NEW`). Readers pin a committed generation; a second
    instance opens read-only or Save As. Reads use positional IO (`pread`/overlapped) — never a
    shared seek cursor. Per-handle-class share flags are a spec checklist: every Windows
    reader/mapping handle opens with `FILE_SHARE_READ|WRITE|DELETE` or compaction's replace
    fails against held maps.
14. **Integrity**: CRC32c over superblock, head slots, commit records, chunk headers, index, and
    stored payloads; per-block CRCs for multi-GB payloads so lazy/partial loads validate only
    consumed ranges. Detects corruption and torn writes; explicitly *not* tamper-evidence or
    rollback protection.
15. **Encoding**: little-endian only; every on-disk field explicitly encoded/decoded — no
    `reinterpret_cast` of packed structs defines the format. Config structs (PPISP, BilateralGrid,
    ControllerPool) move to field-wise serialization; `static_assert(sizeof)` freezes are not
    schema versioning.
16. **Alignment**: 64 B chunk headers; 4096 B tensor payloads; Windows maps from the 64 KiB
    allocation-granularity floor with pointer adjust.
17. **Checkpoint layout**: `CKPT` embeds the LFKP stream byte-verbatim through a *bounded*
    random-access window (`base+length`, never a raw `istream`). Partial display-load is gated on
    the P0 benchmark; if scanning LFKP magics is too slow, the section table goes **inside** the
    LFKP trailer (versioned there) — no container-side `CKPX` whose absolute offsets can dangle.
18. **Honest performance claims**: mapped bounded input + CPU→GPU streaming. "Zero-copy tensor
    hydration" is not claimed for v1 — existing deserializers allocate/copy on CPU before upload;
    a tensor-native aligned layout is future work.
19. **GUI state**: framework-agnostic only (PanelRegistry/RmlUi/WindowManager). Zero ImGui state
    (removal branch pending). Theme, language, UI scale, VRAM HUD = user-global config, **not**
    project state — opening someone else's project must not switch your language. Unsaved editor
    buffers are embedded but flagged in sharing/export UI (secret-leak surface).

## 2. Container specification

Single little-endian file, extension `.licht`. Exact byte-offset tables are a P0 deliverable
(`docs/licht_format_spec.md`); the shapes below are normative for structure, not yet for offsets.

### 2.1 Physical layout

```
[Superblock 256 B @ 0, immutable]
[Head slot A 4096 B @ 4096]  [Head slot B 4096 B @ 8192]
[append region @ 65536: chunks … index … commit record | chunks … index … commit | …]
```

- **Superblock** (written once at creation): magic `\x89LFS\r\n\x1a\n`, format major/minor,
  byte-order tag, `project_uuid`, `file_uuid` (new on every compaction), head-slot geometry,
  creation time, CRC.
- **Head slot** (A/B alternate; the active slot is never overwritten): slot id, `head_sequence`,
  generation, `commit_uuid`, commit offset/bytes, `committed_file_end`, commit CRC echo, head CRC.
  `head_sequence` is the **sole** comparison key (monotonic across A/B alternation); `generation`
  is display/chain metadata, never a tiebreaker.
- **Commit record** (256 B, last bytes of its generation): kind
  `EXPLICIT|AUTOSAVE|RECOVERED|COMPACTION` (`AUTOSAVE` is valid **only inside the sidecar file**;
  a master commit bearing it is corruption), project/file/commit UUIDs, generation, parent commit
  UUID+offset, explicit-ancestor commit, snapshot UUID, wallclock, index offset/size/CRCs,
  `committed_file_end == commit_offset + commit_bytes`, `min_reader_version`,
  `min_safe_writer_version`, 128-bit reader/writer capability bitmaps, CRC. Physical EOF may
  exceed `committed_file_end` (orphan tail from an interrupted append — ignored, reclaimed by
  the next successful append or compaction).
- **Chunk**: 64 B header `{fourcc, chunk_version, instance_uuid[16], flags, compression,
  stored/uncompressed sizes, payload CRC, per-block CRC table ref for large payloads,
  header CRC}` + payload. Tensor-bearing payloads page-aligned.
- **Index**: complete live table per commit — one row per chunk `{fourcc, chunk_version,
  instance_uuid, flags, compression, header/payload offsets, sizes, source_generation,
  payload CRC, header CRC}`, canonically sorted by `{fourcc, instance_uuid}`, zstd-compressed,
  CRC'd. Unchanged rows keep prior offsets; normal append copies no unchanged payload bytes.
- **Compression**: zstd for KB-scale JSON chapters; raw for tensor chunks (Adam moments already
  uint8-quantized; splat attributes near-incompressible).

### 2.2 Commit sequence (crash-sound by construction)

1. Acquire writer lock; validate current head.  2. Assign generation + commit/snapshot UUIDs (every commit stamps a fresh non-null
   snapshot UUID — not only training-snapshot commits).
3. Preflight disk for payloads+index+commit+padding+reserve (before any heavy write).
4. Append dirty chunks (streamed, incremental CRC; a chunk enters the index only after its CRC
   is final).  5. Append complete index.  6. Append commit record.  7. `fdatasync`/
   `FlushFileBuffers`.  8. Write the *inactive* head slot in one aligned op.  9. Flush again.
10. Only now report committed. Crash at any point yields exactly the old or the new generation.

Normative commit rules (correctness-bearing, not implementation hints):

- **Durability total order**: every indexed payload byte durable (CRC final, range flushed) →
  index durable → commit record durable → flush → inactive head → flush → publish. No head may
  ever reference an index whose payload bytes are not yet on media.
- **Clean-proof rule**: an index row may reference a prior generation's physical span **only**
  with a positive clean-proof bound to that content's snapshot/mutation epoch. No proof → the
  payload is rewritten. Explicit save treats every matrix-owned chapter as dirty unless an
  audited clean-proof API attests otherwise. (Append-by-reference makes dirty tracking a
  *correctness* dependency — a false clean bit produces a CRC-valid generation pointing at stale
  bytes, a bug class full rewrites cannot have. Gate: forcibly clear a dirty flag after mutation;
  the save must not reuse the old payload CRC.)
- **Write cursor**: appends start exactly at the active head's `committed_file_end`. Readers and
  parsers never read past `committed_file_end`; physical EOF is not an authority (orphan tails
  from interrupted appends are invisible). Tail reclaim is either
  `ftruncate(committed_file_end)` under the writer lock *before* any new bytes, or left to
  compaction — never mid-append. The repair path never auto-promotes tail data.

### 2.3 Open sequence

Validate superblock → read both heads → validate each fully: head CRC, UUIDs, commit record CRC
and linkage, index CRC, **every index row's header CRC and bounds ⊆ `[0, committed_file_end)`**,
no row overlaps or duplicate keys → choose the valid head with the greater `head_sequence`.
Equal sequences with different commit UUIDs = split-brain, fail loudly; equal sequence, same
UUID, same CRC = duplicate slot write, accept; same UUID, different CRC = corruption. Full
payload CRC verification may stay lazy (per-block on consumed ranges + background sweep), but a
torn region can only exist past `committed_file_end`, where no parser looks. Falling back to the
older head surfaces a recovery warning — never silent. Both heads dead → explicit repair path
(scan + ask), never automatic.

### 2.4 Chunk registry

| Fourcc | Content | Encoding |
|---|---|---|
| `PROJ` | Manifest: versions, timestamps, project UUID, dataset linkage, embed/reference log | JSON |
| `PRMS` | ParameterManager snapshot (project/UI overrides only — exact resume params live in CKPT; precedence per ownership matrix) | JSON |
| `SCNG` | Scene graph: per-node UUID/name/parent/order/type/transform/visibility/lock/training-enabled/cropbox/ellipsoid/camera + context; durable project DTO, **not** the undo snapshot shape | versioned binary |
| `SELM` | Selection groups + per-node mask slices, keyed by node UUID | binary |
| `REFS` | External refs: relative-preferred paths + fingerprints for dataset root, `.rad`+`.rad.meta`, bg image, env HDR, PLY-sequence dirs. Fingerprint = size+mtime+xxh3 head/tail, documented as a *heuristic*; full-hash verify on demand | JSON |
| `SPLT` | Embedded splat payload per non-training dirty/generated node — LFSP verbatim. *P0-audit open point (matrix U1): "dirty/generated" here vs decision 9's "non-training splats = SPLT" leaves clean imported and live-RAD nodes unresolved — owner decision pending.* | binary, page-aligned |
| `CKPT` | The LFKP checkpoint stream embedded unchanged, via bounded windows | binary, page-aligned |
| `PPIS` | `.ppisp` bytes — non-checkpoint sessions only (matrix rule) | binary |
| `GUIL` | Dock/splitter dims (incl. `left_dock_width`), panel visibility/rects/stack order, active tab, window geometry/fullscreen. Framework-agnostic; zero ImGui; no theme/language/scale/HUD (user-global) | JSON |
| `EDTR` | Code-editor session: open files, unsaved buffers (flagged for sharing), cursor/scroll/folds, vim mode | JSON |
| `VIEW` | RenderSettings superset, `PanelCameraState` (R[9]+t+pivot+home+speeds+ortho) ×2, nav mode, split state, camera bookmarks, gizmo/tool prefs | JSON |
| `SEQR` | `Timeline::saveToJson` inline + PLY-sequence clips + playhead/loop/speed (KEYFRAME nodes regenerate via `KeyframeSceneSync`) | JSON |
| `METR` | Loss/PSNR history, accumulated training time, last eval — resume shows a populated graph | binary |

Deferred beyond v1: `THMB` thumbnail, zstd archive-save mode, ZIP export, RNG chunk, tensor-native
zero-copy layout. Deleted outright: `CKPX`, sections-only backup sidecar, `auto0/auto1` rotation,
`SUPERSEDED` flag, in-container CKPT retention (rollback comes from append generations until
compaction).

### 2.5 Versioning

Container `format_major/minor` + per-commit `min_reader_version` / `min_safe_writer_version` +
capability bitmaps; per-chunk `chunk_version`; embedded payloads keep their own magic+version
(LFKP/LFSP/LFAD/…) with legacy readers forever. Rules: any semantic change bumps
`chunk_version`; LFKP flag/structure additions bump the LFKP version (the pending `HAS_SPARSITY`
flag must bump it — current worktree violates this); "new state = new fourcc" is the default but
not an absolute — capability bits gate anything cross-cutting.

## 3. Save semantics

- **Explicit save (Ctrl+S)**: append-generation commit per §2.2 on a background writer thread.
  Metadata-only saves: p95 ≤ 250 ms including both flushes. A changed training checkpoint is
  inherently ~10 GB of appended bytes — that is physics, stated honestly, but it never blocks
  the UI and never rewrites unchanged data.
- **Training snapshot** (`ProjectSnapshotBundle`): at the trainer safe point — wait on *every*
  stream that can mutate persisted state, freeze the optimizer, stamp one snapshot UUID +
  iteration over all CPU chapters and GPU tensor descriptors, then banded D2H through a pinned
  ring (3×128 MiB bands, ≤512 MiB pinned ceiling) into prefaulted *pageable* snapshot storage.
  Training resumes when the last D2H event completes; serialization/IO run afterwards from host
  staging. RAM preflight requires `snapshot_bytes + max(4 GiB, 20 % of RAM)` available or the
  autosave defers with a visible notice. One snapshot + one writer job in flight; newer requests
  coalesce. The current synchronous per-tensor `tensor.cpu()` path is forbidden in this flow.
- **Autosave**: dirty-epoch triggered, replaces the single `project.licht.autosave` sidecar
  atomically. Sidecar records `{project_uuid, base_explicit_commit_uuid, autosave_sequence,
  snapshot_uuid}` and is complete relative to its explicit base commit (logical chunk keys,
  never inherited physical offsets; never chained on an older autosave). Explicit save deletes
  the sidecar only after the master head is durable; compaction merges-or-discards it first.
- **Recovery predicate** (exact, no discretion): offer recovery **iff**
  `sidecar.project_uuid == master.project_uuid && sidecar.base_explicit_commit_uuid ==
  master.head.commit_uuid && sidecar is complete and CRC-valid`. Among multiple valid candidates
  (sidecar + its temp/backup — validate all, trust no pathname) pick the unique highest
  `autosave_sequence`; never merge two sidecars. A leftover sidecar whose base is not the current
  head (e.g. after a durable explicit save whose sidecar-delete didn't run, or after an
  interrupted compaction) is **ignored and deleted**, never offered — offering an
  ancestor-based sidecar over a newer explicit commit rolls the user back. Compaction must
  merge-or-discard the sidecar *before* old commit UUIDs become unreachable.
- **Compaction / Save As / first save**: write temp sibling → flush → validate independently →
  `ReplaceFileW` (dest exists) / `MoveFileExW` (first publication) on Windows, rename on POSIX →
  validate destination before deleting backup; on `ReplaceFileW` errors 1175–1177 retain and
  validate all candidates. New `file_uuid`; no physical offset survives by assumption.
- **Save-on-close**: async save + "quit when save completes" state machine inside `allowclose()`
  **before** `beginShutdown()` (`flush_and_exit` skips destructors); never block the UI thread on
  a multi-GB flush without progress UI. Slots into the existing `pending_training_action_`
  continuation.

## 4. Load & restore semantics

- **Open**: §2.3 head selection → KB-scale chapters decoded → window/docks/editor/split/scene
  tree (placeholder nodes)/sequencer on screen in <100 ms. Tensor hydration streams in the
  background into `SplatTensorAllocator` (mmap raw payloads, per-block CRC on consumed ranges,
  remainder verified in background), swapped in via the proven `swapNodeModel` placeholder
  pattern. `CKPT` hydrates SplatData first for display; full trainer state loads on resume;
  failed hydration leaves a coherent inspectable project, never a half-replaced session.
- **Restore ordering** (hard constraints from the audit):
  1. Mirror the CLI startup path: construct params → `setParameters()` → stage restore work
     behind the existing `gui_frame_rendered_` gate in `update()`.
  2. Scene: nodes parent-first in saved child order → `training_model_node` before trainer
     construction → selection groups → selection mask (per-node slices, only after the node set
     matches) → node selection last.
  3. Rendering: scene/checkpoint load → PPISP companion (before `ppisp_mode=AUTO` restore) →
     RenderSettings (after CLI overrides, `visualizer_impl.cpp:222-228`) → split mode only via
     `SplitViewService::toggleMode` (never the raw enum) → panel cameras via `setViewMatrix` +
     pivot (store R directly; eye/target/up loses roll) → GT camera id →
     `depth_range_initialized_` → `markDirty(ALL)`.
  4. GUI panels: event-driven, after native + Python plugin panel registration completes (a
     concrete panels-ready signal, tested with delayed plugin registration — not prose).
- **Restore-last-session**: argless startup opens the MRU project; `--project` and drag-drop open
  explicitly; crash recovery per §3.
- **Legacy import**: opening `checkpoint.resume`, `.ppisp`, or a bare PLY/dataset folds it into a
  new untitled project (readers kept forever); nothing writes those formats again.

## 5. Known-gap fixes (small, independently landable — land in P1)

Each is silent data loss today and blocks fidelity regardless of container outcome:

1. `SplatData` frozen-ranges serialization + `add_splat_paths`/`add_splat_freeze` in checkpoint
   params (freeze state dropped on resume today). LFSP/LFKP version bumps done properly.
2. Sparsity ADMM tensors (`z_`, `u_`) serialization (+ the LFKP `HAS_SPARSITY` version bump).
3. `loadColmapCamerasOnly` sparse-path record.
4. SceneManager member for the loaded `.ppisp` path.
5. Per-node dirty flag on Scene nodes (embed decision input).
6. `left_dock_width` omitted by `LayoutState::save/load`.
7. `RenderSettingsProxy` missing fields: `depth_view`, `split_view_offset`,
   `lod_auto_enable_rad`, `lod_behind_camera_penalty`.
8. Collapse the three `getConfigDir` duplicates (hygiene; not format-blocking).
9. `disabled_camera_uids` into the scene chapter (checkpoint-only today). *Owner decision
   2026-07-18: deferred to P3 — the SCNG chapter is its authority and no pre-SCNG surface
   exists worth building; reviews confirmed the crashed-worker diff never implemented it.*
10. UI language persisted — in **user-global config**, not the project (per decision 19).
11. Camera bookmarks: currently absent from any persistence; owned by `VIEW`.
12. Point-cloud / mesh payload ownership decided in the matrix (unspecified today).

## 6. Explicit non-goals / documented semantics

- **Statistical resume, not bit-exact**: no RNG state exists today; the format documents
  statistical-resume semantics (optional RNG chunk is future work). Strategy caches rebuild via
  `initialize()` — one-window behavioral discontinuity, as today.
- **Datasets never embedded**; fingerprints are loud heuristics with on-demand full-hash verify.
- **Not tamper-evident**: CRCs target corruption, not adversaries; malicious rollback is out of
  scope for v1. CRC32c collision odds on multi-GB payloads are documented (per-block CRCs shrink
  the window); crash testing is process-kill grade — hardware that acknowledges FLUSH dishonestly
  is documented residual risk, as for every application-level format.
- Trainer transient telemetry beyond `METR` is rebuilt, not persisted.

## 7. Implementation phases

Every phase gates on: guarded build (`lfs-build-guard`), targeted gtests, live GUI validation for
GUI-touching phases, compute-sanitizer memcheck for CUDA-touching phases.

- **P0 — Invariants & feasibility gate** (no chapter schema or released file freezes before it):
  field-level ownership matrix; UUID semantics (create/copy/import/duplicate); exact byte grammar
  + commit state machine; capability/opaque-dependency model; CUDA snapshot prototype; Windows
  append/lock/mapped-reader/replace prototype; independent parser + golden fixtures.
  **Exit numbers** (reference rig, PCIe4 x16 + NVMe ≥2.5 GiB/s): crash injection at every append/
  flush/head boundary yields old-or-new generation only (process-kill semantics; syscall-order
  assertions in debug builds; optional device-mapper fault injection; devices that lie about
  FLUSH are documented residual risk); ≥10 000 randomized corruption cases scored against a
  **spec oracle** (each mutation has an expected outcome: open gen N / open gen N+1 / hard fail /
  repair-only) — dual-parser agreement is necessary, not sufficient; 10 GiB snapshot safe-point
  **pause** — one clock, defined as optimizer-enters-safe-point (including all stream syncs)
  through last-D2H-complete until optimizer-may-mutate, cold first-autosave path included —
  p95 ≤ 750 ms (max ≤ 1 s), D2H ≥ 80 % of pinned baseline, pause insensitive (≤ 100 ms delta) to
  a disk throttled to 500 MB/s, pinned ≤ 512 MiB, extra host RAM ≤ snapshot + 768 MiB, **and**
  step-time regression over the 100 iterations following resume ≤ 10 % (background serialization
  must not throttle training after the pause ends); 10 GiB append+publish p95 ≤ 5 s and ≥ 80 % of
  raw volume speed; metadata-only commit p95 ≤ 250 ms; open-to-GUI p95 < 100 ms cold; no
  save-attributable viewer wait > 16 ms, p99 frame-time regression ≤ 10 % during the pause and
  during background writing; snapshot consistency proof (single snapshot UUID across all
  tensors+CPU state); Windows matrix incl. `ReplaceFileW` 1175–1177.
  **Additional mandatory fixtures** (each a way P0 could otherwise pass with a wrong format;
  ownership: read-side cases → packet (b) parser/fixtures; disk-full mid-append/mid-sidecar,
  `min_safe_writer` refuse-write, metadata-only-with-dirty-CKPT escalation → P2 writer tests;
  Windows compaction-vs-held-map, torn-write/crash-kill cases → packet (d) OS prototype):
  orphan-tail file (`physical_size > committed_file_end`) opens correctly and repair never
  auto-promotes the tail; index rows with out-of-bounds/overlapping offsets fail loudly;
  a reader pinned at generation N observes nothing mixed while N+1 publishes; stale sidecar
  (base ≠ head) after a durable explicit save is ignored+deleted; disk-full mid-append and
  mid-sidecar-replace preserve all priors; forced false-clean dirty flag must not reuse a prior
  payload span (clean-proof rule); the `min_safe_writer_version`/capability refuse-write matrix;
  Windows compaction against a held mapped reader; a metadata-only save with a dirty CKPT
  escalates rather than silently committing without it; `head_sequence` monotonic across A/B
  alternation. **If the snapshot gate fails, training autosave stops here** — next step is
  storage versioning/D2D cloning or renegotiating the SLA, not shipping the stall.
  *Measured 2026-07-18 (RTX 4090 dev rig): all rig-independent gates PASS decisively — pinned
  peak exactly 512 MiB, extra host RAM = 0 beyond staging, step-time regression −0.07 %,
  disk-throttle pause delta 26 ms, consistency proof all cycles, D2H efficiency 84.1 % (single
  non-temporal drain thread; multi-threaded drains measurably regress every metric). Absolute
  pause gates FAIL on this rig for hardware reasons: independently measured raw pinned D2H =
  15.3 GiB/s (H2D 17) vs the 25 GiB/s reference assumption — 10 GiB cannot cross the bus in
  <670 ms; measured pause p95 = 1059 ms; on-reference projection ≈ 700–750 ms. OWNER DECISION
  PENDING: bandwidth-scaled SLA (recommended) vs D2D-cloning engineering vs hard-gate+defer.
  Bench: tools/licht_p0/snapshot_bench (contract mode, drain_threads=1, 4×128 MiB).*
- **P1 — State & serializer foundations**: node UUIDs through scene/undo/selection/sequencer;
  §5 gap fixes; field-wise Config serialization; retained-DOM chapter plumbing; delete duplicate
  authorities. Exit: every matrix row round-trips independently; no state has two writers.
- **P2 — Container core rewrite** (`src/io/project/`): superblock, head slots, commit records,
  append writer, generation-pinned reader, positional IO, bounded stream/mmap read APIs,
  writer lockfile, compaction + atomic replace, opaque carry-forward, capability gating.
  Exit: golden fixtures, parser parity, crash matrix, mmap tests, P0 perf numbers on both OSes.
  *Existing prototype disposition*: keep CRC32c module, `Crc32cCountingStreambuf`, streaming
  `begin_chunk`/`end_chunk` concept, zstd path, error plumbing; rewrite open/find/create/
  finalize/copy_chunk/read_at per the new grammar; delete `prev_index_offset` + footer authority
  + the offset-identical prev-index test.
- **P3 — Core chapters**: `PROJ`/`REFS`/`SCNG`/`SELM`/`SPLT` + non-training `PRMS`.
  Exit: representative multi-node edited scene round-trips with stable UUIDs, relink flow,
  unknown-field preservation.
- **P4 — Training snapshot & chapter**: productionize the P0 snapshot service; bounded-window
  checkpoint embed; resume migration; lazy display-vs-trainer loading. Exit: every strategy +
  auxiliary component resumes at the saved iteration; P0 stall/memory gates hold in-app;
  post-snapshot mutations cannot alter serialized bytes.
- **P5 — RmlUi session chapters**: `GUIL`/`VIEW`/`EDTR`/`SEQR`/`METR`. Exit: layout, windows,
  cameras, bookmarks, split state, editor buffers, timeline restore after plugin registration;
  zero ImGui state.
- **P6 — Lifecycle & partial open**: Save/Load/SaveAs commands, File menu, MRU, `--project` CLI +
  headless `.licht` output, drag-drop, restore-last-session, save-on-close state machine,
  transactional project switching, background hydration, MCP tools (postWork). Exit: <100 ms
  shell restore; failed hydration leaves a coherent project.
- **P7 — Autosave, recovery, compaction**: single sidecar, dirty epochs, base-commit validation,
  recovery prompt, merge-on-explicit-save, idle + threshold compaction. Exit: 24 h × 5 min
  autosave sim holds steady disk ≤ `master + one autosave`; peak transient ≤
  `master + 2 × autosave + ε` during sidecar replacement (full-checkpoint autosaves are ~an
  autosave-sized temp — measured separately from metadata-only cycles, no gaming the gate with
  KB-scale sims); every crash point yields master + a complete autosave (old or new); disk-full
  preserves both priors; sidecar removed only after a durable master head.
- **P8 — Compatibility & release hardening**: old/new binary matrices, fuzzing, malformed-chapter
  tests, migration fixtures, independent-parser CI, Windows/Linux recovery tests, published spec.
  Exit: every released fixture opens as promised; every prohibited old-writer scenario is
  read-only, never lossy.

## 8. Honest v1 product promises (owner-facing)

1. One explicitly saved `.licht` moves the whole session; the autosave sidecar is disposable.
2. Datasets and live `.rad` sources stay external; stale refs get a relink flow, never silence.
3. Ctrl+S appends changed state only; unchanged multi-GB payloads are never rewritten. A changed
   training checkpoint is inherently large — appending doesn't shrink physics.
4. After any crash, a committed save resolves to exactly the previous or the new generation.
5. Saving during training pauses the optimizer for a measured sub-second D2H capture; UI and
   renderer stay responsive; disk IO happens after training resumes.
6. Released `.licht` files remain readable by all future versions.
7. Older apps inspect newer files read-only; they write only when the file declares it safe.
8. Unknown chunks survive declared-safe writes and compaction; unknown JSON fields survive
   semantically.
9. Second instance = read-only or Save As; concurrent writers unsupported.
10. Restore is 1:1 for the ownership matrix; training resume is statistical, not bit-exact.
11. No ImGui internals, theme, language, or machine-local HUD state in the file.
12. CRCs catch corruption, not malice.
13. Compaction may need temp disk ≈ live project size; it is explicit or idle maintenance.

## 9. Top risks & mitigations

| Risk | Mitigation |
|---|---|
| Snapshot gate unachievable on low-end PCIe | P0 measures before anything ships; fallback = renegotiated SLA or storage versioning, decided on numbers |
| Missed dirty epoch persists stale state | Chapters default dirty without positive clean-proof; explicit save is always full-fidelity |
| Head-slot write bug corrupts master | Active slot never rewritten; commit record cross-checks; repair path never auto-selects |
| Windows mapped readers block replace | Delete-sharing on all handles; release gate covers held-map replacement |
| Ownership matrix drifts as features land | Matrix is a checked-in artifact; P8 matrix tests fail on undeclared state |
| Sidecar/master lineage confusion | UUID lineage validation; recovery never trusts filenames or mtimes |
| Old-writer data loss via known-chapter rewrite | Version-gated: newer chapter versions are opaque to old writers, capability bits enforced |

## 10. Execution state & resume instructions (2026-07-22, P0 exited + P1 landed)

Branch `licht_format`. The three-way adversarial review (Codex max ×2 rounds, Grok ×2 rounds,
synthesis) is **done**; this document is the frozen output. Verdict: SOUND-WITH-FIXES, all fixes
folded in above. Do not reopen §1 decisions without new evidence.

**Landed on the branch (checked in 2026-07-22, four commits):**

1. **P0 artifacts** — this plan (`PROJECT_FORMAT_PLAN.md`),
   `docs/licht_format_spec.md` (byte grammar + state machines),
   `docs/licht_ownership_matrix.md` (109 rows), `docs/licht_uuid_semantics.md`,
   `docs/licht_p1_node_uuid_migration.md`, `tools/licht_inspect/` (reference parser +
   `spec_byte_verifier.py` second reader + golden fixtures + oracle corpus + conformance
   battery `run_conformance.py`: 137,730 full + 475,952 fuzz + 4,349 quick — all green),
   `tools/licht_p0/` (snapshot bench + OS-semantics prototype, POSIX 8/8 PASS).
2. **P1 foundations** — node UUID identity migration S1–S5 (`core/uuid.*`, scene/undo/
   selection/sequencer/python/MCP/TCP, additive APIs only) + all §5 gap fixes (LFKP v2 +
   `CHECKPOINT_MIN_SUPPORTED_VERSION` + per-version flag whitelists, frozen-ranges reapply
   after adoption, ADMM restore, `markPayloadDiverged`, `getConfigDir` collapse, …) +
   P1.2 field-wise component config serialization (`training/components/config_serialization.*`).
   Built, 131 C++ + 7 Python targeted gtests green, 7k-iter smoke green, dual-reviewed.
3. **P1.3** retained-DOM JSON chapter layer (`io/json_chapter_dom.*`, `lfs::Result` API).
4. **v1 container prototype** — `src/io/project/`, implements the *refuted* v1 grammar
   (footer authority, `prev_index_offset`). **Input to the P2 rewrite only** per the §7
   disposition list — do not build features on it, do not extend it.

**Resume checklist, in order:**

1. ~~Gap-fix review/build/test~~ **DONE** (landed, see above).
2. ~~P0 packets (a)–(e)~~ **DONE** (landed, see above; Windows column of packet (d) still
   pending — needs a Windows box/CI, blocks P2 *exit*, not P2 start).
3. ~~P0 exit review~~ **DONE** — verdict below; conditions (A)–(C) remain live gates on
   later phases.
   *EXIT VERDICT (2026-07-18, fable synthesis): PROCEED to P1/P2 freezes. Read-side format
   contract complete (grammar/matrix/UUID note/two independent parsers/613k conformance+fuzz
   cases). Conditions: (A) Windows os_semantics execution blocks P2 exit; (B) owner snapshot-SLA
   decision blocks P4 autosave productionization; (C) writer-side gates (G9–G12, disk-full,
   clean-proof reuse, refuse-write enforcement, dirty-CKPT escalation, real process-kill crash
   matrix) are P2/P4/P6 acceptance criteria measured against the conformance battery; the P2
   C++ reader must agree with both Python parsers on the full corpus (decisive third
   implementation). Bench numbers to be re-proven on the production snapshot service, ≥2 rigs.*
4. ~~P1~~ **DONE** (landed; formal matrix-row round-trip proof arrives with P3 chapters).
5. **P2 — NEXT.** Dispatch the container core rewrite (§7 P2 scope + prototype disposition)
   against the frozen spec. Acceptance (all mandatory): (i) reproduces every golden fixture in
   `tools/licht_inspect/fixtures/` byte-for-byte; (ii) every file the writer produces passes
   the full conformance battery; (iii) the C++ reader agrees with both Python parsers on the
   complete corpus; (iv) writer-side gates from condition (C) incl. process-kill crash matrix;
   (v) `lfs::Error`/`Result<T>` conventions + `tools/error_debt_census.py` ratchet clean.
6. **OPEN OWNER DECISIONS** (ask before the affected phase, not before P2 start):
   (a) snapshot pause SLA — bandwidth-scaled (recommended) vs D2D clone vs hard-gate+defer;
   blocks P4 autosave only. (b) gap #12 point-cloud/mesh chunks — distinct `PCLD`/`MESH`
   fourccs (recommended) vs broadened `SPLT`; blocks the P3 chunk registry freeze.
   (c) §2.4 `SPLT` scope for imported/live-RAD nodes (matrix U1); blocks P3.
7. P3..P8 in order per §7.

**Working agreements:** fable plans/audits (≤2 agents, hardest judgments only), Codex/Grok
execute frozen specs (user directive for this campaign); `lfs-build-guard` behind
`flock /tmp/lfs_build.lock`, max `-j2`, builds deprioritized behind higher-priority sessions
on this box; targeted gtests only; live GUI validation for GUI phases; compute-sanitizer for
CUDA changes; commit only when asked, plain messages without Claude trailers. Review artifacts
(Codex session `019f744a-b3ba-71b0-80b9-632308e9c89a`, round logs) were session-scratchpad
only — the surviving authority is this document.
