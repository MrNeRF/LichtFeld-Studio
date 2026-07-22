# LichtFeld UUID Semantics (`.licht`)

**Status:** P0 packet (e) — normative intent for P1 implementation.  
**Plan reference:** `PROJECT_FORMAT_PLAN.md` §1 decisions 6–7, §2.1, §3, §4.  
**Scope:** How every 128-bit UUID behaves across create, import, duplicate, paste,
undo/redo, save, compaction, and recovery. Schema freezes and chapter codecs are
out of scope here.

This note is checked in **before** any chapter schema freezes. Where live code
contradicts the plan, the contradiction is flagged as a **TENSION** block rather
than silently resolved.

---

## 0. Vocabulary

| Term | Meaning |
|---|---|
| `project_uuid` | Stable identity of the *project* across Save As / compaction / path moves. Superblock (§2.1). |
| `file_uuid` | Identity of *this on-disk incarnation*. New on every compaction / Save As / first publication (§3). |
| `commit_uuid` | Identity of one committed generation (head slot + commit record, §2.1). |
| `snapshot_uuid` | Identity of one coherent training/UI capture stamped at the trainer safe point (§3). |
| `node_uuid` / `instance_uuid` | Persistent identity of one scene graph node (and of multi-instance chunks keyed by that node). Replaces v1's 32-bit `instance_id`. |
| Display name | Human-facing label (`SceneNode::name` today). **Not** identity. |

All of the above are RFC 4122 128-bit UUIDs stored as 16 raw bytes (see §8).

---

## 1. Decisions (one rule each)

These eight rules are the normative outcomes of this packet. Items marked
**PROPOSED** need owner sign-off before P1 freezes schema fields that depend on
them; unmarked items are treated as settled for implementation planning.

### D1. Node created in-app → UUIDv4 (random)

**Rule:** Every new scene node minted by application code receives a fresh
**UUIDv4**. The same generator is used for `project_uuid`, `file_uuid`,
`commit_uuid`, and `snapshot_uuid`.

**Rationale:** Node and commit ordering is already carried by explicit structure
(`parent` + child order in `SCNG`, `head_sequence` for commits — plan §2.1).
UUIDv7's timestamp ordering therefore buys little, while embedding creation time
in every node of a *shared* project file is a mild privacy leak. Random v4 keeps
identity generation independent of wall clocks and of sortability expectations.

### D2. Node duplicated in-app → fresh UUID for the copy

**Rule:** `Scene::duplicateNode` (and any operator/MCP/Python path that ends in
it) assigns a **new** `node_uuid` to every node in the duplicated subtree. The
source UUIDs are never reused.

**Provenance (PROPOSED):** `SCNG` may carry an optional
`duplicated_from_uuid` (16 B, zero/absent = none) on the root of a duplicate only.
It is non-authoritative telemetry for the UI ("copy of …") and is **not** used
for identity, selection, or chunk lookup. v1 may omit the field entirely;
implementations must tolerate its absence.

### D3. External import → always a fresh UUID (never deterministic from path)

**Rule:** Importing a PLY / SOG / SPZ / RAD / mesh / COLMAP dataset / checkpoint /
PLY-sequence into a project **always** mints new `node_uuid`s, even if the same
filesystem path was imported earlier into the same project.

**Rationale & interactions:**

- Re-import is a new graph object (the user may edit, delete, or keep both).
- **Relink** of an external artifact (plan `REFS` fingerprints) does **not**
  change `node_uuid`; it rebinds path/fingerprint on an existing node. Identity
  therefore cannot be `f(path)`.
- **Open the same `.licht` twice** restores UUIDs from the file — that is not an
  import. Second concurrent writer is read-only or Save As (plan §1 decision 13);
  both views share the same on-disk UUIDs for the generation they pin.
- Dataset node trees (dataset root, camera groups, cameras named from COLMAP
  image names, `Model` / `PointCloud`) also get fresh UUIDs per load; display
  names may collide with prior imports and are uniquified as today
  (`makeUniqueNodeName`).

### D4. Undo/redo across delete/recreate → UUID survives

**Rule:** Restoring a deleted node via undo **must** reattach the **same**
`node_uuid` that the node had before deletion. Redo of a create undoes that
identity again (tombstone in live graph; UUID reserved in the history entry).

**Constraint on undo snapshot shape (plan §2.4):**

- `SCNG` is the **durable project DTO** only. It is **not** the undo snapshot
  shape.
- The undo system must keep a **history-private** capture that includes, per
  node at minimum: `node_uuid`, parent UUID (not name), type, transform,
  visibility/lock/training flags, payload handles, and per-node selection-mask
  slices when present.
- Replaying history must call an identity-preserving restore path
  (`restoreNodeWithUuid(...)`), **not** the public `addSplat`/`addGroup`/… path
  that mints a new UUID.

See §3 (current code) and **TENSION-UNDO** below.

### D5. Project-level UUIDs

| Event | `project_uuid` | `file_uuid` | `commit_uuid` | `snapshot_uuid` |
|---|---|---|---|---|
| New untitled project | fresh | fresh (on first durable write) | — | — |
| Explicit save (append generation) | unchanged | unchanged | **fresh** | **fresh, always non-null** — every commit stamps a fresh snapshot UUID regardless of whether a training snapshot ran (spec §5; the parser rejects null) |
| Training autosave sidecar | same as master | **fresh non-null** (not the recovery key — the lineage tuple is) | **fresh non-null** (ditto) | **fresh** per capture; sidecar stores `{project_uuid, base_explicit_commit_uuid, autosave_sequence, snapshot_uuid}` (§3) |
| Compaction | unchanged | **fresh** | **fresh** (compaction commit) | **fresh, non-null** |
| Save As | unchanged | **fresh** | **fresh** | **fresh, non-null** |
| Explicit "Duplicate Project" | **fresh** | **fresh** | **fresh** | **fresh, non-null** |
| Open / recover | restored from file | restored | restored from active head | restored from commit/sidecar |

**When does `project_uuid` change?** Only on an **explicit "Duplicate Project"**
(or equivalent "Save a copy as a new project" that deliberately forks lineage).
Path rename, move, Save As, compaction, crash recovery, and second-instance
open **must not** change it. Sidecar recovery predicate (plan §3) depends on
`sidecar.project_uuid == master.project_uuid`.

**Uniqueness scope:**

- `commit_uuid`: unique among all commits ever written for that `project_uuid`
  (no reuse after compaction). Collision stance: random 128-bit, no coordination.
- `snapshot_uuid`: unique among snapshots for that project; one snapshot UUID is
  stamped across **all** CPU chapters + GPU tensor descriptors of a single
  capture (plan §3 consistency proof).
- `file_uuid`: unique per file incarnation; used to distinguish two compactons of
  the same project sitting side by side.

### D6. Selection groups / `SELM` keyed by `node_uuid`

**Rule (target format):**

- `SELM` stores selection groups (global) + **per-node mask slices keyed by
  `node_uuid`** (plan §2.4).
- **Delete:** live `SELM` rows for that UUID are dropped with the node. The undo
  history entry retains the mask payload under the same UUID so undo can
  reinsert the row.
- **Undo-restore:** mask returns under the **same** UUID; no re-keying.
- **Duplicate:** the copy gets a **new** UUID and either (a) an empty mask, or
  (b) a **cloned** mask written under the new UUID. It must **not** share or
  alias the source UUID's slice. Default for v1: **clone mask if source had
  selection; else empty**.

### D7. Cross-project paste

**Rule:** Today clipboard paste is **in-process only**
(`SceneManager::copySelectedNodes` / `pasteNodes` /
`copySelectedGaussians` / `pasteGaussians` — see §2.5). There is no
cross-project or cross-process paste.

**Reserved future semantics:** if/when paste crosses project boundaries (OS
clipboard serialization, multi-window, etc.), pasted nodes **always** receive
**fresh** UUIDs in the destination project. Source UUIDs may be recorded only as
non-authoritative provenance (same rules as D2). Never insert a foreign
`node_uuid` into a live project graph.

### D8. Generation source, concurrency, byte order

| Topic | Rule |
|---|---|
| RNG | OS CSPRNG: `getentropy` / `getrandom` (POSIX), `BCryptGenRandom` (Windows). Fallback: `std::random_device` only where it is documented to be non-deterministic. **Not** a seeded PRNG shared with training. |
| Thread-safety | Generator is process-wide and mutex- or OS-serialized; safe to call from UI, IO, and training helper threads. |
| Collision stance | 128-bit random space; **no** local uniqueness table, **no** coordination across processes. Birthday bound is accepted. Debug builds may assert on in-process duplicates of `node_uuid` within one live scene as a programming-error check only. |
| Canonical bytes | **RFC 4122 layout (big-endian field order) stored as 16 raw bytes.** The container is little-endian for multi-byte integers (§1 decision 15), but UUID bytes are **not** byte-swapped: the on-disk sequence is exactly the RFC 4122 binary representation (same as the conventional hex string `8-4-4-4-12` when printed). Readers never reinterpret the 16 bytes as host-endian `uint64_t[2]` without an explicit encode/decode step. |
| Variant/version | Version nibble = 4; variant bits = RFC 4122. |

---

## 2. Current runtime identity (code as of `licht_format`)

There is **no** persistent 128-bit node UUID in the live scene graph today.
Identity is a dual system of **display name** + **session-local `NodeId`**.

### 2.1 Scene core

| Symbol | Location | Behavior |
|---|---|---|
| `using NodeId = int32_t` | `src/core/include/core/scene.hpp` | Session-local dense id. |
| `SceneNode::id`, `::name`, `::parent_id` | same | Name is unique in `name_to_id_`; id is primary for parent/child links. |
| `Scene::insertNode` | `src/core/scene.cpp` | Allocates `id = next_node_id_++`; rejects duplicate **names**; registers `name_to_id_[name] = id`. |
| `Scene::removeNode` / `removeNodeInternal` | same | Lookup by **name**; frees the name key; does **not** recycle `NodeId`s. |
| `Scene::duplicateNode` | same | Deep-copies subtree; `generate_unique_name(base + "_copy")`; calls `addGroup` / `addSplat` / … → **new** `NodeId`s. Sets `payload_diverged = true` on splat copies. |
| `Scene::renameNode` | same | Rebinds `name_to_id_`; **preserves** `NodeId`. |
| `Scene::clear` | same | Clears maps; resets `next_node_id_ = 0`. |
| `makeUniqueNodeName` | anonymous namespace in `scene.cpp` | Display-name uniquifier (`base`, `base_2`, …). |

### 2.2 SceneManager (imports, clipboard, groups)

| Path | Symbol | Location | Identity effect |
|---|---|---|---|
| Open/replace splat or mesh | `SceneManager::loadSplatFile` | `src/visualizer/scene/scene_manager.cpp` | `clear()` then `addSplat`/`addMesh` with `path.stem()` as name. Handles `.ply`/`.sog`/`.spz`/`.rad` via extension. |
| Add alongside existing | `SceneManager::addSplatFile` | same | Uniquifies name; fresh `NodeId`. |
| COLMAP / dataset | `SceneManager::loadDataset` → `lfs::training::loadTrainingDataIntoScene` | `scene_manager.cpp`, `src/training/training_setup.cpp` | Builds dataset tree: `addDataset(filename)`, `Model`/`PointCloud`, camera groups, `addCamera(image_name, …)`. All fresh `NodeId`s after `clear()`. |
| COLMAP cameras only | `SceneManager::loadColmapCamerasOnly` | `scene_manager.cpp` | Camera nodes only. |
| Checkpoint resume | `SceneManager::loadCheckpointForTraining` | same | Validates, `clear()`, reloads dataset, replaces point cloud with checkpoint splat as `"Model"`. |
| PLY sequence node | `SceneManager::addPlySequenceNode` | same | `scene_.addPlySequence(...)` with unique name. |
| Duplicate (UI/MCP) | `cmd::DuplicateNodeById` → scene path; MCP `scene.duplicate_node` | `src/app/mcp_gui_tools.cpp` | Ends in `Scene::duplicateNode`. |
| Copy / paste nodes | `copySelectedNodes`, `pasteNodes` | `scene_manager.cpp` | In-process clipboard; paste mints `"Pasted_N"` names and **new** `NodeId`s; no cross-project channel. |
| Copy / paste Gaussians | `copySelectedGaussians`, `pasteGaussians` | same | New splat node `"Selection_N"`. |
| Delete with undo | `removePLY` / related + `SceneGraphPatchEntry` | `scene_manager.cpp`, `undo_entry.cpp` | History keyed by **name**. |

### 2.3 Undo / redo

| Piece | Location | What is captured | Identity on restore |
|---|---|---|---|
| `UndoHistory` | `src/visualizer/operation/undo_history.{hpp,cpp}` | Stack of `UndoEntry`; transactions compound. | N/A |
| `SceneGraphNodeSnapshot` | `src/visualizer/operation/undo_entry.hpp` | **`name`, `parent_name`, type, transform, payload clones** — **no UUID, no `NodeId`**. | — |
| `restoreNodeSnapshot` | `undo_entry.cpp` | Calls `scene.addGroup/addSplat/…` by **name** | **New `NodeId`** via `insertNode` |
| `SceneGraphPatchEntry::applyState` | same | Remove by name, restore from snapshot | Name continuity; **id discontinuity** |
| `SceneSnapshot` | same | Selection mask + transforms + deleted-gaussian masks keyed by **node name** | Name lookup; does not recreate missing nodes |
| Operator wiring | `src/visualizer/operator/operator_registry.cpp` | `TransactionGuard` + `undoHistory().commitTransaction` | — |
| Edit undo/redo ops | `src/visualizer/operator/ops/edit_ops.cpp` | `UndoOperator` / `RedoOperator` | — |

### 2.4 Selection today (pre-`SELM`)

| Piece | Location | Keying |
|---|---|---|
| Gaussian selection mask | `Scene::selection_mask_` | **Global** concatenated tensor over visible/topology order — not per-node UUID slices. |
| Selection groups | `Scene::selection_groups_` | Group id 1–255; not node-keyed. |
| Node multi-select | `SelectionState::selected_nodes_` (`NodeId` set) | `src/visualizer/scene/selection_state.hpp` — **breaks** if a node is deleted and undo-restored (new `NodeId`). |

### 2.5 v1 container prototype (`instance_id`)

The unreleased v1 prototype (`src/io/project/project_container.cpp`,
`src/io/include/io/project_container.hpp`) uses a **32-bit** `instance_id` on
chunk headers/index rows. Callers pass it explicitly; **there is no
`instance_id = hash(name)` implementation in the tree**. The plan's phrase
"v1's `instance_id = hash(name)` is deleted" is a **design intent** (refuted
grammar + weak identity), not a live code path to grep for.

Plan v2 replaces it with `instance_uuid[16]` (decision 6, §2.1 chunk header).

---

## 3. Lifecycle matrix (target behavior)

| Event | `node_uuid` | Display name | Notes |
|---|---|---|---|
| Create (group/splat/mesh/…) | **fresh v4** | caller / uniquified | `insertNode` path gains UUID assignment. |
| Duplicate | **fresh** per node in subtree | `name_copy` uniquified | Optional `duplicated_from_uuid` on root only (PROPOSED). |
| Rename | **unchanged** | new label | Name is not identity. |
| Reparent / reorder | **unchanged** | unchanged | |
| Delete | removed from live graph | freed | History retains UUID + payload. |
| Undo delete | **same UUID restored** | same name if free; conflict policy: restore name or uniquify **without** changing UUID | Name conflict after unrelated create is a UI edge case; UUID wins. |
| Redo delete | removed again | | |
| Import external file | **fresh** | stem / dataset rules | Even for identical path. |
| Relink `REFS` target | **unchanged** | usually unchanged | Fingerprint/path update only. |
| Load `.licht` | restored from `SCNG` | restored | Placeholders then hydrate (plan §4). |
| Save As / compact file | **unchanged** (nodes) | unchanged | New `file_uuid` only at file level. |
| Duplicate Project | all node UUIDs **fresh** (fork) | may keep labels | New `project_uuid`. **PROPOSED** detail: deep-copy payload, re-key `SELM`/`SPLT` to new UUIDs. |
| In-process paste | **fresh** | `Pasted_N` / `Selection_N` | Same as duplicate for identity. |
| Cross-project paste | **fresh** (future) | | See D7. |

---

## 4. File-level UUID placement (plan crosswalk)

| UUID | Lives in | Written when |
|---|---|---|
| `project_uuid` | Superblock (immutable after create) | Project creation; Duplicate Project |
| `file_uuid` | Superblock | Create, compaction, Save As (§3) |
| `commit_uuid` | Head slot + commit record | Every successful publish (§2.2 step 2) |
| Parent / explicit-ancestor commit UUIDs | Commit record | Append chain |
| `snapshot_uuid` | Commit record; autosave sidecar lineage | Training safe-point stamp; sidecar replace |
| Node / instance UUID | `SCNG` nodes; chunk `instance_uuid`; `SELM` keys; `SPLT` rows | Node mint; save of dirty nodes |

**Autosave recovery predicate** (plan §3) — exact:

```
sidecar.project_uuid == master.project_uuid
&& sidecar.base_explicit_commit_uuid == master.head.commit_uuid
&& sidecar complete + CRC-valid
```

Stale sidecars (base ≠ head) are ignored and deleted — never offered.

**Restore ordering** relevant to UUIDs (plan §4): nodes parent-first in saved
child order → training model node before trainer → selection groups → selection
masks only after the node set matches (by UUID) → node selection last.

---

## 5. Name-hash / weak-identity sites for P1 to kill

### 5.1 No live `hash(name)` 

A full-repo search finds **zero** call sites that compute
`instance_id = hash(name)` (or equivalent). The plan's deletion target is the
**v1 design + prototype field**, not an existing hash helper.

### 5.2 Prototype `instance_id` (uint32) — delete / replace with `instance_uuid[16]`

These are the concrete symbols that still encode the weak multi-instance key.
P2 rewrites the container; P1 should stop inventing any new `uint32` instance
keys.

| File | Symbols / lines (approx.) |
|---|---|
| `src/io/include/io/project_container.hpp` | `ChunkHeader::instance_id`; `IndexRow::instance_id`; `ChunkOptions::instance_id`; `ChunkInfo::instance_id`; `ProjectReader::find(fourcc, instance_id)` |
| `src/io/project/project_container.cpp` | Index parse assigns `info.instance_id`; `find` compares `instance_id`; writer `record_row` / `write_chunk` / `copy_chunk` propagate `instance_id`; INDX self-row uses `instance_id = 0` |
| `tests/test_project_container.cpp` | Write options `{.instance_id = 1/2/seed/4}` |

**Not** project identity (do not confuse):

- `src/python/lfs/py_gizmo.{hpp,cpp}` — `TransformGizmo` session `instance_id` counter (UI gizmo registry). Unrelated to `.licht`.

### 5.3 Name-as-identity (runtime) — migrate to `node_uuid` in P1

These are the **de facto** identity surfaces that must stop treating display
names as durable keys once UUIDs land. Non-exhaustive but P1-critical:

| Area | Sites |
|---|---|
| Scene maps | `Scene::name_to_id_`, `getNode`/`getMutableNode`/`getNodeIdByName`, `removeNode(name)`, `renameNode`, `setNodeTransform(name)`, `training_model_node_` string |
| SceneManager | `splat_paths_` keyed by name; `removePLY(name)`; `setPlyPath`/`clearPlyPath`/`movePlyPath`; selection APIs that take `string`; history capture root names |
| Undo | `SceneGraphNodeSnapshot::{name,parent_name}`; `deleted_masks_before_` map by name; `transforms_before_` by name; `restoreNodeSnapshot` name lookup; `SceneGraphNodeMetadataSnapshot::name` |
| Events | `PLYAdded`/`PLYRemoved` name fields; many `cmd::*` events keyed by path/name |
| Training | Hard-coded `"Model"`, `"PointCloud"`, `"loaded_model"`, `"Cameras"` in `training_setup.cpp`; `setTrainingModelNode(string)` |
| Sequencer / UI | Sequence node by name (`getNodeIdByName(sequence_node)` in sequencer UI) |
| Python / MCP | Name-based node APIs; `scene.duplicate_node` takes name |

P1 exit criterion (plan §7): *node UUIDs through scene / undo / selection /
sequencer* with stable round-trip — names remain labels only.

---

## 6. TENSION blocks (code vs plan)

### TENSION-IDENTITY — no persistent node UUID yet

- **Plan:** decision 6 — persistent 128-bit UUID per scene node; names are labels.
- **Code:** `NodeId` + unique `name` (`scene.hpp` / `scene.cpp`).
- **Resolution direction:** add `SceneNode::uuid` (16 B); assign in `insertNode`
  (or a dedicated mint path); keep `NodeId` as ephemeral runtime handle;
  stop requiring name uniqueness for identity (uniqueness may remain a UX
  constraint).

### TENSION-UNDO — restore mints a new `NodeId`

- **Plan / D4:** undo of delete must restore the **same** identity.
- **Code:** `restoreNodeSnapshot` → `addSplat`/`addGroup`/… → `insertNode` →
  `next_node_id_++` (`undo_entry.cpp`). Snapshot struct has no UUID field.
  `SelectionState` holds `NodeId`s that go stale across topology undo.
- **Resolution direction:** extend `SceneGraphNodeSnapshot` with `node_uuid` +
  `parent_uuid`; add `Scene::restoreNode(uuid, …)` that reinserts with a fixed
  UUID; rebind selection by UUID. Do **not** overload durable `SCNG` as the
  undo buffer (plan §2.4).

### TENSION-INSTANCE-ID — prototype still uint32

- **Plan:** chunk key `{fourcc, instance_uuid}`; "at most one live row per
  `{fourcc, instance_uuid}`" (decision 7).
- **Code:** v1 prototype `uint32_t instance_id` + footer/`prev_index_offset`
  grammar (explicitly refuted; keep only CRC/streaming pieces per plan §10).
- **Resolution direction:** P2 rewrite; no new features on `instance_id`.

### TENSION-HASH-NAME — design deleted before it was implemented

- **Plan:** "v1's `instance_id = hash(name)` is deleted".
- **Code:** no hash-name helper exists; only free-form `instance_id` and
  name maps.
- **Resolution direction:** treat as already-deleted design; P1/P2 implement
  UUIDs only. Do not introduce a temporary hash(name) shim.

### TENSION-SELM — selection not keyed by node UUID

- **Plan:** `SELM` per-node mask slices keyed by node UUID; restore only after
  node set matches (§2.4, §4).
- **Code:** one global `selection_mask_` sized by total Gaussians; node
  selection is a `NodeId` set.
- **Resolution direction:** P1/P3 introduce per-node slices; migration from
  global mask at save/load boundaries.

### TENSION-IMPORT-STABLE-NAMES — re-import looks like "same" node

- **Plan / D3:** re-import = new UUID.
- **Code:** re-import after `clear()` reuses display names like `"Model"` and
  `path.stem()`, so users perceive continuity that the file format must **not**
  treat as identity.
- **Resolution direction:** document in UI if needed; format always mints
  fresh UUIDs on import.

---

## 7. Implementation notes for P1 (non-normative checklist)

1. Introduce `lfs::core::Uuid` (16 B) + `generate_uuid_v4()` in core, OS CSPRNG,
   unit tests for version/variant bits and encode/decode hex.
2. `SceneNode::uuid`; assign in `insertNode` unless caller supplies a restore
   UUID (history / `.licht` load only).
3. Extend undo snapshots with UUIDs; identity-preserving restore API.
4. Selection + events: dual-write name for UI strings, UUID for authority.
5. Do not teach the v1 container `instance_id`; wait for P2 `instance_uuid`.
6. Golden tests: create → rename → save → load (UUID stable, name changed);
   delete → undo (UUID stable, `NodeId` may change); duplicate (UUIDs differ);
   import twice (UUIDs differ); Save As (`project_uuid` same, `file_uuid` new).

---

## 8. Decision summary (one line each)

1. **Create:** fresh **UUIDv4** (random; no timestamps in shared files).  
2. **Duplicate:** fresh UUID for every copied node; optional PROPOSED
   `duplicated_from_uuid` provenance only.  
3. **Import:** always fresh UUIDs; never derive from path; relink keeps UUID;
   open `.licht` restores file UUIDs.  
4. **Undo/redo:** UUID survives delete/restore; undo snapshots are
   history-private and must store UUID (SCNG ≠ undo shape).  
5. **Project UUIDs:** Save As/compaction → new `file_uuid`, same
   `project_uuid`; `project_uuid` changes only on explicit Duplicate Project;
   new `commit_uuid` / `snapshot_uuid` at their generation points.  
6. **SELM:** keyed by `node_uuid`; undo restores same key; duplicate clones
   mask under new UUID (or empty).  
7. **Cross-project paste:** none today; future = always fresh destination
   UUIDs.  
8. **Generator:** OS CSPRNG, thread-safe, no coordination; RFC 4122
   big-endian 16 bytes stored verbatim in the little-endian container.

---

*End of P0 (e) UUID semantics note.*
