# Splat crop workflow and operation contracts

The scene tree consumes `core/splat_operations.hpp` for direct splat-owned
crop volumes. The tree is the only element list. There is no extra inspector,
button panel, persistent modifier stack, or second operator registry.

## Before and after

| Before | After |
| --- | --- |
| Selecting a splat exposed New crop buttons | No additional controls below the tree |
| Toolbar could reuse a splat's existing crop | Toolbar activation creates a fresh volume on the selected splat or crop owner |
| Single-clicking a crop entered editing immediately | Single click selects; double click enters editing |
| Eye and Enable represented two easily confused states | Crop row has one small effect switch; outlines follow selection |
| Footer duplicated Edit, Duplicate and numeric controls | Viewport toolbar edits; context menu provides secondary actions |
| Checkmark could bake geometry | Checkmark / Enter / pie Finish retain the editable volume |

```text
Select splat -> toolbar crop -> new independent crop -> edit -> Finish
Select crop  -> outline only -> double click -> edit that instance -> Finish
Crop row switch -> enable/disable its filtering effect
Crop context menu -> Edit / Duplicate / Apply to geometry... / Delete
```

Selecting the splat does not expose creation controls in the scene tree or
its context menu. The viewport toolbar is the creation path. Selecting a crop
shows its outline without entering editing. Double click and the context-menu
Edit action use the same explicit crop-edit entry point. Crop shapes retain their
existing toolbar parameters, fitting and reset actions.

The switch represents only the crop effect's enabled flag. Helper visibility
is driven transiently by selection; it does not enable or disable filtering.
Ordinary objects retain their visibility eye. The bulk visibility eye is hidden
when the selection includes crops, avoiding a second ambiguous crop control.

## Minimal descriptors with real consumers

Fit and Reset remain toolbar actions and are not duplicated as unused descriptor fields.
The runtime table contains only the four existing splat element types.
There are no new `splat.*` classification strings and no capability
boolean matrix. Empty action IDs mean that no registered action is exposed.

| Element | Category | Set operator | Fit | Reset | Repeat |
| --- | --- | --- | --- | --- | --- |
| Crop Box | Crop volume | `crop_box.set` | `crop_box.fit` | `crop_box.reset` | `crop_volume.duplicate` |
| Ellipsoid | Crop volume | `ellipsoid.set` | `ellipsoid.fit` | `ellipsoid.reset` | `crop_volume.duplicate` |
| Crop Polygon | Crop volume | `crop_polygon.set` | Not exposed here | Not exposed here | `crop_volume.duplicate` |
| Collision | Generated result | None | None | None | None |

The scene tree invokes the live registered operators for executable controls. Invocation still passes through existing poll, validation and undo.
Tests register the actual scene and polygon operators and check every referenced
ID. This avoids validating a duplicate list of expected IDs or parsing Python
source as a substitute for registry discovery.

## Controls and lifecycle

- On a selected splat or one of its crops, activating the viewport crop tool creates a fresh
  volume using the chosen shape, without replacing existing volumes. Polygon
  creation enters drawing; cancellation retains the previous scene.
- A single click selects an existing crop and shows its outline. Double click
  enters viewport editing. Selecting another item leaves crop editing.
- The crop row switch toggles only its filtering effect through its registered
  setter, explicit UUID, validation and undo. There is no Enable footer button.
- Duplicate copies the existing volume's settings through the context menu.
- Checkmark, Enter and crop pie Finish preserve the volume, existing enabled
  flag and source geometry. Normal project Save persists the scene to disk.
- Apply to geometry is a separate crop context-menu action with confirmation,
  target name, and UUID/owner/lock revalidation. It bakes only the selected volume
  and removes it, preserving the other volumes and the existing undo transaction.
- Collision remains a generated result and retains its ordinary visibility eye.

Names are RML-escaped. Locks inherited from ancestors, disabled editing modes and
modal operations gate mutations. Numeric fields are not duplicated in the tree.
The Python edit/create/finish bindings marshal mutations onto the viewer thread.

## Repeated crop instances on one splat

`crop_volume.duplicate` is a real registered operator with a concrete scene-tree
consumer. It copies the selected box, ellipsoid or polygon into another direct
child of the same SPLAT, with a new UUID, unique name and independent payload.
It copies local transform, volume parameters, enabled state and helper visibility;
it never clones the splat geometry. The source and other crop instances remain
unchanged. The existing scene-graph patch history provides undo/redo.

The initial composition rule is **intersection**: a Gaussian must pass every
enabled crop attached to its splat. An inverse volume excludes its region.
Disabling one instance removes only its constraint. No union, arbitrary Boolean
expression or order-dependent modifier evaluation is introduced. The renderer
and render-crop selection filter already apply their lists with this semantics.

The crop context menu exposes Duplicate for all three shapes. The viewport
toolbar on a selected splat creates a fresh instance; Duplicate copies settings.
Repeated polygons retain an independent copy of their outline and local plane.
Point clouds and groups do not gain the repetition command.

Crop filters use a dynamically sized GPU storage-buffer table. A sorted owner
index points to separate contiguous box, ellipsoid and polygon ranges. Shader
lookup is logarithmic in the number of owners; filter evaluation visits only
its owner and optional global filters. There is no 16-volume truncation in
creation, duplication, conversion, clipboard, viewport or video export.

The fixed overlay prefix retains legacy point-cloud/single-volume filters,
selection, view and reveal controls. Dynamic offsets/counts are exact float
integers; oversized tables fail explicitly before allocation instead of losing
filters. Upload layout, descriptor range and per-frame byte cache follow the
actual initialized payload, including shrink/removal. The GPU storage-buffer
limit is checked before upload; allocation failures propagate normally.

Request-side crop filters carry CPU-only source node IDs. A gizmo preview replaces
its exact source, preserving siblings and inversion; a parent-only ambiguous
legacy preview cannot overwrite an arbitrary instance. A disabled source is not
reintroduced as an extra render filter. Source node identity stays CPU-only and is not uploaded as filter payload.

Video snapshots retain every ellipsoid, as they already retain boxes/polygons.
The legacy point-cloud active-ellipsoid field remains separate. Splat duplication
and node clipboard paths preserve repeated crop children; project serialization
continues to store their existing UUID-bearing scene nodes. The collective gate adds the optional boolean documented below.

Training-region bounds/loss semantics remain the existing training contract;
this increment does not claim multi-region training support or change it.
Applying one crop to geometry remains the explicit existing action for that
volume, distinct from baking an entire stack or exporting filtered geometry.

## Exact targets and standard errors

`readSplatOperation(owner, element)` returns `lfs::Result<SplatOperationSnapshot>`.
Errors use `lfs::Error`, Core domain, a standard code and diagnostic detail:
unsupported owner/element, invalid identity, inconsistent ownership, or missing
payload. No separate error enum is introduced.

The reader checks non-nil distinct UUIDs and valid distinct node IDs, SPLAT owner
type and reciprocal direct parent/child links. It copies parameters, local
transform and local visibility/lock flags. It contains no Gaussian tensors or
voxel words, never hydrates geometry and never generates identities or falls back
to names, selection or ancestors. Collision `enabled` is absent.

Read live nodes on the scene-owning thread or under the caller's existing read
protection. Snapshots are detached; later mutations must re-resolve identities.

The scene tree re-resolves owner and element identities before mutations.
Context-menu bake confirmations revalidate the captured UUIDs and ownership. The existing volume Set/Fit/Reset
operators now accept optional `node_uuid`; it takes precedence over `node` and
selection. An invalid, deleted or wrong-type explicit UUID cancels the command,
without falling back. Legacy callers that omit it retain their current behavior.

## Inventory versus runtime contract

The broader inventory still includes transform, selection, depth/crop selection
filters, Gaussian clipboard/deletion/field edits, simplification and scene
organization. They remain commands or shared tool/selection state, rather than
persistent object-owned modifier instances. They do not need unused descriptor
rows in the runtime table. All eight native toolbar tools remain
unchanged; cataloging them does not require another registry.

| Inventory category | Existing functions |
| --- | --- |
| Transform | Translate, rotate, scale, set/bake transform, align, scale by distance |
| Selection | Selection tool and submodes, all/none/invert, grow/shrink, opacity/scale queries and selection groups |
| Selection filters | Depth and crop filters |
| Direct edit | Mirror, Gaussian delete/copy/cut/paste and field writes |
| Reduction | Splat simplification producing another splat; no persistent editable recipe inferred |
| Organization | Duplicate/reparent splat and group selection; no new group-owned effects |

Group owners, point-cloud changes, new camera models, sequencing, postprocess,
training and Mesh-to-Splat remain outside this increment. The crop GPU parameter layout is updated;
Saved crop hierarchy is unchanged. The collective gate described below adds one optional project field and a SceneNode flag.

## Editing versus creation

Double click (or context-menu Edit) reopens the saved crop with its type fixed.
The current type stays selected; the other two types use the standard disabled
style. The native entry point rejects shape changes during that session.
Transform operations and Finish remain available. A new creation starts unlocked
and does not inherit the previous session's type lock. The main toolbar creates
a fresh instance even if a finished crop remains selected in the tree.

## Review cleanup — 30 October 2026

The splat-specific effect switch only intercepts direct SPLAT children.
Point-cloud crop eyes retain ordinary helper visibility and do not toggle the
payload's enabled flag. Removed the unreachable legacy box/ellipsoid drag state
and rollback blocks, unused Fit/Reset descriptor fields, duplicate Python
fallback checks and 28 translations per language left by the removed inspector.

Dynamic crop tables are transient rendering data. Scene node identity, crop
payloads remain unchanged. The collective gate below extends project serialization with an optional boolean. The shared switch style
continues to serve Preferences and the MCP popup; the scene-tree variant only
reduces its dimensions.

## Verification

Enable `BUILD_SPLAT_WORKFLOW_TESTS=ON` in an application build for
`lichtfeld_splat_crop_workflow_tests` / CTest `lichtfeld.splat_crop_workflow`.
The ordinary suite uses the real operator registry and retained RML tree without
requiring a window or GPU renderer. Its optional clipboard hydration regression
uses CUDA; opt in with `LFS_TEST_CROP_CLIPBOARD_GPU=1` on a GPU validation host.
The core readers have their own `lichtfeld_splat_operation_tests` target.

Final local Windows validation:

- 25 crop workflow/tree/packing tests, including the opt-in GPU clipboard test;
- 11 core reader/descriptor tests and 219 Python toolbar/preferences/UI tests;
- successful application/shader builds with sccache and FAST_COMPILE;
- clang-format, diff whitespace, I/O discipline and error-debt gates pass;
- English/Italian keys match, with no new undefined localization references.
  The broad localization scanner still reports 127 existing undefined references
  elsewhere; its baseline comparison is recorded separately.

Real GPU image comparisons used an isolated 64-Gaussian fixture and 120 crops
(40 per type) in both 3dgs and 3dgut. The fortieth box, ellipsoid and polygon each
changed the pixels; disabling it restored the reference exactly. Duplicating the
splat preserved all 120 crop children. Hiding the original through its crops and
removing it produced identical images, exercising owner isolation and index
remapping; undo restored the subtree. Removing 240 crop children shrank the
buffer without changing the nonrestrictive image, and regrowing one restrictive
crop reproduced the owner-isolated reference exactly.

The native GPU clipboard regression separately verifies duplicate and paste of
40 crops of each type, parent/UUID ownership and paste undo/redo. The pixel
checks cover the interactive viewport, not a new exhaustive matrix of video,
training, VR, stochastic transparency or every graphics driver.

## Virtual Crop container — 30 October 2026

Each SPLAT with direct crop children has one collapsible `Crop (N)` row in the
scene tree. This is a presentation container, not a Scene GROUP: crop UUIDs,
real parents, origins and transforms remain unchanged. Point-cloud rows keep
their previous behavior. The container is initially collapsed; expansion is
session-only. It is excluded from scene counts, selection ranges, Select All,
drag/reparent destinations and transform tools. Double click only folds it.

The container switch suspends all its crop effects without changing individual
`enabled` flags. Child switches are dimmed while suspended. Viewport filters and
video export honor the same gate; helper visibility remains independent.
The SPLAT flag `crop_effects_enabled` is saved as an optional SCNG boolean;
missing fields default to true, and malformed types are rejected. Duplicate,
clipboard and undo snapshots preserve it.

Right-click actions:

| Action | Result | Undo |
| --- | --- | --- |
| Enable/disable | Changes the collective gate only | One entry |
| Apply active crops to geometry | Prepares the intersection on detached data, applies its deletion mask, removes enabled volumes; disabled volumes remain | One transaction |
| Delete all crops | Removes every crop volume, preserves source geometry and unrelated children, resets the empty container gate | One transaction |

Bake requires an enabled container, resident SPLAT data and valid, unlocked
active crops. Delete requires every affected crop to be unlocked. Both commands
show a confirmation; its callback checks the exact owner UUID, member UUIDs and
scene revision before invoking the manager validation. Training/tool/modal locks
are checked again at execution. Individual volumes retain their existing edit
and removal commands.

The UI container does not introduce group-owned effects or change crop order:
all enabled crop volumes continue to combine by intersection, with inverse
volumes excluding their regions.

Validation of the collective container: 31 native workflow/tree/packing tests
(including CUDA clipboard), 11 core tests and 178 Python tests pass; 72 Python
API-completeness cases skip outside the embedded application. The complete
Windows application builds with sccache and FAST_COMPILE. Real GPU comparisons
confirm the collective gate in both 3dgs and 3dgut. Runtime checks cover reopening
a saved disabled gate, active-only bake retaining a disabled volume, delete-all
preserving geometry and undo/redo of both operations. These checks do not claim
an exhaustive manual review of every menu or graphics driver.

## General solidity review — 30 October 2026

The review found and corrected three selection inconsistencies:

- Python `Scene.apply_crop_filter` now checks the collective gate as well as each
  volume's enabled flag, matching viewport and export behavior.
- Native scene selection filtering evaluates every enabled crop with its own
  model index. It no longer chooses a single volume according to the selected
  tree row or applies that volume globally to other splats. Explicit selection
  volumes keep their global scope and replace the scene filters of their own
  shape, preserving that workflow.
- Point rendering's live preview rechecks the persistent volume/gate, so a stale
  gizmo flag cannot reactivate a suspended or individually disabled crop.

The training load rollback descriptor now preserves the collective flag. Removed
its now-unused selection lookup helper. The virtual tree's organization pass
uses constant-time snapshot lookups instead of repeatedly searching the entire
crop list for every child.

New native regressions exercise repeated crop intersection, selection with
another owner or no owner selected, isolation of the other model, gate off/on,
all three crop shapes and stale point-preview state. The final suites pass
33 workflow/tree/packing/CUDA tests and 11 core tests. Runtime Python checks on
an isolated 64-Gaussian project exercise each of the three shapes with gate
on/off. The complete application builds with sccache and FAST_COMPILE.

The ordinary Python toolbar/preferences suite passes 178 tests, with 72 native
API checks skipped by its external Python 3.11 runner. A supplementary attempt
to include the histogram suite cannot initialize its native fixture in that
runner (`ModuleNotFoundError: lichtfeld`, whose built module targets Python
3.12); those histogram cases are not reported as passing. Native selection and
embedded Python crop-filter behavior are covered separately above.

Scope of assurance: source review, native/RML event tests, real CUDA selection
and GPU image checks; no claim of exhaustive manual GUI review or allocation
failure injection. Existing user files were not modified during runtime checks;
the test application uses a separate LFS_HOME and a copied project.

Review GPU finale: ripetuti 120 crop (40 per tipo) in 3dgs e 3dgut; ogni quarantesimo crop modifica i pixel e spegnerlo ripristina esattamente il riferimento (delta 0). Istanza e progetto di test isolati, copia finale salvata e istanza chiusa.


### Crop su gruppi normali disabilitato — 30 ottobre 2026

Il supporto ai modificatori resta limitato ai singoli splat. Il comando Crop è
ora disabilitato quando la selezione include un GROUP della scena, anche insieme
a splat o volumi crop. Il controllo è presente nei poll Python e nativo;
l'attivazione diretta della toolbar rifiuta la stessa selezione prima di cambiare
strumento, forma o creare nodi. I percorsi già esistenti per splat, point cloud,
dataset e singoli volumi non sono modificati. Il contenitore virtuale Crop (N)
conserva il comportamento organizzativo già descritto.

Verifica: 106 test Python di toolbar, definizioni e registro strumenti passati,
compresi i nuovi casi GROUP e selezioni miste. Build con cache e FAST_COMPILE.

Verifica runtime finale: scena utente ripristinata con 2.997.954 Gaussiane; sul gruppo reale poll Python, poll nativo e record del pulsante Crop risultano disabilitati. Passati anche i 33 test C++ workflow. Preview aggiornata lasciata aperta sulla copia salvata prima della build.


## Local master alignment — 30 October 2026

Integrated upstream/master `b47363a6b9ecd2573dbc2030e5712fab2386afc3`
(31 upstream commits) using a merge, preserving the published branch history.
Local checkpoint before integration: `d321c45eb`; recovery branch:
`backup/splat-crops-before-master-20261030`. Merge commit: `6278a5072`.
No push and no PR metadata changes.

Integration decisions:

- Keep master's shader-aware, per-frame video-export pipeline and retain the
  complete list of repeated ellipsoid crops in its snapshots.
- Custom shader parameters keep their fixed address immediately after the
  overlay prefix. The dynamic crop directory/payload moves after that tail;
  both count toward the exact-address and GPU descriptor-range checks.
- Use master's shared full/incremental SCNG writer. Persist the crop gate only
  when false; true is the backward-compatible default, so camera-only records
  do not acquire an extra key. Upsert and sparse-update paths are covered by
  the crop persistence test.
- Preserve current master stubs, adding only the new crop declarations and
  avoiding duplicate gamepad signatures.
- Keep all new master locale keys. Crop controls, confirmations and the renamed
  Finish Crop Editing action are translated in all ten shipped locales.
- vcpkg manifest/baseline did not change between the branch base and this master.
  The existing installation already provides shader-slang 2026.5; no unrelated
  dependency upgrade was introduced.

Completed checks so far: complete Windows build with sccache and FAST_COMPILE;
290 Python tests; 34 native crop workflow/packing/CUDA tests and 11 core tests.
The new custom-tail packing regression and full/upsert/sparse gate round-trip
pass. Real GPU comparisons cover box, ellipsoid and polygon with an active
parameterized custom shader in 3dgs and 3dgut. Switching the backend can compile
its shader variant asynchronously: image references are taken only after that
variant demonstrably affects pixels. Every crop then restores the reference
exactly when suspended. Shader and disabled-gate state survive project reopen.
Four 4-frame video exports verify both renderers with crop on/off and a scene-
time callback, exercising per-frame snapshot capture; decoded output changes
with the collective gate.


The complete format suite passes: 137 passed, 3 skipped (UNC share and opt-in
large scale simulation), plus one disabled test. Both collision suites pass
(4 preview-upload and 5 volume-validation tests). The 20k-camera budget is kept
unchanged. Its first runs under other validation activity exceeded 5 seconds;
the isolated test passed in 1.8 seconds. After omitting the default crop key and
closing the test preview, the full suite passes without relaxing any budget.
The final focused suite also checks that enabling the gate removes its optional
JSON override. Final large-stack GPU validation is recorded in the external
scene-organization validation register.


## GUI follow-up and single-bake undo fix — 30 October 2026

The deeper GUI pass exposed a blocking case not covered by the earlier focused
suite: baking one crop while sibling crops remained restored the deleted volume
at the end of the child list. The geometry snapshot rejected that changed
ordering, so undo could fail to restore the splat. Single-volume removal now uses
the existing ordered batch graph patch, with payload capture scoped to the
removed volume. The bake transaction commits only after successful removal.
Legacy box and ellipsoid commands now group parameter changes, bake and removal
into the same undo transaction, matching the polygon and active-tool paths.

The native bake regression now includes repeated sibling crops, all three
shapes, active-tool and legacy-command paths, and locked target/ancestor cases.
It verifies one undo step, exact child order, restored geometry, sibling retention
and redo. Older tests were aligned with explicit editing and preservation of
sibling crop instances. Windows-only test build issues were fixed without
exporting private implementation methods; the persistence fixture uses positive
ellipsoid radii rather than a degenerate flat ellipsoid.

Completed follow-up checks:

- Native GUI crop/persistence filter: 21 passed.
- General GUI suite: 614 passed; RmlUI: 82 passed; sidebar input: 7 passed.
- Final focused workflow/packing/CUDA suite: 34 passed, clipboard GPU opt-in on.
- Python controller/toolbar tests: 162 passed; real-app numeric gesture history:
  12 passed, covering single/multiple selections, local/world space and all
  three transform operations.
- Actual mounted toolbar controller: 34 phases, including creation, type changes
  during creation, locked type on reopen, selection without editing, repeated
  instances, numeric transforms and single deletion with undo/redo.
- Fit, percentile fit, reset and invert on every shape; actual polygon modal
  drawing through native mouse events; single and collective bake, collective
  deletion, disabled gate, disabled siblings and undo/redo.
- Native registry and Python polls reject normal groups and mixed group/splat
  selections, while an individual splat inside a normal group remains editable.
  The native poll is queried through `ui_tool_describe`; the older Python
  `is_tool_available` mapping does not expose Crop and is not a Crop poll oracle.
- Real project reopen preserves six crop UUIDs, child order, individual enabled
  flags, collective gate and all 64 fixture Gaussians. GUI screenshots inspected.
- Final normal application build uses sccache and FAST_COMPILE, with the temporary
  `BUILD_VISUALIZER_TESTS` configuration restored to OFF. The isolated application
  was saved and closed through File > Exit. No push or PR metadata changes.

Physical desktop mouse automation remains unavailable in this Windows session:
`GetCursorPos` returns access denied. Native input events, retained-tree event
coverage, real controller callbacks and MCP window captures were exercised;
physical mouse drag/double-click/context-menu interaction is not claimed as
verified. The portable fixture also references an unavailable environment asset;
the final visual capture uses a solid background in its isolated copy.

Validation artifacts: `work/scene-organization/pr01-integrated/`, especially
`gui-native-crop-results.json`, `gui-ctest-details.log`, `gui-retest-routes-final.log`,
`gui-final-expected.json` and `gui-followup-results.json`.


### Crop localization completion — 30 October 2026

Translated all 15 crop inspector/container strings in DE/ES/FR/JA/KO/NL/PL/ZH
and updated the existing Finish Crop Editing action in the same eight languages.
Italian confirmations now explicitly identify the original splat geometry and
undoing the whole action. No locale keys were removed. All ten languages have
the same 3,542 keys and compatible placeholders; the eight completed languages
have no English fallback among the new crop strings. The Italian label Crop is
retained as the existing tool term.

Validation: locale completeness, all 21 headless localization contracts, the
hardcoded-UI audit, UTF-8/JSON layout and whitespace checks pass. This change is
limited to localized resources and documentation.
