# `.licht` project-state ownership matrix

Status: **normative for format 1.0**. P3–P8 implement this ownership model. The human-readable audit columns are historical; the machine-readable runtime inventory is current. U3 and U5 are v1 non-goals; U4 and U6 are resolved. A format field is not releasable until it appears here with one semantic authority.

“Serialized where today” describes the preimplementation `licht_format` checkout audited for this matrix; it is historical evidence, not permission to duplicate that state in `.licht`. A legacy value may remain physically present inside an opaque, byte-verbatim `CKPT` or `PPIS` payload while another chapter is its semantic authority. Such values must be ignored or reconciled as stated below.

## Normative vocabulary

- **Authority** is the only chapter allowed to decide a field's restored value. Mirrors, cached projections, provenance strings, and fields trapped inside an opaque legacy payload are not additional authorities.
- **Role-qualified fields are different state.** For example, active-trainer `means_lr` is exact-resume state in `CKPT`; next-run/preset `means_lr` is project UI state in `PRMS`.
- **Dirty trigger** means a project-persistence dirty event, not `RenderingManager::markDirty`, a GPU cache invalidation, or an undo-history mutation by itself.
- Every JSON chapter (`PROJ`, `PRMS`, `REFS`, `GUIL`, `EDTR`, `VIEW`, `SEQR`) uses the retained DOM required by plan decision 11. Typed accessors mutate that DOM in place. Unknown object members and unknown array-element members survive. UUID-addressed array elements are never rebuilt from a parallel typed vector.
- Paths stored in state chapters are logical reference keys or dataset-relative paths. Machine paths and their fingerprints live in `REFS`, except editor file locators, which are part of `EDTR`'s external-file session contract.

## Settled precedence and reconciliation

These rules are mandatory and take precedence over incidental duplication in today's serializers.

1. **Training model and optimizer are `CKPT` only.** A project containing a checkpoint must not emit a `SPLT` instance for its training node. The training model, optimizer, scheduler, strategy state, iteration, active mathematical parameters, bilateral grid, PPISP training state, PPISP controller, and sparsity ADMM state are restored only from `CKPT`.
2. **Non-training splats are `SPLT`.** Their resident LFSP state is never restored from `CKPT`. Imported PLY/SPZ/SOG nodes are always embedded; live-RAD nodes are never embedded (read-only, explicit bake) — resolved U1, owner decision 2026-07-30.
3. **Exact-resume parameters are `CKPT`; project/UI overrides are `PRMS`.** On project open, `CKPT` first hydrates the active trainer. `PRMS` then restores strategy presets and pending next-run UI values only; it must not mutate the active trainer. A later explicit user edit may go through the trainer's supported live-update path, but that is a new mutation, not load precedence. Today's blanket `ParameterManager::dirty_` application does not define format semantics.
4. **Training PPISP is `CKPT`; `PPIS` is only for a session without `CKPT`.** A commit must not contain an authoritative `PPIS` beside an authoritative training PPISP. Viewer overrides remain `VIEW`, not PPISP model state.
5. **Camera enablement is `SCNG`.** Load the checkpoint, build the saved scene, then apply each camera node's `training_enabled`. A legacy `TrainingParameters::disabled_camera_uids` inside LFKP is ignored for project restore and may only be used by standalone checkpoint import.
6. **External reference resolution is `REFS`.** Resolved dataset, RAD, background-image, environment-map, COLMAP, and PLY-sequence paths replace legacy path strings found in `CKPT`, `PPIS`, `VIEW`, or runtime managers. A fingerprint mismatch is loud; it never makes a stale legacy path win.
7. **Unsaved editor buffers are embedded in `EDTR` and flagged.** For an entry marked modified, the embedded bytes win over disk and the project/share UI must warn about the embedded-buffer secret-leak surface. A clean external file follows U5.
8. **Derived mirrors never win.** `RenderSettings::raster_backend` is canonical over its compatibility `gut` mirror; `SequencerController`'s playback speed and PLY clip FPS are canonical over `SequencerUIState` mirrors; selection-group counts and scene counts are recomputed.

### Mandatory restore order

1. Construct parameters, call the normal `setParameters()` path, and stage project restoration behind the existing first-GUI-frame gate.
2. Restore `SCNG` nodes parent-first and in saved child order; designate `training_model_node` before trainer construction; then restore selection groups, per-node selection slices after the node set matches, and node selection last.
3. Load scene/`CKPT`; load a non-checkpoint `PPIS` companion before restoring `ppisp_mode=AUTO`; apply `VIEW` after CLI overrides; enter split mode only through `SplitViewService::toggleMode`; restore each panel camera with `Viewport::setViewMatrix`, then pivot/home/speeds/orthographic scale while storing `R` directly; restore GT camera identity; set the one-shot `InputController::depth_range_initialized_` guard; finally call render `markDirty(ALL)`.
4. Restore `GUIL` only after native and Python/plugin panels have emitted a concrete panels-ready signal. Unmatched panel IDs stay in the retained DOM for a later plugin registration.

## `PROJ` — project manifest

| Field | C++ owner (struct/class @ file) | Serialized where today | Chapter authority | Precedence/reconciliation rule | Dirty trigger |
|---|---|---|---|---|---|
| Format/application versions, minimum reader/writer versions, required/optional capability declarations | No manifest DTO yet; closest current fields are `FileHeader::{container_version,min_reader_version}` and `ProjectReader` @ `src/io/include/io/project_container.hpp` | Current prototype file header only; no `PROJ` chapter | `PROJ` | Container envelope versions validate physical readability; `PROJ` owns semantic manifest compatibility. The two must be mutually compatible or open read-only/fail as the plan requires. | Application/schema capability change or migration |
| `project_uuid` | No semantic manifest owner yet; `FileHeader::project_uuid` and `ProjectReader::project_uuid()` @ `src/io/include/io/project_container.hpp` | Current prototype file header | `PROJ` | The immutable superblock is the physical duplicate. `PROJ.project_uuid` must equal it; mismatch is corruption, not precedence. | Project creation only |
| Created and last-modified timestamps | No current project DTO; current save generation is `FileHeader::save_generation` @ `src/io/include/io/project_container.hpp` | Not serialized as timestamps | `PROJ` | `created_at` is immutable. `modified_at` is generated by a successful commit and does not itself cause another dirty cycle. | Project creation; successful commit |
| Dataset linkage (manifest-level reference identity, not a path) | `TrainingParameters::dataset` @ `src/core/include/core/parameters.hpp`; `SceneManager::dataset_path_` @ `src/visualizer/scene/scene_manager.hpp` | Dataset path is in checkpoint JSON and `SceneManager` memory | `PROJ` | Links a project/dataset scene to the dataset-root record in `REFS`; `REFS` owns the locator and fingerprint, `SCNG` owns dataset/camera nodes. | Dataset association, detach, or relink |
| Per-node embed/reference decision log | `SceneNode::payload_diverged` @ `src/core/include/core/scene.hpp`; source paths in `SceneManager::splat_paths_` @ `src/visualizer/scene/scene_manager.hpp` | Only runtime source map and divergence bit | `PROJ` | This is an auditable decision record, not payload state. The chosen payload authority must agree with the live index (`SPLT`/proposed geometry payload versus `REFS`); disagreement is corruption. | Source bind/relink, generated/imported payload, or embed-mode decision |
| Import/source provenance, including legacy viewer `view_paths` and the loaded non-checkpoint `.ppisp` source path | `TrainingParameters::view_paths`; `SceneManager::{splat_paths_,dataset_path_,colmap_sparse_path_,ppisp_path_}` @ `src/core/include/core/parameters.hpp`, `src/visualizer/scene/scene_manager.hpp` | `view_paths` may be in checkpoint JSON; manager paths are runtime only | `PROJ` | Provenance never overrides `REFS`, `SPLT`, `CKPT`, or `PPIS`. The `.ppisp` bytes are authoritative in `PPIS`; original paths are informational after embedding. | Import, relink, or source replacement |
| Georeference block (added 2026-07-30): `crs` opaque UTF-8 string (never parsed, absent by default), `world_origin` f64[3] such that `world = local + world_origin`, `world_unit_scale` f64 (metres per project unit, default 1.0), `world_origin_provenance` enum `none/centralize_by_pointcloud/centralize_by_cameras/user/import` | Computed-then-discarded today: `centralize_scene` @ `src/io/loaders/loader_utils.hpp` (offset is a local, only logged); USD import scale @ `src/io/formats/usd.cpp` applied and dropped | Not serialized anywhere — the world offset of a centralized project is currently unrecoverable | `PROJ` | f64 is mandatory: f32 ULP at geo magnitudes is 3 cm–0.5 m. Loaders populate origin/scale/provenance at import; `none` provenance means the block is inert. The origin is a *different role-qualified field* from the derived scene-centre statistic and never rebuilt from payload means. | Import, centralization, explicit user georeference edit |

## `PRMS` — pending project/UI parameter state

`PRMS` snapshots `ParameterManager`, not the active trainer. The same C++ scalar can therefore occur once in `CKPT` as active exact-resume state and once in `PRMS` as a different, role-qualified next-run/preset value.

| Field | C++ owner (struct/class @ file) | Serialized where today | Chapter authority | Precedence/reconciliation rule | Dirty trigger |
|---|---|---|---|---|---|
| Pending UI `active_strategy` | `ParameterManager::active_strategy_` @ `src/visualizer/core/parameter_manager.hpp` | Memory only | `PRMS` | Selects which pending preset the UI edits. A running trainer's actual strategy comes from `CKPT`. | User selects a pending strategy |
| MCMC/MRNF/IGS+ session-default and current/pending copies of the complete mathematical `OptimizationParameters` set enumerated field-by-field in the CKPT parameter rows below | `ParameterManager::{mcmc_session_,mrnf_session_,igs_session_,mcmc_current_,mrnf_current_,igs_current_}` and `OptimizationParameters` @ `src/visualizer/core/parameter_manager.hpp`, `src/core/include/core/parameters.hpp` | Some values can be imported from config/checkpoint JSON; the six role-qualified copies are not serialized as a project snapshot | `PRMS` | Restore as pending presets only. `CKPT` wins for every corresponding active-trainer value. A pending background image or PPISP sidecar is a logical reference to `REFS`/`PPIS`, not a second raw-path authority. Process controls `headless`, `auto_train`, `no_splash`, `debug_python`, `debug_python_port`, and `config_file` are excluded below. | Preset import/reset or pending UI edit |
| Pending dataset/import UI values: `images`, `resize_factor`, `test_every`, `timelapse_images`, `timelapse_every`, `max_width`, `min_track_length`, `invert_masks`, `mask_threshold`, `centralize_dataset` | `ParameterManager::dataset_config_`; `DatasetConfig` @ `src/visualizer/core/parameter_manager.hpp`, `src/core/include/core/parameters.hpp` | `DatasetConfig::to_json` omits both timelapse fields and `centralize_dataset` | `PRMS` | These are next-run values. Corresponding values already governing a resumed trainer come from `CKPT`; dataset root comes from `REFS`. | Pending dataset option edit or dataset preset import |
| Pending loading policy: `use_cpu_memory`, `min_cpu_free_memory_ratio`, `min_cpu_free_GB`, `use_fs_cache`, `print_cache_status`, `print_status_freq_num`, `use_16bit_color` | `LoadingParams` inside `ParameterManager::dataset_config_` @ `src/core/include/core/parameters.hpp` | Dataset JSON via `LoadingParams::to_json` | `PRMS` | Resource/cache policy is pending project UI state. The active trainer's input-precision choice `use_16bit_color` is also captured role-qualifiably in `CKPT`; other active cache knobs are not exact mathematical resume state. | Pending loading-policy edit |

`ParameterManager::{loaded_,dirty_}` are runtime synchronization flags and are excluded; they are not project fields.

## `SCNG` — scene graph

| Field | C++ owner (struct/class @ file) | Serialized where today | Chapter authority | Precedence/reconciliation rule | Dirty trigger |
|---|---|---|---|---|---|
| Persistent 128-bit node UUID | No owner yet; current identity is runtime `NodeId SceneNode::id` and name maps @ `src/core/include/core/scene.hpp` | Not serialized | `SCNG` | UUID is the only persistent identity and the chunk instance UUID. Runtime `NodeId` is regenerated; names never identify references. | Node creation/duplication only |
| `type`, display `name` | `SceneNode::{type,name}` @ `src/core/include/core/scene.hpp` | Memory only | `SCNG` | Unknown future node types remain opaque. Name collisions do not alter UUID identity. | Add, rename, or type migration |
| `parent_uuid`, ordered child position | `SceneNode::{parent_id,children}` and `Scene::moveNode/reparent` @ `src/core/include/core/scene.hpp` | Memory only | `SCNG` | Restore parent-first in saved child order. Runtime parent IDs are UUID-resolved. | Add/remove/reparent/reorder |
| Local transform | `SceneNode::local_transform` @ `src/core/include/core/scene.hpp` | Memory/undo snapshots only | `SCNG` | Store local matrix. `world_transform` and `transform_dirty` are derived after hierarchy restore. | Local transform mutation |
| Optional per-node `georef_pose` (added 2026-07-30): f64[4] quaternion + f64[3] translation, node-local → project-world, absent by default | No runtime owner yet — `local_transform` is `glm::mat4` (f32) and cannot carry it | Not serialized | `SCNG` | Only for nodes imported in a frame different from the project's (E57 per-scan `pose` model on top of the `PROJ` file-level frame). Never a substitute for `local_transform`; when absent, the node lives in the project frame. Splitting transform authority further is forbidden. | Import with foreign frame, explicit georeference edit |
| `visible`, `locked` | `SceneNode::{visible,locked}` @ `src/core/include/core/scene.hpp` | Memory/undo snapshots only | `SCNG` | Effective visibility is derived through parents. | Visibility or lock edit |
| Payload source-divergence flag (`payload_diverged`) | `SceneNode::payload_diverged` @ `src/core/include/core/scene.hpp` | Memory only | `SCNG` | Means resident payload no longer matches its external source; it remains true across project saves until a verified external rebind/reload. It is distinct from an internal “changed since last commit” epoch. | Edit, generate, paste, merge, crop, or verified rebind |
| Training-model node UUID | `Scene::training_model_node_` (currently a name) @ `src/core/include/core/scene.hpp` | Memory only | `SCNG` | Resolve by UUID before trainer construction. Its payload is `CKPT`, never `SPLT`. | Training model designation/change |
| Per-camera `training_enabled` | `SceneNode::training_enabled`, `Scene::setCameraTrainingEnabled`, `Scene::getTrainingDisabledCameraUids` @ `src/core/include/core/scene.hpp` | Runtime node bool; duplicate disabled UID list in checkpoint JSON | `SCNG` | Apply after `CKPT` load; SCNG wins over `disabled_camera_uids`. | Camera enable/disable |
| Crop box: `min`, `max`, `inverse`, `enabled`, `color`, `line_width` | `CropBoxData` @ `src/core/include/core/scene.hpp` | Memory/undo snapshots only | `SCNG` | `flash_intensity` is transient and excluded. Viewer show/use toggles are `VIEW`. | Crop-box property or transform edit |
| Ellipsoid: `radii`, `inverse`, `enabled`, `color`, `line_width` | `EllipsoidData` @ `src/core/include/core/scene.hpp` | Memory/undo snapshots only | `SCNG` | `flash_intensity` is transient and excluded. Viewer show/use toggles are `VIEW`. | Ellipsoid property or transform edit |
| Camera identity/calibration/pose: `uid`, `camera_id`, `R`, `T`, `focal_x`, `focal_y`, `center_x`, `center_y`, radial/tangential distortion, `camera_model_type`, camera width/height | `Camera` @ `src/core/include/core/camera.hpp`; `SceneNode::camera_uid` @ `src/core/include/core/scene.hpp` | Rebuilt from dataset/checkpoint import | `SCNG` | Camera context is saved independently of the dataset root. Computed FoV, world-view transform, camera position, undistortion products, and CUDA state are rebuilt. | Camera import, calibration/pose edit, or context migration |
| Camera/image context: `image_name`, dataset-relative `image_path`, `mask_path`, `depth_path`, `normal_path`, image width/height, `has_alpha`, train/eval `split` | `Camera` @ `src/core/include/core/camera.hpp`; compatibility path mirrors `SceneNode::{image_path,mask_path,depth_path}` @ `src/core/include/core/scene.hpp` | Rebuilt from dataset; path mirrors held in memory | `SCNG` | Paths are logical and relative to the `REFS` dataset root when possible. The duplicate C++ path owner and in-memory mask case are U3/U4. | Camera asset/context edit or relink |
| Structural nodes with no independent payload (`GROUP`, `DATASET`, `CAMERA_GROUP`, `IMAGE_GROUP`, `IMAGE`, `PLY_SEQUENCE`) and payload bindings for `SPLAT`, `POINTCLOUD`, `MESH` | `NodeType`, `SceneNode::{model,point_cloud,mesh,image_path}` @ `src/core/include/core/scene.hpp` | Memory only | `SCNG` | SCNG owns node existence and the UUID binding to another chapter, never geometry bytes. `KEYFRAME_GROUP`/`KEYFRAME` nodes are omitted and regenerated from `SEQR`. | Structural node or payload-binding change |

## `SELM` — selection state

| Field | C++ owner (struct/class @ file) | Serialized where today | Chapter authority | Precedence/reconciliation rule | Dirty trigger |
|---|---|---|---|---|---|
| Selection groups: `id`, `name`, `color`, `locked` | `SelectionGroup`, `Scene::selection_groups_` @ `src/core/include/core/scene.hpp` | Memory/undo snapshot only | `SELM` | IDs 1–255 are persisted. `count` is recomputed from restored slices. | Group add/remove/rename/color/lock edit |
| `active_selection_group`, `next_group_id` | `Scene::{active_selection_group_,next_group_id_}` @ `src/core/include/core/scene.hpp` | Memory/undo snapshot only | `SELM` | Validate against restored groups; repair an invalid active ID explicitly and mark the project dirty. | Active group or allocator change |
| Per-node selection mask slices keyed by node UUID, including splat and point-cloud slices | Current flat `Scene::selection_mask_` and `Scene::currentSelectionCapacity()` @ `src/core/include/core/scene.hpp`, `src/core/scene.cpp` | One flat splat-only tensor; no point-cloud slice serializer | `SELM` | Restore only after the complete node set and payload sizes are known. Concatenate/project into runtime caches; no positional node-order authority. Current point-cloud absence is contradiction C7. | Selection paint/write, topology change, group removal, or mask remap |
| Ordered selected-node UUID set | Runtime `SelectionState::selected_nodes_` @ `src/visualizer/scene/selection_state.hpp` | Runtime `NodeId` set only | `SELM` | Restore SCNG-owned UUIDs last. Capture drops generated keyframe and keyframe-group UUIDs as settled in U6. | Node select/add/remove/clear |

`SelectionGroup::count`, `Scene::has_selection_`, cached node masks, generation counters, and group-count dirty flags are derived and excluded.

## `REFS` — external references

Each row uses the same normative reference record: a stable reference key, relative-preferred locator with a portable base, and the plan's heuristic fingerprint `{size, mtime, xxh3(head), xxh3(tail)}`. A full hash is optional on-demand verification. No fingerprint is a security boundary.

| Field | C++ owner (struct/class @ file) | Serialized where today | Chapter authority | Precedence/reconciliation rule | Dirty trigger |
|---|---|---|---|---|---|
| Reference-record locator and fingerprint fields | No DTO yet; current paths are `std::filesystem::path` fields in `TrainingParameters`, `SceneManager`, `RenderSettings`, and `PlySequenceClip` @ files cited below | Paths only; no common fingerprint | `REFS` | Prefer project-relative, then configured relocation/search roots, then stored absolute fallback. A changed fingerprint requires a warning/relink/full verify; never silently accept by matching filename. Component precedence (2026-07-30): `mtime` is a fast path only — an mtime mismatch with matching `size`+xxh3 head/tail is NOT a relink event (refresh the stored mtime silently); size or xxh3 mismatch is decisive. | Relink, locator-base change, or observed source fingerprint change |
| Dataset root | `DatasetConfig::data_path`; `SceneManager::dataset_path_` @ `src/core/include/core/parameters.hpp`, `src/visualizer/scene/scene_manager.hpp` | Checkpoint dataset JSON and runtime member | `REFS` | REFS replaces LFKP/PPIS dataset path strings before dataset validation. Datasets are never embedded. | Dataset association/relink |
| COLMAP sparse/camera-only root | `TrainingParameters::import_cameras_path`; `SceneManager::colmap_sparse_path_` @ `src/core/include/core/parameters.hpp`, `src/visualizer/scene/scene_manager.hpp` | Runtime member; checkpoint JSON may contain import path | `REFS` | Reference is authoritative; SCNG owns the camera snapshot. On mismatch, retain the inspectable SCNG scene and surface relink failure. | Camera-source import/relink |
| Live `.rad` source and optional `.rad.meta` cache hint | `SceneManager::splat_paths_`; `rad_meta_sidecar_path/open_rad_meta_sidecar` @ `src/visualizer/scene/scene_manager.hpp`, `src/io/formats/rad.hpp` | Runtime path map; sidecar beside RAD | `REFS` | RAD is the live external payload under U1. `.rad.meta` is never semantic state: record it only as an optional acceleration hint and rebuild when absent/mismatched because code declares it a derived cache. | RAD bind/relink or source fingerprint change |
| Training background image | `OptimizationParameters::bg_image_path` @ `src/core/include/core/parameters.hpp` | Optimization JSON inside checkpoint/config | `REFS` | REFS resolves the file; `CKPT` owns `bg_mode`/`bg_color` and the active requirement to use the image. Legacy checkpoint path is ignored after resolution. | Background image choose/relink/change |
| Viewer environment HDR/map | `RenderSettings::environment_map_path` @ `src/visualizer/rendering/rendering_types.hpp` | Runtime settings/proxy only | `REFS` | VIEW owns mode/exposure/rotation and a logical reference. REFS injects the resolved path. Built-in packaged environment IDs need no external record. | External environment choose/relink/change |
| PLY-sequence directory | `PlySequenceClip::directory` and frame paths @ `src/visualizer/sequencer/sequencer_controller.hpp` | Runtime clip only | `REFS` | SEQR owns clip ordering/names/FPS; REFS owns directory and fingerprint. Rescan must be reconciled against saved clip entries and never silently reorder them. | Sequence directory choose/relink/change |

## `SPLT` — non-training splat payloads

Every row below is scoped to a **non-training** splat node. The equivalent training-model fields are different, role-qualified state owned by `CKPT`.

| Field | C++ owner (struct/class @ file) | Serialized where today | Chapter authority | Precedence/reconciliation rule | Dirty trigger |
|---|---|---|---|---|---|
| LFSP schema version, `active_sh_degree`, `max_sh_degree`, `scene_scale` | `SplatData` @ `src/core/include/core/splat_data.hpp`; `SplatData::serialize` @ `src/core/splat_data.cpp` | LFSP v4 | `SPLT` | Embed LFSP byte-verbatim in the node-UUID instance. Do not synthesize from scene counts. | SH degree or scene-scale mutation |
| Positions `_means` | `SplatData::_means` via `SplatData::serialize` @ `src/core/splat_data.cpp` | LFSP | `SPLT` | Node transform remains SCNG; means remain payload-local. | Geometry edit/topology change |
| Appearance `_sh0` and canonical `_shN` | `SplatData::{_sh0,_shN}` via `SplatData::serialize` @ `src/core/splat_data.cpp` | LFSP; swizzled runtime `_shN` is canonicalized on write | `SPLT` | Restore canonical LFSP tensors, then build runtime layout. | SH/colour edit or topology change |
| `_scaling`, `_rotation`, `_opacity` | `SplatData` via `SplatData::serialize` @ `src/core/splat_data.cpp` | LFSP | `SPLT` | Payload values win over renderer caches. | Attribute edit or topology change |
| Optional soft-deletion mask `_deleted` | `SplatData::_deleted` via `SplatData::serialize` @ `src/core/splat_data.cpp` | LFSP | `SPLT` | `_deleted_count` and `_deleted_mask_version` are rebuilt. | Soft delete/undelete/apply-deleted |
| Optional `_densification_info` | `SplatData::_densification_info` via `SplatData::serialize` @ `src/core/include/core/splat_data.hpp`, `src/core/splat_data.cpp` | LFSP | `SPLT` | Preserved byte-verbatim even when semantically idle for a non-training node; no strategy chapter may override it. | Payload edit that changes the tensor |
| Frozen ranges `{start,count}` | `SplatData::_frozen_ranges` via LFSP v4 `SplatData::serialize` @ `src/core/splat_data.cpp` | LFSP v4 | `SPLT` | Ranges are payload-relative and remap with topology; scene lock is a separate SCNG field. | Freeze/unfreeze or topology remap |

`lod_tree`, allocator, deletion counts/versions, GPU exportable storage, and consolidated render copies are derived and excluded.

## `CKPT` — exact training resume

`CKPT` is one bounded, byte-verbatim LFKP stream. The table names its semantic contents so no other chapter claims them.

| Field | C++ owner (struct/class @ file) | Serialized where today | Chapter authority | Precedence/reconciliation rule | Dirty trigger |
|---|---|---|---|---|---|
| LFKP header: magic, format version, `iteration`, `num_gaussians`, `sh_degree`, component flags, parameter JSON offset/size | `CheckpointHeader`, `CheckpointFlags`; `serialize_checkpoint` @ `src/core/include/core/checkpoint_format.hpp`, `src/training/checkpoint.cpp` | LFKP v2 embedded in `CKPT` | `CKPT` | Header counts validate the embedded model. Container metadata never substitutes for an LFKP field. | Any training step or component-presence change |
| Training-model SplatData: active/max SH, scene scale, means, SH0/SHN, scaling, rotation, opacity, deleted mask, densification info, frozen ranges | `SplatData::serialize` called by `serialize_checkpoint` @ `src/core/splat_data.cpp`, `src/training/checkpoint.cpp` | LFSP stream nested in embedded LFKP | `CKPT` | Exactly the training model; never emit a `SPLT` duplicate. | Any training-model value/topology/freeze mutation |
| Strategy type tag | `IStrategy::strategy_type`; `serialize_checkpoint` @ `src/training/checkpoint.cpp` | Length-prefixed name in embedded LFKP | `CKPT` | Must match the strategy state and active checkpoint parameters. PRMS's selected strategy is next-run UI state. | Strategy creation/migration |
| Adam configuration and per-parameter state: global `lr`, `beta1`, `beta2`, `eps`, `growth_factor`, `initial_capacity`, named parameter LRs, and each state's `name`, `step_count`, `capacity`, `size`, quantized `exp_avg`, `exp_avg_sq`, and their scales | `AdamOptimizer::serialize` @ `src/training/optimizer/adam_optimizer.cpp` | LFAD v2 nested in strategy | `CKPT` | Exact optimizer state; no PRMS or model-derived reconstruction may replace it when present. | Every optimizer step/topology/state reset |
| Exponential scheduler `gamma`, parameter IDs; warmup scheduler `gamma`, `warmup_steps`, `warmup_start_factor`, `current_step`, `initial_lr`, parameter IDs | `ExponentialLR::serialize`, `WarmupExponentialLR::serialize` @ `src/training/optimizer/scheduler.cpp` | LFSE/LFSW v1 nested in strategy | `CKPT` | Scheduler state wins over recomputation from UI params. | Scheduler step/config mutation |
| MCMC component-presence flags and optimizer/scheduler state | `MCMC::serialize` @ `src/training/strategies/mcmc.cpp` | LFMC v1 | `CKPT` | No additional MCMC state is serialized today; absent caches rebuild. | Optimizer/scheduler mutation |
| MRNF component-presence flags, optimizer/scheduler, optional `_free_mask`, `_mean_lr_unscaled`, `_scale_lr_current` | `MRNF::serialize` @ `src/training/strategies/mrnf.cpp` | LFBR v3 | `CKPT` | `_refine_weight_max`, `_vis_count`, precomputed edge scores, bounds and decay caches rebuild in `deserialize`; they are not project fields. | Training step, topology, free-mask, or LR-state mutation |
| IGS+ component-presence flags, optimizer/scheduler, `_initial_points`, `_current_step`, `_total_steps`, `_budget_schedule`, optional `_free_mask` | `ImprovedGSPlus::serialize` @ `src/training/strategies/improved_gs_plus.cpp` | IGS+ strategy stream | `CKPT` | Precomputed scores/error caches rebuild. | Training step, budget, topology, or free-mask mutation |
| Bilateral grid dimensions/configuration, `step`, current/initial LR, total iterations, grids, Adam moments | `BilateralGrid::serialize` @ `src/training/components/bilateral_grid.cpp` | Optional LFKP `HAS_BILATERAL_GRID` block | `CKPT` | Accumulated gradients and temporary buffers rebuild. | Bilateral optimizer step/state mutation |
| Training PPISP dimensions/configuration, `step`, current/initial LR, total iterations, exposure/vignetting/colour/CRF tensors and Adam moments, camera/frame ID maps | `PPISP::serialize` @ `src/training/components/ppisp.cpp` | Optional LFKP `HAS_PPISP` block | `CKPT` | Wins over `PPIS`. Viewer manual overrides remain `VIEW`. Gradients/finalization caches rebuild. | PPISP optimizer step, mapping, or tensor mutation |
| PPISP controller dimensions/configuration, `step`, current/initial LR, total iterations, CNN/FC weights and Adam moments | `PPISPControllerPool::serialize` @ `src/training/components/ppisp_controller_pool.cu` | Optional LFKP `HAS_PPISP_CONTROLLER` block | `CKPT` | Inference-only `PPIS` controller weights cannot override an active checkpoint controller. Runtime prediction buffers/last-camera cache rebuild. | Controller optimizer step/state mutation |
| Sparsity ADMM `z`, `u`, `opa_sigmoid` | `ADMMSparsityOptimizer::serialize` @ `src/training/components/sparsity_optimizer.cpp` | Optional LFKP `HAS_SPARSITY` block | `CKPT` | Required for mid-sparsification resume. The C2 v2 block is covered by checkpoint and P3 matrix round-trip tests. | ADMM initialize/step/reset/topology change |
| Active core/schedule params: `iterations`, `sh_degree_interval`, `means_lr`, `means_lr_end`, `shs_lr`, `opacity_lr`, `scaling_lr`, `scaling_lr_end`, `rotation_lr`, `lambda_dssim`, `min_opacity`, `refine_every`, `start_refine`, `stop_refine`, `sh_degree`, `opacity_reg`, `scale_reg`, `init_opacity`, `init_scaling`, `max_cap` | `OptimizationParameters`; `OptimizationParameters::to_json`; `serialize_checkpoint` @ `src/core/include/core/parameters.hpp`, `src/core/parameters.cpp`, `src/training/checkpoint.cpp` | Embedded LFKP parameter JSON | `CKPT` | Active exact-resume values win. PRMS contains only role-qualified pending copies. `sh_degree` here is the training param — distinct from the identically named VIEW render setting. | Active trainer parameter mutation |
| Active evaluation/output schedule and strategy behavior: `eval_steps`, `save_steps`, `bg_modulation`, `enable_eval`, `enable_save_eval_images`, `strategy` | `OptimizationParameters::to_json` @ `src/core/parameters.cpp` | LFKP parameter JSON | `CKPT` | Resume behavior uses checkpoint values; any explicit post-open edit is a new mutation. `headless` is excluded despite being serialized today. | Active trainer parameter mutation |
| Active mask params: `mask_mode`, `invert_masks`, `mask_threshold`, `mask_opacity_penalty_weight`, `mask_opacity_penalty_power`, `use_alpha_as_mask` | `OptimizationParameters::to_json` @ `src/core/parameters.cpp` | LFKP parameter JSON | `CKPT` | DatasetConfig's duplicated invert/threshold values are normalized to these active values for the trainer; pending dataset UI copies remain PRMS. | Active mask-policy mutation |
| Active supervision/filter params: `use_depth_loss`, `depth_loss_weight`, `depth_loss_mode`, `use_normal_loss`, `normal_loss_weight`, `normal_consistency_weight`, `normal_flatten_weight`, `normal_loss_space`, `mip_filter` | `OptimizationParameters::to_json` @ `src/core/parameters.cpp` | LFKP parameter JSON | `CKPT` | Exact active training behavior. `mip_filter` here is the training param — distinct from the identically named VIEW render setting. | Active supervision/filter mutation |
| Active background behavior: `bg_mode`, `bg_color`, and binding to the background-image reference (currently represented by `bg_image_path`) | `OptimizationParameters::{bg_mode,bg_color,bg_image_path}` @ `src/core/include/core/parameters.hpp` | LFKP parameter JSON, including legacy raw path | `CKPT` | CKPT owns whether/how the selected reference is used; the referenced record's locator/fingerprint is the REFS row above. | Background mode/color/reference-binding mutation |
| Active bilateral config: `use_bilateral_grid`, `bilateral_grid_X`, `bilateral_grid_Y`, `bilateral_grid_W`, `bilateral_grid_lr`, `tv_loss_weight` | `OptimizationParameters::to_json` @ `src/core/parameters.cpp` | LFKP parameter JSON | `CKPT` | Must agree with the optional bilateral block; inconsistency is invalid. | Active bilateral config mutation |
| Active PPISP config: `use_ppisp`, `ppisp_lr`, `ppisp_reg_weight`, `ppisp_warmup_steps`, `ppisp_freeze_from_sidecar`, `ppisp_use_controller`, `ppisp_freeze_gaussians_on_distill`, `ppisp_controller_activation_step`, `ppisp_controller_lr` | `OptimizationParameters::to_json` @ `src/core/parameters.cpp` | LFKP parameter JSON | `CKPT` | `ppisp_sidecar_path` is source provenance/embedded `PPIS`, not exact path authority. Once full PPISP state exists in CKPT, it wins. | Active PPISP config mutation |
| Active densification/shared controls: `prune_opacity`, `reset_every`, `gut`, `undistort`, `steps_scaler` | `OptimizationParameters::to_json` @ `src/core/parameters.cpp` | LFKP parameter JSON | `CKPT` | Exact active strategy behavior. `gut` here is the training param — distinct from the VIEW compatibility mirror of `raster_backend`. | Active densification config mutation |
| Active MRNF controls: `growth_grad_threshold`, `grow_fraction`, `grow_until_iter`, `opacity_decay`, `scale_decay`, `means_noise_weight`, `bounds_percentile`, `use_error_map`, `use_edge_map` | `OptimizationParameters::to_json` @ `src/core/parameters.cpp` | LFKP parameter JSON | `CKPT` | Exact active strategy behavior; derived edge/error caches rebuild as documented. | Active MRNF config mutation |
| Active random-init and sparsity controls: `random`, `init_num_pts`, `init_extent`, `enable_sparsity`, `sparsify_steps`, `init_rho`, `prune_ratio` | `OptimizationParameters::to_json` @ `src/core/parameters.cpp` | LFKP parameter JSON | `CKPT` | Preserve even after initialization because they govern resume validation/schedules. | Active config mutation |
| Active dataset/decode behavior: `images`, `resize_factor`, `test_every`, `timelapse_images`, `timelapse_every`, `max_width`, `min_track_length`, `invert_masks`, `mask_threshold`, `use_16bit_color` | `DatasetConfig`, `LoadingParams`; `DatasetConfig::to_json` @ `src/core/include/core/parameters.hpp`, `src/core/parameters.cpp` | LFKP dataset JSON, except timelapse fields are currently dropped | `CKPT` | Dataset root comes from REFS. Cache/printing knobs are PRMS policy, not exact state. Missing timelapse persistence is contradiction C5. | Active dataset/decode mutation |
| Initialization/frozen-add state: `init_path`, `add_splat_paths`, `add_splat_freeze`, `exclude_frozen_add_splats_from_export`, plus model frozen ranges | `TrainingParameters`; `serialize_checkpoint`; `SplatData::serialize` @ `src/core/include/core/parameters.hpp`, `src/training/checkpoint.cpp`, `src/core/splat_data.cpp` | Embedded LFKP parameter JSON and nested model LFSP | `CKPT` | Paths are retained as exact-resume provenance for legacy-import compatibility; the already materialized model/frozen ranges are operative on project resume. Do not replay initialization. | Add/freeze/export-policy mutation |

RNG state is absent. `CKPT` therefore promises statistical, not bit-exact, continuation.

## `PPIS` — standalone inference PPISP for non-checkpoint sessions

| Field | C++ owner (struct/class @ file) | Serialized where today | Chapter authority | Precedence/reconciliation rule | Dirty trigger |
|---|---|---|---|---|---|
| PPIS header magic/version, `num_cameras`, `num_frames`, controller/metadata flags | `PPISPFileHeader`, `PPISPFileFlags` @ `src/training/components/ppisp_file.hpp` | `.ppisp` v2 | `PPIS` | Valid only when the project has no authoritative training PPISP in CKPT. | Standalone PPISP load/replace |
| Inference exposure, vignetting, colour, and CRF tensors | `PPISP::serialize_inference` @ `src/training/components/ppisp.cpp` | LFPI v1 inside `.ppisp` | `PPIS` | Model inference values win; VIEW applies non-destructive viewer overrides afterward. | Inference model replace/edit |
| Optional controller CNN/FC inference weights | `PPISPControllerPool::serialize_inference` @ `src/training/components/ppisp_controller_pool.cu` | Controller inference stream inside `.ppisp` | `PPIS` | No optimizer moments/steps are implied; a CKPT controller supersedes the whole row. | Controller model replace/edit |
| Frame/camera mapping metadata: `images_folder`, `frame_image_names`, `frame_camera_ids`, `camera_ids` | `PPISPFileMetadata` @ `src/training/components/ppisp_file.hpp`, `src/training/components/ppisp_file.cpp` | Metadata JSON in `.ppisp` v2; it also physically carries legacy `dataset_path` | `PPIS` | Validate mapping cardinalities against SCNG cameras. The physical legacy `dataset_path` is non-authoritative and ignored after the REFS dataset-root row resolves. | Mapping metadata change |

## `GUIL` — framework-agnostic GUI layout

| Field | C++ owner (struct/class @ file) | Serialized where today | Chapter authority | Precedence/reconciliation rule | Dirty trigger |
|---|---|---|---|---|---|
| Dock/splitter dimensions: `right_panel_width`, `scene_panel_ratio`, `python_console_width`, `bottom_dock_height`, `left_dock_width` | `LayoutState`; `PanelLayoutManager` @ `src/visualizer/gui/layout_state.hpp`, `src/visualizer/gui/panel_layout.hpp` | Project `GUIL`; legacy `layout.json` reader is import-only | `GUIL` | Project copy wins for project layout; clamp to the current window. Theme/UI scale remain user-global. | Splitter/dock resize |
| Dock/panel visibility, including sequencer visibility | `LayoutState::{show_sequencer,window_visibility}`; `PanelInfo::enabled` @ `src/visualizer/gui/layout_state.hpp`, `src/visualizer/gui/panel_registry.hpp` | Project `GUIL` and runtime registry; legacy `layout.json` reader is import-only | `GUIL` | Restore by stable panel ID after panels-ready. Missing plugin IDs remain in retained DOM. | Panel show/hide/close/open |
| Dock placement and active tab IDs, including main scene/history/logging tab | `PanelInfo::{id,parent_id,space,order}`; `PanelLayoutManager::active_tab_id_`; `NativeScenePanel::active_tab_` @ `src/visualizer/gui/panel_registry.hpp`, `src/visualizer/gui/panel_layout.hpp`, `src/visualizer/gui/scene_panel_native.hpp` | Mostly runtime; active tabs not in `LayoutState` | `GUIL` | Registration metadata supplies defaults; saved placement/tab state wins for matching IDs. | Dock/reorder/tab selection |
| Floating panel rectangle, auto-center state, and stack order | `PanelInfo::{float_x,float_y,float_user_height,float_last_x,float_last_y,float_last_w,float_last_h,float_auto_center,float_stack_order}` @ `src/visualizer/gui/panel_registry.hpp` | Runtime only | `GUIL` | Clamp to visible monitor work area; preserve requested rect in DOM so temporary clamping does not destroy it. Drag/resize interaction fields are transient. | Floating move/resize/restack |
| Main window position, windowed size, fullscreen, maximized/restore geometry | `WindowManager::{window_size_,is_fullscreen_,windowed_pos_,windowed_size_,is_borderless_maximized_,borderless_restore_pos_,borderless_restore_size_}` @ `src/visualizer/window/window_manager.hpp` | Runtime only | `GUIL` | Validate/clamp against current displays. Framebuffer size and monitor geometry are derived. | Move/resize/fullscreen/maximize |
| Python console output/terminal active tab and editor font scale | `PythonConsoleState::{active_tab_,font_scale_}` @ `src/visualizer/gui/panels/python_console_panel.hpp` | Runtime only | `GUIL` | These are panel presentation, not editor document state. | Console tab/font-scale change |

No ImGui ini/window/dock IDs are allowed in `GUIL`.

## `EDTR` — code-editor session

| Field | C++ owner (struct/class @ file) | Serialized where today | Chapter authority | Precedence/reconciliation rule | Dirty trigger |
|---|---|---|---|---|---|
| Ordered open-file locators | `PythonEditorWorkspaceSessionState::open_files`; displayed normal Zep buffers in tab/window order @ `src/visualizer/gui/editor/python_editor.hpp`, `src/visualizer/gui/editor/python_editor.cpp` | Runtime only | `EDTR` | Capture every real displayed editor buffer. `PythonConsoleState::script_path_` is only the legacy active-buffer locator fallback. Locator/content conflict for clean files is U5. | Open/close/reorder/Save As |
| Active editor file | `PythonEditorWorkspaceSessionState::active_file`; `ZepEditor::GetActiveBuffer` @ `src/visualizer/gui/editor/python_editor.hpp`, `src/visualizer/gui/editor/python_editor.cpp` | Runtime only | `EDTR` | Resolve after open-file entries; missing files remain visible as unresolved tabs. The real Zep active buffer wins over stale console metadata. | Active-file change |
| Embedded unsaved buffer bytes and explicit modified/share-warning flag | `PythonEditorSessionFile::{text,modified}`; Zep `FileFlags::Dirty`; legacy active-buffer fallback `PythonConsoleState::is_modified_` @ `src/visualizer/gui/editor/python_editor.hpp`, `src/visualizer/gui/panels/python_console_panel.hpp` | Runtime only | `EDTR` | Modified embedded bytes win over disk. The flag is persisted even if bytes equal disk, and drives share/export warnings. | Text mutation, revert, save, or modified-flag change |
| Per-buffer cursor byte position/selection anchor | `PythonEditorSessionState::{cursor_byte,selection_anchor_byte}` captured from each displayed Zep window/buffer @ `src/visualizer/gui/editor/python_editor.hpp`, `src/visualizer/gui/editor/python_editor.cpp` | Runtime Zep state only | `EDTR` | Store framework-neutral byte offsets, clamp to restored UTF-8 buffer; do not serialize Zep objects. | Cursor/selection change |
| Per-buffer scroll position and collapsed fold ranges | `PythonEditorSessionState::{scroll_x,scroll_y,folds}` captured from each displayed Zep window/buffer @ `src/visualizer/gui/editor/python_editor.hpp`, `src/visualizer/gui/editor/python_editor.cpp` | Runtime editor state only | `EDTR` | Store framework-neutral offsets/ranges. Reconcile invalid ranges against restored text and retain unknown fields. | Scroll or fold toggle |
| Vim mode | `PythonEditor::Impl::vim_mode_enabled`, `PythonEditor::setVimModeEnabled` @ `src/visualizer/gui/editor/python_editor.cpp` | Runtime only | `EDTR` | Apply after buffer/cursor restoration. | Vim-mode toggle |

Editor execution status, LSP state, completion lists, diagnostics, output terminal, shell process, command history, and terminal scrollback are excluded.

## `VIEW` — renderer, cameras, split view, and tool preferences

| Field | C++ owner (struct/class @ file) | Serialized where today | Chapter authority | Precedence/reconciliation rule | Dirty trigger |
|---|---|---|---|---|---|
| Core render settings: `focal_length_mm`, `scaling_modifier`, `antialiasing`, `mip_filter`, `sh_degree`, `render_scale`, `camera_metrics_mode` | `RenderSettings` @ `src/visualizer/rendering/rendering_types.hpp`; proxy conversion @ `src/visualizer/ipc/render_settings_convert.hpp` | Runtime/proxy only | `VIEW` | Apply after scene/PPISP and CLI overrides; sanitize unsupported values. `mip_filter`/`sh_degree` here are viewer settings — distinct from the identically named CKPT training params. | Render setting edit |
| Crop/selection presentation: `show_crop_box`, `use_crop_box`, `show_ellipsoid`, `use_ellipsoid`, `desaturate_unselected`, `desaturate_cropping`, `hide_outside_depth_box`, `crop_filter_for_selection` | `RenderSettings` @ `src/visualizer/rendering/rendering_types.hpp` | Runtime/proxy only | `VIEW` | Geometry stays SCNG; these are viewer toggles only. | Toggle/edit |
| Appearance UI: `apply_appearance_correction`, `ppisp_mode`, `PPISPOverrides::{exposure_offset,vignette_enabled,vignette_strength,wb_temperature,wb_tint,color_red_x,color_red_y,color_green_x,color_green_y,color_blue_x,color_blue_y,gamma_multiplier,gamma_red,gamma_green,gamma_blue,crf_toe,crf_shoulder}` | `RenderSettings`, `PPISPOverrides` @ `src/visualizer/rendering/rendering_types.hpp` | Runtime/proxy only | `VIEW` | Load PPIS/CKPT PPISP first, then apply. Overrides do not mutate PPISP model bytes. | Appearance-mode/override edit |
| Background presentation: `background_color`, `environment_mode`, environment reference binding, `environment_exposure`, `environment_rotation_degrees` | `RenderSettings` @ `src/visualizer/rendering/rendering_types.hpp` | Runtime/proxy stores raw path | `VIEW` | VIEW owns selection of the environment reference and its presentation; the bound record's locator/fingerprint is owned by REFS and injected after resolution. | Background presentation/reference-binding edit |
| Axes/grid: `show_coord_axes`, `axes_size`, `axes_visibility`, `show_grid`, `grid_plane`, `grid_opacity` | `RenderSettings` @ `src/visualizer/rendering/rendering_types.hpp` | Runtime/proxy only | `VIEW` | Per-panel grid plane below overrides the shared mirror for independent panels. | Overlay edit |
| Point/ring/camera/pivot display: `point_cloud_mode`, `voxel_size`, `show_rings`, `ring_width`, `show_center_markers`, `show_camera_frustums`, `camera_frustum_scale`, `train_camera_color`, `eval_camera_color`, `show_pivot` | `RenderSettings` @ `src/visualizer/rendering/rendering_types.hpp` | Runtime/proxy only | `VIEW` | Display only; never changes SCNG payload or selection. | Display setting edit |
| Split/projection/backend: `split_view_mode`, `gt_comparison_mode`, `split_position`, `split_view_offset`, `raster_backend`, `equirectangular`, `orthographic`, `ortho_scale`, `depth_view`, `depth_view_min`, `depth_view_max`, `depth_visualization_mode` | `RenderSettings`; `SplitViewService::toggleMode` @ `src/visualizer/rendering/rendering_types.hpp`, `src/visualizer/rendering/split_view_service.cpp` | Runtime/proxy only | `VIEW` | Enter split state only via `toggleMode`; sanitize depth/GT/backend. `gut` is a compatibility mirror derived from `raster_backend`. | Mode/divider/projection/backend/depth edit |
| Selection colours and depth clipping: `selection_color_committed`, `selection_color_preview`, `selection_color_center_marker`, `depth_clip_enabled`, `depth_clip_far` | `RenderSettings` @ `src/visualizer/rendering/rendering_types.hpp` | Runtime/proxy only | `VIEW` | Presentation/filter state only; selection bytes remain SELM. | Colour/clip edit |
| Mesh rendering: `mesh_wireframe`, `mesh_wireframe_color`, `mesh_wireframe_width`, `mesh_light_dir`, `mesh_light_intensity`, `mesh_ambient`, `mesh_backface_culling`, `mesh_shadow_enabled`, `mesh_shadow_resolution` | `RenderSettings` @ `src/visualizer/rendering/rendering_types.hpp` | Runtime/proxy only | `VIEW` | Does not own mesh material/geometry. | Mesh-view setting edit |
| Selection depth filter: `depth_filter_enabled`, `depth_filter_min`, `depth_filter_max`, `depth_filter_transform` rotation/translation | `RenderSettings` @ `src/visualizer/rendering/rendering_types.hpp`; proxy conversion @ `src/visualizer/ipc/render_settings_convert.hpp` | Runtime/proxy only | `VIEW` | Tool filter, not SCNG crop geometry. | Filter bounds/transform/toggle edit |
| LOD: `lod_enabled`, `lod_auto_enable_rad`, `lod_max_splats`, `lod_render_scale`, `lod_behind_camera_penalty`, `lod_cone_foveation`, `lod_cone_inner_degrees`, `lod_cone_outer_degrees`, `lod_page_pool_splats`, `lod_pool_vram_fraction`, `lod_fade_frames`, `lod_debug_colors` | `RenderSettings` @ `src/visualizer/rendering/rendering_types.hpp`; proxy conversion @ `src/visualizer/ipc/render_settings_convert.hpp` | Runtime/proxy only | `VIEW` | Runtime page residency/tree is rebuilt from SPLT/RAD. Clamp machine-dependent budgets without overwriting retained requested values. | LOD setting edit |
| Left/right panel grid planes | `RenderingManager::panel_grid_planes_` @ `src/visualizer/rendering/rendering_manager.hpp`, `src/visualizer/rendering/rendering_manager.cpp` | Runtime only | `VIEW` | Restore after split mode; focused panel projects its value into `RenderSettings::grid_plane`. | Per-panel grid-plane edit |
| Primary and secondary `PanelCameraState`: `R[9]`, `t`, `pivot`, `home_R`, `home_t`, `home_pivot`, `home_saved`, zoom/max-zoom, rotate/centre/roll/translate speeds, WASD/max-WASD speed, per-panel orthographic scale | `Viewport::CameraMotion`, `Viewport::ortho_scale_override`, `Viewport::setViewMatrix` @ `src/visualizer/internal/viewport.hpp`; secondary viewport @ `src/visualizer/rendering/split_view_service.hpp` | Runtime only; no `PanelCameraState` DTO yet | `VIEW` | Store `R` directly to preserve roll. Restore with `setViewMatrix`, then pivot/home/speeds/ortho. Window/framebuffer sizes are not camera state. | Camera move/home/speed/ortho edit |
| Camera navigation mode and view-snap preference | `InputController::{camera_navigation_mode_,camera_view_snap_enabled_}` @ `src/visualizer/input/input_controller.hpp` | Runtime only | `VIEW` | Apply after panel cameras; transient momentum is cleared. Input bindings remain user-global. | Navigation-mode/snap toggle |
| Focused split panel and GT comparison camera identity | `SplitViewService::focused_panel_`; `GTComparisonContext::camera_id` @ `src/visualizer/rendering/split_view_service.hpp`, `src/visualizer/rendering/rendering_types.hpp` | Runtime only; GT texture context rebuilt | `VIEW` | Resolve GT camera against SCNG after split mode. Texture handles, dimensions, transforms, and render camera are derived. | Focused-panel or GT-camera change |
| Ordered camera bookmarks | No bookmark DTO exists; source pose fields are `Viewport::CameraMotion::{R,t,pivot}` @ `src/visualizer/internal/viewport.hpp` | Not serialized | `VIEW` | Bookmarks must reference/store framework-neutral camera state; schema details require implementation but chapter ownership is settled. | Bookmark add/remove/reorder/update |
| Active tool/submode and gizmo preferences: active tool ID, selection submode, gizmo operation, transform space, pivot mode, multi-transform mode, crop-tool shape/operation | `UnifiedToolRegistry::{active_tool_id_,active_submode_id_}` @ `src/visualizer/tools/unified_tool_registry.hpp`; `GizmoManager` fields @ `src/visualizer/gui/gizmo_manager.hpp` | Runtime/AppStore only | `VIEW` | Resolve stable tool IDs after tool/plugin registration; unavailable IDs stay retained but inactive. | Tool/submode/gizmo preference change |
| Selection-tool preferences: brush radius and `SelectionFilterState::{crop_filter,depth_filter,restrict_to_selected_nodes}` | `SelectionOp::brush_radius_`; `SelectionService::InteractiveSelectionState::{brush_radius,filters}` @ `src/visualizer/operator/ops/selection_ops.hpp`, `src/visualizer/selection/selection_service.hpp` | Runtime/operator properties only | `VIEW` | Persist last committed preference, never an in-progress stroke/polygon. | Preference change outside active gesture |
| Sequencer viewport preferences: `show_camera_path` | `SequencerUIState::show_camera_path` @ `src/visualizer/gui/sequencer_ui_state.hpp` | Runtime only | `VIEW` | Affects viewport overlay only; keyframe data remains SEQR. | Camera-path visibility toggle |

`InputController::depth_range_initialized_` is not serialized: project restore sets it true after restoring saved depth ranges. Camera momentum, drag/orbit state, scene extent, split texture handles, and render caches are transient.

## `SEQR` — sequencer/timeline

| Field | C++ owner (struct/class @ file) | Serialized where today | Chapter authority | Precedence/reconciliation rule | Dirty trigger |
|---|---|---|---|---|---|
| Timeline JSON version, `clip_duration`, ordered camera keyframes `{time,position,rotation,focal_length_mm,easing}` | `Timeline`, `Keyframe`; `Timeline::saveToJson` @ `src/sequencer/timeline.hpp`, `src/sequencer/keyframe.hpp`, `src/sequencer/timeline.cpp` | Standalone timeline JSON | `SEQR` | Inline the semantic JSON in retained DOM. Loop-point keyframe and runtime keyframe IDs are regenerated. | Keyframe/clip-duration edit |
| Animation clip `name`; tracks `{id,type,target}`; generic keyframes `{time,value,easing}` | `AnimationClip::toJson`, `AnimationTrack`, `GenericKeyframe` @ `src/sequencer/animation_clip.cpp`, `src/sequencer/animation_track.hpp` | Nested in standalone timeline JSON | `SEQR` | Values and order survive unknown fields; stable timeline-keyframe identity is post-v1. | Clip/track/keyframe edit |
| PLY-sequence clip node name, ordered frame entries/node names, and `fps` | `PlySequenceClip`, `PlySequenceFrame` @ `src/visualizer/sequencer/sequencer_controller.hpp` | Runtime only | `SEQR` | Saved order wins over directory enumeration. Its external directory record is separately owned and resolved by REFS. | Clip import/reorder/name/FPS change |
| `playhead`, `loop_mode`, `playback_speed` | `SequencerController::{playhead_,loop_mode_,playback_speed_}` @ `src/visualizer/sequencer/sequencer_controller.hpp` | Runtime only | `SEQR` | Restore stopped at saved playhead. Controller values win over `SequencerUIState` mirrors. | Seek, loop-mode, or speed change |
| Editing/preview preferences: `snap_to_grid`, `snap_interval`, `follow_playback`, `show_pip_preview`, `pip_preview_scale`, `show_film_strip` | `SequencerUIState` @ `src/visualizer/gui/sequencer_ui_state.hpp` | Runtime only | `SEQR` | Project sequencer-session state. Export preset/dimensions/framerate/quality and preview `equirectangular` are excluded. | Preference change |

`KEYFRAME_GROUP` and `KEYFRAME` scene nodes are not serialized in SCNG. `KeyframeSceneSync::syncToSceneGraph` @ `src/visualizer/gui/keyframe_scene_sync.cpp` regenerates them after SEQR restore.

## `METR` — resumable training graphs

| Field | C++ owner (struct/class @ file) | Serialized where today | Chapter authority | Precedence/reconciliation rule | Dirty trigger |
|---|---|---|---|---|---|
| Loss history samples | Full `{iteration,loss}` samples in `CommandCenter::loss_history_`; capped value cache in `TrainerManager::loss_buffer_` @ `src/training/control/command_api.hpp`, `src/visualizer/training/training_manager.hpp` | Runtime only | `METR` | Persisted graph source/retention is U7. Any restored canonical history rebuilds the capped UI deque; never persist both as authorities. | New/revised loss sample |
| PSNR history samples | `MetricsReporter::all_metrics_` (`EvalMetrics`) and capped `TrainerManager::psnr_buffer_` @ `src/training/metrics/metrics.hpp`, `src/visualizer/training/training_manager.hpp` | Runtime plus external CSV/report | `METR` | Persisted graph source/retention is U7. External report files do not override project history. | New/revised evaluation sample |
| Accumulated training time in seconds | `TrainerManager::accumulated_training_time_`, `getElapsedSeconds()` @ `src/visualizer/training/training_manager.hpp`, `src/visualizer/training/training_manager.cpp` | Runtime only | `METR` | Store elapsed duration, never a `steady_clock` time point. Running interval is captured at checkpoint safe point. | Pause/resume/stop or snapshot capture |
| Last evaluation `{iteration,psnr,ssim}` | `TrainerManager::EvaluationMetricsSnapshot`, `last_eval_metrics_` @ `src/visualizer/training/training_manager.hpp` | Runtime only | `METR` | Rebuild `last_psnr_` and UI observables from this row. Per-image elapsed time and Gaussian count are report telemetry, not the plan's last-eval field. | Evaluation completion/clear |

## Gap #12 — point-cloud and mesh payload ownership

**Status: NORMATIVE (owner sign-off 2026-07-30) — payloads live in distinct `PCLD`/`MESH` chunks with their own versioning; see resolved U2.**

1. Expand the logical scope of `SPLT` from “splat payload” to “embedded resident geometry payload,” still keyed by scene-node UUID.
2. Keep LFSP byte-verbatim for splats; add separately versioned inner payload formats for `PointCloud` and `MeshData` rather than pretending either is LFSP.
3. Embed point-cloud and mesh payloads whenever such a node is persisted; `REFS` may retain import provenance but never becomes geometry authority.
4. Point-cloud payload covers `means`, `colors`, `normals`, `sh0`, `shN`, `opacity`, `scaling`, `rotation`, `attribute_names`; mesh covers `vertices`, `normals`, `tangents`, `texcoords`, `colors`, `indices`, `Submesh::{start_index,index_count,material_index}`, `TextureImage::{pixels,width,height,channels}`, and `Material::{base_color,emissive,metallic,roughness,ao,albedo_tex,normal_tex,metallic_roughness_tex,emissive_tex,ao_tex,albedo_tex_path,normal_tex_path,metallic_roughness_tex_path,double_sided,name}`. `MeshData::generation_` is derived.
5. Rationale: a fixed single payload authority preserves the one-file session promise and avoids authority flipping when a clean imported node is first edited; the cost is a deliberate §2.4 registry wording change and possible dataset-derived duplication.
6. *Audit alternative (2026-07-18, weigh at sign-off):* keep the single-authority principle but give point clouds and meshes their own fourccs (`PCLD`/`MESH`) instead of overloading `SPLT` — one `chunk_version` per payload family matches plan §2.5's "new state = new fourcc" default and keeps opaque-preservation semantics per type.

Real state surfaces: `PointCloud` @ `src/core/include/core/point_cloud.hpp`; `MeshData`, `TextureImage`, `Submesh` @ `src/core/include/core/mesh_data.hpp`; `Material` @ `src/core/include/core/material.hpp`; node bindings and `payload_diverged` @ `src/core/include/core/scene.hpp`. Neither payload type has a serializer today.

## Exclusions

These fields must not appear as authoritative `.licht` project state.

| Excluded state | C++ owner / current persistence | Boundary rule |
|---|---|---|
| Active theme ID, custom palette/sizes | `g_current_theme_id`, theme preference @ `src/visualizer/theme/theme.cpp` | User-global. Opening a project must not change it. |
| Language | `LocalizationManager::current_language_`, `language_preference` @ `src/core/event_bridge/localization_manager.hpp`, `src/core/event_bridge/localization_manager.cpp` | User-global. |
| Global UI scale | `ui_scale` preference @ `src/visualizer/theme/theme.cpp` | User-global. Project layout is clamped under the current scale. |
| VRAM HUD position/size/tab/collapsed paths and enablement | `LayoutState::vram_hud_*` @ `src/visualizer/gui/layout_state.hpp`; profiler toggle in `src/visualizer/gui/rml_status_bar.cpp` | User-global/machine diagnostic in `ui_preferences.json`; legacy `layout.json` is read-only migration input. |
| Input bindings and current input profile | `InputBindings::{bindings_,current_profile_name_}` @ `src/visualizer/input/input_bindings.hpp`, `src/visualizer/input/input_bindings.cpp` | User-global controls. VIEW stores only project camera/tool mode preferences. |
| OS file association | `LayoutState::file_association` @ `src/visualizer/gui/layout_state.hpp` | User-global/machine integration. |
| Process/CLI controls: `headless`, `auto_train`, `no_splash`, `debug_python`, `debug_python_port`, `config_file`, `ServerConfig::{tcp_server_connection_port,tcp_broadcast_connection_port,tcp_connection}`, `resume_checkpoint`, `render_path` (camera/load/output paths, width, height, FPS, CRF), `python_scripts`, `cli_bg_color_set`, dataset `output_path`/`output_name` | `OptimizationParameters`, `ServerConfig`, `RenderPathConfig`, `TrainingParameters`, `DatasetConfig` @ `src/core/include/core/parameters.hpp` | Invocation/machine/output policy. Runtime/CLI wins. Legacy copies trapped in CKPT are ignored for `.licht` restore. |
| `ParameterManager::{loaded_,dirty_}` and trainer pending-update synchronization | `ParameterManager::{loaded_,dirty_}`, `TrainerManager::{pending_opt_params_,pending_dataset_params_}` @ `src/visualizer/core/parameter_manager.hpp`, `src/visualizer/training/training_manager.hpp` | Runtime synchronization; rebuilt from loaded chapters. |
| Runtime scene IDs/maps/caches, `SceneNode::{world_transform,transform_dirty,gaussian_count,centroid}`, derived scene-centre statistic, `images_have_alpha`, consolidated models | `Scene`, `SceneNode` @ `src/core/include/core/scene.hpp` | Rebuild from UUID hierarchy, payloads, and cameras. The *world origin* removed by centralization is NOT derivable and is owned by the `PROJ` georeference block (2026-07-30) — the mean of shifted payloads is ≈0, not the original origin. |
| Scene-wide `point_cloud_modified_` compatibility flag | `Scene::{point_cloud_modified_,setPointCloudModified}` @ `src/core/include/core/scene.hpp` | Derive from per-node payload/source state; `SceneNode::payload_diverged` is the durable SCNG field. |
| Crop/ellipsoid flash intensity | `CropBoxData::flash_intensity`, `EllipsoidData::flash_intensity` @ `src/core/include/core/scene.hpp` | Animation feedback only; restore as zero. |
| Camera GPU/image/mask/depth/normal/undistortion caches, CUDA stream, computed FoV/world-view/camera position | `Camera` private cache fields @ `src/core/include/core/camera.hpp` | Rebuild lazily from SCNG plus REFS. In-memory raw mask payload itself is U3, not excluded by this cache row. |
| Splat LOD tree/page residency, allocator, deleted-count/version, renderer/export storage | `SplatData` @ `src/core/include/core/splat_data.hpp` | Rebuild from LFSP or live RAD. |
| RNG state | `serialize_checkpoint`/`load_checkpoint` have no RNG field @ `src/training/checkpoint.cpp` | Absent by plan §6: statistical resume only; optional RNG chapter is future work. |
| Strategy caches and temporary training buffers | `MRNF::deserialize`, `ImprovedGSPlus::deserialize`, `BilateralGrid::deserialize`, `PPISP::deserialize`, `PPISPControllerPool::deserialize` @ `src/training/strategies/mrnf.cpp`, `src/training/strategies/improved_gs_plus.cpp`, `src/training/components/bilateral_grid.cpp`, `src/training/components/ppisp.cpp`, `src/training/components/ppisp_controller_pool.cu` | Rebuild as today; one-window behavioral discontinuity is documented. |
| Selection counts, cache masks, dirty/generation counters | `Scene` selection internals and `SelectionState` @ `src/core/include/core/scene.hpp`, `src/visualizer/scene/selection_state.hpp` | Recompute from SELM. |
| Undo/redo history, clipboard, active drags/strokes/polygons, hover/popup/modal state | `UndoHistory`, `SceneManager::clipboard_`, `SelectionService::InteractiveSelectionState` @ `src/visualizer/operation/undo_history.hpp`, `src/visualizer/scene/scene_manager.hpp`, `src/visualizer/selection/selection_service.hpp` | Transient interaction state, not durable project DTO. |
| Editor execution/LSP/completion/diagnostics/output/terminal process/history/scrollback | `PythonEditor::Impl`, `PythonLspClient`, `PythonConsoleState`, `TerminalWidget` @ `src/visualizer/gui/editor/python_editor.cpp`, `src/visualizer/gui/editor/python_lsp_client.hpp`, `src/visualizer/gui/panels/python_console_panel.hpp`, `src/visualizer/gui/terminal/terminal_widget.hpp` | Rebuilt or intentionally session-transient; only EDTR document session persists. |
| Sequencer playback state (`PLAYING`/`PAUSED`/`SCRUBBING`), reverse direction, revisions, loop-point keyframe, export preset/dimensions/FPS/quality | `SequencerController`, `Timeline`, `SequencerUIState` @ `src/visualizer/sequencer/sequencer_controller.hpp`, `src/sequencer/timeline.hpp`, `src/visualizer/gui/sequencer_ui_state.hpp` | Restore stopped at the persisted playhead; regenerate keyframe nodes and drop their prior selection as resolved in U6. Export dialog choices are not v2 project state. |
| Generated `KeyframeData`/keyframe scene nodes, including derived `keyframe_index` | `KeyframeData`; `KeyframeSceneSync::syncToSceneGraph` @ `src/core/include/core/scene.hpp`, `src/visualizer/gui/keyframe_scene_sync.cpp` | Regenerate from SEQR; never serialize in SCNG. |
| Trainer telemetry beyond METR (FPS, ETA, instantaneous phase, transient snapshots, per-image report details) | `TrainingSnapshot`, `EvalMetrics`, AppStore fields @ `src/training/control/command_api.hpp`, `src/training/metrics/metrics.hpp`, `src/visualizer/include/visualizer/app_store.hpp` | Rebuilt or external report output per plan §6. |
| Dataset contents | `loadTrainingDataIntoScene`; `Camera` asset paths @ `src/training/training_setup.cpp`, `src/core/include/core/camera.hpp` | Never embedded. REFS stores root/fingerprint; SCNG stores logical camera context. |
| `.rad.meta` contents | `rad_meta_sidecar_path/open_rad_meta_sidecar` @ `src/io/formats/rad.hpp` | Derived cache. REFS may record a locator/fingerprint hint only. |
| Window framebuffer size, monitor geometry, render texture handles, frame serials | `WindowManager`, `SplitViewService`, `RenderingManager` @ `src/visualizer/window/window_manager.hpp`, `src/visualizer/rendering/split_view_service.hpp`, `src/visualizer/rendering/rendering_manager.hpp` | Recompute for the current machine. GUIL stores logical window geometry only. |
| Container transaction/envelope metadata: file/commit/snapshot UUIDs, head sequence, physical offsets/sizes, generations, CRCs, tombstones, compression/alignment | Current prototype surfaces are `FileHeader`, `ChunkHeader`, `IndexRow`, `Footer` @ `src/io/include/io/project_container.hpp` | Physical container authority, outside application-state chapters. Only semantic `project_uuid` is mirrored in PROJ and must match the superblock. |

## Remaining policy and explicit v1 non-goals

These are not permissions to choose opportunistically. Resolved entries record the selected
policy; non-goals record a deliberate v1 boundary.

### ~~U1~~ — RESOLVED (owner decision 2026-07-30): SPLT scope

Every imported splat node (PLY/SPZ/SOG) is **always embedded** in `SPLT`, clean or dirty —
projects are self-contained for imported splats and no relink flow exists for them. Live-RAD
nodes are **never embedded**: `REFS` external, read-only; destructive edits are refused until
the user explicitly bakes the node to an embedded resident splat (the bake is a new `SPLT`-owned
node, the RAD reference is dropped from it). `SceneManager::splat_paths_` records the original
import source for provenance only — it is never load-bearing for restore of an embedded node.

### ~~U2~~ — RESOLVED (owner decision 2026-07-30): gap #12 payloads

Distinct `PCLD` and `MESH` chunks with their own versioning (plan §2.2). The PROPOSED
geometry-payload block above is hereby normative with that chunk assignment.

### U3 — V1 NON-GOAL: camera in-memory mask payload

`Camera::_in_memory_mask_raw` can supersede `mask_path` for direct-scene plugins
(`src/core/include/core/camera.hpp`). V1 deliberately does not serialize this tensor.
`capture_scene_graph_state` returns a typed `FailedPrecondition` error instead of silently dropping
it. A future embedded camera-mask payload requires a new ownership and compatibility decision.

### ~~U4~~ — RESOLVED: camera asset-path authority

The `Camera` value is canonical. `capture_camera` writes its image, mask, depth, and normal paths to
the SCNG camera record, and hydration reconstructs the `Camera` from that record. Scene-node path
mirrors are not a second serialized authority.

### U5 — V1 NON-GOAL: clean editor file recovery snapshots

EDTR embeds modified buffers only. A clean entry is locator-only external state: restore reads the
current disk bytes, and a missing, unreadable, or oversized file restores an empty tab. V1 has no
clean-buffer fingerprint, embedded recovery snapshot, or conflict UI; adding those is future work.

### ~~U6~~ — RESOLVED: selected generated keyframe nodes

V1 drops selection of generated `KEYFRAME` and `KEYFRAME_GROUP` scene nodes during SELM capture.
Those nodes remain derived from SEQR and omitted from SCNG, so they cannot reappear as restored
node selection after save and reopen. Stable timeline-keyframe identity and selection restoration
are a post-v1 format feature.

### ~~U7~~ — RESOLVED (owner work order 2026-07-31): METR canonical history and retention

`METR` stores binary, ordered `{i32 iteration, f32 value}` loss and PSNR histories (up to 10,000,000 samples each), f64 accumulated seconds, and optional last-eval `{i32 iteration, f32 psnr, f32 ssim}`. Full histories are canonical; capped `TrainerManager` deques and report projections are rebuilt.

## Historical plan/code contradictions — resolved

1. **Container grammar — resolved in P2:** the production container uses the immutable superblock,
   dual head slots, complete indices/tombstones, explicit encoding, and UUID chunk keys.
2. **LFKP versioning — resolved in P1:** sparsity state is version-gated by LFKP v2; the current
   build and checkpoint tests exercise the accepted version range.
3. **Camera enablement duplicate — resolved in P3/P4:** SCNG wins for project restore. The LFKP
   field remains only for standalone legacy-checkpoint import.
4. **Retained-DOM preservation — resolved in P1/P3/P5:** every JSON project chapter mutates a
   retained DOM and the P8 compatibility tests prove unknown-field carry-forward.
5. **Exact dataset fields — resolved in P4:** CKPT and PRMS capture the role-qualified timelapse
   and dataset fields covered by the ownership proofs.
6. **Process mode in exact params — resolved in P3/P4:** project chapter adaptation strips
   `headless`; invocation policy remains runtime-owned.
7. **Point-cloud selection — resolved in P3:** SELM stores and hydrates per-node splat and
   point-cloud slices.
8. **Component config layouts — resolved in P1:** bilateral-grid, PPISP, and controller configs
   use field-wise versioned encoding with legacy readers.
9. **Persistent identity — resolved in P1/P3/P5/U6:** project references use UUIDs. V1 drops
   selection of regenerated keyframe nodes; stable timeline-keyframe identity is post-v1.
10. **Legacy writers — resolved in P6:** standalone checkpoint/LFKP, `.ppisp`, and `layout.json`
    writer paths were removed from product surfaces. Their readers are transitional importers;
    new project persistence is chapter-backed `.licht`.

## Runtime serialization inventory

This exact inventory is generated from the maximally populated
`P8OwnershipMatrixRatchet.MaximallyPopulatedSerializedOutputMatchesOwnershipMatrix`
project. JSON array indices and PRMS strategy names are normalized schema dimensions; keys
are not. Every marker must have exactly one authority matching its serialized chapter.
`PPIS` is the sole intentional absence from this CKPT-owning project because the format
refuses those two authorities in one generation; `P4MatrixProof.PpispSurvivesSaveLoadSave`
is its separate round-trip proof.

<!-- P8-NOT-SERIALIZED chapter=PPIS reason=mutually-exclusive-with-CKPT proof=P4MatrixProof.PpispSurvivesSaveLoadSave -->
<!-- P8-RUNTIME-INVENTORY-BEGIN -->
<!-- P8-RUNTIME chapter=CKPT kind=chunk path=CKPT authority=CKPT -->
<!-- P8-RUNTIME chapter=EDTR kind=chunk path=EDTR authority=EDTR -->
<!-- P8-RUNTIME chapter=EDTR kind=json path=active_file authority=EDTR -->
<!-- P8-RUNTIME chapter=EDTR kind=json path=contains_embedded_secrets authority=EDTR -->
<!-- P8-RUNTIME chapter=EDTR kind=json path=open_files authority=EDTR -->
<!-- P8-RUNTIME chapter=EDTR kind=json path=open_files[] authority=EDTR -->
<!-- P8-RUNTIME chapter=EDTR kind=json path=open_files[].cursor_byte authority=EDTR -->
<!-- P8-RUNTIME chapter=EDTR kind=json path=open_files[].embedded_buffer authority=EDTR -->
<!-- P8-RUNTIME chapter=EDTR kind=json path=open_files[].folds authority=EDTR -->
<!-- P8-RUNTIME chapter=EDTR kind=json path=open_files[].folds[] authority=EDTR -->
<!-- P8-RUNTIME chapter=EDTR kind=json path=open_files[].folds[].collapsed authority=EDTR -->
<!-- P8-RUNTIME chapter=EDTR kind=json path=open_files[].folds[].end_byte authority=EDTR -->
<!-- P8-RUNTIME chapter=EDTR kind=json path=open_files[].folds[].start_byte authority=EDTR -->
<!-- P8-RUNTIME chapter=EDTR kind=json path=open_files[].locator authority=EDTR -->
<!-- P8-RUNTIME chapter=EDTR kind=json path=open_files[].modified authority=EDTR -->
<!-- P8-RUNTIME chapter=EDTR kind=json path=open_files[].scroll_x authority=EDTR -->
<!-- P8-RUNTIME chapter=EDTR kind=json path=open_files[].scroll_y authority=EDTR -->
<!-- P8-RUNTIME chapter=EDTR kind=json path=open_files[].selection_anchor_byte authority=EDTR -->
<!-- P8-RUNTIME chapter=EDTR kind=json path=open_files[].share_warning authority=EDTR -->
<!-- P8-RUNTIME chapter=EDTR kind=json path=version authority=EDTR -->
<!-- P8-RUNTIME chapter=EDTR kind=json path=vim_mode authority=EDTR -->
<!-- P8-RUNTIME chapter=GUIL kind=chunk path=GUIL authority=GUIL -->
<!-- P8-RUNTIME chapter=GUIL kind=json path=layouts authority=GUIL -->
<!-- P8-RUNTIME chapter=GUIL kind=json path=layouts[] authority=GUIL -->
<!-- P8-RUNTIME chapter=GUIL kind=json path=layouts[].active authority=GUIL -->
<!-- P8-RUNTIME chapter=GUIL kind=json path=layouts[].areas authority=GUIL -->
<!-- P8-RUNTIME chapter=GUIL kind=json path=layouts[].areas[] authority=GUIL -->
<!-- P8-RUNTIME chapter=GUIL kind=json path=layouts[].areas[].active_space authority=GUIL -->
<!-- P8-RUNTIME chapter=GUIL kind=json path=layouts[].areas[].rect_or_split_position authority=GUIL -->
<!-- P8-RUNTIME chapter=GUIL kind=json path=layouts[].areas[].rect_or_split_position.height authority=GUIL -->
<!-- P8-RUNTIME chapter=GUIL kind=json path=layouts[].areas[].rect_or_split_position.kind authority=GUIL -->
<!-- P8-RUNTIME chapter=GUIL kind=json path=layouts[].areas[].rect_or_split_position.width authority=GUIL -->
<!-- P8-RUNTIME chapter=GUIL kind=json path=layouts[].areas[].rect_or_split_position.x authority=GUIL -->
<!-- P8-RUNTIME chapter=GUIL kind=json path=layouts[].areas[].rect_or_split_position.y authority=GUIL -->
<!-- P8-RUNTIME chapter=GUIL kind=json path=layouts[].areas[].spaces authority=GUIL -->
<!-- P8-RUNTIME chapter=GUIL kind=json path=layouts[].areas[].spaces[] authority=GUIL -->
<!-- P8-RUNTIME chapter=GUIL kind=json path=layouts[].areas[].spaces[].opaque_payload authority=GUIL -->
<!-- P8-RUNTIME chapter=GUIL kind=json path=layouts[].areas[].spaces[].opaque_payload.active_tab authority=GUIL -->
<!-- P8-RUNTIME chapter=GUIL kind=json path=layouts[].areas[].spaces[].opaque_payload.active_tabs authority=GUIL -->
<!-- P8-RUNTIME chapter=GUIL kind=json path=layouts[].areas[].spaces[].opaque_payload.active_tabs.main_panel authority=GUIL -->
<!-- P8-RUNTIME chapter=GUIL kind=json path=layouts[].areas[].spaces[].opaque_payload.active_tabs.scene_panel authority=GUIL -->
<!-- P8-RUNTIME chapter=GUIL kind=json path=layouts[].areas[].spaces[].opaque_payload.bottom_dock_height authority=GUIL -->
<!-- P8-RUNTIME chapter=GUIL kind=json path=layouts[].areas[].spaces[].opaque_payload.font_scale authority=GUIL -->
<!-- P8-RUNTIME chapter=GUIL kind=json path=layouts[].areas[].spaces[].opaque_payload.left_dock_width authority=GUIL -->
<!-- P8-RUNTIME chapter=GUIL kind=json path=layouts[].areas[].spaces[].opaque_payload.panels authority=GUIL -->
<!-- P8-RUNTIME chapter=GUIL kind=json path=layouts[].areas[].spaces[].opaque_payload.panels[] authority=GUIL -->
<!-- P8-RUNTIME chapter=GUIL kind=json path=layouts[].areas[].spaces[].opaque_payload.panels[].enabled authority=GUIL -->
<!-- P8-RUNTIME chapter=GUIL kind=json path=layouts[].areas[].spaces[].opaque_payload.panels[].float_auto_center authority=GUIL -->
<!-- P8-RUNTIME chapter=GUIL kind=json path=layouts[].areas[].spaces[].opaque_payload.panels[].float_last_bounds_valid authority=GUIL -->
<!-- P8-RUNTIME chapter=GUIL kind=json path=layouts[].areas[].spaces[].opaque_payload.panels[].float_last_h authority=GUIL -->
<!-- P8-RUNTIME chapter=GUIL kind=json path=layouts[].areas[].spaces[].opaque_payload.panels[].float_last_w authority=GUIL -->
<!-- P8-RUNTIME chapter=GUIL kind=json path=layouts[].areas[].spaces[].opaque_payload.panels[].float_last_x authority=GUIL -->
<!-- P8-RUNTIME chapter=GUIL kind=json path=layouts[].areas[].spaces[].opaque_payload.panels[].float_last_y authority=GUIL -->
<!-- P8-RUNTIME chapter=GUIL kind=json path=layouts[].areas[].spaces[].opaque_payload.panels[].float_stack_order authority=GUIL -->
<!-- P8-RUNTIME chapter=GUIL kind=json path=layouts[].areas[].spaces[].opaque_payload.panels[].float_user_height authority=GUIL -->
<!-- P8-RUNTIME chapter=GUIL kind=json path=layouts[].areas[].spaces[].opaque_payload.panels[].float_x authority=GUIL -->
<!-- P8-RUNTIME chapter=GUIL kind=json path=layouts[].areas[].spaces[].opaque_payload.panels[].float_y authority=GUIL -->
<!-- P8-RUNTIME chapter=GUIL kind=json path=layouts[].areas[].spaces[].opaque_payload.panels[].id authority=GUIL -->
<!-- P8-RUNTIME chapter=GUIL kind=json path=layouts[].areas[].spaces[].opaque_payload.panels[].order authority=GUIL -->
<!-- P8-RUNTIME chapter=GUIL kind=json path=layouts[].areas[].spaces[].opaque_payload.panels[].parent_id authority=GUIL -->
<!-- P8-RUNTIME chapter=GUIL kind=json path=layouts[].areas[].spaces[].opaque_payload.panels[].space authority=GUIL -->
<!-- P8-RUNTIME chapter=GUIL kind=json path=layouts[].areas[].spaces[].opaque_payload.panels[].vendor_extension authority=GUIL -->
<!-- P8-RUNTIME chapter=GUIL kind=json path=layouts[].areas[].spaces[].opaque_payload.python_console_visible authority=GUIL -->
<!-- P8-RUNTIME chapter=GUIL kind=json path=layouts[].areas[].spaces[].opaque_payload.python_console_width authority=GUIL -->
<!-- P8-RUNTIME chapter=GUIL kind=json path=layouts[].areas[].spaces[].opaque_payload.right_panel_width authority=GUIL -->
<!-- P8-RUNTIME chapter=GUIL kind=json path=layouts[].areas[].spaces[].opaque_payload.scene_panel_ratio authority=GUIL -->
<!-- P8-RUNTIME chapter=GUIL kind=json path=layouts[].areas[].spaces[].opaque_payload.sequencer_visible authority=GUIL -->
<!-- P8-RUNTIME chapter=GUIL kind=json path=layouts[].areas[].spaces[].opaque_payload.window authority=GUIL -->
<!-- P8-RUNTIME chapter=GUIL kind=json path=layouts[].areas[].spaces[].opaque_payload.window.fullscreen authority=GUIL -->
<!-- P8-RUNTIME chapter=GUIL kind=json path=layouts[].areas[].spaces[].opaque_payload.window.height authority=GUIL -->
<!-- P8-RUNTIME chapter=GUIL kind=json path=layouts[].areas[].spaces[].opaque_payload.window.maximized authority=GUIL -->
<!-- P8-RUNTIME chapter=GUIL kind=json path=layouts[].areas[].spaces[].opaque_payload.window.restore_height authority=GUIL -->
<!-- P8-RUNTIME chapter=GUIL kind=json path=layouts[].areas[].spaces[].opaque_payload.window.restore_width authority=GUIL -->
<!-- P8-RUNTIME chapter=GUIL kind=json path=layouts[].areas[].spaces[].opaque_payload.window.restore_x authority=GUIL -->
<!-- P8-RUNTIME chapter=GUIL kind=json path=layouts[].areas[].spaces[].opaque_payload.window.restore_y authority=GUIL -->
<!-- P8-RUNTIME chapter=GUIL kind=json path=layouts[].areas[].spaces[].opaque_payload.window.width authority=GUIL -->
<!-- P8-RUNTIME chapter=GUIL kind=json path=layouts[].areas[].spaces[].opaque_payload.window.x authority=GUIL -->
<!-- P8-RUNTIME chapter=GUIL kind=json path=layouts[].areas[].spaces[].opaque_payload.window.y authority=GUIL -->
<!-- P8-RUNTIME chapter=GUIL kind=json path=layouts[].areas[].spaces[].type authority=GUIL -->
<!-- P8-RUNTIME chapter=GUIL kind=json path=layouts[].areas[].spaces[].version authority=GUIL -->
<!-- P8-RUNTIME chapter=GUIL kind=json path=version authority=GUIL -->
<!-- P8-RUNTIME chapter=MESH kind=chunk path=MESH authority=MESH -->
<!-- P8-RUNTIME chapter=METR kind=chunk path=METR authority=METR -->
<!-- P8-RUNTIME chapter=PCLD kind=chunk path=PCLD authority=PCLD -->
<!-- P8-RUNTIME chapter=PRMS kind=chunk path=PRMS authority=PRMS -->
<!-- P8-RUNTIME chapter=PRMS kind=json path=active_strategy authority=PRMS -->
<!-- P8-RUNTIME chapter=PRMS kind=json path=dataset authority=PRMS -->
<!-- P8-RUNTIME chapter=PRMS kind=json path=dataset.centralize_dataset authority=PRMS -->
<!-- P8-RUNTIME chapter=PRMS kind=json path=dataset.images authority=PRMS -->
<!-- P8-RUNTIME chapter=PRMS kind=json path=dataset.invert_masks authority=PRMS -->
<!-- P8-RUNTIME chapter=PRMS kind=json path=dataset.loading_params authority=PRMS -->
<!-- P8-RUNTIME chapter=PRMS kind=json path=dataset.loading_params.min_cpu_free_GB authority=PRMS -->
<!-- P8-RUNTIME chapter=PRMS kind=json path=dataset.loading_params.min_cpu_free_memory_ratio authority=PRMS -->
<!-- P8-RUNTIME chapter=PRMS kind=json path=dataset.loading_params.print_cache_status authority=PRMS -->
<!-- P8-RUNTIME chapter=PRMS kind=json path=dataset.loading_params.print_status_freq_num authority=PRMS -->
<!-- P8-RUNTIME chapter=PRMS kind=json path=dataset.loading_params.use_16bit_color authority=PRMS -->
<!-- P8-RUNTIME chapter=PRMS kind=json path=dataset.loading_params.use_cpu_memory authority=PRMS -->
<!-- P8-RUNTIME chapter=PRMS kind=json path=dataset.loading_params.use_fs_cache authority=PRMS -->
<!-- P8-RUNTIME chapter=PRMS kind=json path=dataset.mask_threshold authority=PRMS -->
<!-- P8-RUNTIME chapter=PRMS kind=json path=dataset.max_width authority=PRMS -->
<!-- P8-RUNTIME chapter=PRMS kind=json path=dataset.min_track_length authority=PRMS -->
<!-- P8-RUNTIME chapter=PRMS kind=json path=dataset.resize_factor authority=PRMS -->
<!-- P8-RUNTIME chapter=PRMS kind=json path=dataset.test_every authority=PRMS -->
<!-- P8-RUNTIME chapter=PRMS kind=json path=dataset.timelapse_every authority=PRMS -->
<!-- P8-RUNTIME chapter=PRMS kind=json path=dataset.timelapse_images authority=PRMS -->
<!-- P8-RUNTIME chapter=PRMS kind=json path=dataset.timelapse_images[] authority=PRMS -->
<!-- P8-RUNTIME chapter=PRMS kind=json path=presets authority=PRMS -->
<!-- P8-RUNTIME chapter=PRMS kind=json path=presets.*.current authority=PRMS -->
<!-- P8-RUNTIME chapter=PRMS kind=json path=presets.*.current.background_image_reference_uuid authority=PRMS -->
<!-- P8-RUNTIME chapter=PRMS kind=json path=presets.*.current.bg_color authority=PRMS -->
<!-- P8-RUNTIME chapter=PRMS kind=json path=presets.*.current.bg_color[] authority=PRMS -->
<!-- P8-RUNTIME chapter=PRMS kind=json path=presets.*.current.bg_mode authority=PRMS -->
<!-- P8-RUNTIME chapter=PRMS kind=json path=presets.*.current.bg_modulation authority=PRMS -->
<!-- P8-RUNTIME chapter=PRMS kind=json path=presets.*.current.bilateral_grid_W authority=PRMS -->
<!-- P8-RUNTIME chapter=PRMS kind=json path=presets.*.current.bilateral_grid_X authority=PRMS -->
<!-- P8-RUNTIME chapter=PRMS kind=json path=presets.*.current.bilateral_grid_Y authority=PRMS -->
<!-- P8-RUNTIME chapter=PRMS kind=json path=presets.*.current.bilateral_grid_lr authority=PRMS -->
<!-- P8-RUNTIME chapter=PRMS kind=json path=presets.*.current.bounds_percentile authority=PRMS -->
<!-- P8-RUNTIME chapter=PRMS kind=json path=presets.*.current.cropbox_loss_weight authority=PRMS -->
<!-- P8-RUNTIME chapter=PRMS kind=json path=presets.*.current.cropbox_lr_scale authority=PRMS -->
<!-- P8-RUNTIME chapter=PRMS kind=json path=presets.*.current.depth_loss_mode authority=PRMS -->
<!-- P8-RUNTIME chapter=PRMS kind=json path=presets.*.current.depth_loss_weight authority=PRMS -->
<!-- P8-RUNTIME chapter=PRMS kind=json path=presets.*.current.enable_eval authority=PRMS -->
<!-- P8-RUNTIME chapter=PRMS kind=json path=presets.*.current.enable_save_eval_images authority=PRMS -->
<!-- P8-RUNTIME chapter=PRMS kind=json path=presets.*.current.enable_sparsity authority=PRMS -->
<!-- P8-RUNTIME chapter=PRMS kind=json path=presets.*.current.eval_steps authority=PRMS -->
<!-- P8-RUNTIME chapter=PRMS kind=json path=presets.*.current.eval_steps[] authority=PRMS -->
<!-- P8-RUNTIME chapter=PRMS kind=json path=presets.*.current.grow_fraction authority=PRMS -->
<!-- P8-RUNTIME chapter=PRMS kind=json path=presets.*.current.grow_until_iter authority=PRMS -->
<!-- P8-RUNTIME chapter=PRMS kind=json path=presets.*.current.growth_grad_threshold authority=PRMS -->
<!-- P8-RUNTIME chapter=PRMS kind=json path=presets.*.current.gut authority=PRMS -->
<!-- P8-RUNTIME chapter=PRMS kind=json path=presets.*.current.init_extent authority=PRMS -->
<!-- P8-RUNTIME chapter=PRMS kind=json path=presets.*.current.init_num_pts authority=PRMS -->
<!-- P8-RUNTIME chapter=PRMS kind=json path=presets.*.current.init_opacity authority=PRMS -->
<!-- P8-RUNTIME chapter=PRMS kind=json path=presets.*.current.init_rho authority=PRMS -->
<!-- P8-RUNTIME chapter=PRMS kind=json path=presets.*.current.init_scaling authority=PRMS -->
<!-- P8-RUNTIME chapter=PRMS kind=json path=presets.*.current.invert_masks authority=PRMS -->
<!-- P8-RUNTIME chapter=PRMS kind=json path=presets.*.current.iterations authority=PRMS -->
<!-- P8-RUNTIME chapter=PRMS kind=json path=presets.*.current.lambda_dssim authority=PRMS -->
<!-- P8-RUNTIME chapter=PRMS kind=json path=presets.*.current.mask_mode authority=PRMS -->
<!-- P8-RUNTIME chapter=PRMS kind=json path=presets.*.current.mask_opacity_penalty_power authority=PRMS -->
<!-- P8-RUNTIME chapter=PRMS kind=json path=presets.*.current.mask_opacity_penalty_weight authority=PRMS -->
<!-- P8-RUNTIME chapter=PRMS kind=json path=presets.*.current.mask_threshold authority=PRMS -->
<!-- P8-RUNTIME chapter=PRMS kind=json path=presets.*.current.max_cap authority=PRMS -->
<!-- P8-RUNTIME chapter=PRMS kind=json path=presets.*.current.means_lr authority=PRMS -->
<!-- P8-RUNTIME chapter=PRMS kind=json path=presets.*.current.means_lr_end authority=PRMS -->
<!-- P8-RUNTIME chapter=PRMS kind=json path=presets.*.current.means_noise_weight authority=PRMS -->
<!-- P8-RUNTIME chapter=PRMS kind=json path=presets.*.current.min_opacity authority=PRMS -->
<!-- P8-RUNTIME chapter=PRMS kind=json path=presets.*.current.mip_filter authority=PRMS -->
<!-- P8-RUNTIME chapter=PRMS kind=json path=presets.*.current.normal_consistency_weight authority=PRMS -->
<!-- P8-RUNTIME chapter=PRMS kind=json path=presets.*.current.normal_flatten_weight authority=PRMS -->
<!-- P8-RUNTIME chapter=PRMS kind=json path=presets.*.current.normal_loss_space authority=PRMS -->
<!-- P8-RUNTIME chapter=PRMS kind=json path=presets.*.current.normal_loss_weight authority=PRMS -->
<!-- P8-RUNTIME chapter=PRMS kind=json path=presets.*.current.opacity_decay authority=PRMS -->
<!-- P8-RUNTIME chapter=PRMS kind=json path=presets.*.current.opacity_lr authority=PRMS -->
<!-- P8-RUNTIME chapter=PRMS kind=json path=presets.*.current.opacity_reg authority=PRMS -->
<!-- P8-RUNTIME chapter=PRMS kind=json path=presets.*.current.ppisp_controller_activation_step authority=PRMS -->
<!-- P8-RUNTIME chapter=PRMS kind=json path=presets.*.current.ppisp_controller_lr authority=PRMS -->
<!-- P8-RUNTIME chapter=PRMS kind=json path=presets.*.current.ppisp_freeze_from_sidecar authority=PRMS -->
<!-- P8-RUNTIME chapter=PRMS kind=json path=presets.*.current.ppisp_freeze_gaussians_on_distill authority=PRMS -->
<!-- P8-RUNTIME chapter=PRMS kind=json path=presets.*.current.ppisp_lr authority=PRMS -->
<!-- P8-RUNTIME chapter=PRMS kind=json path=presets.*.current.ppisp_reference_uuid authority=PRMS -->
<!-- P8-RUNTIME chapter=PRMS kind=json path=presets.*.current.ppisp_reg_weight authority=PRMS -->
<!-- P8-RUNTIME chapter=PRMS kind=json path=presets.*.current.ppisp_use_controller authority=PRMS -->
<!-- P8-RUNTIME chapter=PRMS kind=json path=presets.*.current.ppisp_warmup_steps authority=PRMS -->
<!-- P8-RUNTIME chapter=PRMS kind=json path=presets.*.current.prune_opacity authority=PRMS -->
<!-- P8-RUNTIME chapter=PRMS kind=json path=presets.*.current.prune_ratio authority=PRMS -->
<!-- P8-RUNTIME chapter=PRMS kind=json path=presets.*.current.random authority=PRMS -->
<!-- P8-RUNTIME chapter=PRMS kind=json path=presets.*.current.refine_every authority=PRMS -->
<!-- P8-RUNTIME chapter=PRMS kind=json path=presets.*.current.reset_every authority=PRMS -->
<!-- P8-RUNTIME chapter=PRMS kind=json path=presets.*.current.rotation_lr authority=PRMS -->
<!-- P8-RUNTIME chapter=PRMS kind=json path=presets.*.current.save_steps authority=PRMS -->
<!-- P8-RUNTIME chapter=PRMS kind=json path=presets.*.current.save_steps[] authority=PRMS -->
<!-- P8-RUNTIME chapter=PRMS kind=json path=presets.*.current.scale_decay authority=PRMS -->
<!-- P8-RUNTIME chapter=PRMS kind=json path=presets.*.current.scale_reg authority=PRMS -->
<!-- P8-RUNTIME chapter=PRMS kind=json path=presets.*.current.scaling_lr authority=PRMS -->
<!-- P8-RUNTIME chapter=PRMS kind=json path=presets.*.current.scaling_lr_end authority=PRMS -->
<!-- P8-RUNTIME chapter=PRMS kind=json path=presets.*.current.sh_degree authority=PRMS -->
<!-- P8-RUNTIME chapter=PRMS kind=json path=presets.*.current.sh_degree_interval authority=PRMS -->
<!-- P8-RUNTIME chapter=PRMS kind=json path=presets.*.current.shs_lr authority=PRMS -->
<!-- P8-RUNTIME chapter=PRMS kind=json path=presets.*.current.sparsify_steps authority=PRMS -->
<!-- P8-RUNTIME chapter=PRMS kind=json path=presets.*.current.start_refine authority=PRMS -->
<!-- P8-RUNTIME chapter=PRMS kind=json path=presets.*.current.steps_scaler authority=PRMS -->
<!-- P8-RUNTIME chapter=PRMS kind=json path=presets.*.current.stop_refine authority=PRMS -->
<!-- P8-RUNTIME chapter=PRMS kind=json path=presets.*.current.strategy authority=PRMS -->
<!-- P8-RUNTIME chapter=PRMS kind=json path=presets.*.current.tv_loss_weight authority=PRMS -->
<!-- P8-RUNTIME chapter=PRMS kind=json path=presets.*.current.undistort authority=PRMS -->
<!-- P8-RUNTIME chapter=PRMS kind=json path=presets.*.current.use_alpha_as_mask authority=PRMS -->
<!-- P8-RUNTIME chapter=PRMS kind=json path=presets.*.current.use_bilateral_grid authority=PRMS -->
<!-- P8-RUNTIME chapter=PRMS kind=json path=presets.*.current.use_depth_loss authority=PRMS -->
<!-- P8-RUNTIME chapter=PRMS kind=json path=presets.*.current.use_edge_map authority=PRMS -->
<!-- P8-RUNTIME chapter=PRMS kind=json path=presets.*.current.use_error_map authority=PRMS -->
<!-- P8-RUNTIME chapter=PRMS kind=json path=presets.*.current.use_normal_loss authority=PRMS -->
<!-- P8-RUNTIME chapter=PRMS kind=json path=presets.*.current.use_ppisp authority=PRMS -->
<!-- P8-RUNTIME chapter=PRMS kind=json path=presets.*.session authority=PRMS -->
<!-- P8-RUNTIME chapter=PRMS kind=json path=presets.*.session.background_image_reference_uuid authority=PRMS -->
<!-- P8-RUNTIME chapter=PRMS kind=json path=presets.*.session.bg_color authority=PRMS -->
<!-- P8-RUNTIME chapter=PRMS kind=json path=presets.*.session.bg_color[] authority=PRMS -->
<!-- P8-RUNTIME chapter=PRMS kind=json path=presets.*.session.bg_mode authority=PRMS -->
<!-- P8-RUNTIME chapter=PRMS kind=json path=presets.*.session.bg_modulation authority=PRMS -->
<!-- P8-RUNTIME chapter=PRMS kind=json path=presets.*.session.bilateral_grid_W authority=PRMS -->
<!-- P8-RUNTIME chapter=PRMS kind=json path=presets.*.session.bilateral_grid_X authority=PRMS -->
<!-- P8-RUNTIME chapter=PRMS kind=json path=presets.*.session.bilateral_grid_Y authority=PRMS -->
<!-- P8-RUNTIME chapter=PRMS kind=json path=presets.*.session.bilateral_grid_lr authority=PRMS -->
<!-- P8-RUNTIME chapter=PRMS kind=json path=presets.*.session.bounds_percentile authority=PRMS -->
<!-- P8-RUNTIME chapter=PRMS kind=json path=presets.*.session.cropbox_loss_weight authority=PRMS -->
<!-- P8-RUNTIME chapter=PRMS kind=json path=presets.*.session.cropbox_lr_scale authority=PRMS -->
<!-- P8-RUNTIME chapter=PRMS kind=json path=presets.*.session.depth_loss_mode authority=PRMS -->
<!-- P8-RUNTIME chapter=PRMS kind=json path=presets.*.session.depth_loss_weight authority=PRMS -->
<!-- P8-RUNTIME chapter=PRMS kind=json path=presets.*.session.enable_eval authority=PRMS -->
<!-- P8-RUNTIME chapter=PRMS kind=json path=presets.*.session.enable_save_eval_images authority=PRMS -->
<!-- P8-RUNTIME chapter=PRMS kind=json path=presets.*.session.enable_sparsity authority=PRMS -->
<!-- P8-RUNTIME chapter=PRMS kind=json path=presets.*.session.eval_steps authority=PRMS -->
<!-- P8-RUNTIME chapter=PRMS kind=json path=presets.*.session.eval_steps[] authority=PRMS -->
<!-- P8-RUNTIME chapter=PRMS kind=json path=presets.*.session.grow_fraction authority=PRMS -->
<!-- P8-RUNTIME chapter=PRMS kind=json path=presets.*.session.grow_until_iter authority=PRMS -->
<!-- P8-RUNTIME chapter=PRMS kind=json path=presets.*.session.growth_grad_threshold authority=PRMS -->
<!-- P8-RUNTIME chapter=PRMS kind=json path=presets.*.session.gut authority=PRMS -->
<!-- P8-RUNTIME chapter=PRMS kind=json path=presets.*.session.init_extent authority=PRMS -->
<!-- P8-RUNTIME chapter=PRMS kind=json path=presets.*.session.init_num_pts authority=PRMS -->
<!-- P8-RUNTIME chapter=PRMS kind=json path=presets.*.session.init_opacity authority=PRMS -->
<!-- P8-RUNTIME chapter=PRMS kind=json path=presets.*.session.init_rho authority=PRMS -->
<!-- P8-RUNTIME chapter=PRMS kind=json path=presets.*.session.init_scaling authority=PRMS -->
<!-- P8-RUNTIME chapter=PRMS kind=json path=presets.*.session.invert_masks authority=PRMS -->
<!-- P8-RUNTIME chapter=PRMS kind=json path=presets.*.session.iterations authority=PRMS -->
<!-- P8-RUNTIME chapter=PRMS kind=json path=presets.*.session.lambda_dssim authority=PRMS -->
<!-- P8-RUNTIME chapter=PRMS kind=json path=presets.*.session.mask_mode authority=PRMS -->
<!-- P8-RUNTIME chapter=PRMS kind=json path=presets.*.session.mask_opacity_penalty_power authority=PRMS -->
<!-- P8-RUNTIME chapter=PRMS kind=json path=presets.*.session.mask_opacity_penalty_weight authority=PRMS -->
<!-- P8-RUNTIME chapter=PRMS kind=json path=presets.*.session.mask_threshold authority=PRMS -->
<!-- P8-RUNTIME chapter=PRMS kind=json path=presets.*.session.max_cap authority=PRMS -->
<!-- P8-RUNTIME chapter=PRMS kind=json path=presets.*.session.means_lr authority=PRMS -->
<!-- P8-RUNTIME chapter=PRMS kind=json path=presets.*.session.means_lr_end authority=PRMS -->
<!-- P8-RUNTIME chapter=PRMS kind=json path=presets.*.session.means_noise_weight authority=PRMS -->
<!-- P8-RUNTIME chapter=PRMS kind=json path=presets.*.session.min_opacity authority=PRMS -->
<!-- P8-RUNTIME chapter=PRMS kind=json path=presets.*.session.mip_filter authority=PRMS -->
<!-- P8-RUNTIME chapter=PRMS kind=json path=presets.*.session.normal_consistency_weight authority=PRMS -->
<!-- P8-RUNTIME chapter=PRMS kind=json path=presets.*.session.normal_flatten_weight authority=PRMS -->
<!-- P8-RUNTIME chapter=PRMS kind=json path=presets.*.session.normal_loss_space authority=PRMS -->
<!-- P8-RUNTIME chapter=PRMS kind=json path=presets.*.session.normal_loss_weight authority=PRMS -->
<!-- P8-RUNTIME chapter=PRMS kind=json path=presets.*.session.opacity_decay authority=PRMS -->
<!-- P8-RUNTIME chapter=PRMS kind=json path=presets.*.session.opacity_lr authority=PRMS -->
<!-- P8-RUNTIME chapter=PRMS kind=json path=presets.*.session.opacity_reg authority=PRMS -->
<!-- P8-RUNTIME chapter=PRMS kind=json path=presets.*.session.ppisp_controller_activation_step authority=PRMS -->
<!-- P8-RUNTIME chapter=PRMS kind=json path=presets.*.session.ppisp_controller_lr authority=PRMS -->
<!-- P8-RUNTIME chapter=PRMS kind=json path=presets.*.session.ppisp_freeze_from_sidecar authority=PRMS -->
<!-- P8-RUNTIME chapter=PRMS kind=json path=presets.*.session.ppisp_freeze_gaussians_on_distill authority=PRMS -->
<!-- P8-RUNTIME chapter=PRMS kind=json path=presets.*.session.ppisp_lr authority=PRMS -->
<!-- P8-RUNTIME chapter=PRMS kind=json path=presets.*.session.ppisp_reference_uuid authority=PRMS -->
<!-- P8-RUNTIME chapter=PRMS kind=json path=presets.*.session.ppisp_reg_weight authority=PRMS -->
<!-- P8-RUNTIME chapter=PRMS kind=json path=presets.*.session.ppisp_use_controller authority=PRMS -->
<!-- P8-RUNTIME chapter=PRMS kind=json path=presets.*.session.ppisp_warmup_steps authority=PRMS -->
<!-- P8-RUNTIME chapter=PRMS kind=json path=presets.*.session.prune_opacity authority=PRMS -->
<!-- P8-RUNTIME chapter=PRMS kind=json path=presets.*.session.prune_ratio authority=PRMS -->
<!-- P8-RUNTIME chapter=PRMS kind=json path=presets.*.session.random authority=PRMS -->
<!-- P8-RUNTIME chapter=PRMS kind=json path=presets.*.session.refine_every authority=PRMS -->
<!-- P8-RUNTIME chapter=PRMS kind=json path=presets.*.session.reset_every authority=PRMS -->
<!-- P8-RUNTIME chapter=PRMS kind=json path=presets.*.session.rotation_lr authority=PRMS -->
<!-- P8-RUNTIME chapter=PRMS kind=json path=presets.*.session.save_steps authority=PRMS -->
<!-- P8-RUNTIME chapter=PRMS kind=json path=presets.*.session.save_steps[] authority=PRMS -->
<!-- P8-RUNTIME chapter=PRMS kind=json path=presets.*.session.scale_decay authority=PRMS -->
<!-- P8-RUNTIME chapter=PRMS kind=json path=presets.*.session.scale_reg authority=PRMS -->
<!-- P8-RUNTIME chapter=PRMS kind=json path=presets.*.session.scaling_lr authority=PRMS -->
<!-- P8-RUNTIME chapter=PRMS kind=json path=presets.*.session.scaling_lr_end authority=PRMS -->
<!-- P8-RUNTIME chapter=PRMS kind=json path=presets.*.session.sh_degree authority=PRMS -->
<!-- P8-RUNTIME chapter=PRMS kind=json path=presets.*.session.sh_degree_interval authority=PRMS -->
<!-- P8-RUNTIME chapter=PRMS kind=json path=presets.*.session.shs_lr authority=PRMS -->
<!-- P8-RUNTIME chapter=PRMS kind=json path=presets.*.session.sparsify_steps authority=PRMS -->
<!-- P8-RUNTIME chapter=PRMS kind=json path=presets.*.session.start_refine authority=PRMS -->
<!-- P8-RUNTIME chapter=PRMS kind=json path=presets.*.session.steps_scaler authority=PRMS -->
<!-- P8-RUNTIME chapter=PRMS kind=json path=presets.*.session.stop_refine authority=PRMS -->
<!-- P8-RUNTIME chapter=PRMS kind=json path=presets.*.session.strategy authority=PRMS -->
<!-- P8-RUNTIME chapter=PRMS kind=json path=presets.*.session.tv_loss_weight authority=PRMS -->
<!-- P8-RUNTIME chapter=PRMS kind=json path=presets.*.session.undistort authority=PRMS -->
<!-- P8-RUNTIME chapter=PRMS kind=json path=presets.*.session.use_alpha_as_mask authority=PRMS -->
<!-- P8-RUNTIME chapter=PRMS kind=json path=presets.*.session.use_bilateral_grid authority=PRMS -->
<!-- P8-RUNTIME chapter=PRMS kind=json path=presets.*.session.use_depth_loss authority=PRMS -->
<!-- P8-RUNTIME chapter=PRMS kind=json path=presets.*.session.use_edge_map authority=PRMS -->
<!-- P8-RUNTIME chapter=PRMS kind=json path=presets.*.session.use_error_map authority=PRMS -->
<!-- P8-RUNTIME chapter=PRMS kind=json path=presets.*.session.use_normal_loss authority=PRMS -->
<!-- P8-RUNTIME chapter=PRMS kind=json path=presets.*.session.use_ppisp authority=PRMS -->
<!-- P8-RUNTIME chapter=PRMS kind=json path=presets.igs+ authority=PRMS -->
<!-- P8-RUNTIME chapter=PRMS kind=json path=presets.mcmc authority=PRMS -->
<!-- P8-RUNTIME chapter=PRMS kind=json path=presets.mrnf authority=PRMS -->
<!-- P8-RUNTIME chapter=PRMS kind=json path=schema_version authority=PRMS -->
<!-- P8-RUNTIME chapter=PROJ kind=chunk path=PROJ authority=PROJ -->
<!-- P8-RUNTIME chapter=PROJ kind=json path=created_at_unix_ns authority=PROJ -->
<!-- P8-RUNTIME chapter=PROJ kind=json path=dataset_reference_uuid authority=PROJ -->
<!-- P8-RUNTIME chapter=PROJ kind=json path=embed_decisions authority=PROJ -->
<!-- P8-RUNTIME chapter=PROJ kind=json path=embed_decisions[] authority=PROJ -->
<!-- P8-RUNTIME chapter=PROJ kind=json path=embed_decisions[].decision authority=PROJ -->
<!-- P8-RUNTIME chapter=PROJ kind=json path=embed_decisions[].node_uuid authority=PROJ -->
<!-- P8-RUNTIME chapter=PROJ kind=json path=embed_decisions[].payload_fourcc authority=PROJ -->
<!-- P8-RUNTIME chapter=PROJ kind=json path=embed_decisions[].reason authority=PROJ -->
<!-- P8-RUNTIME chapter=PROJ kind=json path=embed_decisions[].reference_uuid authority=PROJ -->
<!-- P8-RUNTIME chapter=PROJ kind=json path=embed_decisions[].uuid authority=PROJ -->
<!-- P8-RUNTIME chapter=PROJ kind=json path=embedded_payloads authority=PROJ -->
<!-- P8-RUNTIME chapter=PROJ kind=json path=embedded_payloads[] authority=PROJ -->
<!-- P8-RUNTIME chapter=PROJ kind=json path=embedded_payloads[].content_xxh3_128 authority=PROJ -->
<!-- P8-RUNTIME chapter=PROJ kind=json path=embedded_payloads[].fourcc authority=PROJ -->
<!-- P8-RUNTIME chapter=PROJ kind=json path=embedded_payloads[].import_fingerprint authority=PROJ -->
<!-- P8-RUNTIME chapter=PROJ kind=json path=embedded_payloads[].import_fingerprint.full_xxh3_128 authority=PROJ -->
<!-- P8-RUNTIME chapter=PROJ kind=json path=embedded_payloads[].import_fingerprint.head_xxh3_128 authority=PROJ -->
<!-- P8-RUNTIME chapter=PROJ kind=json path=embedded_payloads[].import_fingerprint.kind authority=PROJ -->
<!-- P8-RUNTIME chapter=PROJ kind=json path=embedded_payloads[].import_fingerprint.mtime_unix_ns authority=PROJ -->
<!-- P8-RUNTIME chapter=PROJ kind=json path=embedded_payloads[].import_fingerprint.size authority=PROJ -->
<!-- P8-RUNTIME chapter=PROJ kind=json path=embedded_payloads[].import_fingerprint.tail_xxh3_128 authority=PROJ -->
<!-- P8-RUNTIME chapter=PROJ kind=json path=embedded_payloads[].import_locator authority=PROJ -->
<!-- P8-RUNTIME chapter=PROJ kind=json path=embedded_payloads[].import_locator.base authority=PROJ -->
<!-- P8-RUNTIME chapter=PROJ kind=json path=embedded_payloads[].import_locator.preferred authority=PROJ -->
<!-- P8-RUNTIME chapter=PROJ kind=json path=embedded_payloads[].node_uuid authority=PROJ -->
<!-- P8-RUNTIME chapter=PROJ kind=json path=embedded_payloads[].uuid authority=PROJ -->
<!-- P8-RUNTIME chapter=PROJ kind=json path=georeference authority=PROJ -->
<!-- P8-RUNTIME chapter=PROJ kind=json path=georeference.crs authority=PROJ -->
<!-- P8-RUNTIME chapter=PROJ kind=json path=georeference.world_origin authority=PROJ -->
<!-- P8-RUNTIME chapter=PROJ kind=json path=georeference.world_origin[] authority=PROJ -->
<!-- P8-RUNTIME chapter=PROJ kind=json path=georeference.world_origin_provenance authority=PROJ -->
<!-- P8-RUNTIME chapter=PROJ kind=json path=georeference.world_unit_scale authority=PROJ -->
<!-- P8-RUNTIME chapter=PROJ kind=json path=manifest authority=PROJ -->
<!-- P8-RUNTIME chapter=PROJ kind=json path=manifest.application_name authority=PROJ -->
<!-- P8-RUNTIME chapter=PROJ kind=json path=manifest.application_version authority=PROJ -->
<!-- P8-RUNTIME chapter=PROJ kind=json path=manifest.application_version.major authority=PROJ -->
<!-- P8-RUNTIME chapter=PROJ kind=json path=manifest.application_version.minor authority=PROJ -->
<!-- P8-RUNTIME chapter=PROJ kind=json path=manifest.application_version.patch authority=PROJ -->
<!-- P8-RUNTIME chapter=PROJ kind=json path=manifest.minimum_reader_version authority=PROJ -->
<!-- P8-RUNTIME chapter=PROJ kind=json path=manifest.minimum_reader_version.major authority=PROJ -->
<!-- P8-RUNTIME chapter=PROJ kind=json path=manifest.minimum_reader_version.minor authority=PROJ -->
<!-- P8-RUNTIME chapter=PROJ kind=json path=manifest.minimum_reader_version.patch authority=PROJ -->
<!-- P8-RUNTIME chapter=PROJ kind=json path=manifest.minimum_safe_writer_version authority=PROJ -->
<!-- P8-RUNTIME chapter=PROJ kind=json path=manifest.minimum_safe_writer_version.major authority=PROJ -->
<!-- P8-RUNTIME chapter=PROJ kind=json path=manifest.minimum_safe_writer_version.minor authority=PROJ -->
<!-- P8-RUNTIME chapter=PROJ kind=json path=manifest.minimum_safe_writer_version.patch authority=PROJ -->
<!-- P8-RUNTIME chapter=PROJ kind=json path=manifest.optional_capabilities authority=PROJ -->
<!-- P8-RUNTIME chapter=PROJ kind=json path=manifest.optional_capabilities[] authority=PROJ -->
<!-- P8-RUNTIME chapter=PROJ kind=json path=manifest.required_capabilities authority=PROJ -->
<!-- P8-RUNTIME chapter=PROJ kind=json path=manifest.required_capabilities[] authority=PROJ -->
<!-- P8-RUNTIME chapter=PROJ kind=json path=manifest.schema_version authority=PROJ -->
<!-- P8-RUNTIME chapter=PROJ kind=json path=manifest.schema_version.major authority=PROJ -->
<!-- P8-RUNTIME chapter=PROJ kind=json path=manifest.schema_version.minor authority=PROJ -->
<!-- P8-RUNTIME chapter=PROJ kind=json path=manifest.schema_version.patch authority=PROJ -->
<!-- P8-RUNTIME chapter=PROJ kind=json path=modified_at_unix_ns authority=PROJ -->
<!-- P8-RUNTIME chapter=PROJ kind=json path=project_lineage authority=PROJ -->
<!-- P8-RUNTIME chapter=PROJ kind=json path=project_lineage[] authority=PROJ -->
<!-- P8-RUNTIME chapter=PROJ kind=json path=project_uuid authority=PROJ -->
<!-- P8-RUNTIME chapter=PROJ kind=json path=provenance authority=PROJ -->
<!-- P8-RUNTIME chapter=PROJ kind=json path=provenance[] authority=PROJ -->
<!-- P8-RUNTIME chapter=PROJ kind=json path=provenance[].kind authority=PROJ -->
<!-- P8-RUNTIME chapter=PROJ kind=json path=provenance[].uuid authority=PROJ -->
<!-- P8-RUNTIME chapter=PROJ kind=json path=provenance[].value authority=PROJ -->
<!-- P8-RUNTIME chapter=PROJ kind=json path=schema_version authority=PROJ -->
<!-- P8-RUNTIME chapter=REFS kind=chunk path=REFS authority=REFS -->
<!-- P8-RUNTIME chapter=REFS kind=json path=references authority=REFS -->
<!-- P8-RUNTIME chapter=REFS kind=json path=references[] authority=REFS -->
<!-- P8-RUNTIME chapter=REFS kind=json path=references[].fingerprint authority=REFS -->
<!-- P8-RUNTIME chapter=REFS kind=json path=references[].fingerprint.full_xxh3_128 authority=REFS -->
<!-- P8-RUNTIME chapter=REFS kind=json path=references[].fingerprint.head_xxh3_128 authority=REFS -->
<!-- P8-RUNTIME chapter=REFS kind=json path=references[].fingerprint.kind authority=REFS -->
<!-- P8-RUNTIME chapter=REFS kind=json path=references[].fingerprint.mtime_unix_ns authority=REFS -->
<!-- P8-RUNTIME chapter=REFS kind=json path=references[].fingerprint.size authority=REFS -->
<!-- P8-RUNTIME chapter=REFS kind=json path=references[].fingerprint.tail_xxh3_128 authority=REFS -->
<!-- P8-RUNTIME chapter=REFS kind=json path=references[].key authority=REFS -->
<!-- P8-RUNTIME chapter=REFS kind=json path=references[].kind authority=REFS -->
<!-- P8-RUNTIME chapter=REFS kind=json path=references[].locator authority=REFS -->
<!-- P8-RUNTIME chapter=REFS kind=json path=references[].locator.absolute_fallback authority=REFS -->
<!-- P8-RUNTIME chapter=REFS kind=json path=references[].locator.base authority=REFS -->
<!-- P8-RUNTIME chapter=REFS kind=json path=references[].locator.preferred authority=REFS -->
<!-- P8-RUNTIME chapter=REFS kind=json path=references[].unresolved authority=REFS -->
<!-- P8-RUNTIME chapter=REFS kind=json path=references[].uuid authority=REFS -->
<!-- P8-RUNTIME chapter=REFS kind=json path=schema_version authority=REFS -->
<!-- P8-RUNTIME chapter=SCNG kind=chunk path=SCNG authority=SCNG -->
<!-- P8-RUNTIME chapter=SCNG kind=json path=nodes authority=SCNG -->
<!-- P8-RUNTIME chapter=SCNG kind=json path=nodes[] authority=SCNG -->
<!-- P8-RUNTIME chapter=SCNG kind=json path=nodes[].camera authority=SCNG -->
<!-- P8-RUNTIME chapter=SCNG kind=json path=nodes[].camera.camera_height authority=SCNG -->
<!-- P8-RUNTIME chapter=SCNG kind=json path=nodes[].camera.camera_id authority=SCNG -->
<!-- P8-RUNTIME chapter=SCNG kind=json path=nodes[].camera.camera_model_type authority=SCNG -->
<!-- P8-RUNTIME chapter=SCNG kind=json path=nodes[].camera.camera_width authority=SCNG -->
<!-- P8-RUNTIME chapter=SCNG kind=json path=nodes[].camera.center_x authority=SCNG -->
<!-- P8-RUNTIME chapter=SCNG kind=json path=nodes[].camera.center_y authority=SCNG -->
<!-- P8-RUNTIME chapter=SCNG kind=json path=nodes[].camera.depth_path authority=SCNG -->
<!-- P8-RUNTIME chapter=SCNG kind=json path=nodes[].camera.focal_x authority=SCNG -->
<!-- P8-RUNTIME chapter=SCNG kind=json path=nodes[].camera.focal_y authority=SCNG -->
<!-- P8-RUNTIME chapter=SCNG kind=json path=nodes[].camera.has_alpha authority=SCNG -->
<!-- P8-RUNTIME chapter=SCNG kind=json path=nodes[].camera.image_height authority=SCNG -->
<!-- P8-RUNTIME chapter=SCNG kind=json path=nodes[].camera.image_name authority=SCNG -->
<!-- P8-RUNTIME chapter=SCNG kind=json path=nodes[].camera.image_path authority=SCNG -->
<!-- P8-RUNTIME chapter=SCNG kind=json path=nodes[].camera.image_width authority=SCNG -->
<!-- P8-RUNTIME chapter=SCNG kind=json path=nodes[].camera.mask_path authority=SCNG -->
<!-- P8-RUNTIME chapter=SCNG kind=json path=nodes[].camera.normal_path authority=SCNG -->
<!-- P8-RUNTIME chapter=SCNG kind=json path=nodes[].camera.radial_distortion authority=SCNG -->
<!-- P8-RUNTIME chapter=SCNG kind=json path=nodes[].camera.radial_distortion[] authority=SCNG -->
<!-- P8-RUNTIME chapter=SCNG kind=json path=nodes[].camera.rotation authority=SCNG -->
<!-- P8-RUNTIME chapter=SCNG kind=json path=nodes[].camera.rotation[] authority=SCNG -->
<!-- P8-RUNTIME chapter=SCNG kind=json path=nodes[].camera.split authority=SCNG -->
<!-- P8-RUNTIME chapter=SCNG kind=json path=nodes[].camera.tangential_distortion authority=SCNG -->
<!-- P8-RUNTIME chapter=SCNG kind=json path=nodes[].camera.tangential_distortion[] authority=SCNG -->
<!-- P8-RUNTIME chapter=SCNG kind=json path=nodes[].camera.translation authority=SCNG -->
<!-- P8-RUNTIME chapter=SCNG kind=json path=nodes[].camera.translation[] authority=SCNG -->
<!-- P8-RUNTIME chapter=SCNG kind=json path=nodes[].camera.uid authority=SCNG -->
<!-- P8-RUNTIME chapter=SCNG kind=json path=nodes[].child_order authority=SCNG -->
<!-- P8-RUNTIME chapter=SCNG kind=json path=nodes[].cropbox authority=SCNG -->
<!-- P8-RUNTIME chapter=SCNG kind=json path=nodes[].cropbox.color authority=SCNG -->
<!-- P8-RUNTIME chapter=SCNG kind=json path=nodes[].cropbox.color[] authority=SCNG -->
<!-- P8-RUNTIME chapter=SCNG kind=json path=nodes[].cropbox.enabled authority=SCNG -->
<!-- P8-RUNTIME chapter=SCNG kind=json path=nodes[].cropbox.inverse authority=SCNG -->
<!-- P8-RUNTIME chapter=SCNG kind=json path=nodes[].cropbox.line_width authority=SCNG -->
<!-- P8-RUNTIME chapter=SCNG kind=json path=nodes[].cropbox.max authority=SCNG -->
<!-- P8-RUNTIME chapter=SCNG kind=json path=nodes[].cropbox.max[] authority=SCNG -->
<!-- P8-RUNTIME chapter=SCNG kind=json path=nodes[].cropbox.min authority=SCNG -->
<!-- P8-RUNTIME chapter=SCNG kind=json path=nodes[].cropbox.min[] authority=SCNG -->
<!-- P8-RUNTIME chapter=SCNG kind=json path=nodes[].ellipsoid authority=SCNG -->
<!-- P8-RUNTIME chapter=SCNG kind=json path=nodes[].ellipsoid.color authority=SCNG -->
<!-- P8-RUNTIME chapter=SCNG kind=json path=nodes[].ellipsoid.color[] authority=SCNG -->
<!-- P8-RUNTIME chapter=SCNG kind=json path=nodes[].ellipsoid.enabled authority=SCNG -->
<!-- P8-RUNTIME chapter=SCNG kind=json path=nodes[].ellipsoid.inverse authority=SCNG -->
<!-- P8-RUNTIME chapter=SCNG kind=json path=nodes[].ellipsoid.line_width authority=SCNG -->
<!-- P8-RUNTIME chapter=SCNG kind=json path=nodes[].ellipsoid.radii authority=SCNG -->
<!-- P8-RUNTIME chapter=SCNG kind=json path=nodes[].ellipsoid.radii[] authority=SCNG -->
<!-- P8-RUNTIME chapter=SCNG kind=json path=nodes[].georef_pose authority=SCNG -->
<!-- P8-RUNTIME chapter=SCNG kind=json path=nodes[].georef_pose.rotation authority=SCNG -->
<!-- P8-RUNTIME chapter=SCNG kind=json path=nodes[].georef_pose.rotation[] authority=SCNG -->
<!-- P8-RUNTIME chapter=SCNG kind=json path=nodes[].georef_pose.translation authority=SCNG -->
<!-- P8-RUNTIME chapter=SCNG kind=json path=nodes[].georef_pose.translation[] authority=SCNG -->
<!-- P8-RUNTIME chapter=SCNG kind=json path=nodes[].local_transform authority=SCNG -->
<!-- P8-RUNTIME chapter=SCNG kind=json path=nodes[].local_transform[] authority=SCNG -->
<!-- P8-RUNTIME chapter=SCNG kind=json path=nodes[].locked authority=SCNG -->
<!-- P8-RUNTIME chapter=SCNG kind=json path=nodes[].name authority=SCNG -->
<!-- P8-RUNTIME chapter=SCNG kind=json path=nodes[].parent_uuid authority=SCNG -->
<!-- P8-RUNTIME chapter=SCNG kind=json path=nodes[].payload authority=SCNG -->
<!-- P8-RUNTIME chapter=SCNG kind=json path=nodes[].payload.fourcc authority=SCNG -->
<!-- P8-RUNTIME chapter=SCNG kind=json path=nodes[].payload.instance_uuid authority=SCNG -->
<!-- P8-RUNTIME chapter=SCNG kind=json path=nodes[].payload.reference_uuid authority=SCNG -->
<!-- P8-RUNTIME chapter=SCNG kind=json path=nodes[].payload.source_kind authority=SCNG -->
<!-- P8-RUNTIME chapter=SCNG kind=json path=nodes[].payload_diverged authority=SCNG -->
<!-- P8-RUNTIME chapter=SCNG kind=json path=nodes[].training_enabled authority=SCNG -->
<!-- P8-RUNTIME chapter=SCNG kind=json path=nodes[].type authority=SCNG -->
<!-- P8-RUNTIME chapter=SCNG kind=json path=nodes[].uuid authority=SCNG -->
<!-- P8-RUNTIME chapter=SCNG kind=json path=nodes[].visible authority=SCNG -->
<!-- P8-RUNTIME chapter=SCNG kind=json path=schema_version authority=SCNG -->
<!-- P8-RUNTIME chapter=SCNG kind=json path=training_model_uuid authority=SCNG -->
<!-- P8-RUNTIME chapter=SELM kind=chunk path=SELM authority=SELM -->
<!-- P8-RUNTIME chapter=SEQR kind=chunk path=SEQR authority=SEQR -->
<!-- P8-RUNTIME chapter=SEQR kind=json path=loop_mode authority=SEQR -->
<!-- P8-RUNTIME chapter=SEQR kind=json path=playback_speed authority=SEQR -->
<!-- P8-RUNTIME chapter=SEQR kind=json path=playhead authority=SEQR -->
<!-- P8-RUNTIME chapter=SEQR kind=json path=ply_sequences authority=SEQR -->
<!-- P8-RUNTIME chapter=SEQR kind=json path=ply_sequences[] authority=SEQR -->
<!-- P8-RUNTIME chapter=SEQR kind=json path=ply_sequences[].directory_hint authority=SEQR -->
<!-- P8-RUNTIME chapter=SEQR kind=json path=ply_sequences[].directory_reference_uuid authority=SEQR -->
<!-- P8-RUNTIME chapter=SEQR kind=json path=ply_sequences[].fps authority=SEQR -->
<!-- P8-RUNTIME chapter=SEQR kind=json path=ply_sequences[].frames authority=SEQR -->
<!-- P8-RUNTIME chapter=SEQR kind=json path=ply_sequences[].frames[] authority=SEQR -->
<!-- P8-RUNTIME chapter=SEQR kind=json path=ply_sequences[].frames[].locator authority=SEQR -->
<!-- P8-RUNTIME chapter=SEQR kind=json path=ply_sequences[].frames[].node_name authority=SEQR -->
<!-- P8-RUNTIME chapter=SEQR kind=json path=ply_sequences[].frames[].node_uuid authority=SEQR -->
<!-- P8-RUNTIME chapter=SEQR kind=json path=ply_sequences[].node_name authority=SEQR -->
<!-- P8-RUNTIME chapter=SEQR kind=json path=ply_sequences[].node_uuid authority=SEQR -->
<!-- P8-RUNTIME chapter=SEQR kind=json path=preferences authority=SEQR -->
<!-- P8-RUNTIME chapter=SEQR kind=json path=preferences.follow_playback authority=SEQR -->
<!-- P8-RUNTIME chapter=SEQR kind=json path=preferences.pip_preview_scale authority=SEQR -->
<!-- P8-RUNTIME chapter=SEQR kind=json path=preferences.show_film_strip authority=SEQR -->
<!-- P8-RUNTIME chapter=SEQR kind=json path=preferences.show_pip_preview authority=SEQR -->
<!-- P8-RUNTIME chapter=SEQR kind=json path=preferences.snap_interval authority=SEQR -->
<!-- P8-RUNTIME chapter=SEQR kind=json path=preferences.snap_to_grid authority=SEQR -->
<!-- P8-RUNTIME chapter=SEQR kind=json path=timeline authority=SEQR -->
<!-- P8-RUNTIME chapter=SEQR kind=json path=timeline.animation_clip authority=SEQR -->
<!-- P8-RUNTIME chapter=SEQR kind=json path=timeline.animation_clip.name authority=SEQR -->
<!-- P8-RUNTIME chapter=SEQR kind=json path=timeline.animation_clip.tracks authority=SEQR -->
<!-- P8-RUNTIME chapter=SEQR kind=json path=timeline.animation_clip.tracks[] authority=SEQR -->
<!-- P8-RUNTIME chapter=SEQR kind=json path=timeline.animation_clip.tracks[].id authority=SEQR -->
<!-- P8-RUNTIME chapter=SEQR kind=json path=timeline.animation_clip.tracks[].keyframes authority=SEQR -->
<!-- P8-RUNTIME chapter=SEQR kind=json path=timeline.animation_clip.tracks[].keyframes[] authority=SEQR -->
<!-- P8-RUNTIME chapter=SEQR kind=json path=timeline.animation_clip.tracks[].keyframes[].easing authority=SEQR -->
<!-- P8-RUNTIME chapter=SEQR kind=json path=timeline.animation_clip.tracks[].keyframes[].time authority=SEQR -->
<!-- P8-RUNTIME chapter=SEQR kind=json path=timeline.animation_clip.tracks[].keyframes[].value authority=SEQR -->
<!-- P8-RUNTIME chapter=SEQR kind=json path=timeline.animation_clip.tracks[].target authority=SEQR -->
<!-- P8-RUNTIME chapter=SEQR kind=json path=timeline.animation_clip.tracks[].type authority=SEQR -->
<!-- P8-RUNTIME chapter=SEQR kind=json path=timeline.clip_duration authority=SEQR -->
<!-- P8-RUNTIME chapter=SEQR kind=json path=timeline.keyframes authority=SEQR -->
<!-- P8-RUNTIME chapter=SEQR kind=json path=timeline.keyframes[] authority=SEQR -->
<!-- P8-RUNTIME chapter=SEQR kind=json path=timeline.keyframes[].easing authority=SEQR -->
<!-- P8-RUNTIME chapter=SEQR kind=json path=timeline.keyframes[].focal_length_mm authority=SEQR -->
<!-- P8-RUNTIME chapter=SEQR kind=json path=timeline.keyframes[].position authority=SEQR -->
<!-- P8-RUNTIME chapter=SEQR kind=json path=timeline.keyframes[].position[] authority=SEQR -->
<!-- P8-RUNTIME chapter=SEQR kind=json path=timeline.keyframes[].rotation authority=SEQR -->
<!-- P8-RUNTIME chapter=SEQR kind=json path=timeline.keyframes[].rotation[] authority=SEQR -->
<!-- P8-RUNTIME chapter=SEQR kind=json path=timeline.keyframes[].time authority=SEQR -->
<!-- P8-RUNTIME chapter=SEQR kind=json path=timeline.version authority=SEQR -->
<!-- P8-RUNTIME chapter=SEQR kind=json path=version authority=SEQR -->
<!-- P8-RUNTIME chapter=SPLT kind=chunk path=SPLT authority=SPLT -->
<!-- P8-RUNTIME chapter=VIEW kind=chunk path=VIEW authority=VIEW -->
<!-- P8-RUNTIME chapter=VIEW kind=json path=camera_bookmarks authority=VIEW -->
<!-- P8-RUNTIME chapter=VIEW kind=json path=camera_bookmarks[] authority=VIEW -->
<!-- P8-RUNTIME chapter=VIEW kind=json path=camera_bookmarks[].R authority=VIEW -->
<!-- P8-RUNTIME chapter=VIEW kind=json path=camera_bookmarks[].R[] authority=VIEW -->
<!-- P8-RUNTIME chapter=VIEW kind=json path=camera_bookmarks[].centre_speed authority=VIEW -->
<!-- P8-RUNTIME chapter=VIEW kind=json path=camera_bookmarks[].home_R authority=VIEW -->
<!-- P8-RUNTIME chapter=VIEW kind=json path=camera_bookmarks[].home_R[] authority=VIEW -->
<!-- P8-RUNTIME chapter=VIEW kind=json path=camera_bookmarks[].home_pivot authority=VIEW -->
<!-- P8-RUNTIME chapter=VIEW kind=json path=camera_bookmarks[].home_pivot[] authority=VIEW -->
<!-- P8-RUNTIME chapter=VIEW kind=json path=camera_bookmarks[].home_saved authority=VIEW -->
<!-- P8-RUNTIME chapter=VIEW kind=json path=camera_bookmarks[].home_t authority=VIEW -->
<!-- P8-RUNTIME chapter=VIEW kind=json path=camera_bookmarks[].home_t[] authority=VIEW -->
<!-- P8-RUNTIME chapter=VIEW kind=json path=camera_bookmarks[].id authority=VIEW -->
<!-- P8-RUNTIME chapter=VIEW kind=json path=camera_bookmarks[].max_wasd_speed authority=VIEW -->
<!-- P8-RUNTIME chapter=VIEW kind=json path=camera_bookmarks[].max_zoom_speed authority=VIEW -->
<!-- P8-RUNTIME chapter=VIEW kind=json path=camera_bookmarks[].name authority=VIEW -->
<!-- P8-RUNTIME chapter=VIEW kind=json path=camera_bookmarks[].ortho_scale authority=VIEW -->
<!-- P8-RUNTIME chapter=VIEW kind=json path=camera_bookmarks[].pivot authority=VIEW -->
<!-- P8-RUNTIME chapter=VIEW kind=json path=camera_bookmarks[].pivot[] authority=VIEW -->
<!-- P8-RUNTIME chapter=VIEW kind=json path=camera_bookmarks[].roll_speed authority=VIEW -->
<!-- P8-RUNTIME chapter=VIEW kind=json path=camera_bookmarks[].rotate_speed authority=VIEW -->
<!-- P8-RUNTIME chapter=VIEW kind=json path=camera_bookmarks[].t authority=VIEW -->
<!-- P8-RUNTIME chapter=VIEW kind=json path=camera_bookmarks[].t[] authority=VIEW -->
<!-- P8-RUNTIME chapter=VIEW kind=json path=camera_bookmarks[].translate_speed authority=VIEW -->
<!-- P8-RUNTIME chapter=VIEW kind=json path=camera_bookmarks[].wasd_speed authority=VIEW -->
<!-- P8-RUNTIME chapter=VIEW kind=json path=camera_bookmarks[].zoom_speed authority=VIEW -->
<!-- P8-RUNTIME chapter=VIEW kind=json path=navigation authority=VIEW -->
<!-- P8-RUNTIME chapter=VIEW kind=json path=navigation.mode authority=VIEW -->
<!-- P8-RUNTIME chapter=VIEW kind=json path=navigation.view_snap authority=VIEW -->
<!-- P8-RUNTIME chapter=VIEW kind=json path=panel_cameras authority=VIEW -->
<!-- P8-RUNTIME chapter=VIEW kind=json path=panel_cameras[] authority=VIEW -->
<!-- P8-RUNTIME chapter=VIEW kind=json path=panel_cameras[].R authority=VIEW -->
<!-- P8-RUNTIME chapter=VIEW kind=json path=panel_cameras[].R[] authority=VIEW -->
<!-- P8-RUNTIME chapter=VIEW kind=json path=panel_cameras[].centre_speed authority=VIEW -->
<!-- P8-RUNTIME chapter=VIEW kind=json path=panel_cameras[].home_R authority=VIEW -->
<!-- P8-RUNTIME chapter=VIEW kind=json path=panel_cameras[].home_R[] authority=VIEW -->
<!-- P8-RUNTIME chapter=VIEW kind=json path=panel_cameras[].home_pivot authority=VIEW -->
<!-- P8-RUNTIME chapter=VIEW kind=json path=panel_cameras[].home_pivot[] authority=VIEW -->
<!-- P8-RUNTIME chapter=VIEW kind=json path=panel_cameras[].home_saved authority=VIEW -->
<!-- P8-RUNTIME chapter=VIEW kind=json path=panel_cameras[].home_t authority=VIEW -->
<!-- P8-RUNTIME chapter=VIEW kind=json path=panel_cameras[].home_t[] authority=VIEW -->
<!-- P8-RUNTIME chapter=VIEW kind=json path=panel_cameras[].max_wasd_speed authority=VIEW -->
<!-- P8-RUNTIME chapter=VIEW kind=json path=panel_cameras[].max_zoom_speed authority=VIEW -->
<!-- P8-RUNTIME chapter=VIEW kind=json path=panel_cameras[].ortho_scale authority=VIEW -->
<!-- P8-RUNTIME chapter=VIEW kind=json path=panel_cameras[].panel authority=VIEW -->
<!-- P8-RUNTIME chapter=VIEW kind=json path=panel_cameras[].pivot authority=VIEW -->
<!-- P8-RUNTIME chapter=VIEW kind=json path=panel_cameras[].pivot[] authority=VIEW -->
<!-- P8-RUNTIME chapter=VIEW kind=json path=panel_cameras[].roll_speed authority=VIEW -->
<!-- P8-RUNTIME chapter=VIEW kind=json path=panel_cameras[].rotate_speed authority=VIEW -->
<!-- P8-RUNTIME chapter=VIEW kind=json path=panel_cameras[].t authority=VIEW -->
<!-- P8-RUNTIME chapter=VIEW kind=json path=panel_cameras[].t[] authority=VIEW -->
<!-- P8-RUNTIME chapter=VIEW kind=json path=panel_cameras[].translate_speed authority=VIEW -->
<!-- P8-RUNTIME chapter=VIEW kind=json path=panel_cameras[].wasd_speed authority=VIEW -->
<!-- P8-RUNTIME chapter=VIEW kind=json path=panel_cameras[].zoom_speed authority=VIEW -->
<!-- P8-RUNTIME chapter=VIEW kind=json path=render_settings authority=VIEW -->
<!-- P8-RUNTIME chapter=VIEW kind=json path=render_settings.antialiasing authority=VIEW -->
<!-- P8-RUNTIME chapter=VIEW kind=json path=render_settings.apply_appearance_correction authority=VIEW -->
<!-- P8-RUNTIME chapter=VIEW kind=json path=render_settings.axes_size authority=VIEW -->
<!-- P8-RUNTIME chapter=VIEW kind=json path=render_settings.axes_visibility authority=VIEW -->
<!-- P8-RUNTIME chapter=VIEW kind=json path=render_settings.axes_visibility[] authority=VIEW -->
<!-- P8-RUNTIME chapter=VIEW kind=json path=render_settings.background_color authority=VIEW -->
<!-- P8-RUNTIME chapter=VIEW kind=json path=render_settings.background_color[] authority=VIEW -->
<!-- P8-RUNTIME chapter=VIEW kind=json path=render_settings.camera_frustum_scale authority=VIEW -->
<!-- P8-RUNTIME chapter=VIEW kind=json path=render_settings.camera_metrics_mode authority=VIEW -->
<!-- P8-RUNTIME chapter=VIEW kind=json path=render_settings.crop_filter_for_selection authority=VIEW -->
<!-- P8-RUNTIME chapter=VIEW kind=json path=render_settings.depth_clip_enabled authority=VIEW -->
<!-- P8-RUNTIME chapter=VIEW kind=json path=render_settings.depth_clip_far authority=VIEW -->
<!-- P8-RUNTIME chapter=VIEW kind=json path=render_settings.depth_filter_enabled authority=VIEW -->
<!-- P8-RUNTIME chapter=VIEW kind=json path=render_settings.depth_filter_max authority=VIEW -->
<!-- P8-RUNTIME chapter=VIEW kind=json path=render_settings.depth_filter_max[] authority=VIEW -->
<!-- P8-RUNTIME chapter=VIEW kind=json path=render_settings.depth_filter_min authority=VIEW -->
<!-- P8-RUNTIME chapter=VIEW kind=json path=render_settings.depth_filter_min[] authority=VIEW -->
<!-- P8-RUNTIME chapter=VIEW kind=json path=render_settings.depth_filter_transform authority=VIEW -->
<!-- P8-RUNTIME chapter=VIEW kind=json path=render_settings.depth_filter_transform.rotation authority=VIEW -->
<!-- P8-RUNTIME chapter=VIEW kind=json path=render_settings.depth_filter_transform.rotation[] authority=VIEW -->
<!-- P8-RUNTIME chapter=VIEW kind=json path=render_settings.depth_filter_transform.translation authority=VIEW -->
<!-- P8-RUNTIME chapter=VIEW kind=json path=render_settings.depth_filter_transform.translation[] authority=VIEW -->
<!-- P8-RUNTIME chapter=VIEW kind=json path=render_settings.depth_view authority=VIEW -->
<!-- P8-RUNTIME chapter=VIEW kind=json path=render_settings.depth_view_max authority=VIEW -->
<!-- P8-RUNTIME chapter=VIEW kind=json path=render_settings.depth_view_min authority=VIEW -->
<!-- P8-RUNTIME chapter=VIEW kind=json path=render_settings.depth_visualization_mode authority=VIEW -->
<!-- P8-RUNTIME chapter=VIEW kind=json path=render_settings.desaturate_cropping authority=VIEW -->
<!-- P8-RUNTIME chapter=VIEW kind=json path=render_settings.desaturate_unselected authority=VIEW -->
<!-- P8-RUNTIME chapter=VIEW kind=json path=render_settings.environment_builtin authority=VIEW -->
<!-- P8-RUNTIME chapter=VIEW kind=json path=render_settings.environment_exposure authority=VIEW -->
<!-- P8-RUNTIME chapter=VIEW kind=json path=render_settings.environment_mode authority=VIEW -->
<!-- P8-RUNTIME chapter=VIEW kind=json path=render_settings.environment_reference_uuid authority=VIEW -->
<!-- P8-RUNTIME chapter=VIEW kind=json path=render_settings.environment_rotation_degrees authority=VIEW -->
<!-- P8-RUNTIME chapter=VIEW kind=json path=render_settings.equirectangular authority=VIEW -->
<!-- P8-RUNTIME chapter=VIEW kind=json path=render_settings.eval_camera_color authority=VIEW -->
<!-- P8-RUNTIME chapter=VIEW kind=json path=render_settings.eval_camera_color[] authority=VIEW -->
<!-- P8-RUNTIME chapter=VIEW kind=json path=render_settings.focal_length_mm authority=VIEW -->
<!-- P8-RUNTIME chapter=VIEW kind=json path=render_settings.grid_opacity authority=VIEW -->
<!-- P8-RUNTIME chapter=VIEW kind=json path=render_settings.grid_plane authority=VIEW -->
<!-- P8-RUNTIME chapter=VIEW kind=json path=render_settings.gt_comparison_mode authority=VIEW -->
<!-- P8-RUNTIME chapter=VIEW kind=json path=render_settings.hide_outside_depth_box authority=VIEW -->
<!-- P8-RUNTIME chapter=VIEW kind=json path=render_settings.lod_auto_enable_rad authority=VIEW -->
<!-- P8-RUNTIME chapter=VIEW kind=json path=render_settings.lod_behind_camera_penalty authority=VIEW -->
<!-- P8-RUNTIME chapter=VIEW kind=json path=render_settings.lod_cone_foveation authority=VIEW -->
<!-- P8-RUNTIME chapter=VIEW kind=json path=render_settings.lod_cone_inner_degrees authority=VIEW -->
<!-- P8-RUNTIME chapter=VIEW kind=json path=render_settings.lod_cone_outer_degrees authority=VIEW -->
<!-- P8-RUNTIME chapter=VIEW kind=json path=render_settings.lod_debug_colors authority=VIEW -->
<!-- P8-RUNTIME chapter=VIEW kind=json path=render_settings.lod_enabled authority=VIEW -->
<!-- P8-RUNTIME chapter=VIEW kind=json path=render_settings.lod_fade_frames authority=VIEW -->
<!-- P8-RUNTIME chapter=VIEW kind=json path=render_settings.lod_max_splats authority=VIEW -->
<!-- P8-RUNTIME chapter=VIEW kind=json path=render_settings.lod_page_pool_splats authority=VIEW -->
<!-- P8-RUNTIME chapter=VIEW kind=json path=render_settings.lod_pool_vram_fraction authority=VIEW -->
<!-- P8-RUNTIME chapter=VIEW kind=json path=render_settings.lod_render_scale authority=VIEW -->
<!-- P8-RUNTIME chapter=VIEW kind=json path=render_settings.mesh_ambient authority=VIEW -->
<!-- P8-RUNTIME chapter=VIEW kind=json path=render_settings.mesh_backface_culling authority=VIEW -->
<!-- P8-RUNTIME chapter=VIEW kind=json path=render_settings.mesh_light_dir authority=VIEW -->
<!-- P8-RUNTIME chapter=VIEW kind=json path=render_settings.mesh_light_dir[] authority=VIEW -->
<!-- P8-RUNTIME chapter=VIEW kind=json path=render_settings.mesh_light_intensity authority=VIEW -->
<!-- P8-RUNTIME chapter=VIEW kind=json path=render_settings.mesh_shadow_enabled authority=VIEW -->
<!-- P8-RUNTIME chapter=VIEW kind=json path=render_settings.mesh_shadow_resolution authority=VIEW -->
<!-- P8-RUNTIME chapter=VIEW kind=json path=render_settings.mesh_wireframe authority=VIEW -->
<!-- P8-RUNTIME chapter=VIEW kind=json path=render_settings.mesh_wireframe_color authority=VIEW -->
<!-- P8-RUNTIME chapter=VIEW kind=json path=render_settings.mesh_wireframe_color[] authority=VIEW -->
<!-- P8-RUNTIME chapter=VIEW kind=json path=render_settings.mesh_wireframe_width authority=VIEW -->
<!-- P8-RUNTIME chapter=VIEW kind=json path=render_settings.mip_filter authority=VIEW -->
<!-- P8-RUNTIME chapter=VIEW kind=json path=render_settings.ortho_scale authority=VIEW -->
<!-- P8-RUNTIME chapter=VIEW kind=json path=render_settings.orthographic authority=VIEW -->
<!-- P8-RUNTIME chapter=VIEW kind=json path=render_settings.point_cloud_mode authority=VIEW -->
<!-- P8-RUNTIME chapter=VIEW kind=json path=render_settings.ppisp_mode authority=VIEW -->
<!-- P8-RUNTIME chapter=VIEW kind=json path=render_settings.ppisp_overrides authority=VIEW -->
<!-- P8-RUNTIME chapter=VIEW kind=json path=render_settings.ppisp_overrides.color_blue_x authority=VIEW -->
<!-- P8-RUNTIME chapter=VIEW kind=json path=render_settings.ppisp_overrides.color_blue_y authority=VIEW -->
<!-- P8-RUNTIME chapter=VIEW kind=json path=render_settings.ppisp_overrides.color_green_x authority=VIEW -->
<!-- P8-RUNTIME chapter=VIEW kind=json path=render_settings.ppisp_overrides.color_green_y authority=VIEW -->
<!-- P8-RUNTIME chapter=VIEW kind=json path=render_settings.ppisp_overrides.color_red_x authority=VIEW -->
<!-- P8-RUNTIME chapter=VIEW kind=json path=render_settings.ppisp_overrides.color_red_y authority=VIEW -->
<!-- P8-RUNTIME chapter=VIEW kind=json path=render_settings.ppisp_overrides.crf_shoulder authority=VIEW -->
<!-- P8-RUNTIME chapter=VIEW kind=json path=render_settings.ppisp_overrides.crf_toe authority=VIEW -->
<!-- P8-RUNTIME chapter=VIEW kind=json path=render_settings.ppisp_overrides.exposure_offset authority=VIEW -->
<!-- P8-RUNTIME chapter=VIEW kind=json path=render_settings.ppisp_overrides.gamma_blue authority=VIEW -->
<!-- P8-RUNTIME chapter=VIEW kind=json path=render_settings.ppisp_overrides.gamma_green authority=VIEW -->
<!-- P8-RUNTIME chapter=VIEW kind=json path=render_settings.ppisp_overrides.gamma_multiplier authority=VIEW -->
<!-- P8-RUNTIME chapter=VIEW kind=json path=render_settings.ppisp_overrides.gamma_red authority=VIEW -->
<!-- P8-RUNTIME chapter=VIEW kind=json path=render_settings.ppisp_overrides.vignette_enabled authority=VIEW -->
<!-- P8-RUNTIME chapter=VIEW kind=json path=render_settings.ppisp_overrides.vignette_strength authority=VIEW -->
<!-- P8-RUNTIME chapter=VIEW kind=json path=render_settings.ppisp_overrides.wb_temperature authority=VIEW -->
<!-- P8-RUNTIME chapter=VIEW kind=json path=render_settings.ppisp_overrides.wb_tint authority=VIEW -->
<!-- P8-RUNTIME chapter=VIEW kind=json path=render_settings.raster_backend authority=VIEW -->
<!-- P8-RUNTIME chapter=VIEW kind=json path=render_settings.render_scale authority=VIEW -->
<!-- P8-RUNTIME chapter=VIEW kind=json path=render_settings.ring_width authority=VIEW -->
<!-- P8-RUNTIME chapter=VIEW kind=json path=render_settings.scaling_modifier authority=VIEW -->
<!-- P8-RUNTIME chapter=VIEW kind=json path=render_settings.selection_color_center_marker authority=VIEW -->
<!-- P8-RUNTIME chapter=VIEW kind=json path=render_settings.selection_color_center_marker[] authority=VIEW -->
<!-- P8-RUNTIME chapter=VIEW kind=json path=render_settings.selection_color_committed authority=VIEW -->
<!-- P8-RUNTIME chapter=VIEW kind=json path=render_settings.selection_color_committed[] authority=VIEW -->
<!-- P8-RUNTIME chapter=VIEW kind=json path=render_settings.selection_color_preview authority=VIEW -->
<!-- P8-RUNTIME chapter=VIEW kind=json path=render_settings.selection_color_preview[] authority=VIEW -->
<!-- P8-RUNTIME chapter=VIEW kind=json path=render_settings.sh_degree authority=VIEW -->
<!-- P8-RUNTIME chapter=VIEW kind=json path=render_settings.show_camera_frustums authority=VIEW -->
<!-- P8-RUNTIME chapter=VIEW kind=json path=render_settings.show_center_markers authority=VIEW -->
<!-- P8-RUNTIME chapter=VIEW kind=json path=render_settings.show_coord_axes authority=VIEW -->
<!-- P8-RUNTIME chapter=VIEW kind=json path=render_settings.show_crop_box authority=VIEW -->
<!-- P8-RUNTIME chapter=VIEW kind=json path=render_settings.show_ellipsoid authority=VIEW -->
<!-- P8-RUNTIME chapter=VIEW kind=json path=render_settings.show_grid authority=VIEW -->
<!-- P8-RUNTIME chapter=VIEW kind=json path=render_settings.show_pivot authority=VIEW -->
<!-- P8-RUNTIME chapter=VIEW kind=json path=render_settings.show_rings authority=VIEW -->
<!-- P8-RUNTIME chapter=VIEW kind=json path=render_settings.split_position authority=VIEW -->
<!-- P8-RUNTIME chapter=VIEW kind=json path=render_settings.split_view_mode authority=VIEW -->
<!-- P8-RUNTIME chapter=VIEW kind=json path=render_settings.split_view_offset authority=VIEW -->
<!-- P8-RUNTIME chapter=VIEW kind=json path=render_settings.train_camera_color authority=VIEW -->
<!-- P8-RUNTIME chapter=VIEW kind=json path=render_settings.train_camera_color[] authority=VIEW -->
<!-- P8-RUNTIME chapter=VIEW kind=json path=render_settings.use_crop_box authority=VIEW -->
<!-- P8-RUNTIME chapter=VIEW kind=json path=render_settings.use_ellipsoid authority=VIEW -->
<!-- P8-RUNTIME chapter=VIEW kind=json path=render_settings.voxel_size authority=VIEW -->
<!-- P8-RUNTIME chapter=VIEW kind=json path=sequencer_view authority=VIEW -->
<!-- P8-RUNTIME chapter=VIEW kind=json path=sequencer_view.show_camera_path authority=VIEW -->
<!-- P8-RUNTIME chapter=VIEW kind=json path=split authority=VIEW -->
<!-- P8-RUNTIME chapter=VIEW kind=json path=split.focused_panel authority=VIEW -->
<!-- P8-RUNTIME chapter=VIEW kind=json path=split.gt_camera_id authority=VIEW -->
<!-- P8-RUNTIME chapter=VIEW kind=json path=split.panel_grid_planes authority=VIEW -->
<!-- P8-RUNTIME chapter=VIEW kind=json path=split.panel_grid_planes[] authority=VIEW -->
<!-- P8-RUNTIME chapter=VIEW kind=json path=tools authority=VIEW -->
<!-- P8-RUNTIME chapter=VIEW kind=json path=tools.active_submode_id authority=VIEW -->
<!-- P8-RUNTIME chapter=VIEW kind=json path=tools.active_tool_id authority=VIEW -->
<!-- P8-RUNTIME chapter=VIEW kind=json path=tools.crop_operation authority=VIEW -->
<!-- P8-RUNTIME chapter=VIEW kind=json path=tools.crop_shape authority=VIEW -->
<!-- P8-RUNTIME chapter=VIEW kind=json path=tools.gizmo_operation authority=VIEW -->
<!-- P8-RUNTIME chapter=VIEW kind=json path=tools.multi_transform_mode authority=VIEW -->
<!-- P8-RUNTIME chapter=VIEW kind=json path=tools.pivot_mode authority=VIEW -->
<!-- P8-RUNTIME chapter=VIEW kind=json path=tools.selection authority=VIEW -->
<!-- P8-RUNTIME chapter=VIEW kind=json path=tools.selection.brush_radius authority=VIEW -->
<!-- P8-RUNTIME chapter=VIEW kind=json path=tools.selection.crop_filter authority=VIEW -->
<!-- P8-RUNTIME chapter=VIEW kind=json path=tools.selection.depth_filter authority=VIEW -->
<!-- P8-RUNTIME chapter=VIEW kind=json path=tools.selection.restrict_to_selected_nodes authority=VIEW -->
<!-- P8-RUNTIME chapter=VIEW kind=json path=tools.selection_submode authority=VIEW -->
<!-- P8-RUNTIME chapter=VIEW kind=json path=tools.transform_space authority=VIEW -->
<!-- P8-RUNTIME chapter=VIEW kind=json path=version authority=VIEW -->
<!-- P8-RUNTIME-INVENTORY-END -->
