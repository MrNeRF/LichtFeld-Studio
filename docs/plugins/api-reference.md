# Python Application API

[Python docs](README.md) · [Development guide](getting-started.md) · [Application API](api-reference.md) · [Plugin and UI API](plugin-reference.md)

This reference describes the bindings in this checkout. The committed
[SDK stubs](../../src/python/stubs/lichtfeld/) list overloads and less common
members; [native bindings](../../src/python/lfs/) and
[Python helpers](../../src/python/lfs_plugins/) define behavior. A name present
in a stub does not imply that it works on every layout or outside the app.

## On this page

- [Python API Surface Map](#python-api-surface-map)
- [Scene API](#scene-api)
- [Training Control](#training-control)
- [Training Status](#training-status)
- [Training Hooks](#training-hooks)
- [Rendering](#rendering)
- [Viewport Control](#viewport-control)
- [Export](#export)
- [File I/O (lf.io)](#file-io-lfio)
- [Project Files (lf.io)](#project-files-lfio)
- [Pipeline Operations (lf.pipeline)](#pipeline-operations-lfpipeline)
- [Logging](#logging)
- [Undo](#undo)
- [Native Transform Gizmos](#native-transform-gizmos)
- [Tensor API](#tensor-api)
- [Application](#application)
- [Native Operators (lf.ops / lf.ui.ops)](#native-operators-lfops--lfuiops)
- [API Errors](#api-errors)
- [MCP Tools (lf.mcp)](#mcp-tools-lfmcp)
- [Packages (lf.packages)](#packages-lfpackages)
- [Scripts (lf.scripts)](#scripts-lfscripts)
- [Keymaps (lf.keymap)](#keymaps-lfkeymap)
- [Animation (lf.animation)](#animation-lfanimation)
- [Mesh (lf.mesh)](#mesh-lfmesh)

## Python API Surface Map

The public Python surface is split between the native `lichtfeld` module and
the pure-Python plugin helpers in `lfs_plugins`.

| Module | Main responsibility |
|---|---|
| `lichtfeld` | Training control, scene shortcuts, tensors, rendering, viewport, transform gizmos, registration, app helpers |
| `lichtfeld.diagnostics` | Best-effort system/CUDA/GPU diagnostics |
| `lichtfeld.app` | Application-level file open helper |
| `lichtfeld.animation` | Tracks, clips, and timeline evaluation |
| `lichtfeld.io` | Load/save splats, point clouds, datasets, images, and supported format queries |
| `lichtfeld.keymap` | Input binding profiles, action lookup, capture, import/export |
| `lichtfeld.log` | Native log sink |
| `lichtfeld.mcp` | Register Python MCP tools and call/read shared MCP capabilities/resources |
| `lichtfeld.mesh` | OpenMesh-style mesh data, handles, iterators, readers/writers, decimaters |
| `lichtfeld.ops` | Native operator invocation, descriptors, built-in operator/tool enums |
| `lichtfeld.packages` | `uv`-backed package install/list/uninstall and stub path helpers |
| `lichtfeld.pipeline` | Chainable selection/edit/transform operation stages |
| `lichtfeld.plugins` | Plugin discovery, lifecycle, registry install/update, capabilities, settings, scaffolding |
| `lichtfeld.scene` | Scene graph, nodes, splat data, point clouds, cameras, selection groups |
| `lichtfeld.scripts` | Script panel state and batch execution |
| `lichtfeld.selection` | Gaussian stroke/preview/brush/lasso/depth/crop selection primitives |
| `lichtfeld.ui` | Panels, menus, immediate UI, dialogs, theme, tools, app UI state, RML bridge |
| `lichtfeld.undo` | Undo/redo stack, transactions, memory accounting, subscriptions |
| `lfs_plugins.*` | Plugin base types, properties, runtime state bindings, tool definitions, capabilities, templates, managers |

The sections below focus on the APIs plugin authors most often call directly.
Large low-level surfaces such as `lichtfeld.mesh.TriMesh` and every tensor
operator are intentionally summarized; inspect the `.pyi` files for exhaustive
method lists.

## Scene API

```python
import lichtfeld as lf
```

### Scene Management

| Function                  | Returns          | Description                       |
|---------------------------|------------------|-----------------------------------|
| `get_scene()`             | `Scene or None`  | Get scene object                  |
| `get_render_scene()`      | `Scene or None`  | Get render scene (PyScene)        |
| `has_scene()`             | `bool`           | Whether scene is loaded           |
| `clear_scene()`           | `None`           | Clear all scene content           |
| `load_file(path, is_dataset=False, output_path="", init_path="", centralize_dataset="off", max_width=None, apply_auto_crop=False, min_track_length=None, stop_training=False, discard_changes=False, replace=False)` | `None` | Load PLY or dataset               |
| `load_config_file(path)`  | `None`           | Load JSON config                  |
| `get_scene_generation()`  | `int`            | Scene generation counter          |
| `list_scene()`            | `None`           | Print scene tree                  |

### Node Operations (on Scene object)

| Method                                            | Returns          | Description                       |
|---------------------------------------------------|------------------|-----------------------------------|
| `add_group(name, parent=-1)`                      | `int`            | Add group node                    |
| `add_splat(name, means, sh0, shN, scaling, rotation, opacity, ...)` | `int` | Add splat node       |
| `add_point_cloud(name, points, colors, parent=-1)`| `int`            | Add point cloud                   |
| `add_camera(name, parent, R, T, focal_x, focal_y, width, height, image_path="", uid=-1, mask=None)` | `int`         | Add camera node                   |
| `remove_node(name, keep_children=False)`          | `None`           | Remove node                       |
| `rename_node(old_name, new_name)`                           | `bool`           | Rename node                       |
| `reparent(node_id, new_parent_id)`                | `None`           | Change parent                     |
| `duplicate_node(name)`                            | `str`            | Duplicate, returns new node name  |
| `merge_group(group_name)`                         | `str`            | Merge group children, returns merged node name |
| `get_node(name)`                                  | `SceneNode or None`      | Get node by name                  |
| `get_node_by_id(id)`                              | `SceneNode or None`      | Get node by ID                    |
| `get_nodes(type=None)` | `list[SceneNode]` | All nodes, optionally filtered by `NodeType`                         |
| `get_visible_nodes()`                             | `list[SceneNode]`| Visible nodes only                |
| `root_nodes()`                                    | `list[int]`      | Root node IDs                     |
| `is_node_effectively_visible(id)`                 | `bool`           | Considers parent visibility       |
| `total_gaussian_count`                            | `int`            | Property. Total gaussians         |
| `invalidate_cache()`                              | `None`           | Clear internal cache (no redraw)  |
| `notify_changed()`                                | `None`           | Invalidate cache + trigger viewport redraw |

### SceneNode Properties

| Property/Method     | Returns              | Description                       |
|---------------------|----------------------|-----------------------------------|
| `id`                | `int`                | Node ID                           |
| `name`              | `str`                | Node name                         |
| `type`              | `NodeType`           | Node type enum (SPLAT, POINTCLOUD, GROUP, etc.) |
| `parent_id`         | `int`                | Parent node ID (-1 for root)      |
| `children`          | `list[int]`          | Child node IDs                    |
| `visible`           | `bool`               | Visibility flag                   |
| `locked`            | `bool`               | Lock flag                         |
| `gaussian_count`    | `int`                | Number of gaussians (splat nodes) |
| `centroid`          | `tuple[float, float, float]` | Node centroid             |
| `world_transform`   | `tuple`              | World-space 4x4 transform, row-major |
| `splat_data()`      | `SplatData or None`  | Splat data for this node (None if not a splat) |
| `point_cloud()`     | `PointCloud or None` | Point cloud data (None if not a point cloud) |
| `cropbox()`         | `CropBox or None`    | Crop box data                     |
| `ellipsoid()`       | `Ellipsoid or None`  | Ellipsoid data                    |

### Selection

| Function                              | Returns          | Description                       |
|---------------------------------------|------------------|-----------------------------------|
| `select_node(name)`                   | `None`           | Select node by name               |
| `deselect_all()`                      | `None`           | Clear selection                   |
| `has_selection()`                     | `bool`           | Any selection active              |
| `get_selected_node_name()`            | `str`            | First selected node name          |
| `get_selected_node_names()`           | `list[str]`      | All selected node names           |
| `can_transform_selection()`           | `bool`           | Selection is transformable        |
| `get_selected_node_transform()`       | `list[float]`    | 16 column-major floats            |
| `set_selected_node_transform(matrix)` | `None`           | Set transform from 16 column-major floats |
| `get_selection_center()`              | `list[float]`    | Local space center                |
| `get_selection_visualizer_world_center()` | `list[float]` | Visualizer world-space center     |
| `get_selection_world_center()`        | `list[float]`    | Deprecated legacy data-world center; use `get_selection_visualizer_world_center()` |
| `capture_selection_transforms()`      | `dict`           | Snapshot for undo                 |

### Scene Shortcuts

Module-level shortcuts for common scene operations (equivalent to `Scene` object methods):

| Function | Returns | Description |
|----------|---------|-------------|
| `set_node_visibility(name, visible)` | `None` | Toggle node visibility |
| `remove_node(name, keep_children=False)` | `None` | Remove node |
| `reparent_node(name, new_parent)` | `None` | Reparent node |
| `rename_node(old_name, new_name)` | `None` | Rename node |
| `add_group(name, parent="")` | `None` | Add group node |
| `get_num_gaussians()` | `int` | Total gaussian count |

### Gaussian-Level Selection (on Scene object)

| Method                          | Returns     | Description                  |
|---------------------------------|-------------|------------------------------|
| `set_selection_mask(mask)`      | `None`      | Apply bool tensor mask       |
| `preview_selection_mask(mask)`  | `None`      | Preview transient selection without an undo step |
| `commit_selection_preview()`    | `None`      | Commit the transient preview as one undo step |
| `cancel_selection_preview()`    | `None`      | Restore the selection before preview |
| `clear_selection()`             | `None`      | Clear gaussian selection     |
| `has_selection()`               | `bool`      | Any gaussians selected       |
| `selection_mask` | `Tensor or None` | Current `[N]` uint8 mask; `None` if unavailable |
| `set_selection(indices)`        | `None`      | Select by index list         |
| `add_selection_group(name, color)` | `int`    | Create a named group         |
| `selection_groups()`            | `list[SelectionGroup]` | Current groups      |
| `active_selection_group`        | `int`       | Property. Active group id    |
| `set_selection_group_locked(id, locked)` | `None` | Lock/unlock group edits |

### Selection Primitives (`lf.selection`)

`lichtfeld.selection` is the lower-level selection-tool API. It operates on
selection strokes, previews, and viewport-space hit data.

| Function | Returns | Description |
|---|---|---|
| `begin_stroke()` / `commit_stroke(mode)` / `cancel_stroke()` | `None` / `bool` / `None` | Manage one undoable selection stroke |
| `get_stroke_selection()` | `Tensor \| None` | Current stroke mask `[N]` uint8 |
| `set_preview(add_mode=True)` / `clear_preview()` | `None` | Show/clear add/remove preview overlay |
| `draw_brush_circle(x, y, radius, add_mode=True)` | `None` | Brush cursor overlay |
| `draw_rect_preview(x0, y0, x1, y1, add_mode=True)` | `None` | Rectangle preview overlay |
| `draw_polygon_preview(points, closed=False, add_mode=True)` | `None` | Polygon preview overlay |
| `draw_lasso_preview(points, add_mode=True)` | `None` | Lasso preview overlay |
| `has_screen_positions()` / `get_screen_positions()` | `bool` / `Tensor \| None` | Viewport-projected positions `[N, 2]` |
| `set_depth_filter_range(enabled, depth_near=0, depth_far=100, frustum_half_width=50)` | `None` | Camera-space selection depth filter |
| `get_depth_filter_range()` | `tuple[bool, float, float, float]` | `(enabled, near, far, half_width)` |
| `set_crop_filter(enabled)` / `apply_crop_filter()` | `None` | Crop-box constrained selection |
| `screen_to_render(screen_x, screen_y)` | `tuple[float, float]` | Convert screen to render coordinates |
| `pick_at_screen(screen_x, screen_y)` | `PickResult \| None` | Depth/world hit at screen point |
| `ring_select(index, add=True)` | `None` | Select/deselect one gaussian |
| `grow(radius, iterations=1)` / `shrink(radius, iterations=1)` | `None` | Spatial grow/shrink selection |
| `by_opacity(min_opacity=0, max_opacity=1)` | `None` | Select by activated opacity range |
| `by_scale(max_scale)` | `None` | Select by activated scale threshold |
| `by_color(gaussian_index, threshold=0.2)` | `None` | Select by SH DC color similarity |

`PickResult.index` is the gaussian under the current cursor, not necessarily
the queried screen coordinate. Use `depth` and `world_position` for the queried
coordinate data.

### Transforms

| Function                                    | Returns        | Description                      |
|---------------------------------------------|----------------|----------------------------------|
| `get_node_transform(name)`                  | `list[float]`  | 16 column-major floats           |
| `set_node_transform(name, matrix)`          | `None`         | Set 4x4 transform from 16 column-major floats |
| `decompose_transform(matrix)`               | `dict`         | Decompose 16 column-major floats; see keys below |
| `compose_transform(translation, euler_deg, scale)` | `list[float]` | Build 16 column-major floats from components (Euler in degrees) |

`decompose_transform` returns a dict with these keys:

| Key | Type | Description |
|-----|------|-------------|
| `translation` | `[x, y, z]` | Position |
| `rotation_quat` | `[x, y, z, w]` | Quaternion |
| `rotation_euler` | `[rx, ry, rz]` | Euler angles (radians) |
| `rotation_euler_deg` | `[rx, ry, rz]` | Euler angles (degrees) |
| `scale` | `[sx, sy, sz]` | Scale |

### Splat Data (combined_model() / node.splat_data())

Accessible via `scene.combined_model()` (visible splats merged, or `None` if
empty) or `node.splat_data()` (per-node). For edits use the per-node payload;
a merged render model is not a write-through view of all source nodes.

| Property/Method       | Returns      | Description                     |
|-----------------------|--------------|---------------------------------|
| `means_raw`           | `Tensor`     | [N, 3] positions (view)        |
| `sh0_raw`             | `Tensor`     | [N, 1, 3] base SH (view)       |
| `shN_raw`             | `Tensor`     | [N, K, 3] higher SH (view)     |
| `scaling_raw`         | `Tensor`     | [N, 3] log-space (view)        |
| `rotation_raw`        | `Tensor`     | [N, 4] quaternions (view)      |
| `opacity_raw`         | `Tensor`     | [N, 1] logit-space (view)      |
| `get_means()`         | `Tensor`     | Positions                       |
| `get_opacity()`       | `Tensor`     | [N] sigmoid applied             |
| `get_scaling()`       | `Tensor`     | Exp applied                     |
| `get_rotation()`      | `Tensor`     | Normalized quaternions          |
| `get_shs()`           | `Tensor`     | SH0 + SHN concatenated         |
| `num_points`          | `int`        | Gaussian count                  |
| `active_sh_degree`    | `int`        | Current SH degree               |
| `max_sh_degree`       | `int`        | Maximum SH degree               |
| `scene_scale`         | `float`      | Scene scale factor              |
| `soft_delete(mask)`   | `Tensor`     | Mark for deletion, returns newly deleted mask |
| `undelete(mask)`      | `None`       | Restore deleted gaussians       |
| `apply_deleted()`     | `int`        | Permanently remove, returns count|
| `clear_deleted()`     | `None`       | Clear deletion mask             |
| `deleted`             | `Tensor`     | Property. [N] bool deletion mask|
| `has_deleted_mask()`  | `bool`       | Whether deletion mask exists    |
| `visible_count()`     | `int`        | Number of non-deleted gaussians |

> After calling `soft_delete()`, `undelete()`, or `clear_deleted()`, call `scene.notify_changed()` to update the viewport.

### Point Clouds, Cameras, and Dataset Nodes

| API | Returns | Description |
|---|---|---|
| `Scene.add_point_cloud(name, points, colors, parent=-1)` | `int` | Add `[N,3]` position/color tensors |
| `SceneNode.point_cloud()` | `PointCloud \| None` | Point-cloud payload for point-cloud nodes |
| `PointCloud.means` / `PointCloud.colors` | `Tensor` | Position and color tensors |
| `PointCloud.normals`, `sh0`, `shN`, `opacity`, `scaling`, `rotation` | `Tensor \| None` | Optional gaussian-like attributes |
| `PointCloud.normalize_colors()` | `None` | Normalize colors to `[0,1]` |
| `PointCloud.filter(mask)` / `filter_indices(indices)` | `int` | Keep matching points, return removed count |
| `PointCloud.set_data(points, colors)` / `set_means(points)` / `set_colors(colors)` | `None` | Update positions/colors without replacing both |
| `Scene.add_camera_group(name, parent, camera_count)` | `int` | Add camera group |
| `Scene.add_camera(name, parent, R, T, focal_x, focal_y, width, height, image_path='', uid=-1, mask=None)` | `int` | Add camera node |
| `SceneNode.load_mask(...)` / `load_depth(...)` | `Tensor \| None` | Load camera mask/depth from node metadata |
| `Scene.get_active_cameras()` | `list[SceneNode]` | Cameras enabled for training |
| `CameraDataset.cameras()` | `list[Camera]` | Dataset cameras as Python objects |
| `Camera.load_image(resize_factor=1, max_width=0, output_uint8=False)` | `Tensor` | `[C,H,W]` CUDA image |
| `Camera.rotation` / `Camera.translation` | `Tensor` | Visualizer camera pose, directly usable with `render_view()` |

Deprecated raw dataset-camera properties are still available on `Camera`:
`R`, `T`, `world_view_transform`, and `cam_position`. Prefer
`rotation`, `translation`, `K`, and `view_matrix` for new code.

## Training Control

| Function                    | Returns          | Description                   |
|-----------------------------|------------------|-------------------------------|
| `start_training()`         | `None`           | Start training                |
| `pause_training()`         | `None`           | Pause                         |
| `resume_training()`        | `None`           | Resume                        |
| `stop_training()`          | `None`           | Stop                          |
| `reset_training()`         | `None`           | Reset to iteration 0          |
| `project_save(wait=False, regenerate_preview=True)` | `bool`           | Save the active `.licht` project |
| `switch_to_edit_mode()`    | `None`           | Enter edit mode               |
| `has_trainer()`            | `bool`           | Trainer loaded                |
| `trainer_state()`          | `str`            | State string                  |
| `finish_reason()`          | `str or None`    | Why training ended            |
| `trainer_error()`          | `str or None`    | Error message                 |
| `context()`                | `Context`        | Training context snapshot     |
| `optimization_params()`    | `OptimizationParams` | Training parameters      |
| `dataset_params()`         | `DatasetParams`  | Dataset parameters            |
| `loss_buffer()`            | `list[float]`    | Loss history                  |
| `load_checkpoint_for_training(checkpoint_path, dataset_path, output_path)` | `None` | Load checkpoint for training |

## Training Status

| Function | Returns | Description |
|----------|---------|-------------|
| `trainer_elapsed_seconds()` | `float` | Elapsed training time |
| `trainer_eta_seconds()` | `float` | Estimated remaining time (-1 if unavailable) |
| `trainer_strategy_type()` | `str` | Strategy type (mcmc, default, etc.) |
| `trainer_is_gut_enabled()` | `bool` | GUT enabled |
| `trainer_max_gaussians()` | `int` | Max gaussians |
| `trainer_num_splats()` | `int` | Current splat count |
| `trainer_current_iteration()` | `int` | Current iteration |
| `trainer_total_iterations()` | `int` | Total iterations |
| `trainer_current_loss()` | `float` | Current loss |

## Training Hooks

| Decorator                    | Description                         |
|------------------------------|-------------------------------------|
| `@lf.on_training_start`     | Called when training starts          |
| `@lf.on_iteration_start`    | Called at start of each iteration    |
| `@lf.on_pre_optimizer_step` | Called before optimizer step         |
| `@lf.on_post_step`          | Called after each step               |
| `@lf.on_training_end`       | Called when training ends            |

Callbacks registered with `lf.on_*` receive one positional hook payload. If
you do not need it, declare the parameter as `_hook`. Keep a `lf.ControlSession()`
for plugin-owned registrations; its `on_training_start`, `on_iteration_start`,
`on_pre_optimizer_step`, `on_post_step`, and `on_training_end` methods register
callbacks, and `clear()` unregisters that session's callbacks on unload.
The module-level decorators register callbacks until app shutdown; setting a
Python variable to `None` does not unregister them.

## Rendering

| Function                                              | Returns          | Description                |
|-------------------------------------------------------|------------------|----------------------------|
| `get_current_view(panel="main")` | `ViewInfo or None`       | Current camera view        |
| `get_viewport_render()` | `ViewportRender or None` | Latest cached CPU-visible capture (no forced readback)     |
| `capture_viewport()` | `ViewportRender or None` | Explicit readback and clone for background use      |
| `export_viewport_image(path, format='', width=0, height=0, transparent=False, jpeg_quality=95, include_provenance=True)` | `dict` | Export active viewport to PNG/JPEG (`include_provenance=False` still embeds a minimal build stamp) |
| `look_at(eye, target, up=(0,1,0))`                    | `(Tensor, Tensor)` | Compute `(rotation, translation)` for rendering |
| `render_view(rotation, translation, width, height, fov=60, bg_color=None, with_depth=False, depth_mode='median')` | `Tensor \| tuple \| None` | Render active scene from camera |
| `render_view_u8(rotation, translation, width, height, fov=60, bg_color=None, orthographic=None, ortho_scale=None)` | `Tensor \| None` | Render active scene as CPU uint8 RGB |
| `render_at(eye, target, width, height, fov=60, up=(0,1,0), bg_color=None)` | `Tensor \| None` | Convenience look-at render |
| `render_asset_preview(path, width=512, height=224, focal_length_mm=35)` | `Tensor \| None` | Offscreen asset thumbnail without mutating live scene |
| `render_asset_preview_from_camera(path, eye, target, ...)` | `Tensor \| None` | Offscreen thumbnail from custom camera |
| `compute_screen_positions(rotation, translation, width, height, fov=60)` | `Tensor` | [N, 2] screen positions |
| `get_render_settings()` | `RenderSettings or None` | Current render settings    |
| `get_render_mode()` / `set_render_mode(mode)` | `RenderMode` / `None`     | Render mode                |

`get_current_view(panel)` accepts `"main"`, `"left"`, or `"right"`.
`render_view()` returns CPU RGB `[H,W,3]`; with `with_depth=True` it returns
`(image, depth)` with CPU float depth `[H,W]`. `bg_color` is accepted for
compatibility, but the Vulkan preview uses the current render settings.
A missing visualizer/capture is represented by `None`.

## Viewport Control

| Function                        | Returns | Description              |
|---------------------------------|---------|--------------------------|
| `reset_camera()`               | `None`  | Reset camera             |
| `toggle_fullscreen()`          | `None`  | Toggle fullscreen        |
| `is_fullscreen()`              | `bool`  | Fullscreen state         |
| `toggle_ui()`                  | `None`  | Toggle UI visibility     |
| `set_orthographic(ortho, extent_world=None)`      | `None`  | Set projection mode      |
| `is_orthographic()`            | `bool`  | Orthographic state       |

## Export

```text
lf.export_scene(
    format: int,             # 0=PLY, 1=SOG, 2=SPZ, 3=HTML, 4=USD,
                             # 5=USDZ NuRec, 6=RAD, 7=COLMAP, 8=SSOG, 13=GLB
    path: str,
    node_names: list[str],
    sh_degree: int,
    rad_flip_y: bool = False,
    rad_streamable: bool = True,
    spz_version: int = 4,    # SPZ only: 4 (zstd) or 3 (legacy gzip)
    include_provenance: bool = True,  # False writes a minimal build stamp; ignored for COLMAP and SPZ v3
    *,
    lod_levels: int = 4,         # SSOG export options
    lod_ratio: float = 0.5,
    chunk_count_k: int = 512,
    chunk_extent: float = 16.0,
    chunk_min_k: int = 8,
    kmeans_iterations: int = 10,
)
lf.save_config_file(path: str)
```

## File I/O (`lf.io`)

Use `lf.io` when you need data objects directly instead of loading into the
live application scene.

| Function | Returns | Description |
|---|---|---|
| `lf.io.load(path, format=None, resize_factor=None, max_width=None, images_folder=None, progress=None, min_track_length=None)` | `LoadResult` | Load splat file, point cloud, mesh-derived data, or dataset |
| `lf.io.load_point_cloud(path)` | `(Tensor, Tensor)` | Load PLY point cloud as positions/colors |
| `lf.io.save_ply(data, path, binary=True, progress=None, extra_attributes=None, include_provenance=True)` | `None` | Save `SplatData` as PLY (`False` still embeds a minimal build stamp) |
| `lf.io.save_point_cloud_ply(point_cloud, path, extra_attributes=None, include_provenance=True)` | `None` | Save `PointCloud` as PLY (`False` still embeds a minimal build stamp) |
| `lf.io.save_sog(data, path, kmeans_iterations=10, use_gpu=True, progress=None, include_provenance=True)` | `None` | Save SOG-compressed splats (`False` still embeds a minimal build stamp) |
| `lf.io.save_ssog(splat, path, lod_levels=4, lod_ratio=0.5, chunk_count_k=512, chunk_extent=16.0, chunk_min_k=8, kmeans_iterations=10, use_gpu=True, progress=None, include_provenance=True)` | `None` | Save a PlayCanvas SSOG bundle/directory with LODs and chunks |
| `lf.io.save_spz(data, path, version=4, include_provenance=True)` | `None` | Save SPZ splats (`version` 4=zstd, 3=legacy gzip; `include_provenance` ignored for v3) |
| `lf.io.save_usd(data, path, include_provenance=True)` | `None` | Save OpenUSD gaussian file (`False` still embeds a minimal build stamp) |
| `lf.io.save_nurec_usdz(data, path, include_provenance=True)` | `None` | Save NuRec-compatible USDZ (`False` still embeds a minimal build stamp) |
| `lf.io.export_html(data, path, kmeans_iterations=10, progress=None, include_provenance=True)` | `None` | Self-contained HTML viewer (`False` still embeds a minimal build stamp) |
| `lf.io.save_image(path, image, include_provenance=True)` | `None` | Save PNG/JPG/TIFF/EXR from `[H,W,C]` or `[C,H,W]` tensor (`False` still embeds a minimal build stamp on PNG/JPEG) |
| `lf.io.is_dataset_path(path)` | `bool` | Dataset directory detection |
| `lf.io.is_gaussian_splat_ply(path)` | `bool` | PLY schema check for gaussian splat attributes |
| `lf.io.get_supported_formats()` / `get_supported_extensions()` | `list[str]` | Loader/exporter support |

`include_provenance=False` writes a minimal build stamp (`app_version` + `build_commit`). The flag is ignored for COLMAP and SPZ v3 (legacy gzip), which have no stamp slot.

`LoadResult` exposes `splat_data`, `point_cloud`, `cameras`,
`scene_center`, `loader_used`, `load_time_ms`, `warnings`, and `is_dataset`.
PLY extra attributes must avoid reserved gaussian names and must match the
visible/raw point count expected by the exporter.

## Project Files (`lf.io`)

These APIs operate on `.licht` files without opening them as the live scene.
Use native inspection instead of parsing chapter bytes in a plugin.

| API | Returns | Purpose |
|---|---|---|
| `inspect_project(path, resolve_preview_fallback=True)` | `ProjectInspection` | Project metadata and preview |
| `classify_project(path)` | `ProjectOpenClassification` | File classification for opening |
| `inspect_project_card(path)` | `ProjectInspectorCard` | Library card metadata |
| `inspect_project_details(path, checkpoint_byte_budget=8388608)` | `ProjectInspectorDetails` | Contents and checkpoint details |
| `project_storage_stats(path)` | `ProjectStorageStats` | Storage accounting |
| `read_preview(path)` | `bytes` | Embedded preview PNG |
| `verify_project_file(path, progress=None, cancel=None)` | `ProjectVerificationResult` | Verify the project |
| `plan_reduce_size(path)` | `ProjectReducePlan` | Plan contents removal |
| `reduce_size(path, options={}, progress=None, cancel=None)` | `ProjectReduceResult` | Apply contents removal options |
| `export_project_as(path, format, destination, progress=None, cancel=None)` | `ProjectExportResult` | Export PLY, SOG, SSOG, or SPZ |

Inspection can raise `lf.Error` when a project is invalid or incompatible;
handle it before displaying success in a panel. Destructive operations such as
`reduce_size()` modify the file; inspecting a plan does not apply it.
Additional backup, repair, reference, preview, and license APIs and their result
fields are in the [IO stubs](../../src/python/stubs/lichtfeld/io.pyi).

## Pipeline Operations (`lf.pipeline`)

Pipelines compose built-in operations and execute them as one chain.

```python
pipe = (
    lf.pipeline.Pipeline("grow-and-move")
    | lf.pipeline.select.grow(radius=0.05)
    | lf.pipeline.transform.translate(delta=(0.0, 0.1, 0.0))
)
if pipe.poll():
    result = pipe.execute()
```

| API | Description |
|---|---|
| `lf.pipeline.Pipeline(name='')` | Create an empty chain |
| `Pipeline.add(stage)` / `Pipeline \| stage` | Append a stage |
| `Pipeline.poll()` / `execute()` | Check and execute all stages |
| `Stage.execute()` | Execute a single stage immediately |
| `lf.pipeline.select.all(**kwargs)` / `none(**kwargs)` / `invert(**kwargs)` / `grow(**kwargs)` / `shrink(**kwargs)` | Selection stages |
| `lf.pipeline.transform.translate(**kwargs)` / `rotate(**kwargs)` / `scale(**kwargs)` / `set(**kwargs)` | Transform stages |
| `lf.pipeline.edit.duplicate(**kwargs)` / `delete_(**kwargs)` | Duplicate/delete stages |

Pipeline stage properties are read by the native operation, so spelling matters:
`translate(delta=(x,y,z))`, `rotate(axis=(x,y,z), angle=degrees, pivot=...)`,
`scale(scale=(x,y,z), pivot=...)`, and `set(transform=matrix)`. An unrecognized
key such as `offset` does not supply the translation delta.

## Logging

| Function           | Description       |
|--------------------|-------------------|
| `lf.log.info(message)` | Info level        |
| `lf.log.warn(message)` | Warning level     |
| `lf.log.error(message)`| Error level       |
| `lf.log.debug(message)`| Debug level       |

## Undo

```text
lf.undo.push(name: str, undo: Callable, redo: Callable, validate: Callable | None = None)
lf.undo.transaction(name: str = "Grouped Changes") -> Transaction
lf.undo.stack() -> dict
```

- `lf.undo.transaction(...)` groups multiple undoable mutations into one history step.
- `lf.undo.stack()` returns structured undo/redo items with `id`, `label`, `source`, `scope`, and `estimated_bytes`.

## Native Transform Gizmos

| API | Returns | Description |
|-----|---------|-------------|
| `lf.TransformGizmo(operation="translate", matrix=[], id="")` | `TransformGizmo` | Reusable native TRS gizmo |
| `lf.TranslationGizmo(matrix=[], id="")` | `TransformGizmo` | Translate handle |
| `lf.RotationGizmo(matrix=[], id="")` | `TransformGizmo` | Rotate handle |
| `lf.ScaleGizmo(matrix=[], id="")` | `TransformGizmo` | Scale handle |
| `lf.get_transform_gizmo_ids()` | `list[str]` | Attached transform gizmo IDs |
| `lf.has_transform_gizmos()` | `bool` | Whether any native TRS gizmos are attached |
| `lf.clear_transform_gizmos()` | `None` | Detach all native TRS gizmos |

`TransformGizmo` properties:

| Property | Type | Description |
|----------|------|-------------|
| `id` | `str` | Stable ID |
| `operation` | `str` | `"translate"`, `"rotate"`, or `"scale"` |
| `space` | `str` | `"local"` or `"world"` |
| `matrix` | `list[float]` | 16 floats, column-major |
| `translation` | `list[float]` | Translation component |
| `visible`, `enabled`, `input_enabled` | `bool` | Runtime draw/input controls |
| `active`, `hovered`, `changed` | `bool` | Last-frame interaction state |
| `snap` | `bool` | Enable snapping |
| `translate_snap`, `rotate_snap_degrees`, `scale_snap_ratio` | `float` | Per-operation snap settings |

| Method | Description |
|--------|-------------|
| `attach()` | Draw without an automatic target |
| `attach_to_callbacks(getter, setter)` | Bind to arbitrary Python transform callbacks |
| `attach_to_node(node_name, visualizer_world=True)` | Bind to a scene node |
| `detach()` | Remove from viewport drawing |
| `set_on_begin(callback)`, `set_on_change(callback)`, `set_on_end(callback)` | Drag lifecycle callbacks |

---

## Tensor API

```python
import lichtfeld as lf
t = lf.Tensor
```

The tables below list the most-used tensor APIs. For the full bound surface, see `src/python/stubs/lichtfeld/__init__.pyi`.

**Creation:**

| Function                                    | Returns  | Description              |
|---------------------------------------------|----------|--------------------------|
| `t.zeros(shape, device='cuda', dtype='float32')` | `Tensor` | Zero-filled tensor   |
| `t.ones(shape, device, dtype)`              | `Tensor` | Ones tensor              |
| `t.full(shape, value, device, dtype)`       | `Tensor` | Constant-filled tensor   |
| `t.eye(n, device, dtype)`                   | `Tensor` | Identity matrix          |
| `t.arange(start, end, step, device, dtype)` | `Tensor` | Range tensor             |
| `t.linspace(start, end, steps, device="cuda", dtype="float32")` | `Tensor` | Linear space          |
| `t.rand(shape, device, dtype)`              | `Tensor` | Uniform random [0, 1)   |
| `t.randn(shape, device, dtype)`             | `Tensor` | Normal random            |
| `t.empty(shape, device, dtype)`             | `Tensor` | Uninitialized tensor     |
| `t.randint(low, high, shape, device)`       | `Tensor` | Random integers          |
| `t.from_numpy(arr, copy=True)`              | `Tensor` | From NumPy array         |
| `t.cat(tensors, dim=0)`                     | `Tensor` | Concatenate              |
| `t.stack(tensors, dim=0)`                   | `Tensor` | Stack                    |
| `t.where(condition, x, y)`                  | `Tensor` | Conditional select       |

`linspace()` applies the requested dtype. For example,
`lf.Tensor.linspace(0, 2, 3, device="cpu", dtype="int32").dtype` is `"int32"`.

**Properties:**

| Property         | Type    | Description              |
|------------------|---------|--------------------------|
| `.shape`         | `tuple` | Tensor dimensions        |
| `.ndim`          | `int`   | Number of dimensions     |
| `.numel`         | `int`   | Total elements           |
| `.device`        | `str`   | `'cpu'` or `'cuda'`     |
| `.dtype`         | `str`   | Data type string         |
| `.is_contiguous` | `bool`  | Memory contiguous        |
| `.is_cuda`       | `bool`  | On GPU                   |

**Methods:**

| Method                              | Returns  | Description              |
|-------------------------------------|----------|--------------------------|
| `.clone()`                          | `Tensor` | Deep copy                |
| `.cpu()` / `.cuda()`               | `Tensor` | Move device              |
| `.contiguous()`                     | `Tensor` | Make contiguous          |
| `.sync()`                           | `None`   | CUDA synchronize         |
| `.numpy(copy=True)`                 | `ndarray`| Convert to NumPy         |
| `.to(dtype)`                        | `Tensor` | Convert dtype            |
| `.size(dim)`                        | `int`    | Size at dimension        |
| `.item()`                           | `float` | Extract scalar           |
| `.sum(dim=None, keepdim=False)`     | `Tensor` | Reduce sum               |
| `.mean(dim=None, keepdim=False)`    | `Tensor` | Reduce mean              |
| `.max(dim=None, keepdim=False)`     | `Tensor` | Reduce max               |
| `.min(dim=None, keepdim=False)`     | `Tensor` | Reduce min               |
| `.reshape(shape)`                   | `Tensor` | Reshape                  |
| `.view(shape)`                      | `Tensor` | View reshape             |
| `.squeeze(dim=None)`                | `Tensor` | Remove size-1 dims       |
| `.unsqueeze(dim)`                   | `Tensor` | Add size-1 dim           |
| `.transpose(dim0, dim1)`            | `Tensor` | Swap dimensions          |
| `.permute(dims)`                    | `Tensor` | Reorder dimensions       |
| `.flatten(start_dim=0, end_dim=-1)`         | `Tensor` | Flatten range            |
| `.expand(sizes)`                    | `Tensor` | Broadcast view           |
| `.repeat(repeats)`                  | `Tensor` | Tile tensor              |
| `.prod()`, `.std()`, `.var()`      | `Tensor` | Additional reductions    |
| `.argmax()`, `.argmin()`            | `Tensor` | Index reductions         |
| `.all()`, `.any()`                  | `Tensor` | Logical reductions       |
| `.matmul()`, `.mm()`, `.bmm()`      | `Tensor` | Matrix products          |
| `.masked_select()`, `.masked_fill()`| `Tensor` | Masked operations        |
| `t.zeros_like(other)`, `t.ones_like(other)` etc.| `Tensor` | Like-constructors        |
| `t.from_dlpack(obj)` / `tensor.__dlpack__()` | `Tensor` / capsule | DLPack interop           |

**Operators:** `+`, `-`, `*`, `/`, `**`, `==`, `!=`, `<`, `>`, `<=`, `>=`, `[]` (indexing/slicing)

## Application

`lf.app.open(path)` opens a dataset or file in the application. Run scene,
training, and UI helpers inside the app; an imported stub package does not
provide an application runtime.


| Function             | Description              |
|----------------------|--------------------------|
| `lf.request_exit()`  | Exit with confirmation   |
| `lf.force_exit()`    | Immediate exit           |
| `lf.run(path)`       | Execute Python script    |
| `lf.on_frame(callback, duration_s=None)` | Per-frame callback; expires after 10 seconds with a one-time warning when no duration is supplied |
| `lf.set_frame_callback(callback, duration_s=None)` | Alias for the bounded per-frame callback API |
| `lf.stop_animation()`| Clear frame callback     |
| `lf.mat4(rows)`      | Create 4x4 matrix        |
| `lf.help()`          | Show help                |

## Native Operators (`lf.ops` / `lf.ui.ops`)

Use `lf.ops` for the native operator registry and descriptor metadata. The
`lf.ui.ops` namespace exposes the same common invoke/poll/modal controls for UI
code.

| API | Returns | Description |
|---|---|---|
| `lf.ops.invoke(operator_id, /, **kwargs)` | `OperatorReturnValue` | Invoke a native or Python operator |
| `lf.ops.poll(id)` | `bool` | Check whether the operator can run |
| `lf.ops.get_all()` | `list[str]` | Registered operator IDs |
| `lf.ops.get_descriptor(id)` | `OperatorDescriptor \| None` | Label, description, icon, shortcut, flags |
| `lf.ops.has_modal()` / `cancel_modal()` | `bool` / `None` | Modal operator state/control |

`OperatorReturnValue` has boolean helpers: `finished`, `cancelled`,
`running_modal`, `pass_through`, and `bool(result)` for successful completion.
Extra return data can be accessed by attribute.

## API Errors

Native failures raise `lf.Error` (a `RuntimeError` subclass) with `code`,
`domain`, `user_message`, `operation_id`, `retryable`, `details`, and `context`.
Specialized subclasses include `InvalidArgumentError`, `NotFoundError`,
`CancelledError`, and `ResourceError`. Optional return values such as a missing
node or viewport capture can instead be `None`; handle both cases in plugins.

## MCP Tools (`lf.mcp`)

Python plugins can register MCP tools into the same local MCP surface used by
automation clients.

```python
@lf.mcp.tool(name="my_plugin.echo", description="Echo a message")
def echo(args):
    return {"message": args.get("message", "")}
```

| Function | Returns | Description |
|---|---|---|
| `register_tool(fn, name='', description='')` | `None` | Register a Python function as an MCP tool |
| `@tool(name='', description='')` | decorator | Decorator form |
| `unregister_tool(name)` | `None` | Remove a Python MCP tool |
| `list_tools()` / `describe_tools()` | `list` | All shared MCP tools/capabilities |
| `list_python_tools()` | `list[str]` | Python tools registered through `lf.mcp` |
| `list_resources()` / `read_resource(uri)` | `list` | Shared MCP resource discovery/read |
| `call_tool(name, args=None)` | `object` | Invoke a registered tool/capability |

## Packages (`lf.packages`)

Package installation is backed by `uv` and the LichtFeld-managed Python
environment.

| Function | Returns | Description |
|---|---|---|
| `init()` | `str` | Initialize `~/.lichtfeld/venv` |
| `install(package)` / `uninstall(package)` | `str` | Synchronous package change |
| `list()` | `list[PackageInfo]` | Installed packages |
| `is_installed(package)` | `bool` | Package presence check |
| `install_async(package)` | `bool` | Start non-blocking install |
| `install_torch(cuda='auto', version='')` | `str` | Install PyTorch with CUDA detection |
| `install_torch_async(cuda='auto', version='')` | `bool` | Non-blocking PyTorch install |
| `is_busy()` | `bool` | Async operation running |
| `is_uv_available()` / `uv_path()` | `bool` / `str` | `uv` discovery |
| `embedded_python_path()` / `site_packages_dir()` / `typings_dir()` | `str` | Runtime paths |

There is no `uninstall_async()` binding; use synchronous
`uninstall()`.

## Scripts (`lf.scripts`)

These functions back the Scripts panel and are useful for plugin-managed script
batches.

| Function | Returns | Description |
|---|---|---|
| `get_scripts()` | `list` | Loaded script records |
| `set_script_enabled(index, enabled)` | `None` | Toggle one script |
| `set_script_error(index, error)` | `None` | Set or clear one script error |
| `clear_errors()` / `clear()` | `None` | Clear errors or all scripts |
| `run(paths)` | `dict` | Run scripts, returns success/error data |
| `get_enabled_paths()` | `list[str]` | Enabled script paths |
| `count()` | `int` | Script count |

## Keymaps (`lf.keymap`)

`lf.keymap` exposes input bindings for tools and global actions.

| Function | Description |
|---|---|
| `get_action_for_key(mode, key, modifiers=0)` / `get_action_for_scroll(mode, modifiers=0, held_keys=[])` | Resolve input to action |
| `get_key_for_action(action, mode=GLOBAL)` / `get_trigger(action, mode=GLOBAL)` | Read current binding |
| `set_binding(mode, action, key, modifiers=0)` / `set_trigger_binding(mode, action, trigger)` | Change binding |
| `clear_binding(mode, action)` / `reset_to_default()` | Remove or reset bindings |
| `find_conflict_for_action(mode, action)` | Detect binding conflicts |
| `get_available_profiles()` / `get_current_profile()` | Profile discovery |
| `load_profile(name)` / `save_profile(name)` / `export_profile(path)` / `import_profile(path)` | Profile persistence |
| `start_capture(mode, action)` / `capture_scroll(...)` / `cancel_capture()` | Interactive capture |
| `is_capturing()` / `get_captured_trigger()` / `bindings_revision()` | Capture and change state |

The enum surfaces include `Action`, `ToolMode`, `Modifier`, `MouseButton`,
`KeyTrigger`, and `MouseButtonTrigger`.

## Animation (`lf.animation`)

| Class | Purpose |
|---|---|
| `AnimationTrack` | Keyframes for one target property path; supports `add_keyframe()`, `remove_keyframe()`, `evaluate()`, and `keyframes()` |
| `AnimationClip` | Multi-track clip with `add_track(value_type, target_path)`, `remove_track()`, `get_track()`, and `evaluate(time)` |
| `Timeline` | Camera keyframe timeline plus optional animation clip; exposes `animation_clip()` and `evaluate_clip(time)` |

Track value types are `"bool"`, `"int"`, `"float"`, `"vec2"`, `"vec3"`,
`"vec4"`, `"quat"`, and `"mat4"`.

## Mesh (`lf.mesh`)

`lf.mesh` is a broad OpenMesh binding surface. The most common plugin entry
points are:

| API | Returns | Description |
|---|---|---|
| `MeshData` | class | Tensor-backed mesh payload used by scene/rendering |
| `read_trimesh(filename, **options)` / `read_polymesh(filename, **options)` | `TriMesh` / `PolyMesh` | Load OpenMesh meshes |
| `write_mesh(filename, mesh, **options)` | `None` | Save `TriMesh` or `PolyMesh` |
| `write_mesh(mesh_data, path)` | `None` | Save tensor-backed mesh data |
| `TriMesh` / `PolyMesh` | classes | OpenMesh-style topology, geometry, and property APIs |
| `TriMeshDecimater` / `PolyMeshDecimater` | classes | Decimation module management |

For exact method coverage, use `src/python/stubs/lichtfeld/mesh.pyi`; that
file is intentionally the canonical exhaustive reference for the mesh binding.

---
