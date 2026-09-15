# Viewport workspace

The working area is a tree of editor areas. Each area can display a 3D viewport, Scene, Rendering, Training, Python Console, or another registered editor.

## Controls

- Drag an area's lower-right corner inward to split horizontally or vertically. Release to commit; Escape cancels.
- Drag a divider to resize adjacent areas. Minimum sizes are measured in pixels; there is no fixed view-count limit.
- Use the header selector to change editor type. Maximize temporarily hides the other areas; restoring preserves their cameras and editor state.

Each 3D viewport has its own camera, projection, depth window, grid plane, and overlays. Scene edits, selection, and tool state are shared. Camera utilities and navigation affect the clicked viewport. A drag keeps its starting viewport when the pointer crosses a divider. Focusing a UI editor retains the last active 3D viewport for camera commands.

## Ownership

| Owner | State |
| --- | --- |
| `ViewportWorkspace` | View IDs, layout, focus, maximization, and view lifetime |
| `WorkspaceLayout` | Binary split tree and rectangle solving |
| `ViewRegistry` | Editor type, camera, projection, depth window, and editor state |
| `WorkspaceFrameSnapshot` | Copied area bounds and camera state for one frame |
| `AreaEditorHost` | Header and panel instance for each area |
| `RmlViewportOverlay` | Overlay document and toolbar controller for one viewport |
| `SceneManager` | Shared models, transforms, visibility, and selection |
| `RenderingManager` | Per-view rendering and published color/depth output |
| `SharedViewportGpuAssets` | Shared mesh, material, and environment resources |

New views clone camera state and clear movement. Closing a view retires its UI and GPU resources after submitted frames finish. Maximized hidden views retain their state and output, but do not contribute LOD streaming requests.

Rml data models are scoped by context and name. The `viewport_overlay` document alias follows the active viewport; changing this alias does not destroy a document registered under its permanent name.

## Rendering

UI, splitter decorations, and scene presentation use the same frame snapshot. Raster submissions run sequentially because projection, sorting, and raster scratch are shared. Each view retains its own published image while waiting for a new render. Guides and depth projection use that image's unjittered camera state.

Orthographic zoom is measured in logical pixels per world unit and scales with raster resolution. Switching between reduced and full resolution during resize therefore preserves the visible scene.

Python draw handlers receive each viewport's camera and bounds. Native overlays are clipped per pane. Transform gizmos render in every viewport, with one interactive call applying the shared edit.

GT and PLY comparison use their existing comparison camera and restoration rules. They remain separate from workspace layout.

## Persistence and automation

`VIEW.workspace` stores the layout and durable view state in native projects. Restore validates the payload before applying it and preserves the view-ID high-water mark. Legacy projects restore their primary camera into a single viewport.

Dataset-camera appearance correction belongs to the viewport that selected the camera. Navigation, scene replacement, or project restore clears that association; camera poses still persist.

Python exposes workspace operations through `lf.ui.workspace_*`. Discover the corresponding MCP tools and `lichtfeld://workspace/state` through the runtime catalog.

`scripts/verify_viewport_workspace.py` checks camera isolation, published output, layout changes, and project round-trip through MCP. Run it against a disposable app session: it loads a generated scene fixture. Use `--help` for arguments. C++ workspace tests cover geometry, input capture, identity, serialization, output lifetime, and projection; Python tests cover overlay and toolbar ownership.
