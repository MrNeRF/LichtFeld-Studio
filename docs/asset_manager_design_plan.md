# LichtFeld Studio Asset Manager Execution Plan

## Goal

Implement a **floating, resizable Asset Manager window** opened from:

```text
Tools → Asset Manager
```

It should organize LichtFeld work around:

```text
Library
└── Project
    └── Scene
        └── Training Run
            └── Assets / Artifacts
```

All Asset Manager metadata and generated previews should live under:

```text
~/.lichtfeld/asset_manager/
```

Use **JSON only** for persistence. No SQLite. No docking/pinning for now.

---

## 1. Core Principles

| Principle                  | Decision                                                                           |
| -------------------------- | ---------------------------------------------------------------------------------- |
| Asset Manager is a catalog | It tracks projects, scenes, runs, and assets; it is not just a file browser        |
| JSON persistence           | Store everything in `~/.lichtfeld/asset_manager/library.json`                      |
| Centralized storage        | Metadata, thumbnails, previews, and cache live under `~/.lichtfeld/asset_manager/` |
| Floating only              | Open as a floating RmlUi window; support move, resize, close, focus                |
| No destructive defaults    | Prefer “Remove from Catalog” over deleting files                                   |
| Visual-first browsing      | Use thumbnails/cards for splats, not only lists                                    |

---

## 2. Storage Layout

```text
~/.lichtfeld/asset_manager/
├── library.json
├── thumbnails/
├── previews/
└── cache/
```

`library.json` stores:

```json
{
  "version": 1,
  "created_at": "2026-04-28T10:30:00Z",
  "modified_at": "2026-04-28T10:30:00Z",
  "projects": {},
  "scenes": {},
  "runs": {},
  "assets": {},
  "collections": {},
  "tags": {},
  "ui_state": {
    "view_mode": "gallery",
    "sort_mode": "recent",
    "search_query": "",
    "window_position": { "x": 200, "y": 120 },
    "window_size": { "width": 980, "height": 620 }
  }
}
```

---

## 3. Data Model

### Project

```python
Project:
    id: str
    name: str
    description: str
    created_at: str
    modified_at: str
    scene_ids: list[str]
    tags: list[str]
    notes: str
    thumbnail_asset_id: str | None
```

### Scene

```python
Scene:
    id: str
    project_id: str
    name: str
    description: str
    created_at: str
    modified_at: str
    dataset_asset_id: str | None
    run_ids: list[str]
    tags: list[str]
    notes: str
    thumbnail_asset_id: str | None
```

### TrainingRun

```python
TrainingRun:
    id: str
    project_id: str
    scene_id: str
    name: str
    status: str  # draft, training, completed, failed, archived, final
    created_at: str
    modified_at: str
    source_dataset_id: str | None
    parent_run_id: str | None
    parent_checkpoint_id: str | None
    parameters: dict
    metrics: dict
    artifact_asset_ids: list[str]
    tags: list[str]
    notes: str
    is_favorite: bool
```

### Asset

Use both `type` and `role`.

```python
Asset:
    id: str
    project_id: str | None
    scene_id: str | None
    run_id: str | None

    name: str
    type: str       # ply, rad, sog, spz, checkpoint, dataset, video, usd, html, json
    role: str       # source_dataset, training_checkpoint, trained_output, preview, export, reference

    path: str
    absolute_path: str

    created_at: str
    modified_at: str
    file_size_bytes: int

    tags: list[str]
    collection_ids: list[str]
    notes: str
    is_favorite: bool

    thumbnail_path: str | None
    preview_path: str | None

    geometry_metadata: dict | None
    training_metadata: dict | None
    dataset_metadata: dict | None
    video_metadata: dict | None

    exists: bool
```

---

## 4. UI Behavior

### Opening

Add menu entry:

```text
Tools → Asset Manager
```

Behavior:

```text
closed      → open floating window
already open → focus / bring to front
hidden      → show and focus
```

The window should support:

```text
move
resize
close
focus
bring to front
```

Do not implement:

```text
dock
pin
side-panel conversion
```

---

## 5. UI Layout

```text
Asset Manager Window
├── Toolbar
├── Summary Row
├── Content Area
│   ├── Left Navigation / Filters
│   ├── Gallery or List View
│   └── Right Info Panel
└── Footer / Batch Actions
```

### Toolbar

```text
[Import ▼] [Search assets...] [Sort by: Recent ▼] [Gallery] [List]
```

Import options:

```text
Import Asset...
Import Dataset...
Import Checkpoint...
Import Folder...
```

### Summary Row

```text
24 assets    4 projects    8 scenes    3 selected
```

### Left Navigation

Keep filters. Use useful grouped filters:

```text
PROJECTS
SCENES
FILTERS
  All Assets
  Trained Outputs
  Checkpoints
  Datasets
  Videos
  Missing Files
  Favorites

FORMATS
  PLY
  RAD
  SOG
  SPZ
  MP4
  CKPT

TAGS
COLLECTIONS
```

### Gallery View

Default view. Each card should show:

```text
thumbnail
asset/run name
format pill
role/status badge
short metadata
tags
quick actions
```

Example:

```text
[Thumbnail]
dante_statue_final     PLY
2.85M Gaussians · 182 MB
statue · outdoor · final

[Preview] [Load] [Info]
```

### List View

Columns:

```text
Name
Type
Role
Project
Scene
Run
Size
Modified
Status
```

### Right Info Panel

Adapt based on selected object.

For asset selected:

```text
Asset Details
Type
Role
Path
Size
Created
Modified
Exists / Missing

Geometry
Gaussians
SH Degree
Compressed
LoD / Streamable

Training Provenance
Run
Iterations
Optimizer
Source Images
Resolution Scale
Train Time

Actions
Open in Viewer
Reveal in Folder
Locate File
Remove from Catalog
```

For run selected:

```text
Run Details
Status
Scene
Dataset
Created
Modified

Training Parameters
Iterations
SH Degree
Optimizer
Densify Interval
Opacity Reset
Learning Rate
Background

Artifacts
Checkpoint
PLY
RAD
Preview MP4
Metrics JSON
```

### Footer

When selected:

```text
3 selected    Total: 1.16 GB
[Load Selected] [Export Selected] [Add Tag] [Remove From Catalog]
```

---

## 6. RmlUi Implementation

Create/update:

```text
src/visualizer/gui/rmlui/resources/asset_manager.rml
src/visualizer/gui/rmlui/resources/asset_manager.rcss
```

RML should only define structure:

```xml
<head>
  <link type="text/template" href="floating_window.rml"/>
  <link type="text/rcss" href="asset_manager.rcss"/>
</head>
```

RCSS owns:

```text
layout
dimensions
spacing
colors
borders
typography
cards
buttons
selected states
scroll behavior
resize behavior
```

Use reusable classes:

```text
.asset-panel
.asset-section-title
.asset-nav-row
.asset-card
.asset-card-thumb
.asset-pill
.asset-tag
.asset-button
.asset-info-section
.asset-parameter-row
.asset-footer-button
```

Design language:

```text
warm dark panels
subtle borders
muted text
teal/green selection accent
compact desktop spacing
no emoji icons
no web-dashboard styling
```

Spacing scale:

```text
2dp  tiny spacing
4dp  small row spacing
6dp  compact padding
8dp  card/control padding
10dp panel padding
14dp section separation
```

---

## 7. Backend Components

### AssetIndex

Create:

```text
src/python/lfs_plugins/asset_index.py
```

Responsibilities:

```text
load/save library.json
create default catalog if missing
add/update/remove projects
add/update/remove scenes
add/update/remove runs
add/update/remove assets
add/update/remove collections
recompute tag counts
query/search/filter/sort assets
mark missing files
```

### AssetScanner

Create:

```text
src/python/lfs_plugins/asset_scanner.py
```

Responsibilities:

```text
detect file type
detect role
read file size
read created/modified timestamps
extract gaussian count if possible
extract checkpoint metadata if possible
detect dataset structure
return metadata dict
```

### AssetThumbnails

Create:

```text
src/python/lfs_plugins/asset_thumbnails.py
```

Responsibilities:

```text
generate placeholder thumbnails
store thumbnails under ~/.lichtfeld/asset_manager/thumbnails/
return thumbnail path
invalidate thumbnails
show missing thumbnail state
```

First version can use placeholder thumbnails.

### AssetManagerPanel

Create/update:

```text
src/python/lfs_plugins/asset_manager_panel.py
```

Responsibilities:

```text
open/close/focus floating RmlUi window
bind catalog data to RmlUi model
handle search/filter/sort
handle selected assets
handle import actions
handle load actions
handle remove-from-catalog
update info panel
refresh gallery
```

---

## 8. Import Workflows

### Import Asset

```text
User clicks Import → Import Asset
Open file dialog
Scan selected file
Create Asset entry
Generate placeholder thumbnail
Save library.json
Refresh gallery
```

Supported:

```text
.ply
.rad
.sog
.spz
.ckpt
.mp4
.mov
.usd
.usdz
.html
.json
```

### Import Dataset

```text
User clicks Import → Import Dataset
Open folder dialog
Validate dataset
Create Dataset Asset
Optionally create Project/Scene
Save catalog
Refresh gallery
```

### Import Folder

```text
User clicks Import → Import Folder
Scan folder
Show found assets
User confirms
Add assets to catalog
Save catalog
Refresh gallery
```

First version can use shallow scan.

---

## 9. Training and Export Integration

### Training starts

```text
create/select Project
create/select Scene
create TrainingRun with status = training
store parameters
save catalog
```

### Checkpoint saved

```text
create Asset:
  type = checkpoint
  role = training_checkpoint
link to run_id
add to run.artifact_asset_ids
save catalog
```

### Training completes

```text
set run.status = completed
update metrics if available
save catalog
```

### Export generated

```text
scan exported file
create/update Asset
type = ply/rad/sog/spz/etc.
role = trained_output or export
link to project/scene/run if known
generate thumbnail
save catalog
refresh UI if open
```

If path already exists:

```text
preserve asset ID
preserve tags/collections/notes/favorite
update modified_at, file size, metadata
```

---

## 10. Missing Files and Conflicts

Use canonical absolute path as first identity check.

Rules:

```text
same path already indexed → update existing asset
file missing → keep metadata, show Missing state
export overwrites indexed path → preserve asset ID, update metadata
duplicate under different path → allow duplicate for now
```

Missing file UI:

```text
gray card
Missing badge
disabled Load button
warning in info panel
Locate File button
Remove From Catalog button
```

---

## 11. Search, Filter, Sort

Search should match:

```text
asset name
project name
scene name
run name
tags
notes
type
role
path
```

Filters:

```text
project
scene
run
type
role
tag
favorite
missing
status
```

Sort modes:

```text
Recent
Name
Type
File Size
Project
Scene
Run
```

---

## 12. Implementation Phases

| Phase | Goal                        | Deliverable                                                           |
| ----- | --------------------------- | --------------------------------------------------------------------- |
| 1     | Floating UI mock            | Tools → Asset Manager opens polished resizable popup with static data |
| 2     | JSON persistence            | `library.json` loads/saves projects, scenes, runs, assets             |
| 3     | Import and scan             | User can import files/datasets/folders into catalog                   |
| 4     | Real UI binding             | Static cards replaced with real catalog data                          |
| 5     | Asset actions               | Load, preview, favorite, tag, locate missing, remove from catalog     |
| 6     | Training/export integration | Training outputs and exports auto-register as run artifacts           |

---

## 13. First Version Scope

Include:

```text
Tools → Asset Manager floating window
JSON catalog under ~/.lichtfeld/asset_manager/library.json
Project / Scene / Run / Asset model
Manual import
Gallery view
Basic list view
Search
Filters
Tags
Favorites
Right info panel
Training parameter display
Missing file state
Load asset into viewer
Remove from catalog
Placeholder thumbnails
```

Exclude:

```text
SQLite
cloud storage
asset marketplace
automatic compression
docking/pinning
advanced run comparison
real rendered thumbnails
deep background watching
destructive file deletion
```

---

## 14. Success Criteria

The first version is successful when:

```text
Asset Manager opens from Tools menu
It appears as a floating resizable window
UI feels native to LichtFeld Studio
Catalog persists in JSON under ~/.lichtfeld/asset_manager/
Users can organize Project → Scene → Run → Asset
Users can import existing assets
Users can inspect metadata and parameters
Users can load assets into viewer
Missing files are handled gracefully
Gallery is visual, not a flat file list
```
Agreed. Remove `ui_state` from `library.json`. Keep UI/window state separate from the asset catalog, or do not persist it at all for now.

The catalog should stay focused on **library data only**:

```json
{
  "version": 1,
  "created_at": "2026-04-28T10:30:00Z",
  "modified_at": "2026-04-28T10:30:00Z",

  "projects": {},
  "scenes": {},
  "runs": {},
  "assets": {},
  "collections": {},
  "tags": {}
}
```

So the storage section becomes:

````md
## 2. Storage Layout

```text
~/.lichtfeld/asset_manager/
├── library.json
├── thumbnails/
├── previews/
└── cache/
````

`library.json` stores only persistent Asset Manager catalog data:

```json
{
  "version": 1,
  "created_at": "2026-04-28T10:30:00Z",
  "modified_at": "2026-04-28T10:30:00Z",

  "projects": {},
  "scenes": {},
  "runs": {},
  "assets": {},
  "collections": {},
  "tags": {}
}
```

Do not store UI state such as window position, window size, search query, selected project, view mode, or sort mode in `library.json`.

Window position/size can be handled by the RmlUi window system or app preferences later. Search/filter/view state can be runtime-only for now.

```
```
0