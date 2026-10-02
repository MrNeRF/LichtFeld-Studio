---
sidebar_position: 7
title: Editor areas
---

# Editor areas

Editor areas divide the main work region into resizable editors. The screen
model owns layout and editor state; the GUI host turns that model into headers,
content rectangles, input gestures, and rendered viewports.

## Model

`ScreenLayout` is a tree of row and column splits. Its leaves are stable
`AreaId` values. `Screen` maps each area to its current editor and to the saved
`SpaceData` for editors previously shown there. A 3D editor owns a
`View3DSpace`, including its camera and `ViewSettings`.

```text
Screen
├── ScreenLayout (split tree -> AreaId leaves)
└── AreaId -> Area
    ├── editor: "view3d" | "scene" | "properties" | ...
    └── spaces: editor id -> SpaceData
        └── "view3d" -> View3DSpace (camera + ViewSettings)
```

Each layout mutation preserves at least one 3D view and prevents duplicate
single-instance editors. Splitting a multi-instance editor copies its space;
splitting a single-instance editor creates a view in the new area. Switching
back to an editor restores that area's saved space.

## Service and locking

`ScreenService` owns the editor registry and current `Screen`. GUI-thread code
may use `screen()` directly. Structural edits use `edit(fn)` and cross-thread
reads use `read(fn)`; both hold the service's recursive mutex so an area or
space cannot be destroyed while another thread reads it. `replace()` and
`resetToDefault()` advance the screen epoch used by view consumers.

Avoid retaining `Area*`, `SpaceData*`, or references returned during a locked
callback after the callback ends. Pass stable area/view ids across async work,
then resolve them again on the GUI thread.

## Gestures and GUI host

`AreaGestures` is a geometry-only state machine. It recognizes divider drags
and corner drags, previews split, join, and swap operations, then returns a
`GestureCommand`. `ScreenHost` applies those commands through `ScreenService`
and solves the layout again before routing later events in that frame.

```text
pointer events + LayoutGeometry
             |
             v
      AreaGestures -------- preview overlay
             |
       GestureCommand
             v
      ScreenHost -> ScreenService -> Screen
```

`ScreenHost` creates one `AreaFrame` per visible area. It renders editor
headers and the screen chrome with the screen RML documents, forwards header
actions to the matching `AreaEditor`, and lets panel-backed editors draw into
their content rectangle. `View3DEditor` owns the 3D header actions;
`PropertiesEditor`, `ScenePanelEditor`, `ConsoleEditor`, and `PanelEditor` host
their corresponding panel content.

## Per-view rendering

`RenderingManager` keeps a `ViewRenderState` per stable `ViewId`. It owns each
view's render targets, cached viewport image, Vulkan presentation state,
overlay services, and dirty mask. Rendering a view uses that area's
`View3DSpace` camera and settings, then presents the result into the content
rectangle supplied by `ScreenHost`. A hidden or off-screen view can keep its
state without sharing mutable render targets with another view.

Scene changes and camera/settings edits mark the affected view dirty. Screen
structure changes invalidate the layout generation; removed view ids are
retired and their render targets are released by the rendering manager. This
keeps an image, capture, or asynchronous result tied to the view that produced
it.

## Input routing

A press inside a 3D area activates that view. Keyboard and wheel navigation
follow the hovered view, with the active view as the fallback when the pointer
is outside a viewport. A drag or other captured interaction keeps its original
`ViewId` until release, even if the pointer crosses into another area. Screen
headers and gesture zones take their own pointer events before viewport input.
Text fields retain keyboard shortcuts while text editing is active.

## Persistence

`Screen::save()` stores a format version, the split tree, areas, each area's
editor id and saved spaces, the active 3D view, and optional maximized area.
Project loading validates ids and editor types, restores available spaces,
repairs invalid references, and falls back to the default screen when the
stored format cannot be loaded. Project screen state therefore travels with
the project.

## Python and MCP APIs

The `lf.ui.screen` Python API can query areas and editor types, split, join,
swap, close, or maximize areas, switch editors, open or close editors, set the
active view, and issue per-view commands. Camera and `ViewSettings` access is
available by 3D view id. Mutations run on the viewer thread and notify the
screen host after completion.

The MCP surface exposes corresponding `screen.get`, `screen.split`,
`screen.join`, `screen.close`, `screen.swap`, `screen.set_editor`,
`screen.maximize`, and `screen.reset` commands. `view.command`,
`view.get_camera`/`view.set_camera`, and `view.get_settings`/`view.set_settings`
query or update one 3D view. Read
`lichtfeld://scene/state` and the runtime catalog before using these tools;
the MCP guide documents tool schemas and GUI-thread behavior.

## Live view verification

`scripts/verify_editor_area_views.py` checks camera, render target, and redraw
isolation through MCP and real X11 mouse input. Start an MCP-enabled build, load
a scene, and split it into at least two visible 3D areas. With `xdotool` installed,
run the script against that app's display and MCP port:

```sh
python3 scripts/verify_editor_area_views.py --display :96 --port 45696 --output /tmp/editor-view-check
```

Use `--window` when the display contains multiple app windows. The tool orbits
each view and writes before/after captures and JSON evidence to the output folder.
