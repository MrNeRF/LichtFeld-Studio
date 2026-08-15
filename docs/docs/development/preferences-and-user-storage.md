---
title: Preferences and user storage
---

# Preferences and user storage

LichtFeld Studio keeps user-global preferences separate from project state.
This distinction matters both for persistence code and for reset or recovery
features: opening a project may restore its workspace, but it must not replace
the user's application preferences or desktop window placement.

## Global preferences

The Preferences panel currently exposes:

- language;
- application theme and UI scale;
- camera navigation mode and axis/view snap;
- per-setting remember options;
- interface, layout, and window reset actions.

The panel can be opened from the Edit menu or with the default `Ctrl+,`
shortcut. The shortcut is a regular keymap action and can be rebound in Input
Settings.

Application preferences are stored in `config/preferences.json`. User-global
UI details that are not project layout, such as HUD state, are written to
`config/ui_preferences.json`. Preferences, UI state, window state, lifecycle
settings, keymaps, and the Asset Manager catalog are published with
same-directory atomic replacement so an interrupted write cannot expose a
partially written destination.

## User storage roots

The default root follows the platform policy implemented by `UserPaths`.
Explicit application roots use one isolated unified storage tree. Portable
builds use a `.lichtfeld` tree next to the executable. `LFS_HOME` remains an
explicit override on every platform and in portable builds. The application
does not scan or import old settings directories automatically. The Asset
Manager is the exception: it can copy an existing legacy catalog into the
resolved writable catalog directory without deleting the source.

| Mode | Config | Durable data | Cache | Logs |
| --- | --- | --- | --- | --- |
| Windows | `%USERPROFILE%/.lichtfeld/config` | `%USERPROFILE%/.lichtfeld/data` | `%USERPROFILE%/.lichtfeld/cache` | `%USERPROFILE%/.lichtfeld/logs` |
| Linux | `${XDG_CONFIG_HOME:-~/.config}/lichtfeld-studio` | `${XDG_DATA_HOME:-~/.local/share}/lichtfeld-studio` | `${XDG_CACHE_HOME:-~/.cache}/lichtfeld-studio` | `${XDG_STATE_HOME:-~/.local/state}/lichtfeld-studio` |
| `LFS_HOME=<root>` | `<root>/config` | `<root>/data` | `<root>/cache` | `<root>/logs` |
| Portable build | `<executable>/.lichtfeld/config` | `<executable>/.lichtfeld/data` | `<executable>/.lichtfeld/cache` | `<executable>/.lichtfeld/logs` |

Plugins and their virtual environment remain under `~/.lichtfeld` on the
default Linux layout for first-generation compatibility. They move under the
unified root for Windows, `LFS_HOME`, and portable builds.

The user tree contains, as applicable:

```text
config/preferences.json
config/ui_preferences.json
config/window.json
config/project_lifecycle.json
config/keymaps/
data/backups/
data/presets/
data/asset_library/
cache/
logs/
plugins/
venv/
```

`config/layout.json` is a legacy, import-only layout source. New project
workspace state is not written there.

## Window and project state boundaries

Desktop window placement, size, and maximized state are user-global and are
stored in `config/window.json`. Fullscreen is deliberately transient and is
never stored. When fullscreen is active at shutdown, the last windowed
rectangle is saved instead of the fullscreen display dimensions.

At startup the saved rectangle is checked against the usable areas of the
currently connected displays. A rectangle that is no longer visible is
centered on the primary display. A rectangle that still intersects a display
but is larger than that display's usable area is clamped and centered on the
display with the largest intersection. This covers reopening a layout saved on
a large monitor after moving the application to a substantially smaller one.

Panel visibility, dock dimensions, active tabs, the sequencer, and other
project workspace state belong to the GUIL chapter of the `.licht` project.
Opening an existing project restores that saved workspace. Main-window
position, size, maximized state, theme, language, and UI/DPI scale are
user-global and never restored from a project. Fullscreen is not persistent.
Window geometry is persisted only in `config/window.json`.

Creating a new project is intentionally different: it clears project-owned
content without applying the default GUIL chapter, opening or closing panels,
or changing the live desktop window geometry. It therefore preserves the
workspace in which the user invoked New Project.

## Startup and reset operations

The related command-line options are:

| Option | Behaviour |
| --- | --- |
| `--safe-mode` | Start with user plugins and automatic user-state persistence disabled for this process. |
| `--no-splash` | Skip the startup splash screen in non-portable builds. |
| `--reset-preferences` | Back up application preferences and write built-in defaults before startup. |
| `--reset-layout` | Back up and remove legacy layout and user-global HUD/UI-layout preferences before startup. |
| `--reset-all-settings` | Apply the preference and layout resets and also back up and remove window and project-lifecycle settings. |

The Preferences panel additionally provides live reset actions for the current
interface layout and desktop window state. Interface reset restores the initial
docks, panel visibility, tabs, sequencer, scene-tree chrome, and HUD state while
keeping the Preferences panel open. If a `.licht` project is open, that live
default layout replaces its GUIL layout the next time the project is saved.
Reset operations retain backups in the user data tree and do not remove
plugins, keymaps, caches, datasets, or project files.

## Safe mode

`--safe-mode` disables user plugin loading and automatic persistence for
preferences, keymaps, UI preferences, window geometry, project lifecycle
settings, and the Vulkan pipeline cache. Asset Manager path resolution also
avoids write probes and migration in safe mode. The process uses built-in
defaults, including English, without overwriting the user's saved values. An
externally supplied `LFS_SAFE_MODE=1` must remain effective when the
application is otherwise started normally.

Explicit operations such as exporting a keymap remain separate from automatic
persistence and can stay available in safe mode. The status bar identifies
safe mode for the lifetime of the process.
