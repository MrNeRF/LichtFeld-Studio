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

Application preferences are stored in `config/preferences.json`. Writes use
the atomic JSON writer so an interrupted shutdown cannot leave a partially
written preferences file. User-global UI details that are not project layout,
such as HUD state, are written to `config/ui_preferences.json`.

## User storage roots

The default root follows the platform policy implemented by `UserPaths`.
Explicit application roots use one isolated unified storage tree. Portable
builds use a `.lichtfeld` tree next to the executable. `LFS_HOME` remains an
explicit override on every platform and in portable builds. The application
does not scan or import old profile directories automatically.

The user tree contains, as applicable:

```text
config/preferences.json
config/ui_preferences.json
config/window.json
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

Desktop window placement, size, maximized state, and monitor association are
user-global and are stored in `config/window.json`. Restoring this file must
validate the saved rectangle against the monitors currently available and
recover to a visible position when displays or DPI settings have changed.

Panel visibility, dock dimensions, active tabs, the sequencer, and other
project workspace state belong to the GUIL chapter of the `.licht` project.
Opening an existing project restores that saved workspace. For compatibility,
GUIL also retains the window state written by the current project format, while
`config/window.json` supplies the user-global startup state.

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
| `--reset-layout` | Back up and remove the legacy `layout.json` file before startup. |
| `--reset-all-settings` | Apply the preference and legacy-layout resets and also back up and remove saved window state. |

The Preferences panel additionally provides live reset actions for the current
interface layout and desktop window state. Reset operations retain backups in
the user data tree and do not remove plugins, keymaps, caches, datasets, or
project files.

## Safe mode

`--safe-mode` disables user plugin loading and automatic persistence for
preferences, keymaps, legacy UI preferences, window geometry, and the Vulkan
pipeline cache. The process uses built-in defaults, including English, without
overwriting the user's saved values. An externally supplied `LFS_SAFE_MODE=1`
must remain effective when the application is otherwise started normally.

Explicit operations such as exporting a keymap remain separate from automatic
persistence and can stay available in safe mode. The status bar identifies
safe mode for the lifetime of the process.
