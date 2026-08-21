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
- viewport scene reconstruction backend and backend-specific preset;
- camera navigation mode and axis/view snap;
- per-setting remember options;
- MCP server enablement, bind scope, port, and opt-in request logging;
- interface, layout, and window reset actions.

The panel can be opened from the Edit menu or with the default `Ctrl+,`
shortcut. The shortcut is a regular keymap action and can be rebound in Input
Settings.

The `mcp` object defaults to an enabled server bound to the loopback interface
on port `45677`; the UI lists both `127.0.0.1` and `localhost` aliases. Confirmed
changes made in Preferences are staged immediately and listener restarts are
serialized by one long-lived background worker. Rapid changes are coalesced to
the newest pending configuration, and the UI reports starting, running, stopping,
and failed states without waiting for listener shutdown. Binding to `0.0.0.0` exposes the unauthenticated HTTP
endpoint to the local network and is therefore an explicit opt-in. Safe mode
forces the MCP server and request logging off for the process. Preferences
identifies that effective state and disables the MCP controls, so opening the
panel cannot accidentally stage or persist an MCP change while safe mode is
active.

The status bar MCP chip reports the effective listener state, usable endpoint
URLs, request/success/error counters, and bind failures. Its power control does
not require opening Preferences. The default input profile uses `Ctrl+Shift+M`
to enable or disable the server and `Ctrl+Shift+N` to switch between loopback
and network binding without enabling a server that is currently off. While the
server is off, the chip still shows whether the next start will use local or
network scope. Both
shortcuts are regular keymap actions and can be rebound in Input Settings.

MCP request logging is disabled by default. When enabled it appends complete
records to a per-session JSONL file created lazily under `logs/mcp/`; prior
records are neither retained in memory nor rewritten for each request. Records contain
transport metadata such as method, request id, outcome, duration, and the source
and destination socket IP addresses and ports. Errors distinguish JSON-RPC
failures from MCP tool-execution failures and record their stage, stable reason,
protocol or application code, application domain, retryability, and operation
id when available. Request parameters, tool names, payloads, free-form messages,
and error details are deliberately not recorded. Because network addresses can
identify devices and interfaces, request logging remains an explicit opt-in.

Application preferences are stored in `config/preferences.json`. User-global
UI details that are not project layout, such as HUD state, are written to
`config/ui_preferences.json`. Preferences, UI state, window state, lifecycle
settings, keymaps, and the Asset Manager catalog are published with
same-directory atomic replacement so an interrupted write cannot expose a
partially written destination.

Scene reconstruction preferences are user-global rather than `.licht` project
state. They affect viewport presentation only and do not alter training or
stored splat data. See [Scene reconstruction](scene-reconstruction) for the
backend, preset, lazy-loading, and fallback contracts.

## User storage roots

The default root is the unified `~/.lichtfeld` tree on Windows and Linux.
Explicit application roots use the same isolated storage layout. Portable
builds use a `.lichtfeld` tree next to the executable. `LFS_HOME` remains an
explicit override on every platform and in portable builds. The application
imports legacy keymaps, theme/language choices, UI preferences, and
`layout.json` once when the unified tree is first used. Existing destination
files are never overwritten and legacy sources are never deleted. The Asset
Manager similarly copies an existing legacy catalog into the resolved writable
catalog directory without deleting the source.

| Mode | Config | Durable data | Cache | Logs |
| --- | --- | --- | --- | --- |
| Windows | `%USERPROFILE%/.lichtfeld/config` | `%USERPROFILE%/.lichtfeld/data` | `%USERPROFILE%/.lichtfeld/cache` | `%USERPROFILE%/.lichtfeld/logs` |
| Linux | `~/.lichtfeld/config` | `~/.lichtfeld/data` | `~/.lichtfeld/cache` | `~/.lichtfeld/logs` |
| `LFS_HOME=<root>` | `<root>/config` | `<root>/data` | `<root>/cache` | `<root>/logs` |
| Portable build | `<executable>/.lichtfeld/config` | `<executable>/.lichtfeld/data` | `<executable>/.lichtfeld/cache` | `<executable>/.lichtfeld/logs` |

Plugins and their virtual environment are also children of the unified root.

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
logs/mcp/
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
When no valid `config/window.json` exists, including the first launch or after
a complete settings reset, the default 1280x720 window is centered in the
primary display's usable area and clamped if that area is smaller.

Panel visibility, dock dimensions, active tabs, the sequencer, and other
project workspace state belong to the GUIL chapter of the `.licht` project.
Opening an existing project restores that saved workspace. Main-window
position, size, maximized state, theme, language, and UI/DPI scale are
user-global and never restored from a project. Fullscreen is not persistent.
Window geometry is persisted only in `config/window.json`.

Older GUIL v1 chapters may still contain a project-owned `window` member.
Readers accept those projects but remove that legacy member before validation.
The normalization does not make an otherwise clean project appear modified or
show a save prompt; the next Save or Save As materializes the GUIL chapter
without the obsolete window data instead of copying its original bytes.

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

Modal backdrops are input boundaries, not only visual dimming. While one is
active or pending, pointer and keyboard input must not reach the application
underneath, including native viewport overlays such as the axis selector.
The startup overlay remains viewport-centered and does not dim the entire
application window; it stops blocking the viewport once plugin loading starts.

## Safe mode

`--safe-mode` disables user plugin loading and automatic persistence for
preferences, keymaps, UI preferences, window geometry, project lifecycle
settings, and the Vulkan pipeline cache. Asset Manager path resolution also
avoids write probes and migration in safe mode. The process uses built-in
defaults, including English, without overwriting the user's saved values. An
externally supplied `LFS_SAFE_MODE=1` must remain effective when the
application is otherwise started normally.

Safe mode also forces the effective MCP listener and MCP request logging off.
The persisted MCP configuration is neither read nor overwritten. The MCP
section remains visible for diagnosis, but it displays a safe-mode notice and
keeps every live configuration control disabled.

Explicit operations such as exporting a keymap remain separate from automatic
persistence and can stay available in safe mode. The status bar identifies
safe mode for the lifetime of the process.
