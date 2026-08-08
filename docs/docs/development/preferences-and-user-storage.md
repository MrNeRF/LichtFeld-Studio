---
title: Preferences and user storage
---

# Preferences and user storage

LichtFeld Studio stores global user preferences separately from project state.
The modern user root contains application preferences, keymaps, backups,
plugins, Python environments and other user-owned data.

## Global preferences

The Preferences panel currently exposes:

- language;
- application theme;
- UI scale;
- camera navigation mode;
- axis/view snap;
- per-setting remember options;
- MCP server enablement, bind scope, port, and opt-in request logging;
- interface and layout reset actions.

Preferences are stored in `config/preferences.json`. Writes must use the
atomic JSON writer so an interrupted shutdown cannot leave a partially written
file.

The `mcp` object defaults to an enabled server bound to the loopback interface
on port `45677`; the UI lists both `127.0.0.1` and `localhost` aliases. Changes
made in Preferences are applied immediately. Binding to
`0.0.0.0` exposes the unauthenticated HTTP endpoint to the local network and is
therefore an explicit opt-in. Safe mode forces the MCP server and request
logging off for the current process and does not persist MCP changes.
The default input profile uses `Ctrl+Shift+M` to enable or disable the server
and `Ctrl+Shift+N` to switch between loopback and network binding without
enabling a server that is currently off. Both shortcuts can be rebound in
Input Settings.

## User storage roots

The default root follows the platform policy implemented by `UserPaths`.
Explicit application roots use one isolated unified storage tree.
Portable builds use a `.lichtfeld` tree next to the executable. `LFS_HOME`
remains an explicit override on every platform and in portable builds.
The application never scans old profile directories and never imports files
from them automatically.

The modern tree contains, as applicable:

```text
config/preferences.json
config/layout.json
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

Project-owned layout and session state is expected to move into the `.licht`
project format. `layout.json` is therefore transitional and must not acquire
new categories of project state.

## Startup and reset operations

The related command-line flags are:

| Flag | Behaviour |
| --- | --- |
| `--safe-mode` | Start with user plugins and automatic persistence disabled; MCP and MCP request logging remain off for the process. |
| `--no-splash` | Skip the startup splash screen in non-portable builds. |
| `--reset-preferences` | Back up and restore application preferences to built-in defaults. |
| `--reset-layout` | Back up and remove the transitional saved UI layout. |
| `--reset-all-settings` | Reset preferences, the transitional layout file, and window state using the same backup policy. |

The GUI and CLI should expose the same semantics and should report an explicit
error when a reset is requested in a mode that cannot execute it. Resets do not
remove plugins, keymaps, caches, datasets or project files unless explicitly
specified by a future feature.

## Safe mode

`--safe-mode` starts the application with user plugin loading disabled and
disables automatic persistence for preferences, keymaps, legacy layout state,
window geometry, and the Vulkan pipeline cache. It must not
overwrite an externally supplied `LFS_SAFE_MODE=1` when starting normally.
Explicit user actions such as keymap export remain separate from automatic
persistence and should remain available in safe mode.

## Project state boundary

Global preferences belong to the user root. Layout, open tabs, panel
arrangement and other project/session state should be owned by the `.licht`
project format once that upstream work lands. This boundary prevents a
project-specific arrangement from unexpectedly changing the user's global UI.
