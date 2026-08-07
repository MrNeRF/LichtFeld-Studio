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
- interface and layout reset actions.

Preferences are stored in `config/preferences.json`. Writes must use the
atomic JSON writer so an interrupted shutdown cannot leave a partially written
file.

## User storage roots

The default root follows the platform policy implemented by `UserPaths`.
Explicit user roots and portable roots use one isolated `.lichtfeld` tree.
The application never scans old profile directories and never imports files
from them automatically.

The modern tree contains, as applicable:

```text
config/preferences.json
config/layout.json
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

## Reset operations

The command-line reset flags are:

| Flag | Behaviour |
| --- | --- |
| `--reset-preferences` | Back up and restore application preferences to built-in defaults. |
| `--reset-layout` | Back up and remove the transitional saved UI layout. |
| `--reset-all-settings` | Perform both preference and layout reset using the same backup policy. |

The GUI and CLI should expose the same semantics and should report an explicit
error when a reset is requested in a mode that cannot execute it. Resets do not
remove plugins, keymaps, caches, datasets or project files unless explicitly
specified by a future feature.

## Safe mode

`--safe-mode` starts the application with user plugin loading disabled and
disables automatic persistence for recovery-sensitive state. It must not
overwrite an externally supplied `LFS_SAFE_MODE=1` when starting normally.
Explicit user actions such as keymap export remain separate from automatic
persistence and should remain available in safe mode.

## Portable mode

Portable storage is supported only when the corresponding command-line option
is registered and wired to `UserPathOptions`. Documentation and parser
behaviour must remain synchronized: an advertised flag must work, and an
internal-only option must not be presented as a user-facing CLI feature.

## Project state boundary

Global preferences belong to the user root. Layout, open tabs, panel
arrangement and other project/session state should be owned by the `.licht`
project format once that upstream work lands. This boundary prevents a
project-specific arrangement from unexpectedly changing the user's global UI.
