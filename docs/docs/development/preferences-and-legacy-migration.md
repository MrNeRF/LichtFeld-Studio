---
sidebar_position: 5
---

# Preferences and legacy GUI data migration

## Scope

This page documents the current application-level Preferences panel and the
one-time migration of recognised legacy GUI settings. It is intentionally not
a roadmap for presets, future reset commands, project state, or other future
configuration work.

## Preferences panel

Open **Edit > Preferences** to change application-level settings. The panel
uses a left sidebar and keeps its action bar at the bottom of the content area.
Section headers are collapsible and open when Preferences is opened.

| Section | Current controls |
| --- | --- |
| General | persistent application language |
| Appearance | theme and UI scale, including automatic scale |
| Input | camera navigation mode and axis-view snap, with independent choices for remembering each setting |
| Interface | backed-up layout reset |
| Legacy & Migration | inspection, manual migration, and guarded archival of recognised legacy GUI files |

Changes apply immediately. The panel uses the active UI theme tokens rather
than fixed colours, and all panel strings are localised in every bundled
language.

### Reset behaviour

The action bar provides a reset for the current applicable section and a
confirmed **Reset all settings** action. The current implementation resets:

- General: language to English.
- Appearance: built-in dark theme and automatic UI scale.
- Input: built-in orbit navigation and disabled axis-view snap, while clearing
  both independent “remember” choices.
- Interface: the saved GUI layout through the layout reset API.

Layout reset backs up the existing layout before removing it, then applies the
runtime default layout immediately. The Legacy & Migration section has no
generic reset because its files are handled explicitly.

## User-owned GUI storage

`UserPaths` is the shared path service for current GUI preferences and layout.
On Windows, the default root is `%USERPROFILE%\.lichtfeld`; configuration
files are stored under `config/`, with data, cache, logs, plugins, and virtual
environment under their corresponding directories. Linux keeps the normal XDG
config/data/cache/state locations for these categories.

An explicit user root, `LFS_HOME`, or portable mode uses one isolated root and
does not inspect the machine default profile for legacy GUI data.

Current GUI files include:

```text
config/preferences.json
config/layout.json
config/keymaps/
data/migrations/legacy-gui-settings-v1.json
data/backups/
```

## Legacy GUI migration

The migration recognises only the former GUI artefacts below. It never treats
an entire old directory as migratable.

| Legacy artefact | Current destination |
| --- | --- |
| `layout.json` | `config/layout.json` |
| `theme_preference` | `config/preferences.json` |
| `ui_scale` | `config/preferences.json` |
| `input_profiles/*.json` | `config/keymaps/` |

On the normal default profile, LichtFeld considers only these legacy candidate
locations:

- Windows: `%APPDATA%\LichtFeldStudio`, `%LOCALAPPDATA%\LichtFeldStudio`,
  and the earlier `%USERPROFILE%\.lichtfeld` location.
- Linux: `$XDG_CONFIG_HOME/LichtFeldStudio` when set,
  `~/.config/LichtFeldStudio`, and the earlier `~/.lichtfeld` location.

Each recognised file is parsed and validated before copying. A valid modern
destination always wins. Invalid legacy input is left in place and reported as
invalid; unrecognised files are never touched.

After an automatic attempt, the migration manifest records the inspected
sources, destinations, and results. Its presence prevents legacy files from
being scanned again automatically, including after a layout reset. This avoids
an old layout being silently imported again.

Automatic migration is disabled for explicit, `LFS_HOME`, and portable roots,
and is skipped in safe mode. The **Legacy & Migration** Preferences section can
still inspect and manually migrate recognised files when the user explicitly
requests it.

### Legacy archival

**Archive legacy files** is available only for recognised files already
represented by a valid current destination. Before removing an old original,
LichtFeld copies it to a timestamped directory under `data/backups/` and
verifies the copy. Invalid, unrecognised, or not-yet-migrated files are never
archived by this action.

## Command-line recovery flags

The following flags are process-local startup controls. They are not stored in
training configuration files, so loading a saved training configuration cannot
make a later normal launch enter recovery mode or reset user settings.

| Flag | Behaviour |
| --- | --- |
| `--safe-mode` | Starts with user plugin loading disabled, skips normal GUI preference and layout persistence, skips automatic legacy migration, and marks the window as Safe Mode. It does not overwrite normal user files. |
| `--reset-preferences` | Backs up the current `config/preferences.json` under `data/backups/` and restores application preferences to their defaults before normal startup. If no preferences file exists, no backup is created. |
| `--reset-layout` | Backs up and removes `config/layout.json`, applies the runtime default layout, and suppresses automatic legacy layout migration for that launch so an old layout cannot immediately return. If no layout file exists, no backup is created. |
| `--reset-all-settings` | Performs both `--reset-preferences` and `--reset-layout` in the same launch. Each existing file receives its own backup, and automatic legacy layout migration is suppressed for that launch. It does not reset keymaps, plugins, caches, datasets, or project files. |

The reset operations write their outcome and the backup path, when one was
created, to the startup log.
