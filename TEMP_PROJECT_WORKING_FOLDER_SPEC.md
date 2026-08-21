# Working folder + temp project: no project creation at training start

Status: frozen spec v2 (2026-08-21, after design critique `.codex_tmp/temp_project/critique/report.md`). Owner decisions from Discord (#lichtfeld-sw-dev, 2026-08-21) and direct instruction.

## Problem

- Starting training in an untitled session force-creates `<output_path>/project.licht` (`ProjectLifecycle::prepareTrainingStartProject`, `src/visualizer/project/project_lifecycle.cpp:2822-2892`). The second run on the same dataset then pops "Overwrite existing project?" (`training_panel.py:2239`, `training.overwrite.existing_*`). Users dismiss it every time.
- Untitled crash scratch already exists at a fixed `<root>/recovery/<uuid>.licht` and is not configurable. Users do not know where LichtFeld Studio keeps its temp data.

## Owner decisions (not negotiable)

1. A configurable **Working folder** preference. Default on both OSes is the existing `.lichtfeld` root: `~/.lichtfeld` (Linux), `%USERPROFILE%\.lichtfeld` (Windows); `LFS_HOME` and portable mode keep overriding the default root exactly as `UserPaths::resolve` does today.
2. The **temp project** lives in `<working folder>/tmp`. It is always overwritten without asking (autosave, training snapshots, new sessions). No dialog is ever shown for writes into it.
3. Training start never creates a project and never asks. An untitled session trains into the temp project. Only the user creates a project (Save / Save As). An explicitly created or opened project has priority: once the session is titled, everything routes to that project as today and the temp file is removed.
4. The user is informed (and offered Save) on application close and on every action that wipes the current work, if the work is dirty. Nothing else prompts.

## Behaviour

### A. Preference `working_directory`

- Stored in `preferences.json` as `"working_directory": "<absolute utf8 path>"`. Absent or empty string means default. Default resolution: `UserPaths::rootDir()` (new accessor, `config_dir_.parent_path()`, same root `recoveryDir()` derives from today).
- C++: `UserPreferences::setWorkingDirectory(path)`, `workingDirectory()` (effective, absolute), `workingDirectoryPreference()` (raw, empty when default), `clearWorkingDirectory()`; free-function wrappers like the existing ones. Add `"working_directory": ""` to `writeDefaultPreferences` so `--reset-preferences` restores the default.
- `UserPaths`: add `rootDir()`. Keep `recoveryDir()` only as the legacy scan location (see F). Remove `recoveryDir()` from `ensureDirectories()`; the temp directory is created lazily on first use.
- Python (`lf.ui`, registered where the other preference bindings live): `get_working_directory()`, `get_working_directory_preference()`, `get_default_working_directory()`, `get_temp_project_directory()`, `set_working_directory(path) -> str` (empty string on success, user-facing error text on failure), `clear_working_directory()`.
- Validation in `set_working_directory`: expand to absolute, `create_directories` if missing, probe writability (create and remove a temp file). On failure nothing is persisted and the error is returned. Never silently fall back.
- Safe mode: preferences are not persisted; default applies. No extra handling.
- GUI: Preferences > General gets a "Working folder" row: label, text input (draft, commit on Enter or the OK/Apply button, same idiom as `mcp_port`), "Browse…" (`lf.ui.open_folder_dialog`), "Use default". Below it a hint line: "Temporary projects: <effective temp path>". Commit failure shows `lf.ui.message_dialog(..., "error")` and reverts the draft. "Reset this section" for General also resets the working folder to default.
- A change takes effect for the next untitled session (New Project or restart). A live session keeps its already-bound temp file; the hint text says so ("Applies to new sessions").
- Locale keys (all 10 files under `src/visualizer/gui/resources/locales/`): `preferences.working_directory`, `preferences.working_directory_hint`, `preferences.working_directory_use_default`, `preferences.working_directory_applies_next_session`, `preferences.working_directory_invalid`. Technical path text stays untranslated.

### B. Temp project directory

- `ProjectLifecycle::scratchAutosaveDirectory()` returns `<working folder>/tmp` resolved from `UserPreferences` at bind time, unless the lifecycle was constructed with an explicit `settings_path` (tests), in which case it stays `{settings.parent}/tmp`. Rename the member `recovery_directory_` to `temp_project_directory_` and the test fixture directory from `recovery` to `tmp`.
- Once `scratch_autosave_path_` is bound for a session it is never re-resolved for that session, even if the preference changes.
- File naming stays `<project_uuid>.licht` with the `.lock` sibling (multi-instance safe). Locks, sweep, scan, `is_scratch_autosave_path` keep working against the resolved temp directory.
- The directory is created before the first scratch write or trainer bind.

### C. Training start in an untitled session

Representation (decided): the temp project is a **titled document internally** (`document_->source_path()` == `<temp>/<uuid>.licht`) that the user sees as untitled. This keeps the only checkpoint-safe protocol in the tree: trainer snapshots append to the master, background light autosave writes the `<temp>/<uuid>.licht.autosave` sidecar. Two writers never replace the same file.

- New predicate `ProjectLifecycle::isTempProject()` = `document_ && document_->source_path() && is_scratch_autosave_path(*source_path, temp_project_directory_)`. Not a sticky bool. `temp_project_directory_` is frozen at construction / `newProject` from the preference (absolute, normalized; case-insensitive compare on Win32 inside the predicate path comparison) so a live session never moves.
- `hasSourcePath()` returns false when `isTempProject()` (keeps `recovered_master_path_` semantics). Every site that reads `document_->source_path()` directly for *user* semantics must use `isTempProject()`: `save()` refuses with the untitled "use Save As" error; `beginOrPollCloseSave()` returns `NeedsPrompt` for temp even with `auto_save_on_close`; `info().path` is empty for temp; user and idle compaction refuse temp; `menuInfo` / title report untitled.
- `rememberProject()` itself skips any path for which `is_scratch_autosave_path` holds, so adopt, settle and open can never put temp in MRU.

`prepareTrainingStartProject()` for `!hasSourcePath()`:

1. No `saveAs` to `dataset.output_path`, no dependency on it.
2. `lockScratchAutosave()` (creates `<temp>`, binds `scratch_autosave_path_`, acquires `scratch_lock_`).
3. If the document is still unbound: one non-user first write to `*scratch_autosave_path_` that binds `source_path` (`leave_unbound=false`, `allow_existing_destination_replacement=true`, writer lease = `scratch_lock_`). It must NOT go through `SaveAs`/`ExplicitSave` settlement: `settleProjectWrite` deletes the scratch and remembers MRU on those purposes. Either a dedicated purpose or an explicit "destination is scratch" branch in settle that skips `removeScratchAutosave` and `rememberProject`.
4. `bindTrainerSnapshotTarget()` (default destination = source path) with the snapshot context lease = recovery lock if a recovery session exists, else `scratch_lock_` when it owns the destination.
5. Grant `{on_completion, at_step_boundaries}` as today.

Titled sessions are unchanged.

- `startAutosave()` on a temp session takes the titled branch (light sidecar) and passes `scratch_lock_` as the sidecar writer lease (today only the recovery lease is passed; without it the sidecar `acquire()` fails against the session-long scratch lock).
- `adoptCompletedTrainingSnapshot` / `adoptSettledTrainerPublishOntoCurrentMaster`: no `forget_source_path`, no MRU (covered by the helper guard). Session stays temp-titled, trainer stays bound.
- `hasDirtyProject()`: after the blank-untitled exception, `isTempProject()` is dirty, and `canFlushFinishedTrainerSnapshot()` is dirty. Without this a completed (Finished, not Paused) temp session looks clean at close because the trainer never emits scene mutations and close suppresses silent adoption; the `NotDirty` branch would then delete the temp file with the checkpoint.
- `trainingStartOverwriteConflict()`: `!hasSourcePath()` (untitled or temp) returns `nullopt` unconditionally; delete the `<output>/project.licht` probe. Titled branches unchanged (a saved project's checkpoint is real data; that dialog stays).
- `openScratchRecovered`: re-lock and keep the temp master bound (temp-titled recovered session) so a recovered checkpoint stays on a bound master.

Headless CLI training (`training_setup.cpp`) is out of scope and keeps binding `<output>/project.licht`.

### D. Explicit project has priority

- Save As from an untitled or temp session (idle, paused, running, completed) migrates the work to the chosen path through the existing routes (`saveAs`, `TrainingExplicitSave`, finished-trainer flush) and `settleProjectWrite` deletes the temp master, its `.autosave` sidecar and the `.lock`. Must work after a training run that already published checkpoints into the temp file: the saved project contains the CKPT chapter.
- Opening a project or New Project removes the session's temp file as today.
- Dirty untitled + `auto_save_on_close` on still prompts (no path to save to). Unchanged.

### E. Prompts on every wipe (Save / Discard / Cancel)

One shared Python helper (new module `src/python/lfs_plugins/unsaved_work.py`):

```
confirm_discard_work_then(title, on_proceed, *, message_key="exit_popup.unsaved_warning")
```

Order: (1) if `lf.project_is_dirty()`: dialog `[Save | Save As…, <title> without saving, Cancel]`. Save on a titled project: `lf.project_save()` then proceed. Save on untitled: `lf.project_save_as("", wait=True)`, then proceed once `lf.project_has_path()` (reuse the `_schedule_start_once_project_bound` idiom from `training_panel.py:2286`); cancelled picker = Cancel. (2) if training is active: the existing stop-training confirm, then `on_proceed(stop_training=True)`.

`file_menu._confirm_discard_then` / `_confirm_switch_then` and `import_panels._confirm_save_before_dataset_load` are replaced by the helper (one dialog shape everywhere; the dataset panel keeps its own button labels if the locale keys already exist, otherwise reuse).

Entry points routed through the helper (from the wipe audit, `.codex_tmp/temp_project/audit_wipes/report.md`):

| Entry point | Change |
|---|---|
| File > New / Open / Open Recent, drag-drop `.licht`, Training panel Clear | already prompt; switch to the shared helper (gain the Save button) |
| Dataset import panel | already prompts; switch to the shared helper |
| File > Import PLY/SOG/RAD/splat, File > Import Mesh, scene-graph Add, drag-drop splat/mesh, Asset Manager Load / Load New, `lf.load_file` from the GUI | gate at the owning layer: `cmd::LoadFile` gains `discard_changes` and `replace` (default false). A shared `VisualizerImpl::preflightLoadFileWipe(const cmd::LoadFile&)` is called from BOTH consumers: `DataLoadingService::handleLoadFileCommand` (splat/mesh; subscriber must forward the full command) and the dataset consumer in `async_task_manager.cpp` before `startAsyncImport` (dataset loads return early from the splat handler today). `would_clear = is_dataset || replace || content == Dataset`. Gate when `would_clear && !discard_changes && (hasDirtyProject() || isTrainingActive() || isCompletionPending())`: do not load; emit `ShowLoadFileConfirmation{paths, is_dataset, replace}` instead. Python subscribes (like `on_request_exit`), shows the helper, and re-issues `lf.load_file(..., discard_changes=True, stop_training=...)` in the original order. Multi-file drops are one confirmation for the batch: `input_controller.cpp` collects the splat/mesh paths and, when the gate would fire, emits one `ShowLoadFileConfirmation{paths}` and no `LoadFile`; Python re-issues them in order with `discard_changes=True` (`replace` only on the first of a Load New batch). Asset Manager "Load New" stops calling `lf.clear_scene()` and passes `replace=True` on the load. Loads that only add to a SplatFiles scene never prompt. |
| Checkpoint resume panel (`import_panels.py:1465`) | helper before `lf.load_checkpoint_for_training` |
| Replace-load (dataset, Asset Manager Load New, splat/mesh onto a Dataset scene, checkpoint resume) | After the wipe helper, C++ starts a fresh untitled session (`newProject` DiscardChanges; live training/import parameters preserved). Adding onto a SplatFiles scene does not. |
| Training panel Reset | helper with title "Reset training" before `lf.reset_training()` |
| Sequencer PLY-sequence load from the sequencer folder picker | route through the same C++ gate/event if it is reusable without new plumbing; otherwise document as follow-up in the report |

Not gated (machine callers, already marked destructive): MCP `scene.load_*`, `project_open(discard_changes=true)`, Python `lf.clear_scene()` / `lf.new_project(True)` from scripts, CLI flags, ForceExit/interrupt.

Exit: unchanged. It already prompts for dirty work (Save / Save As, Discard, Cancel) and for active training; a completed or paused-progressed untitled training must be dirty so the prompt appears (test required, see below).

### F. Startup hygiene and legacy recovery folder

- Startup sweeps and scans the configured temp directory and, if it exists and differs, the legacy `UserPaths::recoveryDir()`. For each scratch Offer, also run `inspect_autosave_recovery(master)` so a `<uuid>.licht.autosave` sidecar overlay is used as `selected_path` when newer; the candidate keeps `untitled_scratch=true`. Newest wallclock wins across both directories.
- After the startup decision (recover, skip, or nothing offered), every other unlocked scratch file (and its sidecar/lock) in both directories is deleted (log one INFO line per file). Files locked by another live instance stay. Temp never accumulates old sessions.
- Nothing else changes about crash recovery.

## Non-goals

- Moving cache, logs, plugins, venv, ONNX models or the pipelined-loader spill. Only the temp project directory follows the working folder.
- Changing `dataset.output_path` semantics (PLY exports, eval images).
- Headless training project binding.
- MCP/Python API wipe prompts.
- `reopen_last_project` (dead setting).

## Implementation map

- `src/core/include/core/user_paths.hpp`, `src/core/user_paths.cpp`: `rootDir()`, default pref, `ensureDirectories` change.
- `src/visualizer/preferences.hpp/.cpp`: working directory accessors + wrappers + validation helper.
- `src/visualizer/project/project_lifecycle.hpp/.cpp`: B, C, D, F.
- `src/visualizer/scene/data_loading_service.cpp`, command struct for `LoadFile` (find it), event for `ShowLoadFileConfirmation`, `src/visualizer/input/input_controller.cpp` (batch drop), Python binding for the subscription and the `discard_changes` / `replace` args in `src/python/lfs/module.cpp`.
- `src/python/lfs/py_ui.cpp` (or `py_ui_theme.cpp`): working directory bindings.
- `src/python/lfs_plugins/preferences_panel.py`, `src/visualizer/gui/rmlui/resources/preferences.rml`, 10 locale files.
- `src/python/lfs_plugins/unsaved_work.py` (new), `file_menu.py`, `import_panels.py`, `training_panel.py`, `asset_manager_panel.py`.
- `src/python/stubs/lichtfeld/__init__.pyi` for every new binding.
- Docs: `docs/preferences-and-user-storage.md` gets the working folder and temp directory.

## Tests (required, real fixtures, no synthetic data)

gtest (`tests/test_visualizer_post_work.cpp`, `VisualizerImplResetTest`):

1. `StartTrainingUntitledBindsTempProjectAndStaysUntitled`: replaces `StartTrainingPreparesProjectAndGrantsSaves`. After prepare: `!hasSourcePath()`, `trainer->bound_project_path()` equals the scratch path inside `{settings.parent}/tmp`, policy granted, no `project.licht` under `output_path`.
2. `UntitledTrainingSnapshotAdoptionKeepsSessionUntitledAndOutOfMru`: trainer publishes one snapshot into the temp file; after adoption `!hasSourcePath()`, `document_->checkpoint_uuids()` non-empty, MRU unchanged, trainer still bound to the scratch path.
3. `SaveAsAfterUntitledTrainingMigratesTempIncludingCheckpoint`: Save As after a published temp snapshot; destination contains CKPT; temp file and `.lock` gone.
4. `UntitledStartConflictNeverReportsExistingOutputProject`: `<output>/project.licht` exists, `trainingStartOverwriteConflict()` is `nullopt`.
5. `CompletedUntitledTrainingBlocksCleanClose`: completed (Finished, not Paused) temp session with an adopted or still-unflushed checkpoint → `hasDirtyProject()` true and `beginOrPollCloseSave()` is `NeedsPrompt` even with `application_close_pending_` set and `auto_save_on_close` on; the temp file is not deleted.
5b. `TempProjectSaveRefusesAndStaysOutOfMru`: `save()` on a temp session fails like untitled; MRU unchanged after settle and adopt; `info().path` empty; `hasSourcePath()` false.
5c. `TempSessionLightAutosaveWritesSidecarWithScratchLease`: during training on a temp session `startAutosave` writes `<uuid>.licht.autosave`, the master keeps its CKPT, no `Unavailable` lock error.
6. `WorkingDirectoryPreferenceChangeAppliesToNextSession`: bound scratch path stays after the preference changes; a new session binds under the new directory.
7. `StartupPrunesOlderUnlockedScratchFilesAfterOffer` and `StartupScansLegacyRecoveryDirectory`.

Python (`tests/python/test_training_panel_regressions.py` and a new `tests/python/test_unsaved_work.py`): the helper's three outcomes, Reset prompt, checkpoint panel prompt, load-file confirmation re-issue with `discard_changes=True`.

Preferences: `tests/test_user_paths.cpp` (or the existing preferences test file) for `rootDir()`, default pref in `writeDefaultPreferences`, `set_working_directory` validation (unwritable path is rejected, nothing persisted).

## Verification (Claude runs these; Grok must run them too and paste the output)

```
cmake --build build -j16
./build/lichtfeld_tests --gtest_filter='VisualizerImplResetTest.*:ProjectLifecycleSettingsTest.*:UserPaths*.*:*Preferences*'
./build/lichtfeld_format_tests --gtest_filter='ProjectRecoveryScratch.*'
./build/vcpkg_installed/x64-linux/tools/python3/python3.12 -m pytest tests/python/test_training_panel_regressions.py tests/python/test_unsaved_work.py -q
clang-format on every touched C++ file
```

GUI validation (Claude): launch, load `data/bicycle` via the import panel, start training, confirm no dialog, confirm `~/.lichtfeld/tmp/<uuid>.licht` grows at a save step, stop, File > Exit → prompt with Save As / Discard / Cancel; Save As → temp gone, project has checkpoint. Change the working folder in Preferences, New Project, train again → new temp location used. Zero error noise in the log.
