# Truly non-blocking startup plugin loading

Design snapshot: `vulkan-hardening` at `be462715df32c92ce15fa1d531ef36ba07cebe36` on 2026-07-11.

Sections 1–6 preserve the analysis reviewed before implementation. The implementation
landed on `plugin-async-loading` from base `f2734dfd3`; the implementation and validation
record is appended below.

## Decision

Use one owned, sequential startup worker that executes the complete existing plugin load pipeline—discovery, virtual-environment preparation, `uv sync`, module execution, and `on_load()`—off the viewer/render thread. The main thread must only read atomic state, render progress, and run explicitly audited, bounded graphics-thread work items.

This is not a novel threading model for the plugin system. The marketplace already calls the same `PluginManager.load()` from a daemon Python thread (`src/python/lfs_plugins/plugin_marketplace_panel.py:1358-1403,1470-1488`). The startup path should adopt that established execution model but give it process-owned lifecycle, cancellation, progress, failure rollback, and shutdown coordination.

Two guardrails are essential:

1. Normal plugin startup code must remain registration-oriented. Panel, operator, menu, capability, tool, and event registration can run on the worker because the relevant registries are GIL-serialized and/or mutex-protected, and retained UI resources are created lazily. APIs that actually touch ImGui, RmlUi documents, Vulkan textures, native dialogs, scene mutation, or renderer state must either marshal a bounded payload through the viewer work queue or reject the call with a clear affinity error.
2. A single CPython interpreter cannot guarantee that arbitrary Python UI work remains runnable while a plugin imports a native extension that holds the GIL. The render loop therefore must not wait for the GIL during a GIL-heavy preload phase. Existing preload gates should be made phase-aware: keep the native shell, viewport, window events, close path, and cached UI responsive; defer Python callbacks when the worker owns the import/activation phase.

## 1. Current-state map

### 1.1 Startup call graph and the freeze

The viewer and renderer run through callbacks installed on one main-loop thread (`src/visualizer/visualizer_impl.cpp:764-770`). `VisualizerImpl` records that thread as `viewer_thread_id_` in its constructor (`src/visualizer/visualizer_impl.cpp:90-98`).

After the first GUI frame, `VisualizerImpl::update()` schedules autoload and immediately processes a preload step (`src/visualizer/visualizer_impl.cpp:1092-1103`):

```text
VisualizerImpl::update()                                      main/viewer thread
  -> python::preload_user_plugins_async()                     runner.cpp:789-801
       only flips scheduled/running flags
  -> python::process_plugin_preload_step()                    runner.cpp:803-900
       -> discover_enabled_plugins_locked()                   runner.cpp:499-595
       -> GilAcquire                                          runner.cpp:876-883
       -> load_single_plugin_locked(name)                     runner.cpp:597-629
       -> lichtfeld.plugins.load(name)                        py_plugins.cpp:51-53
       -> PluginManager.load(name)                            manager.py:276-336
```

The work-queue drain comes *after* the preload call (`src/visualizer/visualizer_impl.cpp:1105-1125`), so a blocked plugin step also delays MCP/viewer work already posted to the main thread.

`preload_user_plugins_async()` is therefore asynchronous in name only. It performs no work and creates no worker (`src/python/runner.cpp:789-801`). `process_plugin_preload_step()` limits the batch to one plugin per frame, but the unit of work is still an unbounded synchronous `PluginManager.load()` (`src/python/runner.cpp:869-883`). A single plugin is enough to stop event processing and rendering for minutes.

### 1.2 Exact dependency-install path

Dependency installation is not an incidental side effect of importing a plugin module. It is already a distinct, separable phase inside `PluginManager.load()`:

```text
PluginManager.load(name)                                     manager.py:276-336
  -> state = INSTALLING                                      manager.py:292-295
  -> PluginInstaller.ensure_venv()                           manager.py:295-297
       -> uv venv ...                                        installer.py:458-528
       -> subprocess.run(...)                                installer.py:499-504
  -> PluginInstaller.install_dependencies(progress_fn)       manager.py:298-300
       -> validate .deps_installed stamp                     installer.py:530-558
       -> uv sync --project ... --python ...                 installer.py:560-581
       -> subprocess.Popen(...)                              installer.py:586-594
       -> blocking stdout.readline loop + proc.wait()         installer.py:595-601
       -> touch .deps_installed only after success            installer.py:603-608
  -> state = LOADING                                         manager.py:302
  -> _load_module(plugin)                                    manager.py:303,354-396
       -> compile and exec entry point                        manager.py:372-387
  -> plugin.module.on_load()                                 manager.py:306-308
  -> state = ACTIVE                                          manager.py:310
```

The per-plugin environment and command are documented consistently in `docs/plugin-system.md:196-205`. An existing `.deps_installed` stamp avoids another sync unless `pyproject.toml` or `uv.lock` is newer (`src/python/lfs_plugins/installer.py:530-545`). A failed or interrupted sync does not write the stamp, so a later launch can repair the environment.

The observed `module.cpp:1839 [Python] [densification] + triton...` line takes this path:

```text
uv stdout line
  -> installer on_progress(line)                             installer.py:595-600
  -> manager default progress_fn logs "[name] line"          manager.py:298
  -> _LfLogHandler.emit() calls lichtfeld.log.info           manager.py:55-69
  -> C++ LOG_INFO("[Python] {}", msg)                        module.cpp:1835-1841
```

The locally installed incident plugin confirms why both dependency and import phases matter. Its manifest declares Torch and other large packages (`~/.lichtfeld/plugins/densification/pyproject.toml:10-22`), its entry point imports the panel package before `on_load()` (`~/.lichtfeld/plugins/densification/__init__.py:9-19`), and that panel imports NumPy and Torch at module scope (`~/.lichtfeld/plugins/densification/panels/densification.py:20-31`). Its `on_load()` itself is short registration work (`~/.lichtfeld/plugins/densification/__init__.py:24-30`). This local path is incident evidence, not a repository contract.

### 1.3 GIL facts

The current branch has `src/python/gil.hpp`, not the `gil_guard.hpp` name in the issue description. Its discipline is:

- `can_acquire_gil()` requires both an initialized interpreter and `g_gil_state_ready` (`src/python/gil.hpp:13-15`).
- `GilAcquire` wraps `PyGILState_Ensure/Release` (`src/python/gil.hpp:17-28`).
- Initialization parks the initializing/main Python thread state with `PyEval_SaveThread()` and only then publishes readiness (`src/python/runner.cpp:703-710`).
- `acquire_gil_main_thread()` restores the saved state and `release_gil_main_thread()` saves it again (`src/python/python_runtime.cpp:646-664`).

There is already a C++-owned Python worker precedent: the embedded REPL starts `std::thread`, enters `GilAcquire`, runs Python, and releases the GIL when its scope ends (`src/python/runner.cpp:1016-1024,1103-1112`).

For the dependency phase, CPython's blocking file-descriptor reads and `waitpid` release the GIL around the OS calls. The relevant CPython 3.12 implementations are `_Py_read` in [`Python/fileutils.c`](https://github.com/python/cpython/blob/v3.12.10/Python/fileutils.c) and `os.waitpid` in [`Modules/posixmodule.c`](https://github.com/python/cpython/blob/v3.12.10/Modules/posixmodule.c); both use `Py_BEGIN_ALLOW_THREADS`. Python's subprocess contract also supports bounded wait/terminate/kill cleanup ([Python 3.12 `subprocess` documentation](https://docs.python.org/3.12/library/subprocess.html#popen-objects)).

That does **not** help the current GUI. The OS thread inside `readline()`/`wait()` is the viewer thread itself; releasing the GIL lets another Python thread run, but it does not make the C++ viewer thread return to `update()` or `render()`. The dependency freeze is primarily an execution-affinity bug, not a GIL-starvation bug.

The import phase is different. Python bytecode normally yields the GIL periodically, but native extension initialization is not required to release it. A worker import of Torch or another extension can therefore delay any main-thread code that tries to acquire the same interpreter GIL. The main render loop can remain responsive only if it does not perform a blocking GIL acquisition during that phase.

The GUI already encodes that intent, but with a coarse batch-wide flag:

- Python animation and viewport overlay demand are disabled during preload (`src/visualizer/visualizer_impl.cpp:1284-1295`).
- Python frame callbacks, signals, and panel sync are skipped (`src/visualizer/visualizer_impl.cpp:1360-1366,1455-1464,1527-1529`).
- Python/UI graphics callbacks are not flushed (`src/visualizer/gui/gui_manager.cpp:5420-5426`).
- Non-native panels, viewport hooks, modals, and popups are suppressed (`src/visualizer/gui/gui_manager.cpp:5683-5691,6263-6268,6344-6350`).

Moving the load to a worker makes those guards useful; today the viewer blocks before they can protect anything.

### 1.4 Registration affinity

The official plugin contract puts runtime registration in `on_load()` (`docs/plugins/getting-started.md:94-115`), and the generated template follows it (`src/python/lfs_plugins/templates.py:33-44`). The current registration paths divide as follows.

#### Worker-compatible registration paths

- **Panels:** `PyPanelRegistry` owns a mutex (`src/python/lfs/py_ui.hpp:672-694`). Registration constructs a Python instance and a lazy adapter, then inserts it into the mutex-protected C++ `PanelRegistry` (`src/python/lfs/py_ui_panels.cpp:310-480`; `src/visualizer/gui/panel_registry.cpp:205-240`). The RmlUi host is not created by the adapter constructor; creation occurs later during draw (`src/python/lfs/rml_python_panel_adapter.cpp:435-448,463-470`; `src/python/lfs/rml_im_mode_panel_adapter.cpp:17-48`). Unregistration already detects an off-graphics-thread caller and schedules graphics cleanup (`src/python/lfs/py_ui_panels.cpp:483-540`).
- **Operators:** Python instances are protected by `g_python_operator_mutex`; operator and property-schema registries have their own mutexes (`src/python/lfs/py_ui.cpp:310-329,895-1037`; `src/visualizer/operator/operator_registry.hpp:41-99`; `src/visualizer/operator/operator_registry.cpp:105-119`; `src/visualizer/operator/property_schema.cpp:14-23`).
- **Menus:** `PyMenuRegistry` builds the Python instance under the GIL and mutates its registry under `mutex_` (`src/python/lfs/py_ui_menus.cpp:298-347`; `src/python/lfs/py_ui.hpp:825-856`).
- **Capabilities and tools:** these are Python registry mutations and are GIL-serialized. The documented full plugin registers both from `on_load()` (`docs/plugins/examples/full_plugin/__init__.py:46-75`).
- **Training/event subscriptions:** `EventBridge` and `ControlBoundary` registrations are mutex-protected (`src/core/event_bridge/event_bridge.cpp:14-29`; `src/core/event_bridge/scoped_handler.cpp:30-35`; `src/core/event_bridge/control_boundary.hpp:47-68`).

The strongest practical evidence is the marketplace: its worker calls `mgr.load()`, which includes `_load_module()` and `on_load()`, not merely dependency installation (`src/python/lfs_plugins/plugin_marketplace_panel.py:1358-1403,1470-1488`). Startup doing the same does not introduce a new registration thread by itself.

#### Graphics/main-thread-bound paths

Registration safety must not be generalized to every `lichtfeld` binding:

- RmlUi document/element mutation and actual panel-host creation happen on the graphics/render path.
- Dynamic texture upload performs CUDA/Vulkan work directly (`src/python/lfs/py_ui.cpp:166-205`).
- Icon loading calls the UI texture service immediately (`src/python/lfs/py_ui.cpp:350-362,418-449`), while texture destruction explicitly marshals to the graphics callback queue when off-thread (`src/python/lfs/py_ui.cpp:452-515`). Creation currently lacks the corresponding affinity guard.
- ImGui state queries and immediate widgets use the shared ImGui context; for example `get_display_size` reads `ImGui::GetMainViewport()` directly (`src/python/lfs/py_ui.cpp:3378-3383`).
- Native file dialogs, direct scene mutations, renderer capture/settings, and input operations have host-specific affinity and cannot be assumed safe merely because the caller owns the GIL.

`register_ui()` records the graphics thread during binding initialization (`src/python/lfs/py_ui.cpp:3387-3401`). The runtime also has a graphics callback queue (`src/python/python_runtime.hpp:673-678`; `src/python/python_runtime.cpp:1124-1141`), and the viewer exposes a more general work queue with wake-up and shutdown cancellation (`src/visualizer/include/visualizer/visualizer.hpp:46-51,67-70`; `src/visualizer/visualizer_impl.cpp:773-793,1775-1798`). Those are the correct mechanisms for audited, bounded main-thread operations.

Posting arbitrary plugin import or arbitrary `on_load()` to that queue is not acceptable: it recreates the original unbounded-main-thread bug. A posted task must contain only a known bounded native operation. If a worker must wait for its result, it must release the GIL first, just as viewport capture releases the GIL before waiting on viewer work (`src/python/lfs/py_rendering.cpp:99-136`).

### 1.5 Progress and startup-overlay behavior

The progress plumbing is already cross-thread-friendly if its lifetime invariant is preserved:

- `notify_startup_plugin_load_state()` invokes the registered callback and requests a redraw (`src/python/python_runtime.cpp:248-260`).
- The callback forwards to `GuiManager::setStartupPluginLoadState()` (`src/visualizer/visualizer_impl.cpp:217-239`).
- `StartupOverlay::setPluginLoadState()` copies state under `plugin_load_mutex_` (`src/visualizer/gui/startup_overlay.cpp:210-229`; mutex declaration at `startup_overlay.hpp:88-93`).
- Active plugin loading keeps the overlay animating (`src/visualizer/gui/startup_overlay.cpp:231-247`).

The callback and wake function are raw function pointers, not atomics. Worker use is safe only if startup preload is fully stopped and joined before `VisualizerImpl` clears callbacks or destroys `GuiManager`. The current callback cleanup runs in reverse registration order (`src/visualizer/visualizer_impl.hpp:222-232`), but there is no worker today to coordinate.

There is also a separate UI-interactivity issue. While the overlay is visible and plugin loading is incomplete, `GuiManager` intentionally blocks underlay input (`src/visualizer/gui/gui_manager.cpp:5429-5440`), and `VisualizerImpl` skips viewport input whenever the startup overlay is visible (`src/visualizer/visualizer_impl.cpp:1385-1389`). A background worker alone would make the window repaint, but it would not satisfy “fully interactive.” The loading phase should switch the startup overlay to a non-modal/compact background-progress mode or expose an immediate “Continue in background” action. Underlay blocking must be a dedicated overlay policy, not `visible && !plugin_load_complete`.

### 1.6 Failure isolation today

`PluginManager.load()` already catches each plugin exception, records `ERROR`, the message, and a traceback, logs it, and returns `False` (`src/python/lfs_plugins/manager.py:331-336`). The C++ loop logs the traceback and continues to the next name (`src/python/runner.cpp:608-628,876-883`). The existing Python tests cover `on_load()` failure and import cleanup (`tests/python/test_plugin_error_recovery.py:74-123`).

One gap matters for stronger isolation: if `on_load()` registers one item and then raises, the manager records `ERROR` but does not roll back partial panels, operators, capabilities, subscriptions, module entries, or `sys.path`. The recommended implementation must factor a best-effort rollback path usable from `LOADING`/`ERROR`, not only `unload()`'s `ACTIVE` state (`src/python/lfs_plugins/manager.py:456-518`).

### 1.7 Synchronous callers and headless behavior

`ensure_plugins_loaded()` is not only a GUI-startup helper. Its direct callers are:

| Caller | Current call site | Calling context/semantics |
|---|---:|---|
| `run_scripts()` | `src/python/runner.cpp:1152-1189` | Training initialization and `lf.scripts.run`; plugins must be active before scripts execute. It is called from trainer setup at `src/training/trainer.cpp:2599-2605` and from the Python binding at `src/python/lfs/py_scripts.cpp:65-85`. |
| `invoke_capability()` | `src/python/runner.cpp:1560-1573` | Capability must exist before invocation. GUI MCP posts this call to the viewer thread (`src/app/mcp_gui_tools.cpp:4257-4279`); headless MCP calls it directly (`src/mcp/mcp_training_context.cpp:112-127,700-723`). |
| `has_capability()` | `src/python/runner.cpp:1645-1649` | Synchronous registry query. |
| `list_capabilities()` | `src/python/runner.cpp:1678-1684` | GUI and headless MCP list calls; GUI list currently runs on the MCP server thread (`src/app/mcp_gui_tools.cpp:4282-4294`). |

`run_scripts()` and `invoke_capability()` currently acquire the GIL *before* calling `ensure_plugins_loaded()` (`src/python/runner.cpp:1157-1159,1187-1188,1560-1572`). If the new synchronous ensure simply waits for the startup worker, that ordering can deadlock: the waiter owns the GIL needed by the worker. Those functions must ensure/wait before their long-lived `GilAcquire`, and a Python binding that waits must use a `gil_scoped_release` guard.

Async startup is GUI-only because only `VisualizerImpl::update()` calls `preload_user_plugins_async()`. Headless routing bypasses the visualizer (`src/app/application.cpp:630-646`). The synchronous path must remain the owner in headless/script contexts; it must not silently turn into fire-and-forget behavior.

`LFS_PLUGIN_AUTOLOAD=off` currently disables only the GUI startup schedule (`src/python/runner.cpp:789-797`; accepted spellings at `runner.cpp:184-198`). It does not prevent a later explicit `ensure_plugins_loaded()` or `lf.plugins.load()`. Preserve that distinction unless product explicitly changes it.

### 1.8 Shutdown ordering today

The GUI sequence after the main loop exits is:

```text
viewer->run()                                                 application.cpp:581
python::finalize()                                            application.cpp:585
viewer.reset()                                                application.cpp:587
std::_Exit(0)                                                 application.cpp:592
```

This ordering is useful: the Python bridge and GUI still exist while `finalize()` runs. `finalize()` calls `join_plugin_preload()` before disabling new GIL acquisition, restoring the main thread state, clearing Python-owning callbacks/registries, and collecting garbage (`src/python/runner.cpp:937-973`). Today `join_plugin_preload()` is empty because preload runs on the main thread (`runner.cpp:937-939`).

The ordering must remain:

```text
request worker stop
terminate/kill and reap any uv child
worker unwinds Python frames and releases GilAcquire
join worker
set_gil_state_ready(false)
restore main PyThreadState
destroy Python-owning registries/callbacks
```

Calling cleanup while a worker still holds `GilAcquire` would race Python refcounts, `sys.modules`, plugin registries, and nanobind objects. Detaching a worker and continuing normal teardown is therefore not safe.

## 2. Candidate designs

### Candidate A — dependency ensure on a worker, existing import on the main thread

#### Threading model

The worker calls `ensure_venv()` and `install_dependencies()` for every enabled plugin. Once dependencies are ready, `process_plugin_preload_step()` retains its current one-plugin-per-frame behavior but skips installation and executes `_load_module()` plus `on_load()` on the viewer thread.

```text
worker:  discover -> venv -> uv sync ---------------------> ready queue
main:    frame -> frame -> frame -> import + on_load -> frame
```

#### GIL choreography

The worker enters `GilAcquire`; blocking `uv` I/O releases the GIL internally. The main thread does no install work. Later, the main thread enters `GilAcquire` for import/activation exactly as today.

#### Cancellation and shutdown

Dependency subprocesses can be made cancellable, and no worker is needed once the ready queue is empty. Shutdown during main-thread import is unchanged and cannot be interrupted safely.

#### Progress and failures

Dependency progress can be rich. Import/activation progress remains one stage per frame. An installation failure can mark the plugin failed and continue.

#### Risk and verdict

This is the smallest affinity change, but it does not meet the requirement. The local densification plugin imports Torch at module scope. A cold or cache-miss import can still consume far more than a frame, and arbitrary `on_load()` remains unbounded. “One plugin per frame” is not a time budget. Reject.

### Candidate B — dependencies and module import on a worker, `on_load()` on the main work queue

#### Threading model

The worker performs dependency setup and `_load_module()`, releases the GIL, then posts a `WorkItem` that reacquires the GIL and calls `on_load()` on the viewer thread. The worker waits for that bounded activation result before continuing.

```text
worker: deps -> import -> release GIL -> post activation ---- wait ---->
main:   frame -> frame -> drain WorkItem -> on_load -> frame
```

#### GIL choreography

The worker must leave its `GilAcquire` scope before posting/waiting. The main task acquires the GIL, looks the plugin up by name, invokes `on_load()`, records the result, and fulfills the promise. `WorkItem.cancel` must fulfill the promise if shutdown rejects or drains the task.

#### Cancellation and shutdown

`uv` cancellation is the same as Candidate A. Posted activation has a clean cancellation path through `VisualizerImpl::beginShutdown()` (`src/visualizer/visualizer_impl.cpp:773-793`). A worker already inside native module initialization remains cooperatively non-cancellable.

#### Progress and failures

Stages are naturally “installing,” “importing,” and “activating.” Failures in any phase remain per plugin. A failed main-thread post is a cancellation, not a plugin traceback.

#### Risk and verdict

This preserves the most conservative interpretation of `on_load()` affinity, but `on_load()` is arbitrary Python. A plugin can import more packages, allocate a model, open a file, or perform synchronous I/O there. Posting it wholesale to the viewer queue recreates the same unbounded-frame defect under a different function name. It also complicates rollback across two threads. Reject as the default; retain it only as a compatibility fallback for a specifically audited legacy plugin and show a warning that it cannot satisfy the frame guarantee.

### Candidate C — one owned worker executes the complete load pipeline (recommended)

#### Threading model

A process-owned `PluginAutoloadCoordinator` starts one `std::jthread` after the first GUI frame. That worker discovers enabled plugins and invokes the complete `PluginManager.load()` sequentially. It never holds a C++ coordinator mutex while acquiring the GIL. The main thread does no plugin load work and never waits from `update()`.

Sequential loading is intentional:

- concurrent imports mutate global `sys.path`, `sys.modules`, and currently even `builtins.__import__` (`src/python/lfs_plugins/manager.py:354-424`);
- multiple large `uv` downloads compete for disk/network and make progress less intelligible;
- per-plugin failure isolation does not require parallel plugins;
- one worker is enough to remove main-thread latency.

The manager should also serialize lifecycle operations from startup, marketplace, and the hot-reload watcher so a plugin cannot be loaded/reloaded twice concurrently.

#### GIL choreography

```text
Main/viewer thread                     Startup worker                    uv child
------------------                     --------------                    --------
Python init
PyEval_SaveThread()  ----------------> saved main PyThreadState
first GUI frame
start worker

frame/event loop                       can_acquire_gil()
                                      GilAcquire
                                      discover enabled plugins
                                      release GilAcquire

frame/event loop                       GilAcquire
                                      ensure_venv / start uv ----------> download/sync
                                      pipe read/wait releases GIL
frame/event loop continues             short progress callbacks
                                      pipe read/wait releases GIL
                                      uv exits <------------------------- exit
                                      module exec/import (GIL-heavy)
main skips blocking Python callbacks   on_load registration
but native/cached UI keeps rendering   release GilAcquire
                                      publish result + wake main

next plugin ...
```

Use `can_acquire_gil()` before every worker entry, then `GilAcquire`, matching `src/python/gil.hpp:13-28` and the REPL precedent. Acquire the existing `g_plugin_init_mutex` only while ensuring the bridge and taking the discovery snapshot, in the existing order “GIL, then mutex” (`src/python/runner.cpp:738-749,819-842`). Never hold it across installation/import.

The current `_exec_with_import_audit()` temporarily replaces process-global `builtins.__import__` (`src/python/lfs_plugins/manager.py:398-424`). That is unsafe when the main UI or marketplace can import concurrently: another thread can observe the wrapper, and nested workers can restore the wrong function. The implementation should remove the monkeypatch and keep total import timing, or replace it with an owner-thread-only profiling mechanism. A global import lock alone does not stop unrelated Python code from seeing the monkeypatch.

#### Cancellation and shutdown

The coordinator owns a stop source, a terminal-state condition variable, and the worker. `VisualizerImpl::beginShutdown()` requests stop without waiting; `python::finalize()` performs the wait/join before changing GIL readiness.

`PluginInstaller` must accept a cancellation predicate and replace both unbounded subprocess paths (`subprocess.run` at `installer.py:499-504` and the `readline`/`wait` loop at `installer.py:586-601`) with one cancellable runner:

1. launch `uv` in its own process group/session;
2. drain stdout without blocking the controlling worker indefinitely (a small pipe-reader thread plus a queue is portable; the controller polls cancellation at <=50 ms);
3. on stop, terminate the process group, wait for a short grace period, then kill and reap it;
4. close/drain the pipe and join the reader;
5. raise a distinct cancellation exception and never write `.deps_installed`.

Cancellation is checked between discovery, environment setup, dependency sync, module exec, and `on_load()`. It is not safe to use `PyThreadState_SetAsyncExc` or to kill a C++ thread during a native extension import. If shutdown arrives while arbitrary native module initialization is running, wait for a short bounded grace period. If it does not unwind, do **not** detach and continue Python/viewer teardown. The GUI process already terminates with `std::_Exit`; the safe fail-stop is to skip Python cleanup and viewer destruction and exit immediately while process-owned objects are still alive. This requires an explicit shutdown result from the coordinator/finalizer and a branch around `application.cpp:585-592`. Normal dependency cancellation should never reach that fallback.

#### Progress plumbing

Keep `notify_startup_plugin_load_state(active, progress, stage)` as the sink so the existing overlay remains connected. Add structured phase reporting inside the Python pipeline while preserving the public `PluginManager.load(name, on_progress=None)` behavior used by the marketplace:

- `discovering`: “Discovering startup plugins…”
- `creating_environment`: “Preparing environment for densification…”
- `installing_dependencies`: “Installing dependencies for densification… + triton==3.6.0”
- `importing`: “Importing densification…”
- `activating`: “Activating densification…”
- `failed`: “densification failed; continuing…”
- terminal: “Loaded 3/4 plugins; 1 failed: densification”

Use one equal progress slot per plugin and monotonic sub-milestones within the slot; do not pretend raw `uv` lines provide exact byte progress. Rate-limit stage redraws (for example, 10 Hz) and cap/sanitize detail length. `StartupOverlay::escapeRmlText()` already escapes stage text (`src/visualizer/gui/startup_overlay.cpp:255-268`). Full tracebacks stay in the log and `PluginManager.get_traceback()`.

The final overlay state must distinguish successful count from attempted count. The current terminal text always reports `total/total` even after failures (`src/python/runner.cpp:893-897`). Keep the overlay visible long enough to show an error summary and expose details through the plugin marketplace; do not open one modal per failed plugin.

#### Failure isolation

For each plugin, catch installation, import, and activation errors, record `ERROR`, log the traceback, roll back partial registration/module state, append a compact failure result, and continue. A startup batch is terminal when all names were attempted; `mark_plugins_loaded()` keeps its existing “autoload attempt completed” meaning, not “every plugin succeeded.”

Cancellation is not a plugin failure. On application shutdown it should be logged at info level and leave the dependency stamp absent. If cancellation becomes user-visible outside shutdown, add a `CANCELLED` lifecycle state rather than misreporting `ERROR`.

#### Risks

- **Arbitrary main-thread-only API calls from top-level code or `on_load()`:** the public docs show registration-only `on_load()`, and marketplace already loads on a worker, but the entire binding surface has no formal affinity contract. Add an explicit startup-worker context, guard known main-only bindings, marshal only bounded native payloads, and fail the offending plugin clearly instead of risking UI corruption.
- **GIL contention with Python-backed UI:** retain phase-aware guards so the render thread never waits behind an import. Native and cached UI stays responsive; newly loading plugin UI appears only after activation. Full concurrent execution of arbitrary Python panels during a native-extension import would require process isolation and is out of scope.
- **Global Python import state:** serialize plugin lifecycle and remove the global import monkeypatch.
- **Shutdown inside a non-cooperative native import:** use the bounded fail-stop described above; never detach and then run Python cleanup.

## 3. Why Candidate C wins

Candidate C is the only incremental design that moves every unbounded existing plugin phase off the viewer thread. It follows a behavior already exercised by the marketplace and by plugin concurrency tests rather than inventing a second plugin runtime. It preserves the synchronous API by sharing one coordinator state machine instead of changing `ensure_plugins_loaded()` into fire-and-forget. It also provides one place to enforce ordering, cancellation, progress, and teardown invariants.

Candidate A loses because heavy module imports remain on the viewer thread. Candidate B loses because arbitrary `on_load()` remains on the viewer thread. A separate plugin-host process would give stronger GIL and kill isolation, but the current API passes in-process Python/C++ objects, panels, tensors, callbacks, and scene access; converting that contract to RPC would be a plugin-system redesign, not a startup hardening change.

## 4. Recommended implementation plan

### 4.1 Introduce one coordinator in `runner.cpp`

Files: `src/python/runner.cpp`, `src/python/runner.hpp`.

Replace the loose globals at `runner.cpp:59-63` with a `PluginAutoloadCoordinator` containing:

```text
enum class State { NotStarted, Discovering, Loading, Completed, Cancelled };
std::mutex mutex;
std::condition_variable cv;
std::jthread worker;
std::stop_source / stop_token;
std::atomic<State> state;
std::atomic<Phase> phase;       // Idle, DependencyIO, PythonImport, Activation
std::vector<PluginResult>;      // protected; name, success, summary
```

Required invariants:

- only one owner may transition `NotStarted -> Discovering`;
- the worker never holds `mutex` while acquiring the GIL or invoking Python;
- a terminal transition publishes results, clears `running`, notifies `cv`, calls the overlay sink, and wakes the main loop;
- `are_plugins_loaded()` becomes true only after a non-cancelled batch finishes attempting all names;
- a failure for one name never terminates the batch;
- `preload_user_plugins_async()` returns in microseconds after starting or observing the worker;
- `process_plugin_preload_step()` is removed; no compatibility poll remains;
- `is_plugin_preload_running()` remains an atomic query for GUI gates.

Factor the current discovery/load loop into one internal `load_enabled_plugins(ProgressSink, StopPredicate)` used by both the worker and the synchronous path. Do not duplicate progress/failure rules between `ensure_plugins_loaded()` and startup preload as they are today (`runner.cpp:729-787` versus `803-900`).

### 4.2 Preserve synchronous `ensure_plugins_loaded()` semantics

`ensure_plugins_loaded()` must:

1. initialize Python and return if the completed flag is set;
2. if no coordinator owner exists, atomically become the synchronous owner and run the shared pipeline inline;
3. if the async worker owns the batch, wait for its terminal state and return only when capabilities/plugins are usable;
4. release any currently held GIL while waiting, then restore it before returning;
5. never wait while holding `g_plugin_init_mutex` or a Python manager lifecycle lock.

Call-site fixes:

- In `run_scripts()` (`src/python/runner.cpp:1152-1189`), call/await `ensure_plugins_loaded()` before the long `GilAcquire` used for script execution.
- In `invoke_capability()` (`runner.cpp:1560-1573`), ensure first, then acquire the GIL for JSON parsing/invocation.
- Add `nb::call_guard<nb::gil_scoped_release>()` or an equivalent split to `lf.scripts.run` (`src/python/lfs/py_scripts.cpp:65-85`) so a Python caller does not wait while owning the interpreter.
- In GUI MCP `plugin.invoke`, wait for startup completion on the MCP server thread *before* posting the actual scene-bound capability invocation to the viewer (`src/app/mcp_gui_tools.cpp:4257-4279`). Do not let a viewer `WorkItem` synchronously wait for startup.
- `plugin.list`, headless MCP, and trainer initialization may continue to block their non-render calling threads; that is the required synchronous behavior.

If a direct synchronous request arrives on the viewer thread while startup is still running, return/queue an explicit “plugins are still loading” continuation rather than blocking that thread. All known GUI call sites should be converted so this is a defensive path, not normal behavior.

### 4.3 Make the Python pipeline cancellable and transactional

Files: `src/python/lfs_plugins/manager.py`, `installer.py`, `errors.py`, and optionally `plugin.py`.

In `PluginManager`:

- factor `load()` into explicit internal stages such as `_ensure_dependencies`, `_import_module`, `_activate_plugin`, and `_finish_plugin_load` while keeping the public synchronous return contract;
- add optional `on_stage(phase, detail)` and `should_cancel()` hooks without breaking existing string `on_progress` callbacks;
- add a manager-wide lifecycle lock or operation queue for load/unload/reload/import because `sys.path`, `sys.modules`, and import instrumentation are process-global;
- make `load(ACTIVE)` idempotently return `True`; require explicit `reload()` for replacement;
- serialize same-plugin startup/marketplace/watcher requests so a second caller joins or queues behind the first;
- snapshot callback lists before invoking them (`manager.py:323-327,506-510,565-569`) so callbacks cannot mutate the active iteration;
- remove or replace the process-global `builtins.__import__` monkeypatch at `manager.py:398-424`;
- add `_rollback_failed_load(plugin)` that best-effort calls `on_unload` if present, removes capabilities/subscriptions/panels, clears module entries and `sys.path`, and leaves the original traceback intact.

In `PluginInstaller`:

- add a distinct `PluginLoadCancelled` exception;
- check cancellation before destructive filesystem steps and before/after each child process;
- replace `subprocess.run()` in `ensure_venv()` and the blocking `Popen.readline()/wait()` in `install_dependencies()` with the shared bounded process controller;
- start a new process group/session and terminate the group on cancel;
- keep the last bounded output tail for errors;
- touch `.deps_installed` only on exit code zero after all output is drained;
- accept that a cancelled `uv sync` may leave a partial `.venv`; absence of the stamp forces repair next launch.

On POSIX, the controller starts each child in a new session and terminates the complete
process group with `killpg`. On Windows, `CREATE_NEW_PROCESS_GROUP` plus
`taskkill /T /F` is the portable stdlib-only best effort used here. It does not provide
the containment guarantee of a Job Object, and the Windows descendant-tree behavior
must be validated on Windows before it is treated as equivalent to POSIX.

The C++ stop token can be exposed to Python through a small internal binding or a Python callable backed by an atomic. Shutdown must be able to request stop without acquiring the GIL.

### 4.4 Formalize registration affinity

Files: `src/python/lfs/py_plugins.cpp`, `py_ui.cpp`, `py_ui_panels.cpp`, other bindings found by the audit, and `src/python/python_runtime.{hpp,cpp}`.

Add a thread-local/plugin-name startup execution context set around module execution and activation. Binding behavior in that context should be one of:

- **Allowed on worker:** registry-only operations proven mutex/GIL-safe (class, menu, operator, capability, tool, hook/subscription registration).
- **Marshaled:** a small audited native operation whose input can be copied without retaining borrowed Python objects. Use `Visualizer::postWork`, provide `WorkItem.cancel`, release the GIL before waiting, and reject work after `accepting_work_` becomes false.
- **Rejected:** immediate drawing, native dialogs, arbitrary scene/renderer mutation, or unbounded operations. Raise a clear `RuntimeError` naming the plugin and API and telling the author to defer it to a user action/frame callback.

Do not make `on_load()` itself a posted work item. Extend the existing off-thread handling for texture destruction to creation only if the posted upload is demonstrably bounded; otherwise require lazy icon/texture creation on first draw.

### 4.5 Keep the UI responsive and informative

Files: `src/visualizer/visualizer_impl.cpp`, `visualizer_impl.hpp`, `src/visualizer/gui/gui_manager.{cpp,hpp}`, `startup_overlay.{cpp,hpp}`, `src/visualizer/gui/string_keys.hpp`, and all locale JSON files under `src/visualizer/gui/resources/locales/`.

- Remove the call that executes plugin work from `VisualizerImpl::update()` (`visualizer_impl.cpp:1101-1103`). Startup should only start the worker.
- Preserve redraw/wake behavior. The overlay state is already mutex-protected.
- Add phase-aware queries, for example `is_plugin_preload_running()` and `is_plugin_preload_python_busy()`. During `uv` I/O, allow normal UI Python work if measurement shows GIL stalls remain within budget; during import/activation, never let the main thread block waiting for the worker's GIL.
- Replace `startup_overlay_.isVisible()` as a blanket input blocker with `startup_overlay_.blocksUnderlayInput()`. Once background plugin loading starts, switch to a compact/non-modal progress presentation or offer immediate “Continue in background.” Window close must always work.
- Preserve the first rendered Python UI frame and use cached/non-Python draw paths during a GIL-heavy phase instead of making the whole shell disappear. Queue or disable Python-backed actions with a visible “plugin loading” state rather than blocking the render thread.
- Add localized strings for environment setup, dependency installation, import, activation, cancellation, and failure summary. Current keys are only discovery/loading/loaded/skipped (`string_keys.hpp:791-803`; English values at `resources/locales/en.json:1381-1392`).
- Show one terminal failure summary with plugin names and a route to details. Keep full traceback logging.

### 4.6 Make shutdown ordering explicit

Files: `src/visualizer/visualizer_impl.cpp`, `src/python/runner.cpp`, `src/app/application.cpp`.

- `VisualizerImpl::beginShutdown()` requests preload stop immediately after it flips `accepting_work_` false; it must not join there.
- `join_plugin_preload()` requests stop idempotently, waits on a terminal condition with a finite deadline, then joins only after the worker reports that its `GilAcquire` scope has ended.
- `python::finalize()` must call that join before `set_gil_state_ready(false)` and before `acquire_gil_main_thread()`, preserving `runner.cpp:941-955` ordering.
- If the bounded wait expires in a non-cooperative native import, return an explicit fatal-shutdown disposition. `runGui()` must skip Python cleanup and `viewer.reset()` and go directly to process exit. Logging must state why. Continuing normal teardown after detaching is forbidden.
- The progress and wake callbacks remain installed until the normal join succeeds. On the fatal-exit path, process-owned state stays alive until `_Exit`.
- Test repeated stop/join/finalize calls for idempotence.

### 4.7 Edge-case policy

- **Plugin enabled mid-session:** the startup denominator is a discovery snapshot. An explicit marketplace load uses the same serialized manager pipeline but is not retroactively added to startup progress. Merely changing `load_on_startup` affects the next launch unless the UI also explicitly loads it.
- **Plugin disabled while queued:** recheck `load_on_startup` immediately before starting that plugin and skip it if disabled.
- **Manual load of a startup plugin already in flight:** join/coalesce with the in-flight operation; never execute it twice.
- **Reload/hot reload during startup:** queue behind the current lifecycle operation. Explicit reload wins after the initial load completes; shutdown cancels both.
- **`LFS_PLUGIN_AUTOLOAD=off`:** create no startup worker and do not show active startup progress. Explicit `lf.plugins.load()` and synchronous `ensure_plugins_loaded()` retain current behavior.
- **No startup overlay / `--no-splash`:** still run the worker; progress goes to logs and any later status surface, with no hidden modal input blocker.
- **No enabled plugins:** publish terminal `0/0`, mark the batch complete, and do not create a long-lived worker.
- **Per-plugin install failure:** leave no success stamp, record error, continue.
- **Per-plugin import/activation failure:** roll back partial registrations/module paths, record error, continue.
- **Headless/CLI:** never start the async coordinator from application startup. Synchronous callers use the shared pipeline inline; plugin CLI scaffolding/check/list paths remain unchanged.

## 5. Validation plan

No build or runtime validation was performed during the design round. The implementation
results are recorded in the implementation log below; the remainder of this section is
the original validation plan.

### 5.1 Deterministic unit/integration tests

Extend existing coverage rather than inventing a separate plugin model:

- `tests/python/test_plugin_system.py` already covers load semantics and installer command/stamp behavior (`:165+`, `:763-1034`). Add a fake `uv` that streams lines, blocks, is cancelled, exits nonzero, and spawns a child; assert bounded cancellation, output capture, child reaping, and no stamp.
- `tests/python/test_plugin_error_recovery.py:74-123` covers `on_load` and import failure. Add a plugin that registers a panel/capability and then raises; assert rollback and that the next valid plugin activates.
- `tests/python/test_plugin_concurrency.py:90-366` already exercises worker-thread load/unload/discovery. Add startup versus marketplace/manual load of the same plugin, startup versus reload, and import-audit isolation.
- `tests/python/test_plugin_callbacks.py:84-190` covers lifecycle callback mutation/failure. Assert callbacks run once from the worker and exceptions do not terminate the batch.
- Add C++ coordinator tests alongside `tests/test_python_integration.cpp`: async completion, synchronous waiter with GIL released, per-plugin continuation, cancellation before/inside `uv`, and no `GilAcquire` surviving terminal state.
- Extend `tests/test_visualizer_post_work.cpp` to prove a posted affinity task wakes the loop, its cancel callback fires during `beginShutdown()`, and a worker waiter cannot hang.

### 5.2 Frame-budget proof

Use a fresh temporary plugin environment and two workloads:

1. a deterministic fake `uv` that emits progress and sleeps for at least 10 seconds;
2. a real cold install of the densification dependency set (including the large Triton wheel) with its `.venv` and dependency stamp absent.

Instrument the viewer loop with a test/perf-only heartbeat around `VisualizerImpl::update()` and GUI presentation. Record:

- maximum `update()` time attributable to the preload pump (target below 1 ms; it should only read atomics);
- presented-frame gap p50/p95/max while install is active;
- input-to-frame latency for mouse move, camera motion, window resize, and close;
- MCP `runtime.state`/non-plugin tool latency while install is active;
- any main-thread GIL wait during worker `PythonImport`/`Activation` phases.

Acceptance criteria should be explicit: no plugin-preload function executes on the viewer thread; no preload-attributed main-thread span exceeds one target frame (16.7 ms at 60 Hz, with a tighter 2 ms budget for the preload pump); the window continues presenting and processing close/resize throughout the 10-second fake install. Report GPU/swapchain stalls separately from plugin stalls.

For the real heavy plugin, log phase durations separately—environment, dependency sync, import, activation—and verify that a long import does not create a main-thread GIL wait because phase-aware UI guards take the cached/native path.

### 5.3 Shutdown and recovery matrix

For Linux and Windows at minimum:

- close during venv creation, active download, `uv` output, module import, and `on_load()`;
- require normal dependency-phase exit within a small bounded interval (for example <=2 s);
- assert no `uv`/child process remains;
- assert no Python cleanup crash, deadlock, or access to a destroyed `GuiManager`;
- relaunch and prove the missing stamp causes a successful repair;
- inject one broken plugin before one valid plugin and prove the valid one loads;
- run with `LFS_PLUGIN_AUTOLOAD=off`, no splash, zero plugins, and a plugin enabled during the batch;
- run headless training with a Python script and headless MCP capability list/invoke to prove synchronous semantics are unchanged.

Use sanitizer/debug runs for registry lifetime and race checks after functional tests pass. A forced fake native import that never returns should exercise the bounded fatal-exit path; never use a real third-party extension as the only shutdown test oracle.

## 6. Open questions and the experiment for each

1. **Does “fully interactive” require Python-backed panels to execute callbacks during a native-extension import, or is a live native/cached shell with those controls temporarily deferred acceptable?** A single non-free-threaded interpreter cannot guarantee both arbitrary native import and arbitrary Python UI execution. Prototype the phase-aware cached UI with the fake 10-second install and a deliberately GIL-holding test extension; measure and review the UX. If concurrent Python execution is mandatory, the real solution is a process-isolated plugin runtime and a new RPC-safe plugin API.

2. **Which third-party plugins perform main-thread-only work at module scope or in `on_load()`?** Instrument every `lichtfeld` binding with the startup worker context in a debug build, classify calls by affinity, and cold-load the installed plugins plus a representative registry corpus. Fail fast with the plugin/API/thread in the log. Static inspection covered the two locally installed plugins and repository examples, not the ecosystem.

3. **How long do Torch/Triton and other real native imports hold the GIL continuously on supported platforms?** Add GIL-acquisition wait telemetry on the viewer thread and phase markers on the worker; run cold and warm imports on Linux and Windows. This determines whether Python panels can remain live during `Importing` or must always take a cached path.

4. **Can the Python `Popen` process-group strategy reliably kill the complete `uv` descendant tree on Windows?** Use a fake `uv` that spawns a child which ignores graceful termination. Close the app mid-run and assert both PIDs are gone. If not, use a Windows Job Object or extend the existing C++ `SubProcess` abstraction rather than shelling out to `taskkill`.

5. **What is the product policy for shutdown during a non-cooperative native import?** Exercise an intentionally stuck test extension and compare a short “finishing plugin shutdown” grace period with immediate fail-stop. The technical invariant is fixed—never detach and continue Python/viewer teardown—but the grace duration and user message need a product decision.

6. **Should `LFS_PLUGIN_AUTOLOAD=off` also prevent later implicit loading by MCP/scripts?** Static reading shows that it currently gates only `preload_user_plugins_async()`; `ensure_plugins_loaded()` ignores it. Validate expected operator intent before changing that behavior. The implementation recommendation is to preserve current semantics.

## Implementation log

Implementation base: `vulkan-hardening` at `f2734dfd3` on 2026-07-11. The branch is
`plugin-async-loading` and was not rebased or pushed during this round.

### Landed commits

- `b4cb87170 feat(plugins): make plugin loading cancellable`
- `3b7407c0e feat(plugins): load startup plugins off render thread`
- `cf80df3a6 fix(plugins): isolate preload frame timing`
- `docs(plugins): record async loading validation` records this implementation log.

### Delivered behavior

- `PluginManager.load()` is one serialized, staged, transactional operation. It reports
  environment, dependency, import, activation, and completion stages; snapshots lifecycle
  callbacks; removes the process-global `builtins.__import__` replacement; and rolls back
  owned registrations, module entries, and `sys.path` state on cancellation or failure.
- Venv creation and `uv sync` share one cancellable subprocess controller. POSIX children
  run in a new session and are terminated/reaped by process group. Windows uses the
  documented `CREATE_NEW_PROCESS_GROUP` plus `taskkill /T /F` best effort. A cancelled or
  failed dependency operation never writes `.deps_installed`.
- Startup loading has one C++ coordinator and one owned `std::jthread`. Discovery and the
  full per-plugin pipeline run off the graphics thread. Status crosses to the graphics
  thread through a revisioned, mutex-protected snapshot; the worker never calls the GUI.
  `process_plugin_preload_step()` and its render-loop call were removed completely.
- `ensure_plugins_loaded()` retains synchronous semantics for scripts, capability queries,
  MCP, and headless callers. A non-graphics caller joins the coordinator after releasing a
  held GIL; a graphics-thread request never waits for the worker. The scripts and GUI MCP
  call sites were moved to the safe side of that contract.
- Shutdown requests stop without the GIL, cancels/reaps dependency children, and joins the
  worker before GIL readiness is cleared or Python teardown begins. A worker stuck in
  non-cooperative native import for five seconds takes the explicit fail-stop path instead
  of detaching into Python teardown.
- The startup overlay is non-modal while loading. Its stage and progress continue to update,
  while viewport/window input and native menus remain live. The Python-visible
  `lf.plugins.startup_load_status()` exposes coordinator state, phase, plugin, detail,
  attempted/total/failed counts, progress, and active state.
- UI texture creation now has one explicit graphics-thread contract used by icon loading
  and dynamic texture upload. Off-thread creation fails with an actionable exception;
  cached texture lookup and the existing marshalled destruction path remain valid.
- Debug builds report the maximum steady-state `VisualizerImpl::update()` duration while
  preload is active. The transition update is deliberately excluded because it may also
  perform unrelated synchronous startup asset loading such as the initial `--view` PLY.

### Automated gates

The required Python command was used throughout:
`./build/vcpkg_installed/x64-linux/tools/python3/python3.12 -m pytest tests/python/ -v`.

| Revision | Build (`-j8`) | Vulkan focused filter | Python suite |
|---|---:|---:|---:|
| Pre-change baseline | pass | 52/55 | 1090 passed, 78 failed, 11 errors, 283 skipped |
| `b4cb87170` | pass | 52/55 | 1098 passed, 78 failed, 11 errors, 283 skipped |
| `3b7407c0e` | pass | 52/55 | 1100 passed, 78 failed, 11 errors, 283 skipped |
| `cf80df3a6` on re-stacked base | pass | 52/55 | 1100 passed, 78 failed, 11 errors, 283 skipped |

The three unchanged focused-test failures are:

- `ViewportFrameLifecycleServiceTest.ResizeActiveDefersFullRefreshUntilDebounceCompletes`
- `ViewportFrameLifecycleServiceTest.PassiveWindowResizeDefersFullRefreshUntilDebounceCompletes`
- `ViewportFrameLifecycleServiceTest.ExplicitRefreshDeferralCompletesAfterStableFrames`

The ten deterministic async-plugin tests pass. They cover stage delivery, cancellation and
module cleanup, activation rollback, concurrent-load coalescing, import-hook ownership,
cancellable streaming process control, platform process-group configuration, the
Python-visible coordinator status contract, texture-affinity rejection, and stamp behavior
after cancellation.

The full Python suite preserves the baseline 78 failures and 11 errors while adding ten
passes. It still terminates with the pre-existing exit 139 after the summary in the
GL-less icon-cache teardown path. Tensor/PyTorch interop also retains its pre-existing
missing-NumPy failures. One intermediate host-wide CUDA initialization outage produced an
invalid 25/55 focused run and a broader Python failure count; both gates were rerun after
CUDA recovered and returned to the counts above.

### Linux cold-start frame-budget proof

The existing 11 GiB `~/.lichtfeld/plugins/densification/.venv` was renamed intact, the
per-plugin `.venv` and its stamp were absent for launch, and the exact application command
was run with debug logging:

```sh
LFS_LOG_LEVEL=debug ./build/LichtFeld-Studio --view splat_64400.ply
```

The global `uv` cache was intentionally left intact, so this was a cold per-plugin
environment but a cache-warm package transfer. Measured stage timings were
`venv=105 ms`, `deps=834 ms`, `module=6395 ms`, `on_load=0 ms`, and `total=7334 ms`.
The long real Torch/Triton import provided a 6.4-second in-flight interval even though
cached wheel installation was fast.

During that interval, automated desktop input moved the camera and opened the File menu
while the overlay visibly read `Importing densification`; the plugin was still active after
the complete 4.144-second interaction sequence. The viewport changed between captures and
the menu rendered over the active progress overlay. The instrumented maximum steady-state
`VisualizerImpl::update()` duration was `0.039 ms`. The app then closed through its normal
confirmation dialog with exit status 0. The log contained no error, critical, assertion,
VUID, device-lost, or validation-error line; the only validation match was the benign
`Vulkan diagnostics: validation_layers=...` info record. The original 11 GiB environment
and its stamp were restored after the run.

### Deliberate scope decisions and remaining validation

- Python frame callbacks and Python-backed panels remain suppressed for the whole preload
  batch, rather than being opportunistically re-enabled during subprocess I/O. This is the
  conservative single-interpreter contract: the graphics thread never waits for the GIL;
  the native/cached shell, viewport, menus, resize, and close remain responsive, and Python
  surfaces resume after the terminal state.
- New stage details use bounded English text and streamed `uv` output. No locale bundle was
  expanded in this focused change. The structured phase/status surface is the single source
  for a future main-thread localization pass.
- The proven unsafe icon and dynamic-texture creation paths now enforce affinity. A generic
  execution-context wrapper over every binding was not added without a complete binding
  audit; ecosystem plugins that perform other main-thread-only work during import remain an
  explicit audit item rather than being hidden behind an arbitrary marshal.
- The startup denominator remains the discovery snapshot. Explicit marketplace/manual loads
  serialize through the same manager lock, but changing `load_on_startup` during an active
  batch is next-launch policy; a mid-batch policy change is not silently reinterpreted.
- Windows process-tree cancellation and GUI behavior remain untested on this Linux host.
  Run the fake child-tree cancellation test and a real GUI close-during-install test on
  Windows. Escalate to a Job Object if `taskkill /T /F` leaves descendants.
- Exercise the five-second fail-stop with a purpose-built native extension that never
  returns from initialization. The required invariant is already explicit: never detach a
  Python-owning worker and continue teardown.
- A truly network-cold Triton install was not forced because that would require bypassing or
  replacing the user's global `uv` cache. To measure it without deleting user data, repeat
  the procedure below with `UV_CACHE_DIR` pointing at a new temporary directory.

### Interactive validation procedure

1. Preserve the current environment and ensure the backup name is unused:

   ```sh
   plugin_dir="$HOME/.lichtfeld/plugins/densification"
   backup="$plugin_dir/.venv.async-validation-backup"
   test ! -e "$backup"
   mv "$plugin_dir/.venv" "$backup"
   ```

2. For a network-cold run, create a disposable cache; otherwise omit these two lines:

   ```sh
   cold_cache="$(mktemp -d)"
   export UV_CACHE_DIR="$cold_cache"
   ```

3. Launch from the repository root and retain the complete log:

   ```sh
   LFS_LOG_LEVEL=debug ./build/LichtFeld-Studio --view splat_64400.ply \
     > /tmp/plugin-async-frame-budget.log 2>&1
   ```

4. While the overlay says `Installing dependencies for densification` or
   `Importing densification`, orbit/zoom the viewport, open a native menu, resize the
   window, and then close through the normal exit dialog. Confirm continuous presentation,
   input response, progress changes, and clean shutdown.
5. Check the timing and error surface:

   ```sh
   rg -n 'load\(densification\) timing|Plugin preload frame budget' \
     /tmp/plugin-async-frame-budget.log
   rg -n -i '\[error\]|\[critical\]|assert(ion)?|validation.*(error|fail)|VUID|device[ -]lost' \
     /tmp/plugin-async-frame-budget.log
   ```

6. Restore the original environment even if validation failed:

   ```sh
   rm -rf -- "$plugin_dir/.venv"
   mv "$backup" "$plugin_dir/.venv"
   test -x "$plugin_dir/.venv/bin/python"
   test -e "$plugin_dir/.venv/.deps_installed"
   test -z "${cold_cache:-}" || rm -rf -- "$cold_cache"
   ```

### Follow-up: terminal overlay interaction regression

Marker: `INTERACTION-HANG-FIX-2026-07-11`.

#### Verdict and root cause

This was **our async-plugin regression**, not a multi-file consolidation, camera-bounds,
or render-demand bug. On the pre-fix branch, the exact three-file directory rendered all
9,000,000 Gaussians and continued producing render-loop diagnostics, but the terminal
`Loaded plugins 1/1` startup overlay remained visible and no camera, menu, or window input
reached the underlay. A single-PLY launch with plugins enabled reproduced the same terminal
state, which excludes the multi-node/consolidation path.

Two coupled overlay contracts caused the apparent hang:

- `StartupOverlay::blocksUnderlayInput()` treated `!plugin_load_state_.active` as modal.
  The coordinator correctly published `active=false` on completion, so its terminal state
  re-enabled the global input guard.
- `GuiManager` masked the frame input for a blocking startup overlay and then supplied that
  same masked input to the overlay. The overlay therefore could not observe the click or key
  that was supposed to dismiss it. The render loop was alive; all interaction was suppressed.

Commit `149ae66c2` (`fix(viewer): keep completed plugin overlay interactive`) makes the
coordinator lifecycle explicit at the overlay boundary. Once startup preload has begun,
the overlay cannot become modal again; monotonic lifecycle assertions enforce that rule.
`GuiManager` preserves an unmasked input snapshot for the overlay before masking the
underlay, and `VisualizerImpl` asserts that a started preload can never latch the startup
input guard. The legacy notification adapter now also publishes `loading`/`completed`
lifecycle state instead of only mutating the `active` bit.

#### GUI validation

- Warm-plugin/cold-storage exact directory run:
  `./build/LichtFeld-Studio -v /media/paja/T7/lcc_images/results/sharpen_ab/plys/`.
  All three files loaded and consolidated, orbit/zoom, File menu, and resize worked, and the
  app closed through its own top-right close control and confirmation with exit status 0.
  The log contained no error, critical, assertion, VUID, device-lost, or validation-error
  line (apart from the benign diagnostics info record).
- Cold-plugin exact directory run: the original 11 GiB densification `.venv` was renamed
  intact, leaving both `.venv` and its `.deps_installed` stamp absent. The in-process monitor
  completed its orbit/zoom/menu/resize sequence within 586 ms of `Splat batch loaded`; the app
  again closed normally with status 0. Plugin import overlapped the longer cold PLY reads and
  happened to finish before the third PLY, so this run proves the terminal-state fix but is
  not used as the in-flight-plugin proof.
- Definitive cold-plugin single-file run:
  `./build/LichtFeld-Studio --view splat_64400.ply`. The file was ready in 0.189 s and GUI
  input began 49 ms later, while the overlay still showed densification dependency progress.
  Camera motion, zoom, the File menu, and a 1280x720 to 1400x860 resize completed in 595 ms.
  Densification did not finish until 5.149 s after the batch-complete record
  (`venv=24 ms`, `deps=812 ms`, `module=4238 ms`, `total=5075 ms`). The maximum measured
  preload contribution to `VisualizerImpl::update()` was 0.103 ms. After completion, the
  next camera click both affected the viewport and dismissed the terminal overlay. The app
  closed through its confirmation dialog with status 0 and a clean diagnostic scan.

The original densification environment and stamp were restored after both cold runs. One
exploratory automation attempt used `xdotool windowclose`; that destroys the X11 window,
caused SDL to receive `BadWindow`, and let Xlib call `exit()` during Python object teardown.
It is not an application close-path result and was discarded. All reported gates used the
application's own close button and confirmation dialog.

#### Follow-up commits and gates

- `149ae66c2` — `fix(viewer): keep completed plugin overlay interactive`
- `e43273316` — `perf(io): accelerate multi-file PLY loading`
- `docs(viewer): record interaction and PLY validation` records this follow-up.

After each source commit, `cmake --build build -j6` passed. The focused Vulkan filter stayed
at its documented 52/55 baseline with only the same three resize-deferral failures;
`*AssertHardening*` passed 19/19; the async-plugin pytest file passed 10/10 using the bundled
Python; and the focused PLY slice passed 10 tests with one expected missing-trained-asset
skip. The pre-existing CUDA-runtime-unload diagnostics still appear after gtest process
teardown and do not change the test outcomes.
