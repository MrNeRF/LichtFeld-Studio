# External training previews

This feature lets LichtFeld observe a trainer running in another LichtFeld
process. Start training through the existing GUI or headless CLI, open another
GUI, select **External training**, choose a running trainer and **Connect preview**.
Both processes must support the same external-training protocol. No Blender
extension or startup script is required.

The preview is actual navigable Gaussian data. The panel and bottom bar distinguish
the training step from the snapshot currently shown. Navigation and visibility
are local. The streamed node is protected from editing, deletion, renaming,
unlocking and training binding at the native scene API, Python and MCP layers.
**Copy preview locally** makes an independent, editable and saveable snapshot.
**Disconnect preview** retains the last received protected snapshot.
**Remove preview** discards that transient node without touching the trainer or
local copies. Switching trainers removes the previous transient preview.
After a terminal state and the final snapshot acknowledgement, the reader closes
its client registration and stops polling. The scene keeps its last preview;
viewing or copying it does not require a running background reader.
Project changes release retained CPU payloads even for disconnected previews,
without requiring the panel to be open. Panel labels, states, tooltips, capture
camera labels and the read-only status bar use every shipped locale and update
when the application language changes.

Protected trainer containers accept only their own transient preview nodes.
Ordinary scene additions and reparenting cannot place local content inside them.
Attached capture cameras remain excluded from training, including bulk camera
selection and the dedicated Python camera-selection APIs. Their Python rotation
and translation tensors are detached copies, so reading camera data cannot
provide mutable access to the protected preview.

Only live trainers with a recent heartbeat appear. A nonresponsive trainer is
not declared completed; an existing observer can recover when its heartbeat
returns. Process creation tokens distinguish reused PIDs on Windows and Linux.
Different protocol versions are rejected explicitly when connecting.

When training ends, the current observer keeps its last received preview. There
is no session history, extra final PLY or implicit export. Open the normally saved
`.licht` project or exported splat for the full result. The built-in publisher
limits previews to 100,000 points and SH1; external adapters may publish different
limits, which the panel displays. A preview is not a replacement for the full
saved result or a checkpoint. Closing an observing GUI does not stop an independent trainer.
Closing the GUI that owns a trainer still ends that process: detachment is not
implemented.

## Dependencies and execution

External previews require NumPy in LichtFeld's Python environment. At first
connection, the observer prepares the environment and installs a compatible NumPy
package on a worker if needed. Setup failures keep the connection alive and retry
automatically. Training hooks never install packages or stop optimization because
an optional preview dependency is unavailable.

Discovery, reading, validation and native model preparation run on workers. UI
callbacks install prepared models and update retained controls. The reader holds
at most one unacknowledged frame; the publisher retains at most three generations.
All observers share the same published preview. Publication is demand-driven;
GPU-to-CPU capture still runs at a safe training hook and costs optimizer time.
The interval targets an amortized capture cost of 2%, with a minimum of two
seconds; this is not a hard latency guarantee for large models.

`SplatData.shN_rows(rows, degree)` gathers only selected SH-rest coefficients
directly from resident FP32, IEEE-f16 or q16 storage into owned canonical Float32
CPU data. Its temporary storage scales with the selected rows and bands, rather
than decoding the full model before applying the preview cap. SH0 previews skip
SH-rest extraction. Call this API at a model read boundary, with rank-1 CPU integer
indices. Invalid rows and unsupported degrees are rejected before GPU access.

Windows uses named memory mappings; Linux uses temporary mapped files. Session
metadata lives in the local training-session registry, overridable with
`LFS_TRAINING_SESSIONS_DIR`. Dead session metadata and protocol-owned temporary
files older than 24 hours are pruned during discovery. Unrelated files are kept.
No network service is used by the publication protocol.

## External adapters

A client application owns its dataset capture, job supervisor, commands and best
result selection. These are not part of this repository. An adapter supplying its
own protocol-1 publisher must launch the trainer with
`LFS_TRAINING_PUBLISHER=external`. This suppresses automatic publication, including
its capture hook, without disabling user scripts. `automatic` explicitly selects
the built-in publisher. For compatibility, the original extension's
`LFS_BLENDER_JOB` also selects external ownership when the generic setting is absent.

Session discovery and shared-memory snapshot layout are the interprocess contract;
LichtFeld's internal Python reader is not a job-control SDK. Clients must ship their
own job/control and saved-result readers. Existing packaged extensions retain their
vendored readers. Do not replace them with this application-specific observer.

## Privacy and terminal-state guarantees

Protocol-owned session directories are private (`0700`); POSIX metadata and
mapped payload files are created exclusively with mode `0600`. Caller-supplied
parent directories are not chmodded. On Windows the bundled runtime
applies a private-directory ACL for `mkdir(mode=0700)`. Named mappings retain the
creator-token default DACL. Use a trusted registry location. Publication code
changes take effect when a trainer process starts.

Metadata reads are bounded. Session field types, finite timestamps and signed
64-bit counters are checked before discovery/UI use; malformed records are
isolated rather than breaking other trainers. Snapshot counters are checked
before model preparation. Declared dimensions, byte counts, layouts and mapping
locations are checked before opening OS resources. Failed stream or mapped-view
construction closes its descriptor/handle; failed mapped-file creation removes
its temporary payload.

TrainingEnd carries an explicit `TrainingOutcome` from the trainer's terminal
save/report path. Errors take precedence over user stop and completion; reaching
the last optimization step does not by itself mean the run succeeded. The Python
hook forwards this outcome instead of inferring it from iteration counts. It
reflects the known result before delivery of the TrainingEnd callbacks.

## Snapshot publication and recovery

Snapshot publication uses one worker and at most one outstanding immutable
payload. GPU-to-CPU capture stays at the safe optimizer boundary; validation,
copying into mapped storage and checksumming run on the publication worker.
Subsequent hooks never wait for an unfinished publication or enqueue another one.
Shutdown/supersession retires the pool after its last writer, including publication
failure or an already-shutting-down executor. A late preview updates its snapshot
iteration without rolling back the live training progress. No extra final export
is created.

The panel states the displayed point count and SH degree, the total model size
when known, and the last measured capture/publication cost. A local copy retains
preview fidelity. Terminal metadata supplies an existing saved project/checkpoint
path, including a custom CLI save target; the panel does not automatically load
or overwrite a project. The observer-close/owner-close distinction is explicit in
the panel and the English/Italian owner exit confirmation. Owner detachment is
not implemented or offered.

Displayed quality metadata changes only after successful native installation;
pending/failed snapshots cannot relabel a retained preview. A total count is
shown only when the received status and snapshot iterations agree. First-use
numerical-runtime preparation stays on a reader worker. Training hooks skip
optional capture while that setup is busy, including package installation.

A repaired session status clears a transient metadata error without requiring a
new Gaussian generation. Readers discard queued payload references when closing;
the viewer isolates scheduling failures and drains retired CPU arrays on exit.
The optional metadata remains protocol 1 and older readers may ignore it. The
Blender job-control protocol remains owned by the extension; this application
continues to expose external previews read-only.

## Discovery and availability

Discovery refreshes automatically while browsing trainers or after disconnecting.
With a connected preview and the browser collapsed, the existing validated reader
supplies the current trainer's state and liveness; no repeated registry scan is
needed. The manual refresh action stays enabled, and a pending scan is reused
instead of queueing another. Closed browser labels do not claim a fresh trainer
count. Heartbeats can probe the native control snapshot, so a pause appears even
when no optimizer step or new splat snapshot is produced. Terminal outcomes win
against an overlapping status probe.

Discovery selects the newest responsive registration per trainer process before
filtering terminal states. A completed latest run therefore cannot expose older
registrations as active trainers. Paused runs remain available. Retained previews
of finished runs remain visible in the current-preview card, independently of the
active-only discovery list. Process identity and fresh heartbeat are still
required; historical folders are never treated as evidence of a running trainer.

## Coordinate frames and attached capture cameras

Capture camera UIDs use a scene allocator advanced by explicit/restored IDs;
restore does not lower its watermark. Camera cleanup can address the owned group
directly, and the status bar selects external splat anchors rather than temporary
camera metadata. Optional geometry-load failures retain the last valid geometry
and retry without blocking splat updates. Python API stubs include the camera and
model-transform arguments used by this integration.

Readers acknowledge the Gaussian generation and validated geometry reference
together. Repaired or removed geometry therefore updates the model transform and
owned cameras even while the trainer is paused and publishes no new splats.

Optional schema-1 `view_geometry.json` metadata carries a model-local-to-data
matrix, data-to-source matrix and OpenCV camera-to-data poses with pixel
intrinsics. The session publishes a bounded size/hash reference; readers validate
it on workers before preparing native payloads. Legacy sessions retain the
existing data-to-visualizer convention. Blender capture keeps the source origin
and units, converting the world Y/Z signs once at the data boundary and once on
return. Gaussian rotations and SH remain local to the model.

For Blender-origin sessions, Studio presents source +Z vertically along its
native visualizer +Y axis. This presentation transform is shared by the splat,
capture cameras and framing bounds; it never rewrites the dataset. Blender
continues to apply the declared data-to-source transform on import, preserving
the original position, scale and orientation.

Native previews retain the producer's model-node world transform. Attached
cameras are separate transient, locked camera nodes owned by the preview UUID;
they cannot enter active training or project persistence. They can be hidden
without affecting training. Camera construction and payload retirement run on
workers, with UI insertion/removal scheduled in short chunks. Discarding a
preview removes only its owned cameras and preserves unrelated scene content.
The panel offers camera visibility and explicit framing of the current preview.
The scene tree groups the preview and its capture cameras under one expandable
trainer. Camera labels are numbered within that trainer; internal ownership
identifiers do not fill every row. The trainer and camera-group visibility
controls affect their own subtree, independently of other scene cameras.

## Development checks

Run the Python contracts with NumPy installed:

```sh
python -m unittest discover -s tests/external_training -p "test_*.py"
```

The external-training CI workflow runs these contracts on Windows and Linux.
`tests/external_training/run_readonly_core.ps1` links the native scene protection
contracts against an existing Windows build. Renderer fixtures use the local MCP
endpoint and require a disposable GUI and project. Training and pressure fixtures
require separate output, runtime and session directories. Keep generated logs,
screenshots, reports and qualification records outside the source checkout.

The same runner with `-Source splat_preview_core.cpp` checks selected SH extraction
on CPU and CUDA across FP32, IEEE-f16 and q16 layouts, SH degrees 0–3, row/block
boundaries, duplicate/unsorted rows, invalid requests and source immutability.
