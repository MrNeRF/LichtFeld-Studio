# External training previews

The **External training** tab in LichtFeld Studio shows a navigable Gaussian preview from a trainer running in another local process. Training continues in its owning process when the viewer disconnects or closes.

Use an empty scene or a model scene for observation. A loaded training dataset uses the native trainer viewport; open a new scene or switch the dataset to Edit Mode before connecting an external trainer.

1. Start training from the LichtFeld Blender extension, or explicitly enable the session publisher in a LichtFeld Studio training script.
2. Open **External training**, refresh the list and choose a running trainer. The list excludes completed runs, dead processes and the current LichtFeld Studio process.
3. Click **View training**. Preview updates arrive automatically. **Frame preview** uses the normal viewport framing action; navigation and visibility remain available during updates.
4. Expand the trainer group in the Scene tree to see the preview model and its capture cameras. **Camera frustums (whole scene)** controls the global camera overlay, including local cameras. Use the camera group's visibility in the Scene tree to hide only the trainer's cameras. The cameras retain their calibration, including the principal point. Their geometry can arrive independently of the Gaussian snapshots.

On first use, the panel shows dependency preparation while the normal package manager installs NumPy if needed. Installation requires network access or a populated package cache. Errors appear in the panel and preparation retries automatically; **Disconnect** cancels it.

The progress line belongs to the trainer. The preview line reports the step, Gaussian count and SH degree of the snapshot actually installed in the viewer. Quality and update frequency follow the publisher's preview settings.

## Ownership and persistence

The preview is read-only. It cannot be transformed, renamed, unlocked, cropped, edited through the Gaussian API or used as a local training model. Capture cameras cannot be enabled for local training. Preview nodes are transient: project saves, exports and document undo history leave them out, and incoming updates do not make the document dirty. To export a trained model, open the file saved by its trainer. Local models remain editable and exportable while observation runs.

**Disconnect** stops observation and keeps the last read-only preview. **Remove preview** also removes its scene group. Neither action pauses, stops or changes the external trainer. A completed or stopped trainer leaves its last installed preview available for inspection. The training output and checkpoints remain the responsibility of the owning process.

Closing a LichtFeld Studio window that owns its own training retains the existing behavior: that training stops. External observation does not transfer ownership or detach a GUI-owned trainer into another process.

## Import the completed model

After a compatible trainer finishes saving, **Import saved model** loads its full best evaluated model, or its last valid saved model if no evaluated candidate is available. The saved-model line reports its step, Gaussian count and SH degree independently of the preview. Preview point/SH limits do not apply to this import.

File integrity checking, loading and renderer preparation run on workers through the normal import system. The result retains its declared coordinate presentation and SH coefficients. It becomes an independent, editable local model that can be saved and exported normally. Once import succeeds, the transient preview is hidden to avoid overlapping models. A failed import leaves the preview and existing local models available.

The trainer's saved PLY is verified in place and loaded into a new group. Importing the same result again creates a separate group; it does not replace an earlier import.

The action becomes available only after the producer publishes a completed saved artifact. Older or preview-only publishers remain observable without offering result import. The import requires the original result file to remain in its output folder; it does not resume or change training.

## Explicit publisher

Ordinary training does not publish sessions automatically. A startup script can opt in using the shared package:

```python
from lichtfeld_sessions.publisher import install

install(dataset="/path/to/dataset", limit=100_000, degree=1, interval=2.0)
```

Pass this script to the normal training command with `--python-script`. `limit` bounds the preview Gaussian count, `degree` bounds its SH degree (0–3), and `interval` sets the minimum publication interval in seconds. These limits affect previews, not the trained model. NumPy is prepared through the normal package manager on a worker when needed.

Sessions use the existing local, per-user snapshot protocol. Gaussian data is copied into viewer-owned renderer storage; this feature does not share GPU allocations between processes. The observer retries transient publication failures automatically and keeps the last installed preview while waiting for a newer one.
