# Offline video reconstruction contract

Video export stores a backend-neutral reconstruction selection independently
from the interactive viewport:

- `backend_id`: stable backend identifier;
- `preset_id`: stable preset identifier;
- `fallback`: either `abort` or `native`.

The persisted object has its own schema version. Projects that do not contain
the object are migrated to `native`/`native` with the `abort` policy, which is
the behavior of video export before reconstruction selection was introduced.
A positive integer schema version newer than the host understands opens the
scene with the same native default and a localized, non-blocking warning.
Saving replaces that unsupported selection with the current version; it does
not preserve a future reconstruction payload. A missing/invalid version,
malformed JSON, an oversized object, or invalid version-1 fields still fails
validation rather than silently accepting corrupted data. The standalone
deserializer reports `unsupported_version`; only project restoration performs
this forward-version migration.
Numeric vendor quality values and input scales are not persisted. Those values
belong to a backend's read-only catalog descriptor and are resolved when an
export starts.

## Resolution boundary

`VideoReconstructionCatalog` is metadata-only. Looking up a descriptor must not
load plugin code, create GPU objects, or probe a device. The resolver snapshots
that metadata into an immutable `VideoReconstructionPlan` before rendering the
first frame. The plan records:

- requested and effective backend and preset identifiers;
- requested and effective provider version and digest;
- output and internal render extents;
- projection and required resource flags;
- fallback policy, resolution issue, and unavailability reason.

`native` is supplied by the host and is always available. It preserves the
exact output extent and does not perform reconstruction. A missing,
unavailable, incompatible, or malformed non-native backend either fails plan
resolution or resolves to native according to the saved fallback policy. No
silent fallback is allowed.

The host localizes the user-facing failure summary and shows a warning toast
with the requested backend/preset when native fallback starts. An ordinary
native export does not produce this warning. Structured issue IDs and
provider-specific unavailability reason IDs remain stable for logs and for
future resolution through owner-scoped plugin catalogs.

The initial catalog intentionally exposes only `native`. DLSS, FSR, and other
implementations are not named or loaded by this contract. Later viewer-only
plugin work can populate the same read-only interface after static discovery
and device availability checks.

That future **viewer-side adapter** must reuse the existing
`SceneUpscalerDescriptor`, `SceneUpscalerPreset`, and `SceneUpscalerSelection`
registry in `src/visualizer/rendering/scene_upscaler_registry.hpp`, including its
stable IDs (`native`, `spatial`, `temporal`, `nvidia-dlss`). Do not introduce a
parallel backend or preset registry. The adapter must copy metadata and add
offline/resource/projection capabilities; viewport availability alone does not
prove offline support. The serializable contract stays in `io`, with no reverse
dependency on the viewer registry. Export retains its own selection and history.

## Offline invariants

- Export selection never inherits viewport selection or viewport history.
- Project parsing and plan serialization never load a plugin.
- Fallback is resolved before scene capture and frame rendering when possible.
- The plan carries the requested/effective pair as the source for later export
  provenance integration.
- The current native export continues to use its existing renderer and encoder;
  this contract does not execute reconstruction yet.
- The contract contains no CUDA types or CUDA-specific behavior and adds no
  CUDA linkage beyond the existing video target.
- Equirectangular support must be declared explicitly by a backend descriptor.
- Unknown projection enum values fail with `invalid_projection`, including for
  native requests and requests that permit native fallback.
- Encoder and plan resolution share output extent validation: positive even
  dimensions and at most `INT_MAX / 3` pixels for packed-RGB signed indexing.
  Invalid dimensions fail before catalog lookup and cannot trigger fallback.
- Splat precision remains a separate concern and is not part of this selection.

## Persisted shape

```json
{
  "version": 1,
  "backend_id": "native",
  "preset_id": "native",
  "fallback": "abort"
}
```

The object is stored under `SEQR.preferences.reconstruction`. Existing SEQR
chapter version 1 remains valid when the field is absent.

Both identifiers match `[a-z0-9][a-z0-9._-]{0,127}`: lowercase ASCII letters or
digits first, then those characters plus `.`, `_`, and `-`; maximum 128 bytes.
The reserved `native` backend requires the `native` preset. Fallback is exactly
`abort` or `native`. The serialized JSON limit is 8 KiB, including whitespace.
Unknown object members are ignored by the selection parser. The project writer
validates through the canonical serializer and preserves same-version unknown
members using the existing SEQR merge rules. Only an unsupported future
selection is replaced as a whole; unrelated SEQR data remains preserved.

## Python control and recovery

The Sequencer button and `lf.ui.export_video(...)` snapshot the same persisted
selection on the viewer thread. Existing export arguments and Native defaults
are unchanged. Python access is marshalled to that thread with the GIL released
while waiting; unavailable or shutting-down viewers raise `RuntimeError`.

```python
import lichtfeld as lf

selection = lf.ui.get_video_reconstruction_selection()
# Metadata only: does not install, load, or probe a backend.
lf.ui.set_video_reconstruction_selection("com.example.reconstruction", "quality", "abort")
# Recovery from a saved missing-backend/abort selection:
lf.ui.reset_video_reconstruction_selection()
assert lf.ui.get_video_reconstruction_selection() == {
    "backend_id": "native", "preset_id": "native", "fallback": "abort"
}
```

The setter validates before mutation and raises `ValueError` for invalid IDs,
native/preset combinations, or fallback policies. Selecting an uninstalled
backend is allowed; availability is checked during export preflight. These APIs
change the live session preference, persisted on the next project save. They do
not modify viewport settings, legacy plugin behavior, or training.

## Regression checks

After building, run the native test filters `VideoReconstruction*`,
`P5SessionChapter*`, `ErrorEventBridgeTest.*`, `VideoExportUtilsTest.*`, and
`VideoEncoderValidationTest.*`. Binding tests are in
`tests/python/test_video_reconstruction_selection.py`; the round-trip test
requires a live viewer and otherwise explicitly skips. No test in that file
starts an export or saves a project, and the live test restores the selection.

Live export checks (a scene with Sequencer keyframes is required):

1. Open and re-save a legacy project: Native defaults and export remain unchanged.
2. Open a project with selection version 2: the scene opens, one warning appears,
   the selection is Native, and re-saving writes version 1.
3. Select an uninstalled backend with `abort`: both the button and Python export
   fail before writing output. Reset to Native and verify both can export again.
4. Select the same uninstalled backend with `native`: both paths export at the
   requested size, show the fallback warning, and log requested/effective IDs.
5. Change the UI language and repeat a failure/fallback: localized messages must
   appear and the error dialog must contain real line breaks, not literal `\n`.

The resolved plan still does not execute a second backend or add reconstruction
fields to the video provenance stamp. Worker reconstruction and durable
requested/effective provenance belong with the future backend adapter.
