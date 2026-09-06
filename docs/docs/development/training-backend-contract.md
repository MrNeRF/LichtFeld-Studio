# Training backend identity and compatibility

Training strategy and raster backend are independent choices. `fastgs` is the
default training backend; `3dgut` selects Gaussian Unscented Transform rendering.
The viewer calls its standard rasterizer `3dgs`: that viewer identifier is not a
third training backend.

`core/training_backend.hpp` owns the backend identifiers, labels, viewer mapping,
and verified training capabilities. Central parameter validation consumes these
capabilities. The current capability set covers IGS+, undistortion, Mip Filter,
depth supervision, and normal supervision. All five are unsupported with 3DGUT.
Mask, segmentation, and appearance support must be assessed separately; a missing
capability key does not imply support or disable an existing feature.

## Compatibility rules

- `OptimizationParameters::raster_backend()` and `set_raster_backend()` adapt the
  existing `gut` storage. There is no second mutable backend field. Existing C++
  and Python writers of `gut` therefore remain effective.
- JSON accepts `raster_backend: "fastgs"` or `"3dgut"`. Files containing only
  `gut` retain their original meaning. Missing both selects the existing default.
- New JSON writes both names consistently. While `gut` remains the compatibility
  storage, its value takes precedence if a previous application preserved a stale
  known `raster_backend` field while changing `gut`. Unknown identifiers and
  incorrect JSON types are still errors.
- `--raster-backend fastgs|3dgut` is additive. `--gut` remains an alias for 3DGUT.
  `--gut --raster-backend fastgs` is an error, independent of argument order.
- Explicit CLI selection overrides a valid configuration's backend. Its captured
  overrides contain both aliases so subsequent checkpoint restoration cannot
  combine a stale config alias with the CLI choice. Invalid configurations still
  fail their existing load-time validation before CLI overrides are applied.
- Checkpoint parameter JSON and `.licht` parameter presets use the same adapters.
  The binary checkpoint format and `.licht` chapter schema are unchanged.
- A stored `.licht` preset may retain an unsupported 3DGUT option so legacy
  projects can still open without silently changing user data. Storage validation
  continues to reject malformed values; trainer validation reports the backend
  conflict when Start is requested.
- Older applications ignore the new field and continue to read `gut`. In `.licht`,
  they can preserve the unknown new field while changing `gut`; the precedence
  rule above ensures that the file still reopens with the legacy application's
  selected backend.

## Python and viewer lifecycle

`lf.optimization_params().raster_backend` and `get/set("raster_backend", ...)`
access the same backend as `gut`. The new setter uses the existing `gut` property
notification path. `backend_capabilities` returns the verified restrictions for
the selected backend. `lf.training_backends()` lists IDs, labels, viewer IDs, and
capabilities without needing to change the current selection.

These are next-run parameter APIs, not an atomic command to replace an active
trainer and viewer. The existing panel change path still updates its viewer
setting; viewer startup receives the resolved compatibility value. Project
restoration preserves the distinction between session defaults, next-run presets,
and active trainer state. The future RmlUi selector should use the descriptors
and the existing scene/lifecycle commands rather than mutating active training
from a parameter setter.

## Adding a backend later

This is a compatibility step, not runtime backend registration. A third backend
needs its implementation, descriptor, capability assessment, explicit dispatch,
and a replacement for binary `gut` storage with legacy accessors. Unknown enum
values are rejected by the compatibility setter. Do not map a third backend to
either boolean value or assume that adding a descriptor installs a rasterizer.

## Verification

After updating the native binaries, run from the repository in the normal
PowerShell development environment:

```powershell
lfsdev
.\build\tests\lichtfeld_tests.exe --gtest_filter="TrainingParametersTest.BackendIdentityCompatibility:TrainingParametersTest.StoredBackendConflictPreservesSettingsButStillRejectsInvalidNumbers:ArgumentParserTest.ExplicitBackendSelectionAndLegacyAlias:ArgumentParserTest.BackendCliOverrideReplacesConfigAliasesTogether:ArgumentParserTest.ViewerBackendSelectionSurvivesParameterDefaults:ProjectChapterTest.TrainingBackendIdentityRoundTripAndCompatibility:ParameterManagerTest.PendingProjectRestoreChangesOnlyRoleQualifiedManagerState" --gtest_color=no
lfspytest tests/python/test_property_system.py -q -p no:cacheprovider
```

Manual checks: open legacy FastGS and 3DGUT projects/checkpoints, save and reopen
them, change the backend through Python and the existing panel, and verify a
valid training run with each backend. Exercise CLI selection, config overrides,
viewer startup, and invalid aliases. Confirm that changing next-run settings
does not replace an active trainer. The parameter/chapter tests do not prove
end-to-end project restoration or GPU training behavior.
