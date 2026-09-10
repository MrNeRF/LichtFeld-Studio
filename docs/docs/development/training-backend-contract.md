# Training backend identity and compatibility

Training strategy and raster backend are independent choices. `3dgs` is the
default training backend; `3dgut` selects Gaussian Unscented Transform rendering.
No additional backend alias is introduced for 3DGS.

`core/training_backend.hpp` owns the backend identifiers, labels, descriptions,
viewer mapping, and high-level training capabilities. Its descriptions provide
technical tooltip text for later UI consumers. The capability map is exhaustive
for the high-level feature groups exposed by the training panel; a consumer does
not need to infer support from a missing key.

Each capability has one of two states:

- `supported`: the backend path is established and may be presented normally;
- `unsupported`: the combination is a known hard incompatibility and may prevent
  training from starting.

Only `unsupported` participates in central backend validation. The current
matrix is:

| Feature group | 3DGS | 3DGUT |
| --- | --- | --- |
| MCMC strategy | supported | supported |
| MRNF strategy | supported | supported |
| IGS+ strategy | supported | unsupported |
| Undistortion | supported | supported |
| Mip Filter | supported | unsupported |
| Depth supervision | supported | unsupported |
| Normal supervision | supported | unsupported |
| Masking | supported | supported |
| Segmentation | supported | supported |
| Background modes | supported | supported |
| Background Improvements | supported | supported |
| Exposure correction | supported | supported |
| Bilateral grid | supported | supported |
| PPISP | supported | supported |
| Sparsity | supported | supported |

This matrix describes backend compatibility, not complete option availability.
Detailed numeric controls, strategy-specific applicability, mutually exclusive
option groups, and dataset requirements still belong to the property registry
and the later resolved UI model. For example, Background Improvements is an MRNF
option even though the backend matrix also reports its backend support state.
Localized conflict reasons continue to come from the existing backend-conflict
result instead of duplicating UI text in the descriptor.

## Compatibility rules

- `OptimizationParameters::raster_backend()` and `set_raster_backend()` adapt the
  existing `gut` storage. There is no second mutable backend field. Existing C++
  and Python writers of `gut` therefore remain effective.
- JSON accepts `raster_backend: "3dgs"` or `"3dgut"`. Files containing only
  `gut` retain their original meaning. Missing both selects the existing 3DGS default.
- New JSON writes both names consistently. If both fields are present, they
  must agree. A disagreement, unknown identifier, or incorrect JSON type is an
  error, including in configs, checkpoint parameters, and project presets.
- For example, `{"gut": true, "raster_backend": "3dgs"}` is rejected;
  `{"raster_backend": "3dgs"}` selects 3DGS. Config authors must update both
  fields consistently or specify only one. Explicit CLI selection overrides a
  valid config, not an invalid file that already failed loading.
- `--raster-backend 3dgs|3dgut` is additive. `--gut` remains an alias for 3DGUT.
  `--gut --raster-backend 3dgs` is an error, independent of argument order.
- Explicit CLI selection overrides a valid configuration's backend. Its captured
  overrides contain both aliases so subsequent checkpoint restoration cannot
  combine a stale config alias with the CLI choice. Invalid configurations still
  fail their existing load-time validation before CLI overrides are applied.
- Checkpoint parameter JSON and `.licht` parameter presets use the same adapters.
  The binary checkpoint format and `.licht` chapter schema are unchanged.
- Stored parameters follow the strict validation established by PR #2047.
  Unsupported 3DGUT combinations are rejected by validating load paths, as
  well as start/resume preflight. There is no separate storage-validation mode,
  inherited-option normalization, or older `.licht` writer precedence rule.

## Python and viewer lifecycle

`lf.optimization_params().raster_backend` and `get/set("raster_backend", ...)`
access the same backend as `gut`. The new setter uses the existing `gut` property
notification path. `backend_capabilities` returns the complete high-level map for
the selected backend using the strings `supported` and `unsupported`.
`lf.training_backends()` lists IDs, labels, descriptions, viewer IDs, and the
same capability map without needing to change the current selection.
`set("raster_backend", value)` rejects non-string values and unknown names with
an actionable `ValueError`, without changing the selected backend. `None` retains
the existing binding-level `TypeError` before setter dispatch.

These are next-run parameter APIs, not an atomic command to replace an active
trainer and viewer. The existing panel change path still updates its viewer
setting; viewer startup receives the resolved compatibility value. Project
restoration preserves the distinction between session defaults, next-run presets,
and active trainer state. The future RmlUi selector should use the descriptors
and the existing scene/lifecycle commands rather than mutating active training
from a parameter setter.

Training and evaluation dispatch use the named identity. The status bar uses
the active trainer's backend or, before restoration, the saved checkpoint's
backend. Missing identity is omitted rather than presented as 3DGS. Labels come
from the backend descriptors, independently of viewer and next-run choices.
The existing viewer hand-off in `application.cpp` and `visualizer_impl.cpp` and
training panel still consume the compatibility `gut` boolean. These consumers
must also migrate before a backend beyond the current two can be installed.
Training and viewer identifiers use the same `3dgs`/`3dgut` vocabulary.

## Adding another backend later

This is a compatibility step, not runtime backend registration. An additional
backend needs its implementation, descriptor, an explicit state for every
high-level capability, explicit dispatch, and a replacement for boolean `gut`
storage with accessors for the existing API. Unknown enum values are rejected by the
compatibility setter. Do not map an additional backend to either boolean value or
assume that adding a descriptor installs a rasterizer.

## Verification

After updating the native binaries, run from the repository with the application's
runtime libraries on the library search path. The chapter test belongs to
`lichtfeld_format_tests`, not `lichtfeld_tests`.

```powershell
.\build\tests\lichtfeld_tests.exe '--gtest_filter=TrainingParametersTest.*:ArgumentParserTest.*Backend*:ArgumentParserTest.Gut*:TrainerConstructionTest.*:CheckpointParamsJsonTest.*:ParameterManagerTest.PendingProjectRestoreChangesOnlyRoleQualifiedManagerState' --gtest_color=no
.\build\tests\lichtfeld_format_tests.exe '--gtest_filter=ProjectChapterTest.TrainingBackendIdentityRoundTripAndCompatibility' --gtest_color=no
.\build\vcpkg_installed\x64-windows\tools\python3\python.exe -m pytest tests/python/test_property_system.py -q -p no:cacheprovider
```

On Linux, with the build's Python module and libraries available:

```sh
./build/tests/lichtfeld_tests --gtest_filter='TrainingParametersTest.*:ArgumentParserTest.*Backend*:ArgumentParserTest.Gut*:TrainerConstructionTest.*:CheckpointParamsJsonTest.*:ParameterManagerTest.PendingProjectRestoreChangesOnlyRoleQualifiedManagerState'
./build/tests/lichtfeld_format_tests --gtest_filter='ProjectChapterTest.TrainingBackendIdentityRoundTripAndCompatibility'
./build/vcpkg_installed/x64-linux/tools/python3/python3.12 -m pytest tests/python/test_property_system.py -q -p no:cacheprovider
```

These paths assume the standard vcpkg triplets. Use the Python interpreter
matching the built extension, with pytest installed and the built module and
its runtime libraries available; a system Python is not interchangeable.
Local shell helpers
are not repository prerequisites. After changing native Python bindings,
generate the committed stubs from the rebuilt module via `refresh_python_stubs`
and run `check_python_stubs`; do not maintain binding stubs by hand.

Manual checks: open valid 3DGS and 3DGUT projects/checkpoints, save and reopen
them, change the backend through Python and the existing panel, and verify a
valid training run with each backend. Exercise CLI selection, config overrides,
viewer startup, and invalid aliases. Confirm that changing next-run settings
does not replace an active trainer. The parameter/chapter tests do not prove
end-to-end project restoration or GPU training behavior.
