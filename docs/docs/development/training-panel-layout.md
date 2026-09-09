# Training panel layout

The compact training-action toolbar is part of this same branch and change series,
not a separate PR D. All actions use the same compact icon-and-text buttons,
with localized short labels and full tooltips. No button stretches to fill the
panel. Save .licht is in the same row as Pause/Resume, not a separate full-width
row. Buttons use the proven RmlUi `inline-flex` sizing pattern and explicit
compact minimum widths; they wrap together only when necessary. The state is an
uppercase colored badge, following the video extractor SDR/HDR badge language,
and remains above the actions in Ready, Starting and every other state. Error
details wrap in a bounded scrollable area.
The existing state visibility, confirmation handlers and Start guard are retained.

| State | Primary action | Other actions |
|---|---|---|
| Ready | Start | Clear; Reset only when iteration > 0 |
| Starting | Pause | Stop |
| Running | Pause | Save Project |
| Paused | Resume | Reset, Stop, Save Project |
| Completed / Stopped | Edit Mode | Reset, Clear |
| Error | None; show error | Reset, Clear |
| Stopping | None; show status | None |

The RmlUi panel separates Training Method, Camera & Rasterization, Masking &
Segmentation, Supervision, Background, Appearance Correction, Dataset, and
Advanced Parameters. It consumes the backend descriptors from the backend
identity change; only installed, implemented descriptors appear in the selector.

Sparsity belongs under Advanced Parameters as a collapsible group, alongside
Initialization and Losses. Its Enable toggle remains inside the group; dependent
parameters appear only when enabled. The header participates in search and saved
section expansion state like the other advanced groups.

Generated rows retain their property metadata, numeric editing, tooltips and
runtime edit locks. Selecting a backend updates the next-run parameters and the
existing viewer setting. Unsupported capabilities are shown beside the selector;
Start is disabled when the central parameter check reports an error. The exact
error and selected unsupported options remain beside Start, outside search and
collapsible sections. Capability values use the `supported`/`unsupported` contract.
Incompatible unchecked Mip/Depth/Normal controls cannot be enabled with 3DGUT;
already-selected conflicts remain switchable off. Native rollback republishes
the effective values. Undistort remains available. No options are silently erased.
Direct Start events are checked before overwrite consent and again after consent;
native checks remain authoritative. The panel's next-run Start gate does not
gate Resume; native resume preflight from PR A still validates effective settings.

Appearance uses Off, Managed Exposure Correction, and Custom Stack. The custom
stack can enable Bilateral Grid, PPISP, or both. Selecting an empty custom stack
does not activate a module. Managed mode clears conflicting standalone enable
flags; numeric tuning and sidecar paths remain available for later use.

## Automatic settings and the padlock

The padlock remains beside Iterations. Locked mode retains the existing
dataset-based scaling and dependent refinement-field locks. Unlocked mode keeps
the manual editing behavior. Relocking uses the existing recalculation path.
The lock preference remains part of panel chrome; searching, expanding sections,
or changing backend/appearance does not toggle it.

## Manual checks

- In Ready at iteration zero, check 3DGS and 3DGUT selection, viewer alignment,
  and the capability notice. Check that unknown/unimplemented backends do not
  appear. Select 3DGUT with multiple unsupported options already enabled: Start must be disabled and list
  selected conflicts even with an unrelated search or collapsed sections. Remove
  conflicts one at a time, or select a compatible backend; Start must recover
  when the configuration is valid. Also check an invalid numeric parameter.
- Test all five mask modes, threshold, inversion, alpha fallback and penalties.
- Select each background mode and exercise the color picker, hex input and image
  browse/clear actions. MRNF background reconstruction remains a separate setting.
- Test Off, Managed, Bilateral-only, PPISP-only and Bilateral+PPISP. Check dependent
  controller, sidecar, grid and tuning fields.
- Toggle the Iterations padlock; change iterations and inspect scaled refinement
  fields. Unlock and edit manually, then relock. Check that search and project
  chrome restoration retain the lock preference.
- Search for Strategy, 3DGS, SH Degree, background image, dataset resize and
  save steps. Clear search and check prior section expansion is restored.
- Check a narrow panel, keyboard selection and the localized headings/tooltips.
- Check toolbar width, text containment and icon contrast with light/dark themes and long
  translations. Hover Reset, Stop and Clear to identify each action, and check
  keyboard focus and activation. Cancel their confirmation dialogs and confirm
  that no work is discarded. Save Project must remain directly visible while
  Running/Paused, including the saved confirmation message.
- Start, pause, resume, stop and restore training. Settings must obey the existing
  edit lock, while project saving and other training actions remain usable.

Run the source Python regressions using the interpreter matching the built
extension, with pytest, the built module and its runtime libraries available.
For the standard Windows triplet:

```powershell
.\build\vcpkg_installed\x64-windows\tools\python3\python.exe -m pytest tests/python/test_training_panel_regressions.py tests/python/test_property_view.py tests/python/test_training_confirm.py tests/python/test_property_system.py -q -p no:cacheprovider
```

These tests cover configuration behavior and RML structure, not native layout
rendering or GPU training.
