# Training panel layout

The compact training-action toolbar is part of this same branch and change series,
not a separate PR D. All actions use the same compact icon-and-text buttons,
with localized short labels and full tooltips. No button stretches to fill the
panel. Save .licht is in the same row as Pause/Resume, not a separate full-width
row. Buttons use RmlUi `inline-flex` sizing. Compact mode compares RmlUi's measured
localized action widths, gaps and status badge against the available toolbar width.
Measurements follow the current language, UI scale, state and panel size. When the captions do not fit,
RCSS hides button captions and uses 28dp icon buttons; tooltips and accessible
names remain. The outer toolbar does not wrap. The state is an
uppercase colored badge, following the video extractor SDR/HDR badge language,
and sits to the right of the actions in the same row. Clear uses
secondary styling without changing its confirmation flow. Error
details wrap in a bounded scrollable area below the toolbar, outside the action
group, so multiline errors cannot move the badge relative to Reset/Clear.
The existing state visibility, confirmation handlers and Start guard are retained.
Project restoration shows its own badge. A restore failure remains visible with
Error actions; once a live trainer exists, its own error takes precedence over
stored restoration feedback.
During stopping, the badge distinguishes model saving from stopping. This
transient state is checked every 100ms because saving has no separate runtime
signal; that polling ends when the trainer leaves stopping.

| State | Primary action | Other actions |
|---|---|---|
| Ready | Start | Clear; Reset only when iteration > 0 |
| Starting | Pause | Stop |
| Running | Pause | Save Project |
| Paused | Resume | Reset, Stop, Save Project |
| Completed / Stopped | Edit Mode | Reset, Clear |
| Error | None; show error | Reset, Clear |
| Stopping / Saving / Restoring | None; show status | None |

The initial RmlUi panel exposes Training Method (strategy, backend, iterations,
padlock, capacity and BG Improvements), Camera &
Rasterization (Exposure Correction/Undistort/Mip), Masking & Segmentation, and Dataset
immediately before Advanced. Dataset starts expanded; a saved collapse preference
still takes precedence. It retains its own edit locks;
searching its fields does not open Advanced. Advanced is a real
collapsible container for optional activation and specialist settings. It consumes
the backend descriptors from the backend
identity change; only installed, implemented descriptors appear in the selector.

Advanced contains one activation checkbox each for Depth, Normal, Bilateral Grid,
PPISP, Sparsity, Evaluation and Random Initialization. These write the existing
training parameters through the property bindings; they are not visibility
preferences and are not duplicated in the detail groups. Enabling a feature opens
its settings. Loaded parameter values drive detail visibility without a separate
UI enable flag; saved expansion preferences remain independent.
Depth, Normal, PPISP, Bilateral Grid, Exposure Correction, Evaluation, Random
Initialization and Sparsity have independent conditional detail sections. Depth
and Normal never share a parameter group. Evaluation owns its interval; random
initialization owns point count and extent. Shared appearance tuning is visible
under Exposure Correction when managed correction is enabled, otherwise under
the enabled standalone PPISP/Grid section. These mutually exclusive views use
the same parameter bindings and do not create additional enable controls. Background modes, SH degree, optimization, losses,
initialization and save steps also live inside Advanced. No artificial enable
flag is added to always-applicable settings.

Generated rows retain their property metadata, numeric editing, tooltips and
runtime edit locks. An unchanged native value does not overwrite a focused draft;
authoritative changes still replace it, including a numeric rollback arriving
after the first post-edit refresh. Native refresh synchronizes buffers before
publishing, without queueing another refresh. Selecting a backend updates the next-run parameters and the
existing viewer setting. Unsupported capabilities are shown beside the selector
as `Not available with {backend}: {features}`: the backend label comes from its
descriptor and the localized feature list follows the capability states, not a
backend-specific sentence. The notice is muted information, not a Start error.
Start is disabled when the central parameter check reports an error. The exact
error and selected unsupported options remain beside Start, outside search and
collapsible sections. Capability values use the `supported`/`unsupported` contract.
Incompatible unchecked Mip/Depth/Normal controls cannot be enabled with 3DGUT;
already-selected conflicts remain switchable off. Native rollback republishes
the effective values. Undistort remains available. No options are silently erased.
Direct Start events are checked before overwrite consent and again after consent;
native checks remain authoritative. The panel's next-run Start gate does not
gate Resume; native resume preflight from PR A still validates effective settings.

Exposure Correction has one checkbox in the main controls. Bilateral Grid and
PPISP have their sole enable controls in Advanced and may be combined. Existing
exclusivity between Exposure Correction and standalone appearance flags is
retained, as are tuning values and sidecar paths. There is no second appearance
mode selector. Rejected enable attempts do not modify other feature flags.

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
- Test Exposure Correction off/on, Bilateral-only, PPISP-only and Bilateral+PPISP. Check dependent
  controller, sidecar, grid and tuning fields.
- Toggle the Iterations padlock; change iterations and inspect scaled refinement
  fields. Unlock and edit manually, then relock. Check that search and project
  chrome restoration retain the lock preference.
- Search for Strategy, 3DGS, SH Degree, background image, dataset resize and
  save steps. Clear search and check prior section expansion is restored.
- With Advanced collapsed, search `means_lr` and `use_normal_loss`: both the
  ancestor and matching section must open temporarily. Search
  `use_exposure_correction` and check its Camera activation remains reachable.
  Search `bg_image` in Color mode: Mode and Browse must remain available.
- Edit Iterations and press Escape: the draft must be discarded even when the
  field immediately loses focus. A later edit must still commit normally.
- In Ready, type Max Gaussians slowly from 5,000,000 to 4,000,000, leaving the
  field empty briefly. No refresh may restore the old value mid-edit. Check Enter,
  blur, Escape and +/- separately, then repeat with Iterations and a learning rate.
  After Enter, make another draft without leaving the field: Escape must return
  to that last committed value. Actual native rejection must still restore the
  authoritative value rather than leaving a misleading draft displayed.
- Reproduce a failed checkpoint trainer restore, then a separate training error:
  the badge and persistent detail must describe the current failure, not stale
  restoration feedback.
- Check narrow and wide panels (including Paused with all four
  actions), keyboard selection, tooltips and the status badge. Also test UI scaling
  and longer translations; native visual verification is still required.
- Enable Depth and Normal together: verify their parameter groups are separate.
  Do the same for standalone Bilateral Grid and PPISP, then enable Exposure
  Correction and check that only its shared tuning view is visible.
- Enable each Advanced feature, verify the real flag and its settings, then
  disable it and confirm tuning values remain available after re-enabling.
  Load a valid configuration with features already active and check their details.
  Confirm each activation exists exactly once and search can reach it while off.
- Check toolbar width, text containment and icon contrast with light/dark themes and long
  translations. Hover Reset, Stop and Clear to identify each action, and check
  keyboard focus and activation. Cancel their confirmation dialogs and confirm
  that no work is discarded. Save Project must remain directly visible while
  Running/Paused, including the saved confirmation message.
- Start, pause, resume, stop and restore training. Settings must obey the existing
  edit lock, while project saving and other training actions remain usable.
  During final model saving, verify SAVING replaces STOPPING and then gives way
  to the final state; neither badge should offer new training actions.

Run the source Python regressions using the interpreter matching the built
extension, with pytest, the built module and its runtime libraries available.
For the standard Windows triplet:

```powershell
.\build\vcpkg_installed\x64-windows\tools\python3\python.exe -m pytest tests/python/test_training_panel_regressions.py tests/python/test_property_view.py tests/python/test_training_confirm.py tests/python/test_property_system.py -q -p no:cacheprovider
```

These tests cover configuration behavior and RML structure, not native layout
rendering or GPU training.
