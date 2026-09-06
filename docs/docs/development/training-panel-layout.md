# Training panel layout

The compact training-action toolbar will continue on this same branch and in the
same change series; it is not a separate PR D. This checkpoint implements panel
grouping and preventive Start feedback, not the compact toolbar layout yet.

The RmlUi panel separates Training Method, Camera & Rasterization, Masking &
Segmentation, Supervision, Background, Appearance Correction, Dataset, and
Advanced Parameters. It consumes the backend descriptors from the backend
identity change; only installed, implemented descriptors appear in the selector.

Generated rows retain their property metadata, numeric editing, tooltips and
runtime edit locks. Selecting a backend updates the next-run parameters and the
existing viewer setting. Unsupported capabilities are shown beside the selector;
Start is disabled when the central parameter check reports an error. The exact
error and selected unsupported options remain beside Start, outside search and
collapsible sections. Options remain editable and are never silently erased.
Direct Start events are checked before overwrite consent and again after consent;
native checks remain authoritative. Resume is not gated by next-run settings.

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

- In Ready at iteration zero, check FastGS and 3DGUT selection, viewer alignment,
  and the capability notice. Check that unknown/unimplemented backends do not
  appear. Enable multiple unsupported options: Start must be disabled and list
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
- Search for Strategy, FastGS, SH Degree, background image, dataset resize and
  save steps. Clear search and check prior section expansion is restored.
- Check a narrow panel, keyboard selection and the localized headings/tooltips.
- Start, pause, resume, stop and restore training. Settings must obey the existing
  edit lock, while project saving and other training actions remain usable.

Run the source Python regressions in the existing LichtFeld environment:

```powershell
lfsdev
lfspytest tests/python/test_training_panel_regressions.py tests/python/test_property_view.py tests/python/test_training_confirm.py -q -p no:cacheprovider
```

These tests cover configuration behavior and RML structure, not native layout
rendering or GPU training.
