# Epic #1496 campaign findings (side discoveries)

## Pre-existing: VUID-vkCmdDispatch-storageBuffers-06936 in vksplat.cumsum (2026-08-06)
GPU-assisted validation (scripts/run_vulkan_validation.sh) reports out-of-range storage buffer
access in the cumsum pipeline, 10+ hits per session, on a tree whose rasterizer code is
bit-identical to master (only image-tracker call sites changed at that point). Not caused by the
epic. Candidate real bug: cumsum shader indexing past the bound descriptor range for some element
counts. USER ORDER: fix on this branch (no filing/postponing).

RESOLVED 2026-08-06 after two investigation rounds (inv_cumsum_report.md, inv_r2_report.md in
the session scratchpad; verdicts mirrored here):
- The app is correct. A `require_backing` contract added to `executeCumsum` (permanent) proves
  every classic scan's input/output/block-sum backings are large enough at record time; it never
  fired while the VUID still appeared. SPIR-V artifacts clean; recording is single-threaded;
  core+sync validation is clean on the same workloads.
- The reports are GPU-assisted-validation false positives in the pinned layer
  (vulkan-validationlayers 1.4.341.0): GPU-AV pairs stale push-descriptor snapshots with
  instrumented dispatches — Vulkan-ValidationLayers issue #11433 (verified open, same VUID,
  same push-descriptor mechanism, reporter confirms errors vanish without push descriptors).
- Fix on this branch: scripts/run_vulkan_validation.sh now defaults to core + synchronization
  validation (the epic's gate) with GPU-AV behind an explicit --gpu-av flag documenting the
  false-positive class. Revisit the flag default when the layer pin advances past a #11433 fix.
