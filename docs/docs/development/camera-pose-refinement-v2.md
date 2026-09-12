# Camera pose refinement v2 — evidence gates

Date: 2026-09-12. Branch: `feature/camera-pose-refinement-v2`.
Base: `upstream/dev` `4841c7f213a3fe5459bd9a040d9cb4cc95475a71`,
verified against the upstream GitHub branch and fetched before branch creation.

## Product requirement

Correct small residual extrinsic errors when the image evidence supports it.
Preserve a useful no-change outcome for already accurate or underconstrained
datasets. Camera motion alone is not evidence of improved reconstruction.

**The user must see the actual cameras move during optimization.** The viewport,
Scene Graph, picking/focus and training must agree on the current pose. Keep the
imported pose immutable; publish UID-indexed, generation-correct CPU snapshots of
the current pose asynchronously. Reuse the existing frustum renderer and selection
behavior. Show an optional original-pose ghost and displacement vector plus net
translation/rotation. No geometric exaggeration or synthetic movement to make the
feature look active. A stopped/reset training must not leave stale pose overlays.

This visible movement is a required later gate, **not implemented in checkpoint A**.
Do not imply that backend plumbing alone makes cameras move in the application.

## Checkpoint A: numerical seam and photometric recovery

Implemented in source; the user passed all eight checkpoint A tests on
2026-09-12 at 13:22:24, and the agent validated the resulting XML:

- Optional per-render FastGS pose tensors `[1,4,4]` and `[3]`, retained by the
  forward context through backward. Caller establishes rigid SE(3), coherent
  center `C=-R^T t`, matching CUDA stream and immutable contents.
- Optional caller-owned `[4,4]` camera gradient, overwritten per backward.
  Shape/dtype/stream and overlap with pose/backward inputs are rejected explicitly.
  The caller must also keep it disjoint from optimizer state and external scratch.
- Direct covariance derivative `J^T dL/d(JR)` and SH camera-center contribution
  added to the optional camera path. The matrix gradient is tangent-consistent
  for `C=-R^T t`; it is not an unconstrained 16-parameter camera optimizer.
- Camera-only backward uses the existing fused dispatch with all per-parameter
  Adam settings disabled; it does not prepare/commit Adam, densification or edge
  statistics. Existing calls keep the default Gaussian-update behavior.
- Reused only the isolated SE(3) math from the historical prototype; the old
  optimizer, scheduler, UI and evidence heuristics are not transplanted.
- Eight dataset-independent CUDA tests with strict inventory and XML validation.

No Trainer hookup, production pose optimizer, UI checkbox, persistence or
real-dataset quality improvement is claimed at this checkpoint.

### Native gate inventory

Suite: `CameraPosePhotometricTest`.

| Test | Evidence |
| --- | --- |
| AnisotropicSH3MatchesSixAxisFiniteDifferences | Actual rendered RGB derivatives at a nonidentity pose; six tangent axes, anisotropic splats and SH3 |
| OnAxisAnisotropicRollHasNonzeroGradient | Covariance rotation isolated where the old center-only roll derivative was zero |
| SHViewDirectionSurvivesGeometryCancellation | One splat: RGB/alpha normalization cancels footprint change, isolating SH view dependence |
| CameraOnlyPreservesModelOptimizerAndSource | Model, packed Adam state, step counters, densification and original camera unchanged |
| TiledGradientMatchesFullImage | Sum of per-tile gradients agrees with full-image gradient |
| OptionalGradientPreservesJointGaussianUpdate | Enabling camera output preserves the ordinary Gaussian update |
| RejectsInvalidContractsBeforeBackward | Invalid tensor, alias, missing-output and unsupported-loss inputs rejected |
| RecoversPerturbedPoseFromImagesWithFixedGeometry | Three predetermined pose perturbations; updates use rendered RGB residual only |

The recovery driver is a bounded **test-only** damped Gauss–Newton reference:
numerical image Jacobian for conditioning, analytical CUDA loss gradient for the
step, at most 24 iterations and 8 backtracks per iteration. It is intentionally
not the production optimizer and its runtime is not a product performance claim.
The true pose is used only to generate reference images and report pose errors,
never as an update gradient or acceptance criterion.

Predeclared recovery thresholds, on all three trials:

- final mean RGB squared error below 10% of initial;
- camera-center and rotation errors each below 30% of initial;
- at least one accepted photometric improvement;
- all eight tests actually run and complete without failures/skips;
- per-trial finite metrics included in the XML.

These are provisional engineering acceptance thresholds, not measured results.
Do not relax them just to pass. Diagnose a failure and change the algorithm or
document a justified scope correction before rerunning.

After the user updates the native test binary, the current combined A+B gate is
one synthetic run (no image datasets or manual viewport inspection):

```powershell
. C:/Users/franz/Documents/PowerShell/Enter-LichtFeldDev.ps1
./build/tests/lichtfeld_tests.exe --gtest_filter="CameraPosePhotometricTest.*:CameraPoseControllerTest.*" "--gtest_output=xml:$env:TEMP/camera-pose-controller.xml"
lfspython tools/check_camera_pose_gate.py "$env:TEMP/camera-pose-controller.xml" --controller
```

Write future test reports/logs outside the repository (Windows TEMP above), so
generated artifacts are not proposed for commits. Existing user reports are
left in place and excluded from staging.

The checker reads the XML only; it does not attest binary/source provenance.
A missing suite in an older binary, missing test, skip, nonfinite metric or
failed threshold is a failed gate. No build/configure/clean is performed by the
agent. The Python validator's own unit tests do not validate CUDA.

### Supported scope of checkpoint A

FastGS RGB/alpha/depth camera gradient seam, with RGB covered by the new gate.
Depth-specific validation still pending. Mip Filter and normal-loss camera
gradients are rejected pending complete derivatives and matching tests. Their
ordinary Gaussian-training paths remain available when no camera gradient is
requested. 3DGUT has no new camera-gradient path in this checkpoint.

The off-path numerical behavior has an added comparison test; its execution and
performance regression checks remain pending. No zero-overhead claim is made.

## Checkpoint B: bounded per-camera controller

`BoundedPoseOptimizer` is a new CPU controller intended for integration with
FastGS camera-only backward. It is not yet connected to Trainer or the viewport.
The reference Gauss-Newton test from A remains; B adds a separate photometric
test using this controller and no numerical image Jacobian/Hessian.

- Source pose is immutable. Current pose and counters are keyed by UID; baseline
  UID/pose/model revisions are checked. Reset invalidates old baselines.
- Camera-center steps are capped at 0.3% of the declared scene scale and rotations
  at 0.25 degrees. Cumulative limits are 3% and 3 degrees from source. Weak source
  priors penalize center displacement and rotation chord distance separately.
- Scaled inverse BFGS supplies a safeguarded descent direction. History is reused
  only while Gaussian geometry, appearance, resolution, mask and objective remain
  fixed. Model changes discard history. It is a direction approximation in
  left-tangent coordinates, not an exact manifold Hessian.
- At most eight candidate-loss renders per call with defaults. Acceptance requires
  actual image-loss decrease and Armijo decrease of image loss plus priors. Rejected
  or nonfinite candidates do not change poses. Callback exceptions roll back all
  controller state. Rejection is not reported as convergence.
- Frozen, anchor and evaluation roles never update or invoke candidate rendering.
  Automatic anchor selection and global frame/scale constraints are still a
  Trainer-integration responsibility, not a claim of this per-camera component.
- Value snapshots contain UID, revision, source/current pose, accepted/rejected
  counts and actual net center/rotation displacement. They are ready for a future
  snapshot publisher; no async publication or visible camera movement exists yet.

The combined gate requires **17 tests**: the original eight, one three-trial
image-driven controller recovery test, and eight controller contract tests.
Controller recovery is capped at 80 steps per trial, with the same 10% image-loss
and 30% pose-error ratios required by A. Candidate renders must be reported and
cannot exceed 640 per trial. This is a test budget, not a training cadence or a
runtime performance claim. `--controller` rejects an older A-only report.

An agent-run double-precision CPU numerical experiment on the same fixture
recovered all three poses in 24 accepted steps and 24 candidate renders per trial.
Center error ratios were 0.08148/0.03288/0.11673 and rotation ratios
0.05044/0.02233/0.06710. This used numerical derivatives on fixed contribution
membership; it supports the algorithm choice but does not execute the C++ controller
or certify its CUDA integration. The user subsequently compiled and ran all 17
native tests successfully at 13:43:17 on 2026-09-12, with no failures or skips.
The C++ controller recovered each of the three poses in 24 accepted steps and
24 candidate renders. Center error ratios were 0.081523/0.032855/0.116696;
rotation error ratios were 0.050466/0.022315/0.067084, and image-loss ratios
0.000064/0.000030/0.000089. Native compilation/execution was performed by the
user, not the agent. Eleven Python report-validator tests passed; static error-debt and
whitespace checks passed. GUI, checkpoint persistence, gauge anchoring, automatic
scene scale, cadence and joint-training quality checks remain pending.

## Subsequent gates — ordered, not concurrent feature expansion

1. **Native checkpoint A evidence (passed).** Passing recovery with fixed geometry proves
   the pose signal can recover small errors; it does not prove joint training
   improves a reconstruction.
2. **Production controller and joint integration.** Small coarse-to-fine pose
   corrections; scene-scale-aware priors and cumulative bounds; rotation and
   translation considered separately. Fix both global frame and scale freedoms
   without overconstraining noisy observations. Count actual per-camera updates;
   make convergence depend on controlled image evidence, not tiny Adam steps.
   Consider brief fixed-geometry acceptance checks and a final pose-freeze tail.
   Test handling of changing camera membership, evaluation exclusions and reset.
3. **Quality experiment.** First one strategy and three paired off/on cases:
   good poses, controlled perturbations of train poses, a real problematic
   dataset. Same seed, initialization, resolution, split and Gaussian budget.
   Never optimize evaluation poses. Measure held-out PSNR/SSIM/LPIPS, known pose
   error, fixed-view artifacts, Gaussian count, wall time and VRAM. Compare
   both equal-work and cost-adjusted outcomes; repeat promising/borderline cases
   with fixed additional seeds. Hold final test views out of runtime acceptance
   and parameter tuning. Predeclare practical benefit/cost thresholds before runs.
   If gains are absent or inconsistent, stop before adding UI complexity.
4. **Compatibility and persistence.** Integrate backend capability reporting,
   supported losses, masks, PPISP/bilateral, tiling and strategy-specific
   densification. Persist source/current poses plus optimizer/controller state;
   verify checkpoint/project resume and Edit Mode ownership. Unsupported cases
   must be explicit and must not silently switch training backends.
5. **Visible camera movement.** Implement the snapshot/frustum requirement above
   on the current viewer architecture, then one consolidated app collaudo:
   current/source agreement, real net displacement, selection/focus, pause,
   reset, reopen and undistort consistency.

Do not restore confidence colors or outlier heuristics merely because they
existed in v1. A statistically narrow COLMAP footprint is not proof of pose error.
Do not advertise universal quality gain, recovery of missing coverage, deblurring
or rolling-shutter correction from a rigid six-DoF pose optimizer.

## Validation status

The user's 2026-09-12 native run executed all eight tests: seven passed, and
`AnisotropicSH3MatchesSixAxisFiniteDifferences` failed with 12.4438% relative
gradient error against the 7% limit. Fixed-geometry recovery passed all three
trials, with accepted-step counts 4/2/5, center error ratios
0.104105/0.069831/0.112211, and rotation error ratios
0.067120/0.045718/0.059006. These are synthetic results, not real-data results.

The agent reproduced the discrepancy with a double-precision CPU mathematical
reference (`tools/diagnose_camera_pose_cutoff.py`, standard-library Python).
At epsilon 1e-3 the alpha=1/255 cutoff changes contribution membership; the CPU
natural secants reproduce the failing native numerical derivatives. Holding
membership fixed reproduces the native analytical gradients. At both 1e-5 and
5e-6 there are zero changed contribution pairs for all six axes and the natural
and fixed-membership derivatives agree. X translation, for example, gives
-0.003330676 at 1e-3 versus local derivative -0.002781019, matching the user log.

The corrected native test keeps 1e-3 as a printed diagnostic and requires BOTH
predetermined fine scales to satisfy the original component and 7% norm bounds,
plus agreement between the numerical estimates. No tolerance was increased.
The recovery driver and production renderer are unchanged by this correction.
The CPU reference omits GPU rounding, tiling/culling optimizations and storage
codecs; it explains the discrepancy but does not certify CUDA.

The user's subsequent native run at 13:22:24 passed all eight A tests with no
skips. Fine-scale gradient errors were 0.005686 and 0.006681 (0.57% and 0.67%);
the original coarse diagnostic stayed at 0.124438. The agent ran the A XML
validator successfully on `camera-pose-gate.xml`. This validates checkpoint A;
checkpoint B also passed all 17 combined native tests in the user's 13:43:17 run.
The app does not yet expose pose refinement v2. Real quality
benefit remains unproven until subsequent gates have measurements.
