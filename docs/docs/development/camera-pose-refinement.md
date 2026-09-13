# Camera pose refinement

Camera pose refinement corrects small residual camera extrinsic errors using
rendered image gradients. Accurate or underconstrained cameras may remain
unchanged. Camera movement and decreasing training loss do not, by themselves,
establish improved reconstruction quality.

The implementation consists of SE(3) operations, a bounded per-camera optimizer,
a multi-camera session and a FastGS evaluator, with opt-in Trainer integration
and embedded checkpoint persistence. Live viewport geometry and Scene Graph
displacement indicators consume published poses.

## Activation

Add `--refine-camera-poses` when starting a new training session. The equivalent
optimization JSON property is `"refine_camera_poses": true`. The option is disabled
by default and is retained in saved training parameters. Existing configuration
files without this property retain their previous behavior.

For example, replace the dataset and output paths with local paths:

```text
LichtFeld-Studio -d DATASET -o OUTPUT --iter 3000 --refine-camera-poses
```

The default schedule starts at iteration 500 and freezes poses at 80% of the
training duration (iteration 2400 in this example). There must be an update window
between warmup and freeze. Each eligible camera visit permits at most two update
steps, with subsequent bursts spaced by eight visits to that camera. The
initialization log reports the camera count, warmup, freeze iteration and cadence.

Enable camera frustums in the viewport and expand the camera nodes in Scene Graph
to observe accepted pose changes and net displacement. Two reference cameras and
all evaluation cameras remain fixed. Well-calibrated cameras may also stay still.
Publication during visits is throttled to 250 ms; movement is shown at its actual
scene scale.

Activation requires Trainer reinitialization; changing the option on an initialized
Trainer is rejected. Saved pose state restores automatically, including its own
schedule, even when the launch flag is absent. The command-line option does not
discard saved poses or override the checkpoint's pose schedule. A dedicated
Training-panel control is not yet connected.

## Pose representation

Poses are row-major world-to-camera matrices with shape `[4,4]`. The camera
center is `C = -R^T t`. Updates use a six-component left tangent,
`[vx, vy, vz, wx, wy, wz]`, and the retraction `T_new = Exp(delta) T_current`.

Imported source poses remain immutable. Current poses, revisions and displacement
are maintained separately and identified by camera UID. Rotation and translation
are not optimized as unconstrained matrix coefficients.

## FastGS interface

An optional per-render override supplies contiguous CUDA float32 tensors:
world-to-camera `[1,4,4]` and camera center `[3]`. Their contents must represent
a coherent rigid transform, use the rendering stream and remain immutable
through backward. The forward context retains their storage.

Camera backward writes a caller-owned CUDA float32 `[4,4]` gradient. This
storage must not overlap pose tensors, backward inputs, optimizer state or
external scratch. Gradients from separate tiles must be summed before applying
a pose update. The matrix derivative includes covariance rotation and
spherical-harmonic camera-center dependence and is converted to a left-tangent
gradient.

Camera-only backward disables Gaussian parameter updates and does not commit
Adam, densification or edge statistics. Calls without camera refinement retain
the ordinary Gaussian-update path. Mip Filter and normal-loss camera gradients
are rejected; these restrictions do not disable their ordinary training paths.
The full-image evaluator does not currently expose depth supervision, tiled
objectives or a 3DGUT camera-gradient path.

## Bounded optimizer

`BoundedPoseOptimizer` owns one camera's source/current pose and optimization
history. Construction requires a UID, rigid source matrix and explicit
`BoundedPoseConfig`; the default role is `Train`.

Default limits are relative to an explicitly supplied scene scale:

- Per-step translation component: 0.3% of scene scale.
- Per-step rotation: 0.25 degrees.
- Cumulative camera-center displacement: 3% of scene scale.
- Cumulative rotation from source: 3 degrees.
- Candidate evaluations per step: at most eight.

Weak source priors penalize normalized center displacement and rotation chord
distance. A scaled inverse-BFGS approximation supplies a safeguarded descent
direction in left-tangent coordinates; it is not an exact manifold Hessian.

Acceptance requires both a decrease in image loss and sufficient Armijo decrease
in image loss plus priors. Rejected or nonfinite candidates leave the current pose
unchanged. Callback exceptions roll back controller state. Frozen, anchor and
evaluation cameras do not invoke candidate rendering.

Baseline UID, pose revision and model revision must match the current state.
The owner must advance the model revision whenever geometry, appearance, masks,
resolution, background or the objective changes. Curvature history is discarded
when that revision changes. Reset restores the source pose and invalidates old
baseline evaluations.

## Multi-camera session

`PoseRefinementSession` owns UID-keyed controllers with immutable membership.
Changing a dataset or split requires a new session with an owner-assigned
generation. Duplicate UIDs and invalid source poses are rejected.

The session requires at least three non-evaluation cameras, at least two fixed
reference cameras and at least one movable camera. Explicit references are
supported. When none are supplied, automatic selection fixes the lowest
training UID and the camera farthest from its source center. Evaluation cameras
never participate in reference selection or refinement.

Fixing two complete poses conservatively constrains world frame and scale but
also overconstrains the minimal similarity freedom. It can retain errors in the
selected references; the selection is not an accuracy ranking. Degenerate
reference baselines are rejected. Scene scale is supplied by the owner and must
not be derived from evolving optimized poses.

Scheduling uses a warmup, per-camera visit cadence, bounded update bursts and a
final frozen phase. The first eligible visit runs immediately. Subsequent visits
are counted independently per camera to avoid starvation from random sampling.
Pause disables updates. Cancellation or evaluator exceptions discard a burst's
pose, history and cadence changes; the observed training iteration is retained.
Candidate-render counters include completed and cancelled visits, but not visits
that throw.

## Image evaluator

`FastGSPoseEvaluator` connects a session to full-image FastGS rendering and
camera-only backward. A baseline forward is reused for backward; candidate
scoring needs forward only. Context lifetime is managed automatically, including
exception paths.

The objective callback evaluates the same frozen model, target, background and
appearance state for baseline and candidates. Its derivatives must refer to raw
raster RGB and optional alpha, with matching shape, dtype and CUDA stream.
It must not update model, camera or appearance parameters.

The provided RGB MSE objective clones its target once and reduces the loss on
GPU. Scalar loss and the camera matrix gradient are transferred to the CPU.
This objective does not implicitly replace the Trainer's L1/SSIM loss.
The Trainer adapter instead uses the production L1/SSIM photometric objective,
with the current SSIM weight. Its existing loss API computes image derivatives
even for candidate scoring, but candidates do not run camera or Gaussian backward.
PPISP/bilateral composition and tiled evaluation require additional integration.
Allocation, synchronization and rendering costs depend on the dataset and update
schedule; no performance improvement is guaranteed.

The evaluator borrows camera, model, optimizer and background references.
The owning training thread must keep them alive and hold the appropriate
training safe point throughout a visit. The evaluator does not acquire model
locks or mutate source Camera tensors.

## Durable session state

The session exposes JSON state serialization independently of display snapshot
timing. It retains scheduling configuration, iteration, pause state, camera UIDs,
roles, source/current poses, revisions and update/candidate counters. Serialized
camera order is not significant.

Restoration requires an exact match with the current session's configuration,
membership, source matrices and reference/evaluation roles. Non-rigid transforms,
out-of-bounds displacements, invalid counters and moved fixed/evaluation cameras
are rejected. Displacement is recomputed from poses rather than trusted from
stored metadata. All records and the replacement snapshot are prepared before
installing the new state, so a malformed record cannot partially restore cameras.

The receiving session retains its own generation and advances snapshot and pose
revisions, invalidating earlier evaluations. Poses, pause state and per-camera
cadence are preserved; inverse-BFGS history is restarted because its model/objective
identity cannot be inferred from pose metadata. This is a controlled warm restart,
not an identical continuation of optimizer internals.

The Trainer captures live session state alongside the matching Gaussian model
in the checkpoint parameter payload, including the CKPT chapter embedded in a
`.licht` project. `HAS_CAMERA_POSES` marks this payload as required: older readers
that do not recognize the flag reject the checkpoint instead of silently loading
the model without its camera corrections. Checkpoints without this flag retain
their existing behavior.

The saved pose clock is aligned to the captured model iteration, including
iterations that skipped an image. Loading verifies flag/payload agreement and
the iteration. Before adopting the decoded model, the Trainer additionally
validates the complete session against the loaded model scale, effective dataset
membership, immutable source transforms and training schedule. A context mismatch
rejects adoption. Loading a checkpoint without poses clears stale pose metadata.

## Trainer integration

`configureCameraPoseRefinement` accepts an optional custom session configuration
before Trainer initialization. The `refine_camera_poses` optimization option uses
the default session configuration when no custom configuration is supplied.
Refinement is disabled by default; a checkpoint containing
pose state restores its saved configuration automatically. Runtime length and
scene scale come from the Trainer and model. Evaluation and disabled cameras are
excluded from the effective training membership.

At scheduled visits, refinement runs before the Gaussian update with the model,
target and background fixed for the complete burst. The normal FastGS forward
then receives the accepted current pose. Imported Camera tensors remain unchanged.
Objective allocation is lazy, so unscheduled visits do not clone targets or
allocate a photometric workspace. Pause and training completion publish snapshots.

This path currently supports raw RGB FastGS training without Mip Filter, masks,
depth/normal supervision, cropbox ROI loss, appearance correction or sparsification.
Unsupported combinations are rejected explicitly. Membership and schedule changes
require reinitialization rather than silently reusing an incompatible session.

## Shared pose state and visualization

Session mutation has one training-thread owner. Cross-thread readers use only
immutable, UID-sorted published snapshots. These contain generation, sequence,
source/current poses, revisions, accepted/rejected counts and net center/rotation
displacement. Publication during visits is throttled to 250 ms; pause and reset
force publication. Reset preserves monotonic snapshot sequencing.

Available states are `waiting`, `ready`, `updated`, `rejected`, `frozen`,
`anchor` and `evaluation`. They describe optimizer activity, not confidence.
An updated pose is not necessarily converged, and a rejected update does not
establish incorrect calibration.

The live viewport uses current poses from the Trainer's published snapshot.
Frustum caches track both session generation and snapshot sequence, including
session removal and replacement. Picking, rectangular selection, camera focus
and camera-based selection projections also resolve current poses by UID.
Geometry uses actual displacement without visual exaggeration. Cameras absent
from the snapshot retain their source transform.

Scene Graph rows use a fixed-width colored symbol. Hovering shows optimizer state
and net camera-center distance in scene units / rotation in degrees. The numeric
values do not consume row width. Viewport frustums have a small, constant-screen-size
marker above their image plane, enlarged on camera hover. The markers reuse the
existing panel projection and camera visibility, including orthographic and
panoramic views. They do not add a selectable camera or alter picking geometry.

Both surfaces share this activity palette:

| State | Color | Graph symbol | Viewport marker |
| --- | --- | --- | --- |
| Waiting | Gray | `.` | Diamond |
| Ready | Cyan | `o` | Diamond |
| Updated | Teal | `+` | Diamond with cross |
| Rejected | Amber | `!` | Cross |
| Anchor | Blue | `A` | Square |
| Evaluation | Light gray | `E` | Square |
| Frozen | Violet | `=` | Square |
| Paused | Yellow | `=` | Two bars |

Pause overrides activity only for movable, non-frozen cameras. Fixed references,
evaluation cameras and frozen poses retain their own categories. Indicators
disappear when there is no matching pose session. Existing reconstruction-loss
colors, selection highlighting and thumbnail tints retain their meanings.
Neither pose distance nor optimizer state is a confidence or quality score.

Source-pose ghosts, displacement vectors, Edit Mode ownership and visualization
without a restored Trainer remain integration work. Scene camera records still
describe imported poses; corrected poses are restored through the matching
training checkpoint, not by overwriting sources.

## Scope and limitations

Rigid six-degree-of-freedom refinement cannot recover missing scene coverage or
correct motion blur and rolling-shutter distortion. Narrow image coverage alone
does not establish a pose error. Concurrent pose movement and densification can
retain obsolete geometry or duplicate surfaces; the final pose-freeze phase
allows subsequent geometry-only refinement.

Unsupported combinations must be explicit and must not silently switch the
training backend. Real reconstruction quality and computational cost remain
dataset-dependent; no universal improvement is implied.
