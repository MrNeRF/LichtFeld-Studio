# Off-Axis Light-Field Rendering

`lichtfeld.render_lightfield_view_u8` lets plugins render parallel off-axis views
for light-field quilts without temporarily changing the desktop viewport or model.
It adds a horizontal principal-point offset and per-call camera-forward clipping
to the RGB uint8 preview path. No display SDK, device calibration, or quilt layout
is part of this host API.

```python
import lichtfeld as lf

rotation, eye = lf.look_at((0, 0, 5), (0, 0, 0))
image = lf.render_lightfield_view_u8(
    rotation, eye, 372, 682,
    fov=30.0,
    shift_x=0.0,
    near_distance=0.0,
    far_distance=100.0,
)
```

The result is a CPU uint8 tensor `[height, width, 3]` in RGB order, or `None` if
arguments are invalid, a renderable scene is unavailable, or rendering fails.
Width/height are limited to 1-2048 and vertical FOV to 1-120 degrees, defaulting to 60.
The principal-point shift must be finite. Finite clipping distances must satisfy
`0 <= near_distance < far_distance`.

## Quilt Camera Geometry

Use one shared camera-to-world rotation for every view. Move the eye along that
rotation's local X axis. For horizontal offset `b`, focus-plane distance `d` and
vertical focal length `fy` in pixels, set `shift_x = fy * b / d`, where
`fy = height / (2 * tan(fov / 2))`. Positive shift moves the principal point right.
This keeps objects on the focus plane aligned without toe-in rotations.

Near/far clipping tests Gaussian centers by camera-forward distance, not Euclidean
radius or complete Gaussian support. Large splat tails can cross the clipping planes.
The function hides selection highlights without clearing the selection. It locally
disables editor overlays and overrides projection, clipping
and splat display scale; other appearance settings, including background, remain
inherited from the current render settings.

Calls dispatch to the viewer thread when needed. Generate views incrementally to
keep the UI responsive. The caller should pause training and cancel a partial quilt
if its scene/model changes; one render call does not freeze an entire multiview run.

## Tests

```powershell
python -m pytest tests/python/test_lightfield_render_runtime.py -q
```

An opt-in GUI test is provided in `test_lightfield_render_runtime.py`. Point
`LFS_LIGHTFIELD_TEST_ENDPOINT` at a **separate, empty** patched GUI instance. It
refuses a loaded scene or trainer, creates synthetic splats, checks clipping,
principal-point focus alignment, selection-highlight isolation, the default FOV
and invalid arguments, then removes its test node. A unit test in
`test_lightfield_test_harness.py` checks that missing NumPy is reported immediately.
Never point this test at a working or training session.
