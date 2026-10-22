# Stochastic rendering

The **Rendering → Quality → Advanced → Transparency** control selects how 3DGS
fragments are combined:

| Mode | Behavior |
| --- | --- |
| Alpha blend | The existing deterministic renderer. This remains the default. |
| Stochastic | One opaque visibility sample per pixel and frame, with cumulative accumulation while the camera stays still. |
| SteadySplats | Stochastic samples with spatial importance resampling and reprojected temporal reuse. |

Each new sample is displayed immediately. **Refinement frames** limits the new
samples rendered after the camera or scene changes (32 by default). New input
starts a new refinement window. The viewer stops requesting extra frames at the
limit; it does not wait for a batch of samples before responding to navigation.
These finite refinement requests can advance as GPU frame slots become available,
with input processed between samples. Periodic GUI animations keep display pacing.
SteadySplats uses an exponential temporal filter, so this limit is a work budget,
not a guarantee of convergence to the alpha-blended image.

Stochastic modes currently apply to 3DGS without LOD. Outlines, rings, center
markers, 3DGUT, offscreen previews, and exports retain alpha blending.
Ordinary selection colors remain supported. The setting does not change training,
Gaussian parameters, saved models, or the default renderer.
Live viewport captures preserve the accumulated history.

## SteadySplats resampling

The implementation follows [SteadySplats](https://arxiv.org/html/2610.05576v2),
Algorithms 1–3 and equations 9, 14, 20, and 21:

- A fragment survives with probability equal to its alpha. The nearest surviving
  fragment supplies the sample; no surviving fragment produces background.
- Four direct pixel neighbors contribute self-normalized importance samples. Each
  source pixel retains four **distinct** occluders in a FIFO. Repeated samples do
  not evict rare occluders or multiply their occlusion factors again.
- Temporal reuse reconstructs the current sampled position and fetches the prior
  color with Catmull–Rom interpolation. YCoCg neighborhood moments and
  velocity-dependent confidence bounds reduce history retention at disocclusions.
  Historical color is not clipped. Nominal retention is 0.70.

The visibility comparison uses LFS's native radial depth keys to preserve its
alpha-blending order in expectation. Actual view-space depth is retained for
reprojection and depth consumers; depth is not averaged with color. The renderer
skips primitive depth sorting, but retains the existing tile grouping operations.
This is not a claim that the complete pipeline performs no sorting.

Both stochastic modes use independent white-noise thresholds. The paper's STBN
sampling optimization is not bundled. Finite occluder history, self-normalized
weights, and temporal filtering introduce the approximations described in the
paper. Models trained with color variance regularization usually provide a more
useful input to reuse than models with strongly different overlapping colors.

## History and memory

History belongs to each logical viewport, including independent split panels.
It is separate from the rotating presentation images. Scene changes, mode changes,
model replacement, live training updates, and resize invalidate history. Basic
stochastic accumulation also resets on camera changes; SteadySplats retains history
for reprojection. Unsupported temporal projections use the current spatial result.

With private rendering scratch, an unchanged camera and scene reuse the last
submitted projection and tile-count prefix. Deferred GPU wave counts are reused
only when their submission belongs to that same projection. Other views, content
changes, input uploads, scratch release, and failed renders invalidate this cache.
Visibility samples, resampling, and the refinement limit remain unchanged.

Color history stores floating-point premultiplied color and transmittance before
final composition and display conversion. It does not average 8-bit screenshots.
SteadySplats adds 144 bytes per pixel for reuse and color history; basic stochastic
adds 32 bytes per pixel. Buffers are allocated when used and released when the scene
or renderer is released.
Stochastic visibility also retains the radial depth keys through rasterization,
which requires four extra bytes per Gaussian in the legacy scratch buffers.

## Automation and validation

Python render settings accept `transparency_mode` identifiers `alpha_blend`,
`stochastic`, and `steady_splats`. `stochastic_max_frames` sets the refinement limit.
The MCP `render_settings_get` and `render_settings_set` tools expose the same fields.

The focused test targets are `stochastic_tests` and `stochastic_viewport_tests`.
They check GPU visibility probabilities, background mass, primitive permutation,
wave partitioning, spatial weights against a double-precision oracle, deduplication,
depth ties, reprojection, soft rejection, and viewport history ownership. GPU tests
skip when a compatible Vulkan device is unavailable.
