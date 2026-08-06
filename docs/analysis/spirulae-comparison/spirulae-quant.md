I'll systematically explore the core state, quant, and optimizer files you named, then the kernels that read/write them, and assemble an exact per-splat memory ledger.# Per-splat memory ledger and quantization inventory

**Scope:** device-resident splat training/render state in `src/core`, `src/engine`, and the kernels that read/write it.  
**Notation:** \(N\) = `max_num_splats` (allocation size); \(K\) = `engine().num_sh` = rest SH bands = \((d+1)^2-1\) for degree \(d\) (default \(d=3 \Rightarrow K=15\)).  
**Default CLI knobs** (relevant to totals): `quantization_level=1`, `use_fused_proj_bwd_optim=true`, `split_batch=true`, `packed=true`, `sh_degree=3` (`src/app/generated/cli_config.h:63,82-84`).  
When both `split_batch` and FPBO are on and batch size can exceed 1, FPBO is disabled (`src/engine/EngineDataManager.cpp:60-99`).

---

## 1. Parameter storage (world splats)

Canonical holder: `WorldSplats` (`src/engine/EngineState.h:80-110`). Seeded once via `set_data_3dgs` (`src/engine/EngineSetup.cpp:11-46`).

### 1.1 Layout model

Attributes live in **separate buffers** (SoA across attributes). Within each buffer, each splat stores a packed vector (`float3`/`float4`/`float`) — **AoS of components per attribute**, indexed by splat id.

Primitive view (`WorldBuffer`) reinterprets these as:

| Attribute | Primitive accessor | File:line |
|---|---|---|
| mean | `float3` at `_data[0][3*i]` | `PrimitiveBase3DGS.cuh:91-92` |
| quat | `float4` at `_data[1][4*i]` | `:93-94` |
| scale | `float3` (log-scale) | `:95-96`, comment `:25` |
| opacity | `float` (logit) | `:97-98`, comment `:26` |
| features_dc | `float3` | `:99-100` |
| features_sh | `float*` stride `3*K` floats | `:101-102`, `num_sh()` at `:30-31` |

### 1.2 Per-attribute ledger (parameters only)

| Attribute | Dtype | Bits/cell | Cells/splat | Bytes/splat (canonical) | Layout | Allocation |
|---|---|---|---|---|---|---|
| **position (means)** | `float3` | 32×3 | 3 | **12** | SoA buffer of AoS float3 | `EngineState.h:82`; `EngineSetup.cpp:39` |
| **rotation (quats)** | `float4` | 32×4 | 4 | **16** | same | `EngineState.h:83`; `:40` |
| **scale (log)** | `float3` | 32×3 | 3 | **12** | same | `EngineState.h:84`; `:41` |
| **opacity (logit)** | `float` | 32 | 1 | **4** | same | `EngineState.h:85`; `:42` |
| **color DC** | `float3` | 32×3 | 3 | **12** | same | `EngineState.h:86`; `:43` |
| **SH rest (`features_sh`)** | see below | see below | \(3K\) scalar cells | see below | see below | `EngineState.h:87-108` |

**Non-SH params are never value-quantized.** Only SH rest bands have a value-quant path.

### 1.3 SH value storage (`features_sh`)

Controlled by `world.sh_value_bits` (`EngineState.h:108-109`). Mapped from CLI:

- `quantization_level == 0` → 32 (`TrainerCore.cpp:325-327`)
- `quantization_level == 1` → **16** (`TrainerCore.cpp:326`)

| Mode | Canonical store | Codec | Bytes/splat (packed) | Bounds | Layout | File:line |
|---|---|---|---|---|---|---|
| **fp32** | `DeviceTensor2D<float3> features_sh` `[N,K]` | none | \(12K\) | none | row-major float3 | `EngineState.h:87`; alloc `EngineOptim.cpp:80-81` |
| **8-bit value** | `QuantizedTensor<8,256>` | linear min-max endpoint-exact | \(1 \times 3K = 3K\) | `float2` × `n_bounds` | cell-block *or* splat-block | `EngineState.h:104-106`; codec `Tensor.h:1288-1376` |
| **16-bit value** (level 1) | `QuantizedTensor<16,256>` | same linear | \(2 \times 3K = 6K\) | `float2` × `n_bounds` | same | `EngineState.h:105,107` |

**Two mutually exclusive bound layouts** (`EngineState.h:89-103`, alloc `EngineOptim.cpp:177-216`):

| Layout | When | Cells | Bounds count | Stride for bound index |
|---|---|---|---|---|
| **Per-cell-block** | non-FPBO (`!use_fused_proj_bwd_optim`) | \(N\cdot K\cdot 3\) | \(\lceil N K 3 / 256\rceil\) | 256 cells (`EngineForward.cpp:90`) |
| **Per-splat-block (FPBO)** | FPBO | \(N\cdot K\cdot 3\) | \(\lceil N/256\rceil\) | \(256\cdot 3\cdot K\) cells (`EngineForward.cpp:87`) |

Bounds storage: 8 B per bound (`float2`); amortized \(\approx 8/256 = 0.03125\) B/splat in FPBO layout (independent of \(K\)), vs \(\approx 8\cdot(3K)/256 = 0.09375K\) B/splat in cell-block layout.

**Decode on load:** projection forward/backward pass packed+bounds into Slang harmonics (`EngineForward.cpp:62-108`; `Primitive3DGUT.cuh:103-124` `sh*_to_color_q{8,16}`).

When value-quant is on, fp32 `features_sh` is **freed** and only a shape descriptor remains (`EngineOptim.cpp:72-79`).

### 1.4 Parameter subtotal

\[
\begin{align*}
B_{\text{params, non-SH}} &= 12+16+12+4+12 = \mathbf{56}\ \text{B/splat} \\
B_{\text{params, SH fp32}} &= 12K \\
B_{\text{params, SH q16}} &= 6K + B_{\text{bounds,v}} \\
B_{\text{params, SH q8}} &= 3K + B_{\text{bounds,v}}
\end{align*}
\]

For default \(K=15\), FPBO, level 1: \(56 + 90 + \approx 0.03 = \mathbf{\approx 146}\) B/splat params.

---

## 2. Optimizer state

### 2.1 Optimizer algorithm

**Adam** with fixed hyperparameters in geometry/FPBO kernels:

- \(\beta_1 = 0.9\), \(\beta_2 = 0.999\), \(\varepsilon = 10^{-15}\)  
  (`FusedGeometryOptim.cu:136-138`; `FusedProjectionBwdOptim_kernel.cuh:364-366`; trust-region path `EngineOptim.cpp:593`)

Standard update (biased moments, then bias-correct):  
\(m \leftarrow \beta_1 m + (1-\beta_1)g\), \(v \leftarrow \beta_2 v + (1-\beta_2)g^2\),  
\(\theta \leftarrow \theta - \mathrm{lr}\cdot \hat m / (\sqrt{\hat v}+\varepsilon)\).

Optional **per-splat step counter** for bias correction: `bias_correction_steps` `[N] int32` (`EngineState.h:209,239`; alloc `EngineOptim.cpp:218-224`).

Optional **color trust-region Adam** for DC/SH when `use_color_trust_region` (`EngineConfig.h:159-166`; `EngineOptim.cpp:591-603,685-694`).

### 2.2 Unquantized Adam state (`sh_optim_bits==32`, `non_sh_optim_bits==32`)

| Buffer | Type | Bytes/splat | File:line |
|---|---|---|---|
| `g1_*` / `g2_*` means | `float3` each | \(2\times 12=24\) | `EngineState.h:211`; `EngineOptim.cpp:88-97` |
| quats | `float4` | \(2\times 16=32\) | same |
| scales | `float3` | \(2\times 12=24\) | same |
| opacities | `float` | \(2\times 4=8\) | same |
| features_dc | `float3` | \(2\times 12=24\) | same |
| features_sh | `float3[K]` each | \(2\times 12K=24K\) | `EngineState.h:216`; `EngineOptim.cpp:120-127` |

**Non-SH optim total (fp32):** \(56\times 2 = \mathbf{112}\) B/splat  
**SH optim total (fp32):** \(24K\) B/splat

### 2.3 Quantized Adam state — codec

`QuantizedAdamState<BITS, BLOCK_SIZE=256>` (`src/core/Tensor.h:916-1113`):

**Transform** (not raw \(g_1,g_2\)):

\[
u = \frac{g_1}{\sqrt{g_2}+\varepsilon},\quad
\log_s = \log1p(\sqrt{g_2}/\varepsilon)
\]

Bounds per block: `float4` = \((u_{\min},u_{\max},\log_s_{\min},\log_s_{\max})\) (`Tensor.h:933-935`).  
Reconstruct: \(\sqrt{g_2}=\varepsilon\cdot\mathrm{expm1}(\log_s)\), \(g_1=u(\sqrt{g_2}+\varepsilon)\), \(g_2=(\sqrt{g_2})^2\) (`:1043-1051`).

**Storage AoS** (`Tensor.h:946-949,970`):

| BITS | Bytes/cell | Packing |
|---|---|---|
| 16 | **4** | `uint16 u_q ‖ uint16 log_s_q` |
| 8 | **2** | `byte[2i]=u_q, byte[2i+1]=log_s_q` |
| 4 | **1** | low nibble \(u\), high nibble \(\log_s\) |

Quantization: **endpoint-exact round-to-nearest** (`roundf`, not stochastic) (`Tensor.h:1064-1065`).  
**Zero fixed-point:** all-zero packed+bounds → \((g_1,g_2)=(0,0)\) (`Tensor.h:921-924,1007-1009`).

### 2.4 Non-SH Adam quant (`non_sh_optim_bits==16`)

- Type: `QuantizedAdamState<16,256>` per attribute (`EngineState.h:227-231`)
- Bundled for kernels as `NonShQuantState` (`src/core/NonShQuantState.h:16-28`)
- Level 1 sets `non_sh_optim_bits=16` (`TrainerCore.cpp:327`)
- Layout: **per-splat-block** — one `float4` bound per 256 splats; cells = \(N\times\text{prims}\) (`EngineOptim.cpp:129-140`)
- Used by FPBO `_NonShQ` (`FusedProjectionBwdOptim_kernel.cuh:97-143`) and non-FPBO geometry path `_OptimNonShQ` (`EngineOptim.cpp:51-57`)

| Attribute | Prim cells | Packed B/splat | Bounds (amortized) |
|---|---|---|---|
| means | 3 | \(3\times 4=12\) | \(16/256\) shared-style |
| quats | 4 | 16 | \(16/256\) |
| scales | 3 | 12 | \(16/256\) |
| opacities | 1 | 4 | \(16/256\) |
| features_dc | 3 | 12 | \(16/256\) |
| **Total non-SH optim q16** | 14 cells | **56** | \(5\times 16/256 \approx 0.31\) |

When enabled, fp32 `g1_/g2_` for these attrs are freed (`EngineOptim.cpp:98-118`).

### 2.5 SH Adam quant (`sh_optim_bits` ∈ {4,8})

Level 1 → **8** (`TrainerCore.cpp:325`). Storage holder always typed `<8,256>` (pessimistic footprint for 4-bit) (`EngineState.h:200-201`).

| Path | Packed B/splat | Bounds | File:line |
|---|---|---|---|
| non-FPBO | \(kBytesPerCell\times 3K\) (8-bit: \(6K\)) | \(\lceil 3NK/256\rceil\) float4 | `EngineOptim.cpp:163-167` |
| FPBO | same packed | \(\lceil N/256\rceil\) float4 | `EngineOptim.cpp:168-171` |

Amortized bounds FPBO: \(16/256\approx 0.0625\) B/splat.

### 2.6 Densification / aux optimizer-adjacent state (per splat)

| Buffer | Type | Bytes | Role | File:line |
|---|---|---|---|---|
| `radii` | `float` | 4 | screen radius via `atomicMax` in proj fwd | `EngineState.h:237`; `EngineForward.cpp:47-60`; `ProjectionPackedFwd_kernel.cuh:182` |
| `accum_buffer` | `float2` | 8 | densify score accumulator (value, count) | `EngineState.h:238`; scoring `DensifyScoring.cu:169-227` |
| `accum_weight` | `float` | 4 | per-splat raster bwd score | `EngineState.h:139` |
| `world_grad_score` | `float` | 0 or 4 | \(\|dL/dmean\|\cdot\max\mathrm{scale}\) if blend enabled | `EngineState.h:140-145`; `EngineOptim.cpp:541-548` |
| `bias_correction_steps` | `int32` | 0 or 4 | optional | `EngineState.h:239` |

---

## 3. Gradient storage

### 3.1 Paths (mutually exclusive allocation logic)

`_alloc_grad_buffers` (`EngineLoss.cpp:20-128`); `quantize_grad` set in `EngineTrainStep.cpp:67-68`:

```
quantize_grad = !use_fused_proj_bwd_optim && (quantization_level != 0)
```

| Path | When | World grad storage |
|---|---|---|
| **A. FPBO** | `use_fused_proj_bwd_optim` | **None** for 3dgs/mip; only means/quats/scales fp32 for **3dgut** (`EngineLoss.cpp:32-49`) |
| **B. Grad-quant** | non-FPBO + level≠0 | Block-wise quantized accumulators (`:52-99`) |
| **C. fp32** | non-FPBO + level 0 | Full fp32 world grads (`:102-127`) |

### 3.2 Path C — fp32 grads (mirror of params)

| Attr | Bytes/splat |
|---|---|
| means+quats+scales+opac+dc | 56 |
| features_sh | \(12K\) |
| **Total** | \(56+12K\) |

Written by raster atomicStore (screen→world for some prims) + projection bwd atomicAdd (`PrimitiveBase3DGS.cuh:75-83`).  
Sub-batch: accumulate across cameras; only first sub-batch zeroes (`EngineLoss.cpp:116-127`).

### 3.3 Path A — FPBO (default when compatible)

- Projection VJP + Adam fused; **no materialized world grad buffer** for 3dgs/mip (`EngineOptim.cpp:287-293`; `EngineLoss.cpp:24-28`).
- Screen-space grads `v_splats_s` stashed for the fused kernel (`EngineState.h:146-149`).
- 3dgut exception: means/quats/scales still need fp32 world grads from raster atomics (**+40 B/splat**).

### 3.4 Path B — GradQuant (signed symmetric)

Codec: `gradq::Codec` in `src/core/GradQuant.cuh` — **not** the min-max `QuantizedTensor` codec.

| Property | Value | File:line |
|---|---|---|
| Domain | signed codes \([-QMax,QMax]\) | `GradQuant.cuh:45-50` |
| Scale | \(a=\max(|\min|,|\max|)\); decode \(q\cdot a/QMax\) | `:52-60` |
| **decode(0)** | **exactly 0.0f** for any bound | `:8-14,44` |
| Rounding | `rintf` (nearest even-style round) | `:66` |
| Stochastic? | **No** | — |
| Non-SH bits | **16** (`int16`, \(QMax=32767\)) | `EngineState.h:186-190` |
| SH bits | **8** (`int8`, \(QMax=127\)) | `:191` |
| Layout | per-**splat**-block; 1×`float2` bound / 256 splats | `GradQuant.cuh:20-24`; `EngineLoss.cpp:60` |
| Bound storage | reuses `QuantizedTensor` as byte+float2 holder | `GradQuant.cuh:16-18` |

**Write path (non-atomic on quant codes):**  
`projection_bwd_quantgrad_kernel` — one thread per splat; register-accumulate over cameras of this sub-batch; then **decode → add → block_reduce minmax → encode** (`ProjectionBwdQuantGrad_kernel.cuh:19-28,153-207`).  
No atomics on quantized codes. Race-free because one owner thread per `gid`.

**3dgut split:** means/quats/scales stay fp32 (raster atomicAdds); only opac/dc/sh quantized (`EngineState.h:172-175`; `EngineLoss.cpp:63-74`).

| Attribute | Bits | Cells | Packed B/splat |
|---|---|---|---|
| means_q | 16 | 3 | 6 |
| quats_q | 16 | 4 | 8 |
| scales_q | 16 | 3 | 6 |
| opacities_q | 16 | 1 | 2 |
| features_dc_q | 16 | 3 | 6 |
| features_sh_q | 8 | \(3K\) | \(3K\) |
| **3dgs/mip total packed** | | | **\(28 + 3K\)** |
| Bounds (6 attrs) | float2×⌈N/256⌉ each | | \(\approx 6\times 8/256 \approx 0.19\) |

Optim step decodes via `GradQuantBuffers` (`EngineOptim.cpp:550-562`; struct `ProjectionBwdQuantGrad.cuh:20-27`).

### 3.5 Why GradQuant is signed-symmetric (not min-max)

Documented correctness requirement (`GradQuant.cuh:10-14`): min-max would map true-zero grads (out-of-view) to half-quantum pseudo-grads; Adam \(m/\sqrt{v}\) would amplify those into floaters. Symmetric codec snaps \(|v|<a/(2QMax)\) to exact 0.

---

## 4. TOTAL bytes per splat

Let \(B_{\text{aux}} = 4+8+4 = 16\) (radii + accum_buffer + accum_weight). Optional +4 world_grad_score, +4 bias steps.

### 4.1 Configuration matrix (packed bytes; bounds amortized in “≈”)

**\(K=15\) (degree 3)** used in numeric examples.

#### Config 0 — all fp32, non-FPBO (level 0, no FPBO)

| Bucket | Formula | Bytes (\(K=15\)) |
|---|---|---|
| Params | \(56+12K\) | 236 |
| Grads | \(56+12K\) | 236 |
| Optim g1+g2 | \(112+24K\) | 472 |
| Aux | 16 | 16 |
| **TOTAL** | \(240+48K\) | **960** |

#### Config 1a — level 1 + **FPBO** + 3dgs/mip (default when FPBO wins)

| Bucket | Formula | Bytes (\(K=15\)) |
|---|---|---|
| Params non-SH | 56 | 56 |
| Params SH q16 | \(6K\) + ~0.03 | 90 |
| Grads world | **0** | 0 |
| Optim non-SH q16 | 56 + ~0.31 | 56 |
| Optim SH q8 | \(6K\) + ~0.06 | 90 |
| Aux | 16 | 16 |
| **TOTAL** | \(\approx 128+12K\) | **≈308** |

vs Config 0: **~3.1×** reduction (~960 → ~308).

#### Config 1b — level 1 + **non-FPBO + GradQuant** (when split_batch disables FPBO)

| Bucket | Formula | Bytes (\(K=15\)) |
|---|---|---|
| Params | \(56+6K\) | 146 |
| Grads quant | \(28+3K\) + ~0.19 | 73 |
| Optim non-SH q16 | 56 + ~0.31 | 56 |
| Optim SH q8 (cell-block bounds) | \(6K\) + ~0.09K | 90 |
| Aux | 16 | 16 |
| **TOTAL** | \(\approx 156+15K\) | **≈381** |

#### Config 1c — FPBO + 3dgut + level 1

Same as 1a plus **+40 B** fp32 means/quats/scales grads → **≈348** B/splat.

#### Rendering-only (viewer / inference)

| Bucket | Bytes |
|---|---|
| Params only (level 1 SH q16) | \(56+6K \approx 146\) |
| Params only (fp32 SH) | \(56+12K = 236\) |
| Optim / grads | **0** |

(Plus transient forward intermediates — §6 — not persistent per-splat state.)

### 4.2 Compact formula summary

\[
\begin{align*}
B_{\text{train, fp32, no FPBO}} &= 240 + 48K \\
B_{\text{train, L1+FPBO, 3dgs}} &\approx 128 + 12K \\
B_{\text{train, L1+GradQ}} &\approx 156 + 15K \\
B_{\text{render, L1}} &= 56 + 6K
\end{align*}
\]

At \(N=10^6\), \(K=15\): Config 0 ≈ **0.96 GB**; Config 1a ≈ **0.31 GB** persistent splat state alone.

---

## 5. Accuracy-preservation tricks

| Trick | What it does | Where |
|---|---|---|
| **Symmetric zero-preserving grad codec** | `decode(0)=0`; prevents Adam floaters from dead splats | `GradQuant.cuh:8-14,56-60` |
| **Joint \((u,\log_s)\) Adam packing** | Stores normalized first moment + log-mapped \(\sqrt{g_2}\); ~20–30 dB better post-step SNR vs linear \(\sqrt{g_2}\) | `Tensor.h:916-944` |
| **log1p / expm1 0↔0 fixed point** | Zero-init of packed+bounds is valid Adam zero without special encode | `Tensor.h:1007-1015,226-229` |
| **Endpoint-exact linear quant** | \(q=0\to lo\), \(q=QMax\to hi\) | `Tensor.h:950,1064-1065,1349-1357` |
| **Block-wise bounds (256)** | Local dynamic range → better SNR than global scale | all quant types |
| **Per-splat-block bounds in FPBO** | One reduction per 256-thread block; all SH cells of a splat share one bound | `EngineOptim.cpp:155-161` |
| **Decode-on-load SH values** | Params stay packed; consumers dequant in registers | `EngineForward.cpp:62-66`; Slang `sh*_q{8,16}` |
| **Requantize-accumulate for split_batch** | Grad quant survives multi-sub-batch without fp32 full buffer | `ProjectionBwdQuantGrad_kernel.cuh:23-26` |
| **No atomics on quant codes** | Avoids torn/racey packed updates; register sum then encode | `:106,153-155` |
| **3dgut keeps geometry grads fp32** | Raster atomics need real fp32; only appearance quantized | `EngineState.h:172-175` |
| **Trust-region color clip** | Bounds DC/SH steps in color space | `EngineConfig.h:159-166` |
| **SH value + optim must both quantize (non-FPBO)** | Doubly-quant kernel only; mixed mode rejected | `EngineOptim.cpp:40-50` |
| **Rounding is deterministic `roundf`/`rintf`** | No stochastic rounding in production codecs | `Tensor.h:1064`; `GradQuant.cuh:66` |

**Explicit non-features:** no stochastic rounding; no zero-point offset (affine); GradQuant is scale-from-amplitude (zero-centered), Adam/value quant is min-max affine.

---

## 6. Activation / intermediate memory (forward & backward)

Not amortized per splat the same way — sizes scale with \(C,H,W,\mathrm{nnz}\), tile occupancy. All pool-backed; high-water mark retains capacity (`Tensor.h:139-143`).

### 6.1 Per-instance (projected splat × camera) — screen buffer

Allocated as `splats_s` / `v_splats_s` (`EngineState.h:129-130,149`; size \(C\cdot N\) or packed `nnz`).

**3dgs / mip** (`ScreenBuffer` 5 tensors, `{2,1,3,1,3}` floats) — `PrimitiveBase3DGS.cuh:187-216`:

| Field | Floats | Bytes |
|---|---|---|
| xy | 2 | 8 |
| depth | 1 | 4 |
| conic | 3 | 12 |
| opac | 1 | 4 |
| rgb | 3 | 12 |
| **Total** | 10 | **40 B / (cam,splat) entry** |

**3dgut** (`ScreenBuffer` 3 tensors `{3,1,3}`) — `PrimitiveBase3DGS.cuh:308-329`:

| Field | Bytes |
|---|---|
| scale (post-exp) | 12 |
| opacity | 4 |
| rgb | 12 |
| **Total** | **28 B / entry** |

Backward allocates a matching `v_splats_s` (another 40 or 28 B/entry).

**Minimization:** `packed=true` (default) stores only visible nnz, not full \(C\cdot N\) (`EngineForward.cpp:110-153`; CLI `packed=true` at `cli_config.h:80`).  
`split_batch` drops peak for these buffers by ~\(1/B\) (`EngineConfig.h:150-152`).

### 6.2 Projection extras

| Buffer | Shape | Bytes | File:line |
|---|---|---|---|
| AABB | packed `[nnz] float4` or `[C,N] float4` | 16/entry | `EngineState.h:128`; `EngineForward.cpp:151-194` |
| sorting depths | `[nnz]` or `[C,N]` float | 4/entry | `EngineForward.cpp:153,196` |
| camera_ids / gaussian_ids | packed only, int32 | 4 each / nnz | `EngineState.h:126-127` |
| radii | `[N]` float | 4/splat | see §2.6 |

### 6.3 Tile intersection

| Buffer | Shape | Bytes | File:line |
|---|---|---|---|
| `isect_ids` | `[n_isects] int64` | 8×n_isects | `IntersectTile.cuh:16-17` |
| `flatten_ids` | `[n_isects] int32` | 4×n_isects | same |
| `tile_offsets` | `[C, tile_h, tile_w] int32` | 4×C×⌈H/8⌉×⌈W/8⌉ | `EngineState.h:131`; tiles 8×8 (`Common.cuh:36-37`) |

\(n_isects\) = sum of tile hits (≫ nnz when large footprints).

### 6.4 Raster forward outputs (per pixel)

| Buffer | Shape | Bytes/pixel | File:line |
|---|---|---|---|
| `renders` RGB | `[C,H,W] float3` | 12 | `EngineState.h:136` |
| depth (if rendered) | float | 4 | primitive `RGB_D` (`EngineState.h:25-28`) |
| `render_Ts` (transmittance) | `[C,H,W] float` | 4 | `EngineState.h:133` |
| `last_ids` | `[C,H,W] int32` | 4 | `EngineState.h:135` |
| `distortions` | only if reg on | 4–12 / active ch | `EngineState.h:137-138` |
| `render_median` | optional | 4 | `EngineState.h:134` |

Backward also needs `v_render_*` of matching footprint (loss path).

### 6.5 Shared-memory / tile scratch (not VRAM-persistent)

Raster kernels load splat batches into shared memory (`RasterizationMomentsFwd.cu:114-115`):

- Tile: \(8\times 8 = 64\) threads (`Common.cuh:36-37`)
- Macro-tile: \(2\times 2\) tiles (`:38-39`) → up to 256 threads / macro block
- Shared: `FragmentFwd splat_batch[TILE_AREA_M]` + hit flags — **register/shared, not per-splat pool**

### 6.6 How intermediates are minimized

| Technique | Effect | Evidence |
|---|---|---|
| **Packed projection** | Screen buffers size nnz ≤ C·N | default `packed=true` |
| **FPBO** | Eliminates world grad buffer entirely (3dgs/mip) | `EngineLoss.cpp:24-49` |
| **Grad quant** | World grads ~half non-SH, ¼ SH vs fp32 | §3.4 |
| **SH value quant** | Drops largest param buffer \(12K\to 6K\) (or \(3K\)) | `EngineOptim.cpp:72-74` |
| **split_batch** | Peak screen/tile/raster ~÷B; trades for quantized/fp32 grad accum | `EngineConfig.h:143-152` |
| **No SH in screen buffer** | Color already evaluated to RGB at project | Screen has `rgb` only (`PrimitiveBase3DGS.cuh:145-149`) |
| **Pool high-water mark** | Avoids realloc thrash; memory never shrinks until free | `Tensor.h:139-143` |
| **Null WorldBuffer slots** | Raster atomicStore no-ops when ptr null | `PrimitiveBase3DGS.cuh:75-80`; `EngineLoss.cpp:27-28` |

### 6.7 Order-of-magnitude: activation vs persistent (example)

For \(N=10^6\), \(K=15\), \(C=1\), \(H=W=1080\), assume nnz \(\approx 0.3N\), \(n_isects\approx 10\,\mathrm{nnz}\):

| Class | Rough size |
|---|---|
| Persistent L1+FPBO splat state | ~0.31 GB |
| Screen fwd+bwd (40 B × nnz ×2) | ~0.024 GB |
| Tile isect (~12 B × n_isects) | ~0.036 GB |
| Render + Ts + last_ids (~20 B × HW) | ~0.023 GB |

Intermediates are image/occupancy-dominated; persistent state is \(N\)-dominated and is the quantization target.

---

## 7. Kernel map (who reads/writes what)

| Kernel / unit | Params | Grads | Optim state |
|---|---|---|---|
| `ProjectionFwd*` | read (SH dequant if q) | — | writes `radii` |
| `Rasterization*Fwd` | via screen | — | — |
| `Rasterization*Bwd` | — | world fp32 and/or screen `v_splats_s` | `accum_weight` |
| `ProjectionBwd` | read | atomicAdd world fp32 | — |
| `ProjectionBwdQuantGrad` | read | decode/encode GradQuant | — |
| `FusedProjectionBwdOptim` (FPBO) | read/write (SH re-encode) | register only (+3dgut world geom) | decode/Adam/encode quant |
| `FusedGeometryOptim` | write | read fp32 or GradQuant | quant or fp32 g1/g2 |
| `FusedAppearanceOptim` | SH write | SH grad | SH quant state |
| Densify / Relocation | copy/encode zeros into quant states | — | relocate moments |

`NonShQuantState` is the host-side pointer bundle for non-SH Adam quant (`NonShQuantState.h:7-15`); `GradQuantBuffers` is the analogous bundle for grads (`ProjectionBwdQuantGrad.cuh:13-27`).

---

## 8. Stale comments vs code (for planners)

| Comment claims | Code actually does |
|---|---|
| `EngineConfig.h:104-108`: SH value quant “not yet plumbed” / will throw | Level 1 sets `sh_value_bits=16`; full alloc + Slang q16 path exists |
| `EngineConfig.h:113-114`: non-SH optim quant “FPBO-only” | `EngineOptim.cpp:51-57` explicitly supports non-FPBO via same buffers |
| `EngineConfig.h:128-132`: FPBO “SH quantization not yet supported” | FPBO kernel templates `QUANT_BITS` / `VALUE_BITS` (`FusedProjectionBwdOptim_kernel.cuh:159-160`) |

Treat **`EngineOptim.cpp` + kernel bodies** as source of truth over older `EngineConfig.h` comments.

---

## 9. One-page cheat sheet (level 1 + FPBO + 3dgs, \(K=15\))

| Component | Bytes/splat |
|---|---|
| means/quats/scales/opac/dc (fp32 params) | 56 |
| SH params q16 | 90 |
| World grads | **0** |
| Non-SH Adam q16 \((u,\log_s)\) | 56 |
| SH Adam q8 | 90 |
| radii + densify accum + accum_weight | 16 |
| **≈ TOTAL** | **≈308** |
| Bounds overhead (all float2/float4 tables) | **≪1** amortized |

**Unquantized non-FPBO baseline at same \(K\):** **960** B/splat.
