I'll systematically map the training-step pipeline and related speed techniques from the engine, kernels, and backend code — read-only only.Next I'll dig into tile intersection, rasterization, optim kernels, and backend CUDA for sorting, precision, and streams.# Training-Speed Techniques: Exhaustive Audit

Scope: engine training loop (`EngineTrainStep` + related), kernels under `src/kernels/`, CUDA backend runtime, and the named core headers. Claims cite concrete `file:line`. Speculative notes are marked.

---

## 1. Full training-step pipeline (one iteration)

### Entry points

| Path | Function | File |
|------|----------|------|
| Managed (CLI/GUI default) | `engine_train_step_managed` | `src/engine/EngineDataManager.cpp:104–222` |
| Homogeneous non-warp | `engine_train_step` | `src/engine/EngineTrainStep.cpp:493–523` |
| Homogeneous warp | `engine_train_step_warped` | `src/engine/EngineTrainStep.cpp:677–729` |
| Split one-cam-per-sub-batch | `_engine_train_step_split_one_per_camera` | `src/engine/EngineTrainStep.cpp:214–367` |
| Hetero multi-res | `engine_train_step_hetero` | `src/engine/EngineTrainStep.cpp:378–489` |
| Warp + split | `_engine_train_step_split_warped` | `src/engine/EngineTrainStep.cpp:534–672` |

Inner core after GT/camera setup: `_engine_train_step_after_setup` → `_engine_step_fwd_bwd_only` → `_engine_step_optim_and_densify` (`EngineTrainStep.cpp:169–200`, `55–104`, `112–162`).

### Ordered stages for a **standard single-batch** iteration

Below is the logical order for one train step with batch size B, packed projection, 3dgs/mip (not 3dgut). Optional stages noted.

#### A. Data / camera staging (host→device or zero-copy)

1. **`set_camera_params`** — viewmats/intrins/dist_coeffs to device (or alias device ptrs)  
   `EngineSetup.cpp:50–84`, helpers `EngineCommon.h:87–120`
2. **`set_training_data`** (or **`set_training_data_warped`**)  
   - Non-warp: H2D byte staging + GPU `uint8/16 → float` conversion (`EngineCommon.h:164–237`, `EngineSetup.cpp:87–122`)  
   - Warp: H2D byte RGB + fused **byte→float + warp** into post-split float GT (`EngineSetupWarped.cpp:1–12`, `71–99`, launches ~143–151)  
   - Optional: linear→ray depth, color-space GT convert (`EngineSetup.cpp:103–121`)
3. **`_set_cur_cam_indices`** + **`engine_viewer_capture_thumbnails`** (viewer side; may D2H/D2D)  
   `EngineTrainStep.cpp:186–187`

#### B. Forward (`forward_3dgs` — `EngineForward.cpp:15–291`)

4. Build world splat views; **zero radii** (unless `skip_grad_zero` sub-batch)  
   `EngineForward.cpp:47–60`
5. **Projection forward** (branch on `packed` + primitive)  
   - **Unpacked**: `projection_{3dgs,mip,3dgut}_forward` → AABB `[C,N,4]`, depths `[C,N]`, screen attrs  
     `EngineForward.cpp:154–197`  
   - **Packed** (`ProjectionPackedFwd.cu:86–150`):  
     a. `projection_packed_mask` — per-(camera,splat) visibility mask  
     b. **CUB InclusiveSum** on mask → `nnz` (D2H of last element: line 119–121)  
     c. `projection_packed_fwd` — write `camera_ids`, `gaussian_ids`, AABBs, depths, screen buffer for **nnz only**
6. **Tile intersection** `do_intersect_tile_generic` (`IntersectTile.cu:286–401`):  
   a. Count tiles (ellipse or AABB)  
   b. **CUB InclusiveSum** → `n_isects` (**D2H** of last cumsum element: 336–341)  
   c. Write isect keys `(tile_id << 32 | depth_u32)` + flatten_ids  
   d. **CUB DeviceRadixSort::SortPairs** on keys (full per-frame sort)  
   e. `intersect_offset_kernel` — per-tile start offsets
7. **Rasterization forward** `rasterize_to_pixels_{3dgs,mip,3dgut}_fwd`  
   `EngineForward.cpp:242–266` → micro-tile kernel (`RasterizationEval3DFwd_kernel.cuh:38–310`)
8. **Background blend** (optional) `_engine_background_forward`  
   `EngineForward.cpp:281–283`
9. **Color space** linear/wide → sRGB `_engine_color_space_forward`  
   `EngineForward.cpp:285–288`

#### C. Appearance modules (order from config)

10. **PPISP and/or bilagrid forward** (order: `cfg.ppisp.run_before_bilagrid`)  
    `EngineTrainStep.cpp:83–93`

#### D. Loss + backward (`engine_compute_loss_backward` — `EngineLoss.cpp:421–857`)

11. **`_alloc_grad_buffers`** — fp32 / quant / FPBO-null layouts; zero only if not `skip_grad_zero`  
    `EngineLoss.cpp:20–128`
12. Multi-scale prep: optional auto `num_loss_scales` from resolution  
    `EngineLoss.cpp:447–462`
13. Optional **depth→normal** forward from render depth  
    `EngineLoss.cpp:525–535`
14. **`compute_multi_scale_per_pixel_losses`** (`PerPixelLoss.cu`):  
    - Per scale: avg-pool pyramid, **fused per-pixel loss + grads**, **fused SSIM inplace**, optional edge-aware densify loss map  
    - Async SSIM/loss scalar readout  
    ~`PerPixelLoss.cu:946–1130`
15. **Color-shift reg** (optional) inject into `v_render_rgb`  
    `EngineLoss.cpp:646–701`
16. **PPISP / bilagrid backward** (inverse of forward order)  
    `EngineLoss.cpp:703–735`
17. **Color-space backward**  
    `EngineLoss.cpp:737–746`
18. **Overexposure reg** (optional)  
    `EngineLoss.cpp:749–767`
19. **Background backward**  
    `EngineLoss.cpp:769–778`
20. **depth_to_normal_backward** (optional)  
    `EngineLoss.cpp:780–805`
21. **Rasterization backward** → screen-space (and world for 3dgut) grads + densify `accum_weight`  
    `EngineLoss.cpp:216–277`
22. **Projection backward** — one of:  
    - **FPBO**: stash `v_splats_s` only (`EngineLoss.cpp:282–283`)  
    - **Grad-quant**: `projection_*_backward_quantgrad` (`EngineLoss.cpp:326–347`)  
    - **Standard**: `projection_*_backward` (`EngineLoss.cpp:348–375`)

#### E. Optim + densify (`_engine_step_optim_and_densify` — `EngineTrainStep.cpp:112–162`)

23. **`engine_optim_step`** (`EngineOptim.cpp:490–707`):  
    - **FPBO path**: single fused proj-bwd+Adam kernel (`engine_fused_proj_bwd_optim_step`, 294–488)  
      - Packed: sort gaussians by id + perm (`FusedProjectionBwdOptim.cu:165–208`) then one big kernel  
    - **Non-FPBO**:  
      - `fused_optim_3dgs_geometry` (means/quats/scales/opac + regs + optional zero_grad)  
      - DC: trust-region Adam **or** fused Adam  
      - SH: quantized value+optim / optim-only / TR / plain fused Adam  
24. **Background optim** (optional)  
25. **Bilagrid optim** — fused image-grad + inline TV + Adam; async TV readout  
    `EngineTrainStep.cpp:123–137`, design in `BilagridFusedAdam.cu:1–21`
26. **PPISP optim** + async reg readout  
    `EngineTrainStep.cpp:139–155`
27. **`engine_densify_step`** — clip scales, update scores, relocate/add (MCMC or long-axis), noise  
    `EngineDensify.cpp:35–348`

### Split-batch / hetero variant

- Loop cameras: each runs A–D with `skip_grad_zero=(k>0)` so grads **atomic-accumulate**; sum `accum_weight` into a persistent buffer (`EngineTrainStep.cpp:294–347`, `429–476`).
- Once: optim+densify with `grad_scale=1/B` (or `1/total_cams`) and `zero_grad_in_optim=true` (`349–364`, `478–486`).

---

## 2. Kernel fusion (logical steps → single kernels)

| Fusion | What is fused | Where |
|--------|---------------|--------|
| **FPBO** | Projection VJP + per-splat regs + Adam update (+ optional SH/value quant encode, densify score) | `EngineOptim.cpp:287–293`, `FusedProjectionBwdOptim*.cu/cuh` |
| **Fused geometry optim** | means + quats + scales + opac (+ DC if non-SH quant) + **per-splat regularizers** + optional **zero-grad** + densify world-grad score | `FusedGeometryOptim.cu:1–9`, kernel `83–207` |
| **Fused appearance Adam** | param/grad/m/v update + L2/SH decay + **zero_grad** + **grad_scale** | `FusedAppearanceOptim.cu:11–56` |
| **Fused SSIM** | SSIM forward + dSSIM/dimg1 + optional densify loss-map contribution in one launch | `FusedSSIM.cu:1020–1060`, `PerPixelLoss.cu:957–981` |
| **Per-pixel multi-loss** | RGB/depth/normal/alpha/distortion/median losses **and** all pixel grads in one kernel family | `PerPixelLoss.cu` + Slang `per_pixel_losses.cuh` |
| **Bilagrid fused Adam** | sparse image grads + **inline TV grad** + Adam (fp32 or q16) — no full-grid dense grad table | `BilagridFusedAdam.cu:1–21` |
| **Warp GT upload** | byte H2D staging + **byte→float + fisheye/equirect warp** into post-split float buffer | `EngineSetupWarped.cpp:1–12`, `77–79` |
| **GT byte convert** | H2D staging + convert kernel (not separate host decode to float) | `EngineCommon.h:164–237` |
| **Projection packed** | Visibility culling → compact nnz screen buffer (mask + scan + project) | `ProjectionPackedFwd.cu:86–150` |
| **Raster densify score** | Backward pass computes `accum_weight` while doing VJP | `RasterizationEval3DBwd_kernel.cuh:263–558` |
| **Color-shift reg** | batch sum + EMA inject into `v_rgb` (few kernels, not full extra forward) | `ColorShiftReg.cu`, `EngineLoss.cpp:689–697` |
| **Overexposure reg** | gradient-only add, no scalar materialization | `EngineLoss.cpp:749–755` |

**Explicitly *not* fused / incompatible**

- `split_batch` **xor** `use_fused_proj_bwd_optim` — documented and auto-resolved in managed path (`EngineDataManager.cpp:60–100`, throws in split paths `EngineTrainStep.cpp:230–235`).
- Loss and raster/proj bwd remain separate launches (loss produces cotangents, then raster, then proj).

---

## 3. Rasterization design

### Tiling hierarchy

From `Common.cuh:36–44`:

- Micro-tile: **`TILE_SIZE_X × TILE_SIZE_Y = 8×8`** (one CUDA block, one thread/pixel).
- Macro-tile for binning: **`MACRO_TILE_SIZE = 2×2`** micro-tiles → **16×16 pixel** bin cells (`TILE_SIZE_IX/IY` in `IntersectTile.cu:19–20`).
- Raster block maps `blockIdx.y/z` → micro-tile, then derives macro `tile_id` (`RasterizationEval3DFwd_kernel.cuh:38–90`).

Rationale in-kernel: binning at macro granularity; many SMs work different micro-tiles of the same macro bin; shared memory stays small for occupancy (`RasterizationEval3DFwd_kernel.cuh:38–43`).

### Sorting (full per-frame, not incremental)

Pipeline in `do_intersect_tile_generic` (`IntersectTile.cu:307–401`):

1. Count tile overlaps per splat (ellipse preferred).
2. Inclusive prefix sum → `n_isects`.
3. Emit keys: **`isect_id = (tile_id << 32) | depth_u32`** with float depth bit-twiddled for radix order (`IntersectTile.cu:164–170`, `191`).
4. **`cub::DeviceRadixSort::SortPairs`** on full `n_isects` list (bits = 32 + ceil(log2(n_tiles))) — **complete rebuild every frame**, not incremental (`379–386`).
5. `intersect_offset_kernel` fills per-tile start indices (`242–278`).

Packed projection also uses CUB scan + D2H of `nnz` (`ProjectionPackedFwd.cu:114–121`). No persistent sorted order across training steps.

### Culling layers

| Stage | Method | Ref |
|-------|--------|-----|
| Packed mask | Frustum/visibility mask before compact | `ProjectionPackedFwd.cu:86–110` |
| Intersect count | Degenerate AABB → 0 tiles | `IntersectTile.cu:119–136` |
| Ellipse tile test | Tighter ellipse-vs-tile than AABB; axis choice by aspect | `IntersectTile.cu:32–89`, `174–218`; engine wires conic/opac `EngineForward.cpp:204–220` |
| Micro-tile footprint | `ellipse_box_overlap_test` before α eval | `Common.cuh:52–77`, fwd kernel `194–211` |
| α threshold | `ALPHA_THRESHOLD = 1/255` | `Common.cuh:45` |
| Transmittance early-out | `next_T <= 1e-4` sets `done`; block exits when all pixels done (`__syncthreads_count`) | fwd `261–279`, `171–174` |

### Per-tile / shared memory

**Forward** (`RasterizationEval3DFwd_kernel.cuh:164–167`):

- `splat_batch[TILE_AREA]` — batch of fragments (64).
- `splat_hit[TILE_AREA]` — micro-tile cull flags.
- Cooperative load: thread `tid` loads one gaussian per batch; all threads process batch (`169–217`).

**Backward** (`RasterizationEval3DBwd_kernel.cuh:133–171`): larger shared for reverse walk — rays, T, colors, distortion moments, median state, optional `accum_weight_map`, survivor list `surv[SURV_CAP]`.

Densify score: `atomicMax` of `map * alpha * T0` into per-splat `o_accum_weight` (`535–558`).

### Distortion closed form

Forward accumulates first/second moments; distortion **`D = W*S - C²`** once after the loop (`RasterizationEval3DFwd_kernel.cuh:271–294`). Depth distortion can use **log-depth** moments then `exp` (`263–301`).

---

## 4. Precision strategy

### Training compute path is **fp32**, not IEEE fp16/bf16 tensor cores

- Raster, projection, loss, SSIM use `float` / `float2/3/4` throughout the kernels inspected.
- `cuda_fp16.h` is included in `Common.cuh:5` and Slang runtime supports half/bf16 (`src/generated/slang.cuh`), but **no `__half` / `bf16` math appears on the training splat/optim path** (grep of kernels shows only incidental “half” wording).
- **Conclusion (fact):** reduced precision is via **integer quantized codecs**, not half activations.

### Integer / packed quantization (VRAM + bandwidth)

| Resource | Bits | Codec / layout | Use |
|----------|------|----------------|-----|
| SH **values** | 8 or 16 | `QuantizedTensor` min-max per block (`Tensor.h:1289+`) | World SH; free fp32 `features_sh` when on (`EngineOptim.cpp:72–82`) |
| SH **Adam state** | 4 or 8 | `QuantizedAdamState` joint `(u, log_s)` AoS (`Tensor.h:916–954`) | g1/g2 for SH |
| Non-SH **Adam state** | 16 | same Adam codec, per-splat-block | means/quats/scales/opac/dc (`EngineOptim.cpp:129–147`) |
| **World grads** (non-FPBO) | 16 (geom/dc), 8 (SH) | **Signed symmetric** `gradq::Codec` — **code 0 → 0.0 exactly** (`GradQuant.cuh:1–68`) | Accumulate across split_batch without dense fp32 |
| Bilagrid values | 16 | bilagrid q16 encode (`BilagridFusedAdam.cu:63–79`) | Grid storage |
| GT images on host | u8/u16 | Converted to fp32 on device | `EngineCommon.h:167–169` |
| Viewer thumbs | u8 | Not training compute | `EngineState.h` |

FPBO LEVEL collapse (`FusedProjectionBwdOptim_kernel.cuh:1042–1117`):

- `LEVEL 0`: 32-bit value, 32-bit optim  
- `LEVEL 1`: 16-bit value + 8-bit optim (2 B/cell optim)

Block size **256** is structural for all quant paths (`EngineOptim.cpp:15`, `GradQuant.cuh:20–24`).

### What stays dense fp32

- Raster render buffers, transmittance, last_ids  
- Means/quats/scales/opacities/features_dc **parameters** (unless value-quant only applies to SH)  
- Screen-space intermediate grads after raster bwd  
- Multi-scale loss pyramids  

---

## 5. Streams, concurrency, graph capture, sync points

### Streams

- Kernel launch macros pin **default stream 0**:  
  `_LAUNCH_ARGS_1D/... ,0,(cudaStream_t)0` (`Common.cuh:163–165`).
- Training path does **not** create multi-stream pipelines for proj/raster/loss.
- **Viewer** uses a high-priority non-blocking stream + events relative to default (`Visualizer.cu` ~56–81) — concurrent with training only for viewer work, not core train kernels.
- **No `cudaGraph` / graph capture** found under `src/` training code (grep over engine/kernels/backend).

### Host-side concurrency (data)

- `DataManager` DISK mode: separate worker pools RGB/mask/depth/normal + prefetch queue (`DataManager.h:19–28`, `76–84`).
- CPU mode: full pre-decode at init (`DataManager.h:12–17`).

### Sync points **per training iteration** (device↔host / device sync)

| Sync | Purpose | Location |
|------|---------|----------|
| H2D `memcpy_sync` | Camera + GT (+ warp staging) | `EngineSetup.cpp`, `EngineCommon.h:100`, `EngineSetupWarped.cpp:41–46` |
| D2H last cumsum | Packed `nnz` | `ProjectionPackedFwd.cu:119–121` |
| D2H last cumsum | Tile `n_isects` | `IntersectTile.cu:338–341` |
| (optional) D2H post-select | `do_intersect_tile_post` count | `IntersectTile.cu:438–439` |
| Event sync on AsyncReadout | Losses, SSIM, bilagrid TV, PPISP reg — **previous** step | `Tensor.h:69–125`, use sites `PerPixelLoss.cu:1103–1129`, `EngineTrainStep.cpp:129–154` |
| Explicit `cudaMemcpy` in non-async SSIM | Only if `return_ssim_val` sync path | `FusedSSIM.cu:1087–1089` (async path preferred at scale 0) |

No full `cudaDeviceSynchronize` in the train-step orchestrator itself; backend exposes it for profiling (`BackendRuntimeCuda.h:201–208`).

### CUB temp storage

`CUB_WRAPPER` uses process-global `DeviceScratch` high-water pool — avoids per-call allocator churn (`Common.cuh:182–190`).

---

## 6. Host↔device traffic per iteration & data loading

### What uploads **every** step (managed C++ path)

DataManager hands **host** `TorchTensorView`s into host `std::vector` buffers (`DataManager.h:118–123`, `EngineDataManager.cpp:146–161`). Engine then:

| Transfer | Content | Notes |
|----------|---------|--------|
| H2D | viewmats `[B,4,4]`, intrins `[B,4]`, dist_coeffs `[B,10]` | Or zero-copy if already device (`EngineSetup.cpp:72–78`) |
| H2D | GT RGB bytes or float | Staging slot + convert (`EngineCommon.h:194–232`) |
| H2D | mask/depth/normal if present | Same |
| H2D | bilagrid cam indices (small int32 host buf) | Built each step `EngineDataManager.cpp:134–149` |
| Warp path extra | input intrins/dist + axes table (axes often already device) | `EngineSetupWarped.cpp:51–68` |

World splat parameters **stay resident** after init (`EngineState.h:71–87`); **not** re-uploaded each step.

### What downloads **every** step

| D2H | Why |
|-----|-----|
| `nnz` (int64) | Packed projection allocation size |
| `n_isects` (int64) | Sort buffer size |
| Loss/SSIM/TV/PPISP scalars via **AsyncReadout** | Display dict — **one iteration lag** (`PerPixelLoss.cu:1103–1107`, `Tensor.h:79–82`) |

Rendered images / dense grads are **not** D2H each train step (stay in pool; eval/viewer have separate copy APIs in `EngineQuery.cpp`).

### Data loading pipeline (`DataManager`)

- Group by image shape so each batch is uniform `[B,H,W,C]` (`DataManager.h:6–8`).
- **CPU cache**: decode all images once to host bytes; step = gather rows (`DataManager.h:12–17`).
- **DISK prefetch**: multi-worker decode, `prefetch_batches` in flight (`DataManager.h:19–28`, default workers 16/8/8, prefetch 4 — lines 78–84).
- Optional **warp_to_pinhole** splits fisheye/equisolid to 5 faces / equirect to 6; GPU fuses byte convert + warp (`DataManager.h:93–99`).
- Heterogeneous packing: multiple resolution groups → `engine_train_step_hetero` (`EngineDataManager.cpp:184–221`).

---

## 7. Unusual / high-value techniques (steal list)

| # | Technique | Why it matters | File:line |
|---|-----------|----------------|-----------|
| 1 | **FPBO**: fold projection bwd into Adam; skip allocating world fp32 grads for 3dgs/mip | Huge VRAM + write savings | `EngineLoss.cpp:24–49`, `EngineOptim.cpp:287–293` |
| 2 | **Signed-symmetric grad quant** with **decode(0)=0** | Prevents Adam floaters from zero-out-of-view grads | `GradQuant.cuh:1–14`, `42–68` |
| 3 | **Adam-state `(u, log_s)` quant** (not raw g1/g2) | Relative precision on √g2; 0↔0 fixed point; 4/8/16-bit | `Tensor.h:916–944` |
| 4 | **split_batch + grad_scale 1/B + fused zero_grad** | Memory-friendly multi-cam without materializing B× full batches | `EngineTrainStep.cpp:203–213`, `349–355`; Adam `FusedAppearanceOptim.cu:41–43` |
| 5 | **Macro 16×16 binning + micro 8×8 raster blocks** | Occupancy / SM distribution vs one giant tile | `Common.cuh:36–43`, `RasterizationEval3DFwd_kernel.cuh:38–43` |
| 6 | **Ellipse-exact tile count + range-in-key radix sort** | Fewer isects than AABB; single CUB sort | `IntersectTile.cu:54–89`, `379–386` |
| 7 | **Packed projection nnz** | Avoid `C×N` dense screen storage | `ProjectionPackedFwd.cu:86–130` |
| 8 | **AsyncReadout** one-iter-behind scalars | Removes blocking D2H from critical path | `Tensor.h:69–125` |
| 9 | **Fused SSIM** (MrNeRF-style shared-mem) + multi-scale pyramid grads | Bandwidth-heavy loss fused | `FusedSSIM.cu:1–2`, `1020–1060` |
| 10 | **Bilagrid sparse cam-axis grads + inline TV in Adam** | Avoids 100MB+ dense grad tables | `BilagridFusedAdam.cu:1–21` |
| 11 | **Warp-fused GT byte→float** | No full-res intermediate float image | `EngineSetupWarped.cpp:1–12` |
| 12 | **DevicePool high-water + CUB DeviceScratch** | Zero realloc thrash mid-training | `Tensor.h:139–143`, `Common.cuh:182–190` |
| 13 | **SH value quant frees largest world buffer** | Explicit free of `features_sh` when quantized | `EngineOptim.cpp:72–79` |
| 14 | **Radii accumulate across sub-batches** | Prevents scale-agnostic mean /0 when splat missed last camera | `EngineForward.cpp:47–56` |
| 15 | **EMA color_shift β^(1/B)** under sub-batching | Keeps EMA schedule batch-invariant | `EngineTrainStep.cpp:263–274` |
| 16 | **Resolution-adaptive multi-scale loss** | `loss_scale_min_pixels` → auto scale count | `EngineLoss.cpp:447–461` |
| 17 | **Bilinear GT at native resolution** + atomic scatter bwd | Depth/normal GT need not match render res | `Interpolation.cuh:1–19`, `120–138`; loss uses it |
| 18 | **Distortion D=W*S−C²** without per-splat extra passes | Cheap regularizer in raster | `RasterizationEval3DFwd_kernel.cuh:271–294` |
| 19 | **World-grad densify score** written in optim (before zero_grad) | Correct under fused zero | `EngineOptim.cpp:535–540` |
| 20 | **FPBO packed: sort by gaussian_id + perm**, not camera_ids | Correct indexing of original-order aabb/screen grads | `FusedProjectionBwdOptim.cu:165–177` |

---

## Pipeline diagram (standard non-split step)

```text
DataManager (host decode / prefetch)
    │ H2D cameras + GT (byte→float [+warp])
    ▼
Projection fwd ──► (packed: mask→scan→nnz project)
    ▼
Tile isect: count → prefix → emit keys → CUB radix sort → offsets
    ▼
Raster fwd (8×8 micro-tiles, 16×16 bins, early-out T)
    ▼
Background → color space → bilagrid/PPISP (cfg order)
    ▼
Multi-scale fused loss + SSIM (+ async scalar D2H issue)
    ▼
Appearance bwd hooks → raster bwd → [proj bwd | stash for FPBO]
    ▼
Optim (FPBO OR geometry+appearance Adam) → densify
```

---

## Gaps / non-goals (not found)

- **No CUDA Graph capture** on the train loop.
- **No training use of Tensor Core fp16/bf16** for raster or optim (quantized ints instead).
- **No incremental sort** (full CUB radix each frame).
- Multi-stream compute for overlapping H2D with kernels is **not** implemented on the train critical path (default stream + sync H2D).

---

*Speculation (not evidenced as intentional design):* default-stream-only training may leave GPU idle during H2D of large GT batches; AsyncReadout and DataManager prefetch mitigate host latency but not GPU-host overlap of the large image upload itself.
