I'll analyze the training-step hot path read-only: starting with the step loop, rasterizer, optimizer, losses, kernels, and tensor library usage.Locating and reading the `train_step` implementation and the core training loop.# Training-step hot path analysis (read-only)

Scope: default **FastGS** path (`!params.optimization.gut`). Gsplat/GUT and PPISP-controller-only phases are noted where they diverge. Line numbers refer to the tree at analysis time.

---

## 1. Ordered pipeline of one training iteration

### A. Outer loop — data fetch (`Trainer::train`)

| # | Op | Alloc? | Sync / materialize? | Refs |
|---|-----|--------|---------------------|------|
| 0 | `train_dataloader->next()` | Host queue; GT tensor already on CUDA from pipeline | **May block host** if prefetch empty | `trainer.cpp:5889–5906` |
| 1 | `cudaStreamWaitEvent(training_stream_, depth/normal ready)` | No | GPU-only wait (not host stall unless event failed) | `trainer.cpp:5908–5934` |
| 2 | Move `example.mask/depth/normal` into `pipelined_*` | No new device alloc if pipeline filled them | `set_stream` only | `trainer.cpp:5936–5945` |
| 3 | `train_step(iter, cam, gt_image, …)` | — | — | `trainer.cpp:5960` |

GT images stay on device; no per-step H2D of the full RGB in the training thread when the pipelined loader is used.

---

### B. `train_step` — setup

| # | Op | Alloc? | Sync? | Refs |
|---|-----|--------|-------|------|
| 4 | Control / pause / `apply_pending_params_at_safe_point` | No | Host only | `trainer.cpp:3777–3784` |
| 5 | `waitForModelReaders()` | No | **GPU event waits** (no host block if readers retired) | `trainer.cpp:3789`, `2237+` |
| 6 | `background_for_step` / optional `bg_image` | Cache miss may allocate resized bg | Cache hit: none | `trainer.cpp:3827–3846` |
| 7 | `loss_accumulator_.zero_()` | Alloc once if invalid | Kernel/memset | `trainer.cpp:3850–3854` |

---

### C. FastGS strategy hooks **at step start** (before forward)

On FastGS and not sparsifying, refinement/noise run **before** rasterization:

| # | Op | Alloc? | Sync? | Refs |
|---|-----|--------|-------|------|
| 8 | `strategy_->pre_step` | strategy-dependent | — | `trainer.cpp:3913` |
| 9 | `strategy_->post_backward` (e.g. MCMC) | | | `trainer.cpp:3935`, `mcmc.cpp:681–757` |
| 9a | SH degree bump (interval) | No | Host | `mcmc.cpp:685–687` |
| 9b | `densification_info` max-reduce + **`zero_()` every iter** until stop | No (reuse buffer) | Memset kernel | `mcmc.cpp:700–714` |
| 9c | If refining: relocate/add Gaussians, `trim_memory_pool`, realloc densif buffers | **Yes** topology | Many kernels; possible host syncs in densify | `mcmc.cpp:717–753` |
| 9d | **`inject_noise()` every iteration** | Reuses `_noise_buffer` | `normal_()` + noise kernel | `mcmc.cpp:755–756`, `641–678` |
| 10 | `install_cropbox_step_damping` | **May rebuild bool mask on GPU** | Mask kernel / H2D if recompute | `trainer.cpp:3936`, `1955–1972` |

---

### D. Regularizers (FastGS: **loss-only** before backward; grads fused later)

| # | Op | Alloc? | Sync? | Refs |
|---|-----|--------|-------|------|
| 11 | `ScaleRegularization::forward_loss_only` | **`empty({num_blocks})` + `empty({1})` every call** | Fused reduction kernel, GPU scalar | `trainer.cpp:3995–4010`, `regularization.cpp:59–87` |
| 12 | `OpacityRegularization::forward_loss_only` | Same pattern | Same | `trainer.cpp:4012–4027`, `regularization.cpp:142–170` |
| 13 | Optional sparsity forward | ADMM buffers | GPU | `trainer.cpp:4029–4050` |

---

### E. Rasterize forward (`fast_rasterize_forward` → `forward_raw` → `forward`)

| # | Op | Alloc? | Sync? | Refs |
|---|-----|--------|-------|------|
| 14 | Thread-local `image`/`alpha`/`depth` ensure size | **Only on resolution/stream change** | No | `fast_rasterizer.cpp:328–362` |
| 15 | Arena `begin_frame(stream)` | Arena offset reset | **GPU wait** on previous frame event; device sync only if chain broken | `rasterization_api.cu:250–251`, `memory_arena.cu:276–344` |
| 16 | `validate_fastgs_forward_cuda_preflight` | No | Host: `cudaPointerGetAttributes` on ~10 pointers every step | `rasterization_api.cu:254–271`, `125–180` |
| 17 | Arena bump: per-primitive blob, per-tile blob, grad helpers, optional normals | Arena (not `cudaMalloc` if capacity ok) | No | `rasterization_api.cu:276–308` |
| 18 | `cudaMemsetAsync` tile ranges + forward status | No | Async | `forward.cu:171–180` |
| 19 | `preprocess_cu` | No | Kernel | `forward.cu:183–217` |
| 20 | CUB `DeviceScan::InclusiveSum` (offsets) | Workspace inside per-prim blob | Kernel | `forward.cu:219–238` |
| 21 | **D2H last offset → `n_instances`** | Host scalar | **`cudaMemcpyAsync` + `cudaStreamSynchronize` — hard host barrier** | `forward.cu:240–259` |
| 22 | If `n_instances > 0`: **`cudaMallocAsync` ×5** (keys, keys_alt, indices, indices_alt, CUB sort WS) | **Yes, every step** | Async alloc | `forward.cu:261–300` |
| 23 | `create_instances_cu` | Writes into sort buffers | Kernel | `forward.cu:311–327` |
| 24 | CUB `DeviceRadixSort::SortPairs` (tile+depth key) | Uses sort WS | Kernel | `forward.cu:328–348` |
| 25 | `extract_instance_ranges_cu` | No | Kernel | `forward.cu:353–364` |
| 26 | `blend_cu` → image/alpha/depth | No | Kernel | `forward.cu:366–393` |
| 27 | Free 4 of 5 sort buffers (`cudaFreeAsync`); keep sorted indices | Free | Async free | `forward.cu:395–403` |
| 28 | Background compose (`launch_fused_background_blend*`) | No | Full-image kernel | `fast_rasterizer.cpp:476`, `46–95` |

On zero instances: early skip, release context (`trainer.cpp:4143–4150`).

---

### F. Appearance + photometric (+ optional geo) loss

Default non-controller phase:

| # | Op | Alloc? | Sync? | Refs |
|---|-----|--------|-------|------|
| 29 | Optional bilateral / PPISP forward | Component buffers | Kernels | `trainer.cpp:4364–4380` |
| 30 | `corrected_image.clamp_(0,1)` | No (in-place) | Full-image kernel | `trainer.cpp:4392`, `tensor.cpp:2421–2450` |
| 31 | Photometric (typical `0 < λ < 1`, no mask): | | | |
| 31a | `prepare_loss_images` → `contiguous()` (+ unsqueeze) | Copy if non-contig | May copy | `loss_tensor_contract.hpp:59–62`, `ssim.cu:1880–1882` |
| 31b | `fusedL1SSIMForwardCUDA` | Workspace reused (fp16 dm_* + ssim_map) | Full-image kernel | `ssim.cu:1896–1903` |
| 31c | `launch_fused_l1_ssim_mean_device` | Reuses `reduction_temp` | Reduction kernel(s) | `ssim.cu:1905–1914` |
| 31d | **`loss_scalar = reduction_result.clone()`** | **New [1] tensor every step** | D2D 4B | `ssim.cu:1916` |
| 31e | `fusedL1SSIMBackwardCUDA` after `grad_img.zero_()` | Workspace `grad_img` | Memset + full-image bwd | `ssim.cu:1956–1973` |
| 32 | Masked / decoupled variants | Own workspaces on trainer | Same structure | `trainer.cpp:1669–1688`, `1749–1827` |
| 33 | Optional depth / normal / consistency | Persistent scalar/partial/grad buffers | Kernels; rare lanczos | `trainer.cpp:4523–4941` |
| 34 | Densification error map from SSIM workspace | Prefer reshape in-place; else `densification_error_map_` | Convert kernel; **if λ=0 still runs full SSIM map path** | `trainer.cpp:4946–5007` |
| 35 | Optional PPISP/bilateral backward | | Kernels | `trainer.cpp:5072–5090` |
| 36 | `raster_grad += tile_grad_raw` (decoupled) | Temp via tensor ops | | `trainer.cpp:5092–5094` |

Mask path may add opacity-penalty / alpha-consistency with extra tensor graph ops (`trainer.cpp:1829–1883`).

---

### G. Rasterize backward + **fused quantized Adam**

| # | Op | Alloc? | Sync? | Refs |
|---|-----|--------|-------|------|
| 37 | Thread-local `grad_alpha` ensure | Only on size change | | `fast_rasterizer.cpp:557–569` |
| 38 | `launch_fused_grad_alpha*` | No | Full-image | `fast_rasterizer.cpp:572–589` |
| 39 | Optional `grad_alpha.add_(extra)` | | Kernel | `fast_rasterizer.cpp:591–597` |
| 40 | **Background unblend** on cached image (inverse of step 28) | No | Full-image | `fast_rasterizer.cpp:662–663` |
| 41 | `prepare_fastgs_fused_adam` | No (pointers into existing state) | Optional crop-mask stream sync | `adam_optimizer.cpp:572–649`, `fast_rasterizer.cpp:665–709` |
| 42 | Arena: opacity/color helpers if missing; zero all helpers | Arena bump | Several memsets | `rasterization_api.cu:526–568` |
| 43 | `blend_backward_cu` | No | Tile reverse-blend | `backward.cu:64–114` |
| 44 | `preprocess_backward_cu` | No | **Fused**: param grads + **quantized Adam** + scale/opacity/flatten/sparsity regs; invisible-prim momentum fold | `backward.cu:117–168`, `kernels_backward.cuh:59–96` |
| 45 | `commit_fastgs_fused_adam` | No | Host step_count++ | `adam_optimizer.cpp:652–668` |
| 46 | Free sorted indices + `arena.end_frame` | FreeAsync | GPU chain event | `rasterization_api.cu:399–408`, `619` |

**No separate full-parameter gradient buffers** are required on the FastGS fused path (`allocate_gradients` comment: `adam_optimizer.cpp:196–197`).

---

### H. Commit / reporting

| # | Op | Alloc? | Sync? | Refs |
|---|-----|--------|-------|------|
| 47 | Camera loss heatmap GPU update | No | Kernel | `trainer.cpp:5148–5149` |
| 48 | Add scale/opacity reg scalars into `loss_accumulator_` | Lazy tensor add on [1] | Usually fused later / small | `trainer.cpp:5166–5211` |
| 49 | Optional bilateral/PPISP Adam steps | | Kernels | `trainer.cpp:5214–5247` |
| 50 | Loss D2H every 10 iters: `submitLossReadback` | Pinned ring | **Async D2H + event**; harvest is **non-blocking** `cudaEventQuery` | `trainer.cpp:5306–5322`, `2109–2170` |
| 51 | `strategy_->step` → `optimizer.step` + `zero_grad` | No | **Both no-ops** after fused Adam (`fused_step_iteration_`, `last_step_zeroed_gradients_`) | `mcmc.cpp:759–773`, `adam_optimizer.cpp:135–145`, `226–237`, `652–668` |
| 52 | Scheduler LR update | No | Host only | `mcmc.cpp:771–772`, `scheduler.cpp:32–78` |
| 53 | Sparsity update/prune | Occasional | | `trainer.cpp:5406–5416` |
| 54 | `recordParamsReady` | No | Event record | `trainer.cpp:5435–5436` |

**Progress / metrics:** `current_loss_` and progress bar update only when a loss-ring slot completes (async); not every step (`trainer.cpp:2155–2168`). Full eval is gated (`evaluator_->should_evaluate`, `trainer.cpp:5440–5446`).

---

## 2. Host–device transfers per step

| Transfer | Frequency | Blocking? | Source |
|----------|-----------|-----------|--------|
| **GT image H2D** | Prefetch path; training thread usually already has CUDA tensor | Prefetch may block earlier I/O; not in `train_step` itself | Dataloader `next()` |
| **Mask / depth / normal** | If pipelined: ready on CUDA; sidecar wait via events | GPU wait on `training_stream_` | `trainer.cpp:5908–5945` |
| **`n_instances` D2H (8 bytes)** | **Every** FastGS forward with primitives | **Yes — `cudaStreamSynchronize`** | `forward.cu:246–251` |
| **Loss scalar D2H (4 bytes)** | Every 10 iters (and iter 1) | Async; harvest non-blocking (sync only if ring full) | `trainer.cpp:5306–5322`, `2114–2120` |
| **Camera-loss EMA** | Publish every 16 iters | Async D2H on copy stream + events | `trainer.cpp:2595–2648`, interval `CAMERA_LOSS_PUBLISH_INTERVAL=16` |
| **Progress / TrainingProgress event** | On harvested loss | Host only after event ready | `trainer.cpp:2155–2168` |
| **FastGS debug status D2H** | Only if `cuda_sync_debug_enabled()` or error path | Sync | `buffer_utils.h:121–191` |
| **Crash dump / eval / checkpoint** | Rare | Full syncs | Not hot path |

**Not** downloaded every step: full images, full gradients, Adam moments.

---

## 3. FastGS rasterizer: buffers, sort, tiles, malloc

### Buffer lifetime

| Buffer | Lifetime | Realloc policy |
|--------|----------|----------------|
| Output image/alpha/depth/normal | **Thread-local persistent** | Resize only when W/H or stream changes | `fast_rasterizer.cpp:21–33`, `328–370` |
| Per-primitive + per-tile + bwd helpers | **Arena frame** per forward | Bump allocator; capacity grows, not free each step | `rasterization_api.cu:247–297` |
| Sorted primitive indices | **Live until backward** then `cudaFreeAsync` | **Allocated every forward** (one of the sort pair buffers `release()`d) | `forward.cu:395–402`, `rasterization_api.cu:30–48` |
| Sort keys ×2 + alt indices + CUB WS | **Transient within forward** | **`cudaMallocAsync` every step with n_instances>0**, free at end of forward | `forward.cu:261–300`, destructor `reset()` |

### Sorting

- Keys: packed **tile id + depth** (`InstanceKey` = `uint32`, depth bits capped at 23) — `buffer_utils.h:217–226`, `forward.cu:159–160`.
- Algorithm: **CUB `DeviceRadixSort::SortPairs`** over `n_instances` — `forward.cu:328–335`.
- Instance list built by `create_instances_cu` after exclusive/inclusive tile-count scan — `forward.cu:311–323`.

### Tile assignment

- Screen tiles: **`16×16`** (`rasterization_config.h:32–35`).
- Grid: `ceil(W/16) × ceil(H/16)`; blend grid uses that dim3 — `forward.cu:154–157`, `366–368`.
- Per-tile: `instance_ranges`, `n_contributions`, `final_transmittance` sized `n_tiles × block_size_blend` — `buffer_utils.h:290–303`.

### Per-step `cudaMalloc` / async malloc

**Yes, every non-empty forward:** five stream-ordered buffers for sort (keys current/alt, indices current/alt, CUB workspace). Sorted indices retained until backward. Arena path avoids raw malloc for the bulk of per-primitive/tile state.

### Extra per-step host work

`validate_fastgs_forward_cuda_preflight` runs **every** forward (`rasterization_api.cu:254–271`) — expensive attribute queries, not just asserts.

### Background double-touch

Forward **blends** bg into the thread-local image; backward **unblends** to recover pre-blend image for the reverse rasterizer (`fast_rasterizer.cpp:476`, `662–663`). Two full-image kernels every step whenever bg is solid or image-based.

---

## 4. Adam optimizer

### State layout (`AdamParamState`)

| Field | Dtype / role | Ref |
|-------|--------------|-----|
| `grad` | fp32, **transient**; often **empty on fused FastGS** | `adam_optimizer.hpp:46`, allocate comment `196–197` |
| `exp_avg` | **uint8** quantized m (zero-point 128) | `hpp:47`, `adam_optimizer.cpp:28` |
| `exp_avg_sq` | **uint8** quantized √v | `hpp:48` |
| `exp_avg_scale` / `exp_avg_sq_scale` | **fp32 per-primitive** scales | `hpp:49–50` |
| `shN` moments | Swizzled layout matching SH storage | `adam_optimizer.cpp:507–531` |

### FastGS path: fused, not per-tensor

- `prepare_fastgs_fused_adam` packs pointers + bias-corrected step sizes for means/sh0/shN/scaling/rotation/opacity (`adam_optimizer.cpp:572–649`).
- Adam update runs **inside** `preprocess_backward_cu` via `adam_step_row` / SH packed updates (`kernels_backward.cuh:59–96`, `291+`).
- Invisible primitives get **momentum decay / reg-only** fold in the same kernel (`kernels_backward.cuh:64–86`, comment `backward.cu:117–119`).
- Scale/opacity **regularization gradients** are folded into that kernel (`scale_regularization_grad`, `opacity_extra_grad`); loss **value** for logging is still computed separately earlier.
- `commit_fastgs_fused_adam` only bumps `step_count` and marks fused iteration (`652–668`).
- Subsequent `optimizer.step` / `zero_grad` are **short-circuited** (`135–140`, `226–230`) — **no separate zero_grad memset of param grads** on the fused path when grads were never allocated.

### Non-fused / gsplat path

- `step_param` launches quantized Adam per param group (`542–563`).
- `zero_grad` uses `cudaMemsetAsync` over each `state.grad` (`231–236`).
- Scale/opacity reg write into `get_grad` via fused kernels that allocate temps every call (`regularization.cpp:36–49`).

### Wasted / duplicate passes (FastGS)

| Issue | Detail |
|-------|--------|
| Reg **loss-only** kernels | Extra full-parameter reductions for scale/opacity loss scalars while grads already computed in fused backward | `trainer.cpp:3995–4027` + `kernels_backward.cuh` |
| `strategy_->step` still calls step/zero_grad | Cheap host no-ops after fusion, but still lock + bookkeeping | `mcmc.cpp:759–769` |
| No separate weight-decay pass | Weight decay not present as a distinct kernel; scale/opacity regs replace that role |

---

## 5. Loss computation

### Default photometric (fused L1+SSIM)

Formula documented: `(1-λ)·L1 + λ·(1-SSIM)` (`photometric_loss.hpp:17–18`).

**Passes over the image (typical λ∈(0,1), no mask, no appearance model):**

1. Optional `clamp_` on corrected image  
2. Fused L1+SSIM **forward** (11×11 Gaussian stats + L1 + SSIM map + dm/d* in fp16) — one grid launch  
3. **Reduction** to scalar (L1+SSIM mean) — second launch  
4. Fused L1+SSIM **backward** — third full-image launch (after `grad_img.zero_`)  
5. Optional densify: `launch_ssim_to_error_map` (cheap) **or**, if λ=0, **another full SSIM map** (`trainer.cpp:4982–4993`)

GT **uint8 is supported in-kernel** (`ssim.cu:51–59`, `dispatch_target_ptr`); no mandatory fp32 GT conversion.

### Workspaces (persistent, resize on resolution change)

- `PhotometricLoss::{fused,ssim}_workspace_` + grad/loss buffers (`photometric_loss.hpp:48–58`)  
- Trainer: masked/decoupled workspaces (`trainer.hpp:539–542`)  
- Fused workspace stores **3×fp16 full maps + fp32 ssim_map + grad_img** (`ssim.cuh:129–163`) — large but stable after first alloc  

### Masked / appearance paths

- Masked fused L1+SSIM when mask or ROI weight present (`trainer.cpp:1818–1821`)  
- Decoupled path splits grad into appearance-corrected vs raw for D-SSIM structure terms (`trainer.cpp:1669–1688`)  
- SegmentAndIgnore / opacity penalty: extra mask tensor ops + penalty kernel (`trainer.cpp:1829–1851`)  
- AlphaConsistent: lazy `abs/sign/mean` graph (`1871–1882`)  

### Fusion status / remaining opportunities

| Already fused | Still multi-pass |
|---------------|------------------|
| L1+SSIM forward stats + dm/d* | Separate reduction launch |
| L1+SSIM backward | Separate from rasterizer / not joint with bg grad_alpha |
| Depth loss launch writes loss+grads together | Depth/normal are separate full-image kernels from photometric |
| Masked L1+SSIM kernels exist | Mask preprocess (`masked_fill` chains) is tensor-lib multi-kernel |

---

## 6. Lazy tensor-lib overhead in the hot loop

### Where graphs appear

- Scalar loss accumulation: `loss_tensor_gpu + tile_loss` / reg terms (`trainer.cpp:5068`, `5171`, …) — builds binary lazy nodes (`tensor_expr_impl.hpp` / `lazy_ir_record_binary`).
- Mask opacity / alpha-consistency path uses `masked_fill`, `pow`, `abs`, `mean`, `*` (`trainer.cpp:1768–1882`).
- Densify fallback L1 error: `(corrected - gt).abs().mean(...)` (`trainer.cpp:4995–5006`) — multi-node graph + materialization.

### Materialization / copies that matter

| Site | Cost | Ref |
|------|------|-----|
| `prepare_image` always `.contiguous()` | Copy if non-contig | `loss_tensor_contract.hpp:59` |
| `reduction_result.clone()` | Alloc [1] every photometric forward | `ssim.cu:1916` |
| Depth/normal optional `.contiguous()` | Full tensor copy when needed | e.g. `trainer.cpp:4549–4567` |
| `clamp_` on corrected image | In-place; **not** lazy | `tensor.cpp:2448–2450` |
| Rasterizer outputs | Raw kernels via `.ptr<float>()` — **no lazy graph** | `fast_rasterizer.cpp:376+` |
| Fused Adam | Raw pointers — **no lazy** | `fast_rasterizer.cpp:710–739` |

### Thresholds / fusion

- Lazy executor can fuse pointwise+reduce on CUDA (`tensor_unified_ops.cpp:1155–1287`) when chains remain deferred until a reduce/sink.
- Hot path mostly **bypasses** lazy for rasterizer/Adam/fused loss kernels; residual cost is small graphs for **scalar loss bookkeeping** and **mask edge cases**.
- Telemetry hooks exist (`lazy_config.hpp/cpp`) but are not on the critical path by default.

---

## 7. Top 10 speed opportunities (expected wall-clock impact)

Ranked for typical FastGS + MCMC training (millions of Gaussians, HD views). Quality must not regress.

| Rank | Opportunity | Impact | Evidence | One-line fix sketch |
|------|-------------|--------|----------|---------------------|
| **1** | **Persistent / pooled sort buffers** (eliminate per-step 5× `cudaMallocAsync`/`FreeAsync`) | **Very high** — allocator + fragmentation + launch overhead dominates variable instance counts | `forward.cu:261–300`, `74–96` | Keep a thread/stream-local double-buffer sized to `max(n_instances)` high-water mark; grow only; never free until capacity shrink epochs. |
| **2** | **Remove hard host barrier on `n_instances`** | **Very high** — full pipeline stall every step | `forward.cu:240–259` | Over-allocate instance buffers to a safe upper bound (e.g. sum of tile bounds or N×max_tiles_per_gauss), run create/sort without D2H; or use device-side CUB with max size + compact. |
| **3** | **MCMC `inject_noise` every iteration** | **High** — full `normal_()` over means + noise kernel every step | `mcmc.cpp:755–756`, `641–678` | Gate noise to densify windows / every k iters if quality allows; cache frozen mask; fuse RNG+add into one kernel. |
| **4** | **Fuse bg blend into blend_cu / unblend into blend_backward** | **High** — 2 full-image passes + grad_alpha often separate | `fast_rasterizer.cpp:476`, `571–589`, `662–663` | Pass solid bg or bg_image ptr into forward blend and write final composite once; bwd accumulate alpha grad without unblend+separate kernel. |
| **5** | **Skip / fold reg loss-only kernels** | **Medium–high** when scale_reg & opacity_reg > 0 | `regularization.cpp:75–86`, `trainer.cpp:3995–4027` vs `kernels_backward.cuh:75–86` | Accumulate reg loss scalars in fused preprocess_backward with a device atomic/block reduce; drop pre-forward loss-only launches (grads already fused). |
| **6** | **Photometric: drop `clone()` + fuse reduction into forward; avoid `grad_img.zero_` if rewrite covers all pixels** | **Medium** | `ssim.cu:1916`, `1956–1957` | Write loss into trainer’s persistent `loss_accumulator_` / pinned-friendly buffer; ensure bwd overwrites all valid pixels. |
| **7** | **`densification_info.zero_()` every step** | **Medium** for large N | `mcmc.cpp:712–713` | Zero only rows touched last view, or reset in densify kernel that writes densif info. |
| **8** | **Strip / compile-out FastGS preflight pointer checks** on release hot path | **Medium** host latency | `rasterization_api.cu:125–180`, `254–271` | `#ifndef NDEBUG` or one-time-per-process validation; never `cudaPointerGetAttributes` per step in release. |
| **9** | **Cache cropbox damping mask** across steps | **Medium** when cropbox LR scale ≠ 1 | `trainer.cpp:1955–1972`, `install` each step `3936` | Recompute only when model size or cropbox transform changes; store on trainer. |
| **10** | **Avoid redundant `clamp_` when PPISP CRF already clamps; skip densify SSIM when λ=0 by reusing L1 error map** | **Low–medium** | `trainer.cpp:4389–4392`, `4982–5007` | Branch: if PPISP active skip clamp; if λ=0 and densify needs error, use abs-diff path only (already present) and never re-enter full SSIM. |

### Honorable mentions (still real)

- **Arena `begin_frame` fallback `cudaDeviceSynchronize`** if event chain fails (`memory_arena.cu:341–344`) — keep chain healthy; treat as regression if it fires in profiles.  
- **Per-step scale/opacity `Tensor::empty` temps** even after reg fusion of grads (`regularization.cpp:75–77`) — use persistent reduction workspaces on `Trainer` like photometric.  
- **Mask `masked_fill` chains** (`trainer.cpp:1768–1839`) — single CUDA mask-band kernel.  
- **`make_frozen_mask` host vector + H2D inside inject_noise** when frozen ranges exist (`strategy_utils.cpp:109–128`, `mcmc.cpp:664–667`) — keep persistent GPU mask.  
- **Gsplat path still `cudaFree`s isect buffers per step** (`trainer.cpp:4171–4180`) — out of FastGS scope but large if gut enabled.

---

## Architecture summary (hot path)

```
dataloader (CUDA GT) → [MCMC post_backward+noise] → FastGS fwd
  (arena + preprocess + scan + ⚠D2H n_inst + ⚠malloc sort + radix + blend + bg)
→ clamp → fused L1+SSIM (fwd+reduce+bwd) [+ depth/normal]
→ bg unblend + grad_alpha → FastGS bwd (blend_bwd + preprocess_bwd+fused Adam)
→ async loss D2H / scheduler / record params-ready
```

**Already strong:** fused L1+SSIM workspaces, quantized moment Adam fused into backward, persistent photometric/raster output buffers, async loss readback, arena for per-primitive/tile state, no full-grad buffers on FastGS.

**Primary remaining waste:** per-step sort malloc, mandatory `n_instances` sync, always-on MCMC noise, dual background image passes, and duplicate regularizer reductions for metrics-only scalars.

---

*Analysis is read-only; no files were modified. Items marked as planning sketches are proposals only — quality-preserving validation would be required before shipping.*
