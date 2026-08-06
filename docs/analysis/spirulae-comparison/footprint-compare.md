I'll explore both repositories read-only and build an exact training memory ledger from the splat data, gradient, optimizer, and rasterization paths.Diving into core splat parameter layouts and memory structures in both repos.I have enough evidence to assemble the full memory ledger. Verifying a few final formulas for raster tile counts and SH byte math.# Training Memory Ledger: LichtFeld-Studio (LFS) vs spirulae-splat (SS)

**Scope:** device-side training state only (parameters, grads, Adam, densify aux, raster transients, allocator policy).  
**Assumptions for totals:** SH degree **3**, capacity **N = 2,000,000**, resolution **1920×1080**, batch cameras **C = 1**, primitive **3dgs**, packed projection where SS uses it.  
**Primary SS config** (model defaults): `quantization_level=1`, `use_fused_proj_bwd_optim=True` — note both `split_batch=True` and FPBO are defaulted, but the engine **throws** if both are active; the ledger uses the **FPBO** branch (not split_batch).

---

## 1. Per-splat parameter bytes

### LFS — `SplatData` (all `float32`)

Evidence: shapes validated at deserialize; SH rest is **swizzled float4**, not canonical `[N,K,3]`.

| Attribute | Layout | dtype | Bytes/splat | Evidence |
|-----------|--------|-------|-------------|----------|
| position (`means`) | `[N,3]` | fp32 | **12** | `splat_data.hpp` params; require `[N,3]` at `splat_data.cpp:1120-1121` |
| rotation | `[N,4]` | fp32 | **16** | `splat_data.cpp:1128` |
| scale (log) | `[N,3]` | fp32 | **12** | `splat_data.cpp:1127` |
| opacity (logit) | `[N,1]` | fp32 | **4** | `splat_data.cpp:1129` |
| SH0 (`sh0`) | `[N,1,3]` | fp32 | **12** | `splat_data.cpp:1126` |
| SH rest (`shN`) | swizzled 1D: `ceil(N/32) × slots(rest) × 32 × 4` floats | fp32 | **192** (deg 3: 12 float4 slots → 48 floats) | layout `sh_layout.cuh:44-93`; slots deg3=`12` at `sh_layout.cuh:76-82`; comment `splat_data.hpp:257-263` |
| **Param subtotal** | | | **248** | |

Swizzle waste vs canonical 15×float3: **192 − 180 = +12 B/splat** for deg 3 (`sh_layout.cuh:26-34` packs 15×float3 into 12×float4 with tail padding).  
**No SH value quantization** in LFS (params always fp32).

Exportable packing confirms same sizes: means/scaling/rotation/opacity/sh0/shN regions at `splat_exportable_storage.cpp:59-72`.

### SS — `WorldSplats` (`EngineState.h:80-110`)

| Attribute | Layout | dtype (level 0) | Bytes/splat L0 | dtype (level 1 default) | Bytes/splat L1 | Evidence |
|-----------|--------|-----------------|----------------|-------------------------|----------------|----------|
| `means` | `DeviceVector<float3>` | fp32×3 | 12 | same | 12 | `EngineState.h:82` |
| `quats` | `DeviceVector<float4>` | fp32×4 | 16 | same | 16 | `EngineState.h:83` |
| `scales` | `DeviceVector<float3>` | fp32×3 | 12 | same | 12 | `EngineState.h:84` |
| `opacities` | `DeviceVector<float>` | fp32×1 | 4 | same | 4 | `EngineState.h:85` |
| `features_dc` | `DeviceVector<float3>` | fp32×3 | 12 | same | 12 | `EngineState.h:86` |
| `features_sh` | `[max_N, K]` float3, **K=15** deg3 | fp32 | **180** | `QuantizedTensor<16,256>` **2 B/cell** × (K×3=45) | **90** + ~0.03 bounds | fp32: `EngineState.h:87`; quant: `EngineState.h:89-108`, codec `Tensor.h:1289-1330`; bits from level: `core.py:104-111` |
| **Param subtotal** | | | **236** | | **146** | |

Level mapping (`core.py:104-111`, `EngineConfig.h:117-125`):
- level 0 → all 32-bit  
- level 1 → `sh_value_bits=16`, `sh_optim_bits=8`, `non_sh_optim_bits=16`

Default model: `quantization_level: int = 1` (`model.py:123-129`).

---

## 2. Per-splat gradient bytes

### LFS

| Path | Storage | Bytes/splat | Evidence |
|------|---------|-------------|----------|
| **Default FastGS fused Adam** | No persistent world-grad tensors; `allocate_gradients` leaves `state.grad` empty | **0** persistent | comment `adam_optimizer.cpp:196-197`, `205-208`; `ensure_grad` only on `get_grad` (`adam_optimizer.cpp:425-447`) |
| Transient FastGS helpers (arena, per step) | `grad_mean2d` 2f, `grad_conic` 3f, `grad_depth` 1f, `grad_opacity` 1f, `grad_color` 3f | **40** | sizes `rasterization_api.cu:288-293`, `532-533`, `556-568` |
| Lazy non-fused / reg `get_grad` | full fp32 matching params | **248** if all allocated | `ensure_grad` zeros fp32 (`adam_optimizer.cpp:442-445`); shapes match params |

Fused backward applies Adam **in-kernel** from screen-space grads (`kernels_backward.cuh:71-86`, `291-311`; `fused_adam_types.h:11-12`).

### SS

| Path | Storage | Bytes/splat | Evidence |
|------|---------|-------------|----------|
| **FPBO** (`use_fused_proj_bwd_optim`, 3dgs/mip) | All world grad buffers released / empty | **0** | `_alloc_grad_buffers` `EngineLoss.cpp:32-49` |
| Non-FPBO + `quantize_grad` (level≠0) | means/quats/scales/opac/dc: **int16** cells; SH: **int8** cells; bounds `float2` per 256 splats | **6+8+6+2+6+45 = 73** + ≪1 bounds | `EngineState.h:164-191`; alloc `EngineLoss.cpp:58-99`; codec `GradQuant.cuh:45-49` (16-bit → 2 B, 8-bit → 1 B) |
| Non-FPBO fp32 | same layout as params | **236** | `EngineLoss.cpp:110-116` |

`quantize_grad` set only when `!FPBO && quantization_level!=0` (`EngineTrainStep.cpp:67-68`).

---

## 3. Per-splat optimizer state bytes

### LFS — **already quantized** (not fp32 m+v)

`AdamParamState` (`adam_optimizer.hpp:45-54`):
- `exp_avg`, `exp_avg_sq`: **uint8**, same shape as param  
- `exp_avg_scale`, `exp_avg_sq_scale`: **fp32**, **one scale per primitive** (not per component)  
- m uses zero-point 128 (`adam_optimizer.cpp:28`, `52-64`)

Allocation: `alloc_quantized_state` (`adam_optimizer.cpp:303-331`); scales length = `splat_data_.size()` (`adam_optimizer.cpp:273-277`).

| Param group | Moment cells/splat | m+v bytes | Scale bytes | Total |
|-------------|-------------------|-----------|-------------|-------|
| means | 3 | 6 | 8 | **14** |
| sh0 | 3 | 6 | 8 | **14** |
| shN (deg3) | 48 floats → 48 u8 each | 96 | 8 | **104** |
| scaling | 3 | 6 | 8 | **14** |
| rotation | 4 | 8 | 8 | **16** |
| opacity | 1 | 2 | 8 | **10** |
| **Optim subtotal** | | | | **172** |

Fused step wires these buffers into FastGS (`adam_optimizer.cpp:614-645`; `fused_adam_types.h:11-12`).

### SS — `SplatOptim` (`EngineState.h:195-263`)

**Level 1 (default) + FPBO layout:**

| Component | Encoding | Bytes/splat | Evidence |
|-----------|----------|-------------|----------|
| non-SH Adam (means,quats,scales,opac,dc) | `QuantizedAdamState<16,256>`: **4 B/cell** joint `(u, log_s)` | 14 cells × 4 = **56** | `Tensor.h:946-970`; resize `EngineOptim.cpp:134-140` |
| SH Adam | `QuantizedAdamState<8>` holder, **2 B/cell** | 45 × 2 = **90** | `Tensor.h:970`; FPBO bounds `EngineOptim.cpp:168-170` |
| Bounds overhead | `float4` per 256 splats/attr | ~0.06–0.3 | `EngineOptim.cpp:135-140,169` |
| **Optim subtotal L1** | | **~146** | |

**Level 0 (fp32 g1/g2):**

| Component | Bytes/splat |
|-----------|-------------|
| g1+g2 all attrs | 2 × (12+16+12+4+12+180) = **472** |

Alloc: `EngineOptim.cpp:87-127`.

---

## 4. Per-splat auxiliary (densification / counters)

### LFS

| Buffer | Shape | Bytes/splat | Evidence |
|--------|-------|-------------|----------|
| `_densification_info` | `[2, N]` fp32 | **8** | `mcmc.cpp:139-147`; bwd atomic to `[2,N]` `RasterizeToPixelsFromWorld3DGSBwd.cu:70,436-437` |
| MCMC `_error_score_max` | `[N]` fp32 | **4** (strategy) | `mcmc.cpp:150-155` |
| MCMC `_noise_buffer` (prealloc) | `[max_cap, 3]` | **12** at capacity | `mcmc.cpp:876-877` |
| Soft-delete `_deleted` | `[N]` bool | **1** optional | `splat_data.hpp:369` |

Default strategy densify core: **8 B/splat** (`densification_info` only).

### SS

| Buffer | Shape | Bytes/splat | Evidence |
|--------|-------|-------------|----------|
| `radii` | `[max_N]` float | **4** | `EngineState.h:237`; resize `EngineOptim.cpp:149-150` |
| `accum_buffer` | `[max_N]` float2 | **8** | `EngineState.h:238`; `EngineOptim.cpp:152-153` |
| `accum_weight` | `[max_N]` float | **4** when densify scoring | `EngineState.h:139` |
| `world_grad_score` | `[max_N]` float | **4** if enabled | `EngineState.h:140-145` |
| `bias_correction_steps` | `[max_N]` int32 | **4** if enabled | `EngineState.h:239`; `EngineOptim.cpp:219-224` |

Always-on optim aux: **12 B/splat** (radii + accum_buffer).

---

## 5. Transient rasterization memory (formulas)

### Common tile geometry

| System | Tile size | Tiles @ 1920×1080 |
|--------|-----------|-------------------|
| LFS FastGS | 16×16 (`rasterization_config.h:32-34`) | `⌈1920/16⌉×⌈1080/16⌉ = 120×68 = **8160**` |
| SS intersect | macro 16×16 (`TILE_SIZE_X/Y=8`, `MACRO=2` → `Common.cuh:36-39`; `IntersectTile.cu:19-20,303-305`) | same **8160** macro-tiles; raster uses 8×8 micro-tiles |

Let:
- \(N\) = #primitives (LFS uses live `n_primitives`; SS may size some pools to `max_num_splats`)
- \(I\) = #cameras in batch  
- \(n_{\mathrm{isect}}\) / \(n_{\mathrm{inst}}\) = total tile-splat instances (data-dependent)

### LFS FastGS (`buffer_utils.h:252-304`, `forward.cu:273-310`, `rasterization_api.cu:277-293`)

**Per-primitive blob** (aligned; ~68 B/N logical + 128 B alignment + CUB scan WS):

| Field | Type | Bytes |
|-------|------|-------|
| depth_keys | uint | 4N |
| depths | float | 4N |
| n_touched_tiles | uint64 | 8N |
| offset | uint64 | 8N |
| screen_bounds | ushort4 | 8N |
| mean2d | float2 | 8N |
| conic_opacity | float4 | 16N |
| color | float3 | 12N |
| + CUB InclusiveSum workspace | | query-dependent |
| **Σ** | | **≈ 68N + cub_scan** |

**Per-tile blob** (`n_tiles = ⌈W/16⌉⌈H/16⌉`):

| Field | Bytes |
|-------|-------|
| instance_ranges | `8 × n_tiles` |
| n_contributions | `4 × n_tiles × 256` |
| final_transmittance | `4 × n_tiles × 256` |
| **Σ** | **2056 × n_tiles** → **~16.8 MB** @ 1920×1080 |

**Per-instance sort** (`forward.cu:273-309`):  
`2 × n_inst × 4` (keys) + `2 × n_inst × 4` (indices) + CUB radix WS  
= **16 × n_inst + cub_sort**

**Backward helpers** (arena): **40N** (see §2).

**Image I/O (caller tensors, not in blob):** typically rendered RGB + α + their grads → **32 B/pixel** if all fp32  
→ `1920×1080×32 ≈ 63.3 MB`.

Arena: VMM, 128 MB initial commit, 2 MB granularity, up to 8 GB max (`memory_arena.hpp:30-35`) — frame-local bump, capacity high-water.

### SS (`PrimitiveBase3DGS.cuh` ScreenBuffer, `IntersectTile.cu`, `EngineForward.cpp`)

**Projected / screen-space** (`ScreenBuffer` channels `{2,1,3,1,3}` → 10 floats) (`PrimitiveBase3DGS.cuh:215-219`):  
- packed: **40 × nnz**  
- non-packed: **40 × I × N**

**AABB + depth for intersect:** float4 + float → **20 × (nnz or I·N)** (`EngineForward.cpp:151-196`).

**Tile intersection** (`IntersectTile.cu:308-357`):

| Buffer | Size |
|--------|------|
| tiles_per_splat | 8 × total_count |
| cum_tiles_per_splat | 8 × total_count |
| isect_ids A+B | **2 × 8 × n_isect** |
| flatten_ids A+B | **2 × 4 × n_isect** |
| offsets | 4 × I × tile_h × tile_w |
| CUB scan/sort WS | via `CUB_WRAPPER` |

**Render outputs retained** (`EngineState.h:132-136`):  
`render_Ts` / `last_ids` / renders RGB(+depth): order **O(I·H·W)** floats.

**World grads under FPBO:** 0 for 3dgs (§2).

**Peak formula (packed, C=1, FPBO):**  
\[
M_{\mathrm{ss,trans}} \approx 40\,nnz + 20\,nnz + 16\,n_{\mathrm{isect}} + 12\,n_{\mathrm{isect}} + O(HW) + \mathrm{CUB}
\]
(with both sort buffers resident → **16+12** bytes/isect for keys+values double-buffers).

---

## 6. Allocator overhead / capacity policy

### LFS

| Mechanism | Policy | Evidence |
|-----------|--------|----------|
| Parameter capacity | MCMC prealloc to `max_cap` (default **1_000_000**) via `zeros_direct` | `parameters.hpp:147`; `mcmc.cpp:830-874,894` |
| Adam capacity | `initial_capacity` / `growth_factor=1.5` | `adam_optimizer.hpp:41-42`; growth `adam_optimizer.cpp:939-944` |
| SizeBucketedPool | Round up to coarse buckets (256 KiB … GiB steps); cache ≤4 entries/bucket; cache budget 64–256 MiB | `size_bucketed_pool.hpp:31-70,88-99` |
| Raster arena | High-water **committed** physical; never shrinks within session without teardown | `memory_arena.hpp:30-35,101-119` |
| Slack meaning | **max_cap − N_live** full param+optim rows often resident; pool bucket **internal waste** on free lists |

At N=2M user must raise `max_cap`; with `max_cap=N`, capacity slack ≈0 aside from SH block pad (N%32).

### SS

| Mechanism | Policy | Evidence |
|-----------|--------|----------|
| World/optim sizing | Always at **`max_num_splats`** (typically `cap_max`, default 1e6; prealloc pads tensors) | `model.py:140-143,602-605,737-744`; `EngineOptim.cpp:65` |
| DevicePool | **Exact `n*sizeof(T)`** when growing; **never shrinks** `cap_bytes` until free | `Tensor.h:139-188` |
| Densify growth | `growth_factor=1.05` on **count**, not buffer growth factor | `model.py:162-163` |
| Slack meaning | **`max_N − cur_N`** entire world+optim+grad capacity held; per-slot HWM from peak `n_isect` / image size |

---

## 7. Side-by-side comparison table (bytes/splat)

**Primary training modes:**  
- LFS: FastGS + fused quantized Adam, SH3  
- SS: quant level 1 + FPBO, SH3, 3dgs  

| Category | LFS | SS (q1+FPBO) | LFS − SS | Notes |
|----------|-----|--------------|----------|-------|
| Position | 12 | 12 | 0 | |
| Rotation | 16 | 16 | 0 | |
| Scale | 12 | 12 | 0 | |
| Opacity | 4 | 4 | 0 | |
| SH0 / DC | 12 | 12 | 0 | |
| SH rest | **192** fp32 swizzled | **90** int16 cells | **+102** | largest param gap |
| **Σ parameters** | **248** | **146** | **+102** | |
| Persistent world grads | **0** | **0** | 0 | both fuse/skip |
| Transient grad helpers | **40** (arena) | ~0 world; screen bwd in place | **+40** (step peak) | LFS helpers always |
| Non-SH Adam | **68** (u8×attrs + scales) | **56** (16-bit joint) | **+12** | |
| SH Adam | **104** | **90** | **+14** | |
| **Σ optimizer** | **172** | **~146** | **+26** | |
| Densify aux | **8** | **12** | **−4** | LFS densify_info vs radii+accum |
| **Σ persistent /splat** | **428** | **~304** | **+124** | params+optim+core aux |
| Transient raster (order) | ~68N + 16 n_inst + 40N + 2056 n_tiles + O(HW) | ~60 nnz + 28 n_isect + O(HW) | scene-dependent | formulas §5 |

**SS level 0 (fp32 SH + fp32 Adam, FPBO):** persistent ≈ 236 + 472 + 12 = **720 B/splat** — **LFS is leaner than SS-L0**.

**SS non-FPBO + grad quant:** add **~73 B** world grads (`EngineLoss.cpp:58-99`).

---

## 8. Totals for N = 2M @ 1920×1080

Using **capacity = live = 2M** (no max_cap slack).

### Persistent model state

| | LFS | SS q1+FPBO | Δ (LFS−SS) |
|--|-----|------------|------------|
| B/splat | 428 | 304 | +124 |
| **Total** | **0.856 GB** | **0.608 GB** | **+0.248 GB** |

Breakdown LFS 2M: params 0.496 GB + optim 0.344 GB + densify 0.016 GB.  
SS 2M: params 0.292 GB + optim 0.292 GB + aux 0.024 GB.

### Step peak (order-of-magnitude, C=1, exclude dataset GT)

| Component | LFS | SS (packed FPBO) |
|-----------|-----|------------------|
| Persistent | 0.856 GB | 0.608 GB |
| Per-prim / screen intermediates | ~68N ≈ 0.136 GB + 40N helpers ≈ 0.080 GB | ~60×nnz (≤N) ≈ ≤0.12 GB |
| Tile fixed | ~0.017 GB | ~0.001–0.03 GB offsets+scratch |
| Instance/isect sort | 16×n_inst | 28×n_isect |
| Image RGB+α+grads | ~0.063 GB | similar O(HW) |

**n_inst / n_isect are view-dependent** (not fixed by N alone). Both keep **double-buffered** sort keys/ids.

If LFS `max_cap=2M` but only 1M live, persistent still ≈ **0.856 GB** (capacity-sized). Same for SS `max_num_splats`.

---

## 9. Ranked gap-closers for LFS (vs SS q1+FPBO)

| Rank | Gap-closer | ~B/splat saved @ SH3 | Why / evidence |
|------|------------|----------------------|----------------|
| **1** | **Quantize SH rest parameters** (fp32→16-bit block, SS-style `QuantizedTensor<16>`) | **~90–102** | LFS always stores 192 B SH rest (`sh_layout.cuh`); SS level1 uses 90 B (`EngineConfig.h:109`, `Tensor.h:1302`) |
| **2** | **Tighter SH rest layout** (drop float4 tail pad / store 15×float3 or 45 scalars) | **~12** param (+ proportional optim moments) | 12 float4 slots pack 45 floats + 3 pad (`sh_layout.cuh:26-34`) |
| **3** | **Joint Adam codec for optim state** (SS `(u,log_s)` 16-bit non-SH / 8-bit SH) | **~26** total optim | LFS 172 vs SS 146; SS joint codec `Tensor.h:916-970`; LFS separate u8 m + u8 √v + **per-primitive scales** (`adam_optimizer.hpp:47-50`) |
| **4** | **Reduce per-group fp32 Adam scales** (6 groups × 8 B = 48 B/splat of scales alone) | up to **~30–40** if block-bounds replace scales | scales always `[N]` fp32×2 per group (`adam_optimizer.cpp:324-330`) |
| **5** | **Fuse away FastGS grad helpers** (or reuse single workspace) | **40** peak (not persistent) | always allocated in forward for bwd (`rasterization_api.cu:288-293`) |
| **6** | **Capacity policy: allocate to live N + small growth, not full max_cap early** | proportional to unused rows | MCMC prealloc max_cap (`mcmc.cpp:830-894`); SS same issue via `max_num_splats` |
| **7** | **Pool / arena HWM discipline** | variable | SizeBucketedPool cache waste (`size_bucketed_pool.hpp:31-36`); arena never shrinks (`memory_arena.hpp`); SS exact slots but also never shrinks (`Tensor.h:139-141`) |
| **8** | **Gradient quantization** (only if non-fused paths allocate full fp32 grads) | up to **248** if lazy grads materialize | SS GradQuant 73 B (`GradQuant.cuh`, `EngineLoss.cpp:58-99`); LFS fused path already 0 persistent |

### Explicit corrections to common assumptions

1. **LFS Adam is not fp32 m+v** — moments are **uint8 + fp32 scales** today (`adam_optimizer.hpp:47-50`).  
2. **Both systems can zero persistent world grads** under fused paths (LFS FastGS fused Adam; SS FPBO).  
3. **SS default is more compressed than LFS on SH values and joint Adam**, not primarily on “grads.”  
4. **LFS densify aux is slightly smaller** than SS radii+accum (8 vs 12).

---

## 10. Evidence index (key files)

| Topic | LFS | SS |
|-------|-----|-----|
| Param structs | `src/core/include/core/splat_data.hpp:191-366` | `src/engine/EngineState.h:80-110` |
| SH layout | `src/core/include/core/cuda/sh_layout.cuh` | `DeviceTensor2D<float3>` / quant in `EngineState.h` |
| Adam state | `src/training/optimizer/adam_optimizer.hpp:45-54` | `EngineState.h:195-231`, `Tensor.h:916-1000` |
| Grad quant | N/A (fused helpers) | `GradQuant.cuh`, `EngineLoss.cpp:20-120` |
| Optim alloc | `adam_optimizer.cpp:167-215,303-331` | `EngineOptim.cpp:18-264` |
| Densify aux | `mcmc.cpp:138-157` | `EngineOptim.cpp:149-153` |
| Raster buffers | `buffer_utils.h:252-304`, `forward.cu:273-310` | `IntersectTile.cu:280-401`, `PrimitiveBase3DGS.cuh:187-225` |
| Quant defaults | always u8 Adam | `model.py:112-129`, `core.py:104-111` |
| Allocators | `size_bucketed_pool.hpp`, `memory_arena.hpp` | `Tensor.h:139-208` DevicePool |

---

**Bottom line for optimization planning:** at SH3 / 2M / FPBO-comparable fused training, LFS is **~+124 B/splat (~0.25 GB)** heavier than SS level-1 mainly because **SH rest stays full fp32 (and slightly padded)** and **Adam state is less compact than SS joint quant**, not because LFS still keeps classic fp32 Adam moments or mandatory full world gradients.
