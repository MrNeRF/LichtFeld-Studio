I'll audit the tensor library's CUDA kernels for training-relevant efficiency. Exploring the kernel sources and related launch/dispatch paths first.Reading the core kernel files and their launch helpers next.Digging into each kernel family with targeted reads of launch paths and hot kernels.# Kernel Efficiency Audit — LichtFeld Tensor Library

**Scope:** `src/core/tensor/` compute kernels for training-relevant ops  
**Mode:** Read-only static analysis (no benchmarks run)  
**Priority:** Speed without regression  

Relative gains below are **engineering estimates** for hot paths (large contiguous fp32 unless noted). Marked **[SPECULATIVE]** where not proven by code comments/benchmarks in-repo.

---

## Executive summary

| Area | Contiguous fp32 quality | Main gap vs CUB/Thrust/cuBLAS |
|------|-------------------------|------------------------------|
| Same-shape elementwise | **Good** — `float4` fast path when aligned & `n>1024` | No multi-element grid-stride reuse; no half2/vectorized half |
| Scalar op | **Good** — scalar by value in register | Threshold `n>512`; small tensors go Thrust |
| Broadcast special patterns | **Mixed** — Row/Col/Scalar/Batch3D vectorized; Channel3D partial | Coalesced/smem Channel3D kernels **exist but are never launched** |
| Generic broadcast | **Poor** — scalar, per-thread full index math | CUB not used; no vectorization |
| Full scalar reduce | **Good** — CUB `DeviceReduce` (warp path disabled by policy) | `sum_scalar` syncs stream + D2H every call |
| Segmented reduce | **Good** — tiny/medium/large + strided specializations | Large-segment grid = one block/segment (uncapped); medium grid often oversubscribed |
| Dot / unary norms | **OK** — float4 + two-stage | Per-call `cudaMallocAsync` partials; `n<100k` uses **1 block only** |
| Matmul | **Weak vs vendor** — hand-rolled tiled FP32 only | **No cuBLAS / no tensor cores** |
| NN (pool, bias+relu) | **Basic** — grid-stride, no float4 | ReLU duplicates slower than vectorized unary path |
| Masking / index | **Mixed** — CUB/Thrust for compact; scatter atomics | Count-nonzero host round-trip; compare kernels not float4 |
| Fusion | **Partial** — lazy pointwise + transform-reduce | Binary–binary (`mul` then `add`) **not fused** |

---

## 1. Elementwise / pointwise (`tensor_vectorized_ops.cuh`, `tensor_generic_ops.cuh`, `tensor_ops.cu` clamp)

### 1.1 Memory access

| Kernel | Vectorized? | Grid-stride? | Coalescing (contig fp32) |
|--------|-------------|--------------|---------------------------|
| `vectorized_unary_kernel` | **Yes** `float4` | **No** — one `float4`/thread, grid covers work | Yes if 16B-aligned |
| `vectorized_binary_kernel` | **Yes** `float4` | **No** | Yes |
| `vectorized_comparison_kernel` | `float4` in / `uchar4` out | **No** | Yes (input); 4B-aligned output |
| `vectorized_scalar_broadcast_kernel` | **Yes** `float4` | **No** | Yes; **scalar by value** (register) |
| Unaligned fallback | Thrust `transform` | Thrust policy | Scalar |
| `clamp_kernel_vectorized` | `float4` | No | Yes |
| `clamp_kernel_optimized` / fused scalar | scalar | **Yes** | Yes |

**Evidence:**

- Vectorized unary/binary: `tensor_vectorized_ops.cuh:33–94`, launch `189–219`
- Scalar broadcast by value: `142–166`, `259–275`
- Dispatch thresholds: aligned + `n > 1024` binary/unary (`tensor_generic_ops.cuh:55–75`, `102–110`); scalar `n > 512` (`137–144`)
- Clamp: `tensor_ops.cu:159–222`, `295–317`

**Issue:** Vectorized kernels do **not** use grid-stride multi-pass; each thread does one `float4`. For large `n` this launches huge grids (occupancy OK, but no L2 reuse / fewer launches than SM-capped + stride). Unaligned or `n≤1024` falls to Thrust.

### 1.2 Launch configuration

- **Block size:** 256 almost everywhere  
- **Grid:** `ceil((n/4)/256)` for vectorized; **not** SM-capped (`GPUConfig` unused here)  
- **Occupancy:** 256 threads is fine; no `__launch_bounds__` on elementwise  

### 1.3 Scalars

- **By value** on fast path (`float scalar` kernel param) — no device memory for the scalar  
- Thrust fallback: `constant_iterator` (`tensor_generic_ops.cuh:147–157`) — also no buffer  

### 1.4 Same-shape contiguous fast path

`BinaryExpr` for float32:

- Same shape → `launch_float_binary_with_numeric_policy` → `launch_binary_op_generic` → **vectorized** if aligned (`tensor_expr_impl.hpp:576–582`, `tensor_ops.hpp:252–262`)  
- Different shapes → broadcast path (`563–574`)

**Cost of generic strided/broadcast:** see §2.

---

## 2. Broadcast (`tensor_broadcast_ops.cuh`, `tensor_broadcast_ops.cu`)

### 2.1 Pattern detection

`detect_broadcast_pattern` (`tensor_broadcast_ops.cuh:29–98`): Scalar, Channel3D, BatchBroadcast3D, Row, Column, else **Generic**.

### 2.2 Specialized kernels (float→float)

| Pattern | Vectorized | Notes |
|---------|------------|-------|
| Scalar | `float4` | Loads scalar once from `a[0]`/`b[0]` **per thread** (`118–119`) — not ideal but small |
| Row `(M×N) op (1×N)` | `float4` | 2D/3D grid for large M |
| Column `(M×N) op (M×1)` | `float4` | Column value in register |
| Channel3D `(H×W×C) op (1×1×C)` | C=4 / C%4 float4; C=3 unrolled | **Per-pixel thread**; poor coalescing for large C (comment `993–997`) |
| Batch3D `(B×H×W) op (1×H×W)` | `float4` | |
| Generic | **Scalar only** | Full stride rebuild **every thread every index** (`741–799`) |

**Critical dead code:** Coalesced and shared-memory Channel3D kernels are implemented (`439–669`) but the launcher **only** calls `broadcast_channel3d_kernel_float` (`999–1001`). Smem/coalesced never run.

### 2.3 Launch config

- Block 256; grid = ceil(work/block) or 2D/3D for large dims  
- **No SM-based grid sizing**  
- Params passed by value in `BroadcastBinaryParams` / strided params — good (no device alloc for meta)

### 2.4 Unary broadcast / pad (`tensor_broadcast_ops.cu`)

- `broadcast_strided_kernel`: scalar, no vectorization, no grid-stride (`48–69`); grid = ceil(n/256)  
- `pad_kernel`: same (`133–154`)  
- `launch_broadcast_generic`: **Thrust** permutation iterator (`251–273`) — heavy for large tensors  

### 2.5 Half / non-float

Explicit `__half` broadcast instantiations (`tensor_broadcast_ops.cu:528+`) go through the same template; vectorized float path is `if constexpr (float)` only → **half uses generic scalar kernel**.

---

## 3. Reductions — warp family (`tensor_warp_reduce.cu`, `warp_reduce.cuh`)

### 3.1 Full reduction to scalar

| Path | When | Method |
|------|------|--------|
| **Production full reduce** | Always for `num_segments==1` | **CUB** `DeviceReduce` — warp **disabled** |
| Two-stage sum | `n>100k` if warp API called | Packed128 + `load128cs` + stage2; **no atomics** |
| Single-stage sum | `n≤100k` if warp API called | float4 + **atomicAdd** |
| Max/min/prod | warp API | float4 + **atomicCAS** |

**Policy (important):**

```1562:1567:src/core/tensor/tensor_warp_reduce.cu
bool should_use_warp_reduce(size_t n, size_t num_segments) {
    // SCALAR REDUCTIONS: Always use CUB DeviceReduce (much faster!)
    if (num_segments == 1) {
        return false; // Use CUB path in tensor_ops.cu
```

So `launch_reduce_op_float32` warp branch at `637–671` is **dead for full reduce**. In-file comment claims CUB is **3–7× faster** than custom warp for scalars.

**Two-stage quality (when used / fused transform-reduce):**

- Warp shuffle + shared[32] block reduce: `warp_reduce.cuh:35–117`  
- Stage1 partials → stage2 single block: `tensor_warp_reduce.cu:104–175`, launch `1053–1096`  
- **Allocates** `cudaMallocAsync` partial buffer every call if not reused (`1078–1081`)  
- Grid = `GPUConfig::optimal_grid_size(256)` = `(max_threads_per_sm * sm_count) / 256` (`gpu_config.hpp:79–83`) — good occupancy, grid-stride in stage1  

**Mean full reduce:** sum then Thrust `transform` of **1 element** (`663–669`) — extra launch for a single multiply.

### 3.2 Segmented (contiguous last-dim style)

| Segment size | Kernel family | Vectorized | Grid |
|--------------|---------------|------------|------|
| `< 32` | tiny: 1 thread/segment | No | ceil(segs/256) |
| `32–2048` | medium: 1 warp/segment, float4 coalesced | Yes if `size%4==0` | `max(need, optimal_grid)` — can over-launch |
| `> 2048` | 1 block/segment, vectorized segment reduce | Yes + double accum for sum | **`grid = num_segments` uncapped** |

Medium float4 loop: `337–376`. Mean **fused** divide in-kernel (tiny/medium/large) — good.

### 3.3 Strided reduce (inner_size > 1)

- Thread-per-output-element, 8× unroll, optional **double** accum for sum (`772–821`)  
- Grid = ceil(outputs/256), **not** SM-capped  
- Used only if `inner_size < 256` (`tensor_ops.cu:771–777`); else host may transpose first (`tensor_unified_ops.cpp:1321–1344`) or CUB segmented  

**Column reduce** `[M,N]→[N]`: 2D grid, optional **atomicAdd** when `grid_y>1`, needs **memset** zero (`1419–1528`). Mean uses extra Thrust scale.

### 3.4 CUB path details (`tensor_ops.cu`)

- Full sum/mean/max/min: CUB workspace via pool (`112–142`, `674–717`)  
- **Prod full:** Thrust reduce → host `result` → `init_scalar_gpu` (`718–724`) — **host sync path**  
- `direct_*_scalar`: CUB + **D2H + `cudaStreamSynchronize`** (`63–85`, `112–118`) — intentional for `sum_scalar()` API  

---

## 4. Dot / scalar norms (`tensor_dot_optimized.cu`)

| Op | Small (`n < 100000`) | Large |
|----|----------------------|-------|
| Dot, sum, L1, L2 square | **1 block** 256 threads, float4 | SM-optimal grid + stage1/2 + **mallocAsync partials** |
| Mean | sum + `div_inplace<<<1,1>>>` | same |
| L2 | sum-of-squares + `sqrt_inplace<<<1,1>>>` | same |
| Count nonzero | **1 block only**, float2 for float | no large path |

- **No alignment check** before float4 reinterpret (`81–85`, `147–149`) — UB / fault if unaligned **[risk]**  
- Separate from `sum_scalar()` which uses CUB (`tensor_impl.hpp:2167–2170`)  
- Dot used from matrix path (`tensor_matrix_ops.cpp` via `launch_dot_product`)

**vs CUB:** CUB/cuBLAS `dot`/`nrm2` would avoid custom malloc and give better large-n schedules; small-n 1-block may beat CUB launch overhead for tiny vectors **[SPECULATIVE]**.

---

## 5. Fused pointwise / transform-reduce (`tensor_fused_pointwise.cu`)

### 5.1 Pointwise chain

- Runtime `switch` over op kinds (`81–104`) — prevents compile-time specialization  
- float4 path when 16B-aligned (`113–165`)  
- Scalars stored in `FusedPointwiseOp.scalar` — **by value** in chain struct (passed by value to kernel)  

### 5.2 Fused transform-reduce

- Two-stage, SM-sized grid, float4 grid-stride (`235–337`)  
- **mallocAsync** partials every call  
- Mean: third kernel via `launch_fused_affine_transform` on 1 element (`330–333`)  
- Segmented last-dim: one block/segment-loop, grid `min(num_segments, 2048)` (`410–413`) — good cap  

Wired from lazy reduce when chain consumed (`tensor_unified_ops.cpp:1153–1288`).

---

## 6. Matrix (`tensor_matrix_ops.cu`)

### 6.1 SGEMM

- Register-tiled `sgemm_optimized_kernel` BM=BN=64, BK=8, TM=TN=4 when `m≥16,n≥64,k≥8` (`371–387`)  
- Else 16×16 double-buffered tile  
- TN / batched / bias+relu variants  
- Shared memory + `__ldg`; **FP32 only**  
- Large M: multi-launch row slicing for grid.y limit (`11–12`, `375–387`)  

### 6.2 What is missing

- **No cuBLAS / cublasLt**  
- **No TF32 / FP16 / BF16 / tensor cores / WMMA / mma.sync**  
- No autotune of tile sizes per GPU  

**[SPECULATIVE]** For training-sized GEMM (e.g. ≥512³), cuBLAS is often **5–50×** faster than a simple shared-memory tile kernel; tensor cores more for half/bf16.

Transpose: classic TILE_DIM shared with bank-conflict padding (`14–40`) — solid.

---

## 7. NN (`tensor_nn_ops.cu`)

| Kernel | Pattern | Vectorized |
|--------|---------|------------|
| max_pool2d | grid-stride over outputs | No |
| adaptive_avg_pool2d | grid-stride | No |
| bias_relu / bias_add | grid-stride; channel via `(idx/spatial)%C` | No float4 |
| relu | grid-stride scalar | No |

- Block 256; grid = ceil(n/256) — grid-stride present but grid oversized so usually one pass  
- **Duplicate ReLU:** `launch_relu` vs functor vectorized unary — training should prefer fused/vectorized path  
- Bias+ReLU fusion is good vs two launches  

---

## 8. Masking / index (`tensor_masking_ops.cu`)

| Op | Implementation | Efficiency notes |
|----|----------------|------------------|
| masked_fill/select | Thrust transform / copy_if | Solid; not custom float4 |
| compare eq/lt/gt | Custom kernel; **fast_path** same-shape direct index | **No float4**; generic uses expensive `compute_broadcast_index` |
| logical and/or/xor | Custom + broadcast | Same |
| scalar compare | Thrust | Scalar by value in functor |
| where | Custom ternary broadcast; 2D grid for large n | No vectorization; always full index math |
| count_nonzero | Thrust count_if → **host** → H2D write | **Stream sync / host round-trip** (`577–611`) |
| index_select / gather | Custom; grid-stride or 2D grid | Coalescing depends on dim |
| scatter / index_add | Custom + **atomicAdd** for add mode (`1041`) | Contention if many collisions |
| gather+unary fused | Template specialization | Good for abs/sqrt/neg |

`launch_count_nonzero_*` in `tensor_dot_optimized.cu` (1-block) is a different path from masking Thrust path.

---

## 9. Strided copy / scatter / upload (`tensor_strided_ops.cu`)

- Grid-stride loops throughout  
- Rank 2/3/4 **immediate** params by value (fast path)  
- Generic rank kernel takes `shape`/`strides` as pointers — if host pointers, device dereference is invalid **[SPECULATIVE risk]**; hot path uses rank specializations  
- Grid capped at `MAX_GRID=65535` (`103`, `111–113`) — for huge `n` with only 65535×256 threads, **grid-stride still covers** all elements  
- Upload kernels: HWC→CHW specialized; pinned host gather  

**No float4** on strided paths (expected for irregular strides).

---

## 10. fp16 / bf16 / tensor cores

| Facility | Status |
|----------|--------|
| `cuda_fp16.h` / `__half` ops | Instantiated binary/unary/broadcast/masked; **Thrust or generic scalar** |
| Vectorized float4 path | **float only** (`tensor_generic_ops.cuh` `is_same_v<float>`) |
| `Packed128<__half>` / `bf128` | Defined in `packed128.cuh:165–167` | **Not used** in these kernels |
| Tensor cores / WMMA / mma | **None** in audited files |
| cuBLAS half GEMM | **None** |

**Where half/TC would apply:** matmul, large elementwise (half2), fused MLP (bias+relu), reductions with promoted accumulators.

---

## 11. Redundant work / double dispatch

1. **Full reduce policy:** warp full kernels maintained but **bypassed** for scalars (`should_use_warp_reduce` → CUB).  
2. **Mean:** sum + separate 1-element divide (reduce + column mean + fused transform-reduce).  
3. **Per-call `cudaMallocAsync`/`FreeAsync`** for partials (dot, warp two-stage, fused TR).  
4. **Channel3D smem/coalesced kernels never launched.**  
5. **Binary fusion gap:** `a.mul(b).add(c)` = two kernels (no binary fusion); only unary composition + lazy scalar/unary chains fuse.  
6. **L2/L1 norms:** multi-pass `mul`/`abs` then reduce unless fused path triggers.  
7. **Prod full reduce:** Thrust → host → GPU scalar init.  
8. **`init_scalar_gpu` `<<<1,1>>>`** for max/min init when warp path used (dead for full reduce).  
9. **NN relu** vs vectorized unary.  
10. **Numeric policy** wrappers add indirection but usually one hop to generic.

---

## 12. Per-op kernel launch counts — training expressions

Assumptions: CUDA float32, contiguous, aligned, `n > 1024`, no lazy pointwise chain unless noted.

### A. `a.mul(b).add(c).sum()` (tensor sum → 0-d tensor)

| Step | Launches | Kernel family |
|------|----------|---------------|
| `mul` | **1** | `vectorized_binary_kernel` (or Thrust if unaligned) |
| `add` | **1** | same |
| `sum` full | **CUB** multi-kernel (~1–2 + workspace) | **not** warp; no mean divide |

**Total: ~3–4 kernel launches** (+ memory allocs for temps).  
**Not fused:** intermediate `mul` and `add` buffers.

### B. Lazy ` (x * s1 + s2).sum()` pointwise chain consumed

| Step | Launches |
|------|----------|
| fused transform-reduce stage1+2 | **2** |
| mean only: + affine 1-el | **+1** |

**Total: 2 (sum) or 3 (mean)** — best case for chain+reduce.

### C. `a.mul(b).sum_scalar()` (host float)

| Step | Launches |
|------|----------|
| mul | 1 vectorized |
| `direct_sum_scalar` | CUB + **stream sync + D2H** |

### D. `x.relu().mean({-1})` segmented

| Without lazy | 1 vectorized relu + 1 segmented mean (tiny/medium/large) |
| With lazy fuse last-dim | 1 `fused_segmented_transform_reduce` |

### E. Matmul + bias + relu

| Path | Launches |
|------|----------|
| Separate | GEMM (maybe multi-row launch) + bias_add + relu |
| Fused | **1** `sgemm_bias_relu_kernel` |

### F. Ideal vs current for A

| Library style | Launches for A |
|---------------|----------------|
| Current | 2 elemwise + CUB reduce |
| Fused binary+reduce (not present) | **1–2** |
| PyTorch/cuBLAS-style | often fused or JIT for epilogues |

---

## 13. Family-by-family checklist (requested dimensions)

### Elementwise / clamp / fused chain

1. **Mem:** float4 yes; grid-stride only on some clamp fallbacks; coalesced contig yes  
2. **Launch:** block 256; grid full-cover; **not** SM-scaled  
3. **Reduce:** N/A (chain reduce is §5)  
4. **Same-shape:** yes via generic; scalar by value  
5. **Redundant:** n≤1024 → Thrust; chain runtime switch  
6. **Half/TC:** half → Thrust; no TC  

### Broadcast

1. **Mem:** special patterns float4; generic scalar + heavy indexing  
2. **Launch:** 256; 2D/3D for large dims; not SM-capped  
3. **Reduce:** N/A  
4. **Same-shape** uses elementwise, not broadcast  
5. **Dead Channel3D optimized kernels**; Thrust single-array broadcast  
6. **Half:** generic only  

### Warp / CUB reduce

1. **Mem:** float4 / Packed128 + `load128cs` on two-stage sum  
2. **Launch:** 256; SM optimal for full two-stage & medium segments  
3. **Reduce:** warp shuffle + smem[32]; two-stage for large sum; atomics for max/min/prod single-stage; CUB for production full  
4. **N/A elementwise**  
5. **Mean divide kernel**; malloc partials; transpose for large strided  
6. **fp32 only** in warp kernels  

### Dot optimized

1. float4 assumed aligned  
2. 256; 1-block small; SM grid large  
3. two-stage sum; no atomics on large path  
4. N/A  
5. mallocAsync each large call; extra 1,1 kernels for mean/sqrt  
6. fp32 only  

### Strided

1. scalar grid-stride; rank specialization  
2. 256; grid cap 65535  
3. N/A  
4. N/A  
5. generic vs rank paths  
6. typed via dtype dispatch  

### Masking

1. mostly scalar; fast same-shape compares  
2. 256; 2D for huge  
3. CUB/Thrust for select/count  
4. fast_path bit for same shape (`masking_ops.cu:130–133`)  
5. count_nonzero H2D  
6. half masked fill/select supported; no vector half  

### NN

1. grid-stride scalar  
2. 256  
3. N/A  
4. N/A  
5. unfused relu vs bias_relu  
6. fp32 only  

### Matrix

1. tiled smem; bank-pad transpose  
2. 2D blocks (e.g. 16×16 or 16×16 for TM/TN config)  
3. N/A  
4. N/A  
5. multi-launch for tall M  
6. **no TC / no cuBLAS**  

---

## 14. Top 10 kernel-level speed improvements

Ranked by expected impact on training throughput, with **file:line** anchors.

| # | Improvement | Location | Expected relative gain | vs CUB/Thrust/cuBLAS |
|---|-------------|----------|------------------------|----------------------|
| **1** | **Replace custom SGEMM with cuBLAS/cublasLt** (keep small-GEMM path optional) | `tensor_matrix_ops.cu:42–105`, launch `371–405` | **[SPECULATIVE] 5–50×** on large GEMMs; smaller on tiny | This is what cuBLAS is for |
| **2** | **Fuse binary–binary–reduce** (e.g. `mul`+`add`+`sum` → 1 kernel) | Expr eval `tensor_expr_impl.hpp:560–582`; reduce `tensor_ops.cu:618+`; fusion only unary/lazy today | **~2–3×** bandwidth on that pattern (2 fewer full-tensor writes) | PyTorch/JAX fusion; CUB doesn't fuse epilogues alone |
| **3** | **Wire Channel3D coalesced/smem kernels** instead of only pixel-serial path | Implemented `439–669`; launched `999–1001` only weak kernel | **[SPECULATIVE] 1.5–3×** for large C (comments claim 2–3× for smem) | Matches better coalescing practices |
| **4** | **Persistent/workspace-pooled partial buffers** for two-stage reduce, fused TR, large dot | `tensor_warp_reduce.cu:1078–1081`; `tensor_fused_pointwise.cu:318–336`; `tensor_dot_optimized.cu:129–133` | **[SPECULATIVE] 10–40%** on reduce-heavy steps (launch+alloc latency) | CUB workspace already pooled (`cub_workspace.hpp`) — match that |
| **5** | **Vectorized half2 / Packed128 half** for elementwise | `tensor_generic_ops.cuh:67–76` float-only; `packed128.cuh` unused | **~1.5–2×** bandwidth for half training | Thrust half is scalar |
| **6** | **Tensor-core GEMM / TF32** for matmul (or cuBLAS with TF32) | `tensor_matrix_ops.cu` entire sgemm family | **[SPECULATIVE] 2–8×** over custom FP32 tile on Ampere+ | cuBLAS default |
| **7** | **SM-capped grid-stride for elementwise** (multiple float4s/thread) + lower n threshold | `tensor_vectorized_ops.cuh:33–76`, launch `189–196` | **[SPECULATIVE] 10–30%** large n; better small-n if lower 1024 cutoff | Matches llm.c / modern practice |
| **8** | **Generic broadcast vectorization + same-shape early out** | `broadcast_binary_kernel` `741–799` | **[SPECULATIVE] 2–5×** when Generic is hit with contig storage | Avoids Thrust-level overhead |
| **9** | **Device-side mean/prod finalize** (no 1-el Thrust; no host prod) | Mean `663–669`; prod `718–724`; fused mean `330–333` | **Small absolute**, large relative for tiny post-ops / latency | CUB can fold scale; avoid host |
| **10** | **count_nonzero / compare float4; scale count_nonzero_float beyond 1 block** | Masking `577–611`; compare `119–192`; dot count `355–382` | **[SPECULATIVE] 2–4×** large masks/compares | CUB `DeviceSelect` / `DeviceReduce` |

### Honorable mentions (just outside top 10)

- Medium segmented reduce: `grid_size = max(min_blocks, optimal_grid)` can over-launch (`1254–1256`) — cap to `min_blocks` when segments few.  
- Large segmented: `grid_size = num_segments` (`1295`) — risk/overhead for huge segment counts; use grid-stride over segments.  
- `direct_*_scalar` always syncs (`83–84`) — fine for host API; avoid in device training loops (use tensor `sum()`).  
- Bias/relu float4 + smem bias broadcast (`tensor_nn_ops.cu:117–128`).  
- Align guarantees from allocator so float4 path always hits.

---

## 15. Regression-sensitive notes (for optimization plan)

1. **Do not silently re-enable warp full reduce for scalars** without re-benchmarking — code claims CUB wins 3–7× (`1563–1566`).  
2. **Double accumulators** in strided/segment sum (`790`, `warp_reduce.cuh:351`) affect numerics if removed.  
3. **IEEE max/min/round** special-cased for NaN policy (`tensor_ops.hpp:246–278`) — fusing must preserve.  
4. **Atomic max/min CAS** bit patterns depend on init (`init_scalar` / `-inf`).  
5. **Fused chain kinds** are fixed enums (`tensor_fused_pointwise.cu:15–17` static_asserts) — new ops need both sides.  
6. **cuBLAS integration** must match current C = A@B layout and multi-launch tall-M behavior.  
7. **Grid caps** (65535) and rank specializations in strided/masking are correctness-critical.

---

## 16. Quick map: “what is already good”

- Contiguous fp32 elementwise float4 + scalar-in-register  
- Pattern-specialized broadcast (row/col/batch)  
- Segmented reduce ladder (tiny/medium/large) + fused mean divide  
- CUB for full scalar reduce and many masking primitives  
- Lazy fused pointwise + transform-reduce  
- Column reduce and transpose-before-reduce for bad strided axes  
- Packed128 + streaming loads infrastructure (partially used)

---

*End of audit. All relative gains marked [SPECULATIVE] are planning estimates; validate with NSight Compute / CUDA graphs on target training shapes before committing ABI-visible behavior changes.*
