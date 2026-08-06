I'll audit the lazy executor and its training-loop call sites read-only, starting with the core lazy sources and then tracing every hot-path usage in `src/training/`.Binary tensor-tensor ops are eager by design; next I'll map which unary/scalar chains actually defer in the training step and quantify their graph sizes.Checking IR recording cost on eager binary ops and whether IDs are always assigned.# Lazy Executor Audit: Training Hot Loop

**Scope:** `src/core/tensor/{lazy_executor,lazy_ir,lazy_config}.cpp`, `internal/{lazy_executor,lazy_ir,lazy_config,tensor_expr,tensor_expr_impl,tensor_impl}.hpp`, and training call sites.  
**Method:** static read-only analysis (no runtime benchmarks). Kernel-count savings are **exact from code structure**; host microsecond costs are **reasoned from code paths** and marked as such.  
**Context:** FastGS is the main training rasterizer; it uses raw parameter pointers and fused loss kernels, so most of the step never builds lazy graphs.

---

## Executive verdict

**For the default FastGS training hot path, the lazy executor does not pay for itself overall.** Photometric L1/SSIM, rasterization, Adam, and regularizers are already custom fused kernels. Lazy only touches a thin set of residual Tensor ops. Real fusion wins exist but are rare and often feature-gated; single-unary defer-then-materialize is pure host overhead. Global IR registration on eager binaries adds always-on mutex cost even when nothing is deferred.

**Recommendation:** **(c) improve with concrete fixes**, with **(b) eager bypass** at listed pure-overhead sites. Not **(a) keep as-is**.

---

## Architecture facts (what lazy actually does)

| Fact | Evidence |
|------|----------|
| Defer decision is **byte-size based** at `TensorExpr::operator Tensor()` | `tensor_expr_impl.hpp:34–46` |
| Default threshold **4096 bytes** | `lazy_executor.cpp:111`, `915–934` |
| **Tensor–tensor binary/compare always eager** (no pointwise fusion) | `tensor_impl.hpp:854–887`, `889–911` |
| Unary/scalar can defer; fusable kinds register fusion recipes | `tensor_impl.hpp:1612–1642`, `1822–1857` |
| `pow` scalar is **not** fusable | `tensor_impl.hpp:1885` vs `1881–1884` |
| Fusion consumer: full reduce or **last-dim** segmented reduce | `tensor_unified_ops.cpp:1153–1288` |
| Planner/memory planner run on materialize | `lazy_executor.cpp:737–815`, `462–542` |
| `lazy_ir_active()` is **always true** | `lazy_ir.cpp:189–191` |
| Deferred tensors always get `id_`; factory `empty` always assigns ids | `tensor.cpp:441`, `tensor_unified_ops.cpp:361` |
| `masked_fill` is **eager** (clone + inplace kernel) | `tensor_masking_ops.cpp:325–328` |
| `clamp` is **eager fused kernel** | `tensor_unified_ops.cpp:2479–2517` |

**Defer threshold meaning (Float32):**

| Shape class | Elements for defer | Typical training |
|-------------|--------------------|------------------|
| Scalar `[1]` (4 B) | never | loss bookkeeping — always eager |
| `[N]` opacities | \(N \ge 1024\) | almost always after densify starts |
| `[N,3]` scales | \(N \ge 342\) | almost always |
| `[N,4]` rotations | \(N \ge 256\) | almost always |
| Image `[3,H,W]` | always (e.g. 1080p ≈ 25 MB) | always over threshold |
| Bool/UInt8 mask `[H,W]` | \(HW \ge 4096\) | always for real images |

---

## 1. Training-loop expressions that hit the lazy path

### 1.1 Every-iteration (or near every-iteration) — FastGS default

#### A. Photometric path — **mostly not lazy**

| Site | Ops | Lazy? | Graph size |
|------|-----|-------|------------|
| `compute_photometric_loss_with_gradient` → fused L1/SSIM | custom CUDA | **No** | 0 |
| `PhotometricLoss::forward` | fused kernels | **No** | 0 |
| Scale/opacity reg | fused kernels | **No** | 0 |
| `corrected_image.clamp_(0,1)` | inplace clamp | **No** | 0 |
| FastGS `forward_raw` / Adam | raw `.ptr<float>()` | **No** | 0 |

Evidence: `trainer.cpp:1663–1699`, `photometric_loss.cpp:54–84`, `regularization.cpp:41–49`, `trainer.cpp:4389–4392`, `fast_rasterizer.cpp:299–301`, `376–386`.

**Scalar loss accumulation** (`tile_loss + …`): tensor–tensor `+` is **eager** (`tensor_impl.hpp:880–885`). Scalar `*` on `[1]` tensors is under 4 KB → **eager** (`tensor_expr_impl.hpp:35–36`). **No deferred graph.**

*Note: prior analysis in `docs/analysis/spirulae-comparison/lfs-hotpath.md:279` claiming scalar loss builds “binary lazy nodes” is **incorrect** relative to current code — binaries always `eval()` immediately.*

#### B. Mask / alpha path — **conditional every step when masks enabled**

| # | Call site | Expression | Deferred nodes | Materialize / fuse sink |
|---|-----------|------------|----------------|-------------------------|
| B1 | `trainer.cpp:1768–1769` | `masked_fill` ×2 | **0** (eager) | N/A; each fill clones + kernel |
| B2 | `trainer.cpp:1837–1839` | `masked_fill` ×3 | **0** | same |
| B3 | `trainer.cpp:1843` | `full - mask_f` | **0** (binary eager) | 1 sub kernel |
| B4 | `trainer.cpp:1844` | `bg_mask.pow(power)` | **1** (if image ≥4 KB; always) | not fusable; materializes on next use |
| B5 | `mask_loss.cpp:74` | `(alpha * effective_weight).mean() * scale` | **0** on mul/mean path after eager mul; scalar `*` on scalar | mul + mean kernels |
| B6 | `trainer.cpp:1874–1880` | `(α−mask).abs()` then `.mean()` | **1** (abs) | **fused transform-reduce** if full mean |
| B7 | `trainer.cpp:1875–1882` | `(α−mask).sign()` then `* scalar` | **1** (sign) | materialize on mul (binary) or scalar mul if deferred |

**Graph sizes (B6):**  
- Build: 1 deferred node (`abs`), fusion recipe `{Abs}`  
- On `.mean()`: `try_consume_pointwise_fusion` → **1 fused kernel** (`tensor_unified_ops.cpp:1153–1205`)  
- Eager equivalent: abs kernel + mean kernel = **2**

**Graph sizes (B4+B5):**  
- `pow`: 1 deferred, **no fusion recipe**  
- `alpha * weight`: binary materializes `pow` result, 1 mul  
- `.mean()`: 1 reduce  
- **Net kernels = 3**, same as eager; host cost only

#### C. ADMM sparsity — **every step when sparsity enabled**

| Site | Expression | Graph | Fusion |
|------|------------|-------|--------|
| `sparsity_optimizer.cpp:104` | `opacities.sigmoid()` | 1 deferred (`Sigmoid`) if \(N\ge1024\) | none if used alone |
| `sparsity_optimizer.cpp:107` | `opa - z + u` | **0** (2× binary eager; materializes sigmoid) | — |
| `sparsity_optimizer.cpp:110` | `diff.square().sum()` | 1 deferred (`Square`) → full reduce | **YES** fuse square+sum |
| `sparsity_optimizer.cpp:113` | `sum * scalar` | 0 (4 B) | — |
| `sparsity_optimizer.cpp:118` | `opa_sigmoid_.ptr()` | materializes if still deferred | — |

**Graph size for win case:** 1 deferred node + reduce consume.  
**Kernel win:** square+sum: 2→1. Sigmoid alone: 1→1 with host overhead.

#### D. Densification error map — **when pixel-error densify and λ_dssim==0**

| Site | Expression | Layout | Fusion? |
|------|------------|--------|---------|
| `trainer.cpp:4998–5002` | `(corrected - gt).abs()` then `mean({0})` or `mean({2})` | FastGS image is **CHW** `[3,H,W]` (`fast_rasterizer.cpp:350`) | `mean({0})` is **not** last dim and **not** full reduce → **fusion miss** (`tensor_unified_ops.cpp:1210–1220`) |

**Graph:** 1 deferred abs (~`3*H*W*4` B) → materialize via reduce path `pin_operands` then separate mean.  
**Kernels:** abs + mean = 2, same as eager. Host overhead only.  
HWC `mean({2})` would fuse; FastGS does not produce HWC here.

#### E. Gsplat-only forward (not FastGS default)

| Site | Expression | Graph | Fusion |
|------|------------|-------|--------|
| `splat_data.cpp:625–626` + `gsplat_rasterizer.cpp:93` | `opacity.sigmoid().squeeze` | 1 deferred + deferred view | no reduce fuse; `pin_operands` materializes (`gsplat_rasterizer.cpp:108`) |
| `splat_data.cpp:639–640` | `scaling.exp()` | 1 deferred | materialize on pin |
| `splat_data.cpp:633–636` | `rot.square().sum(1).sqrt()` then `div(clamp_min(...))` | square deferred → **last-dim sum fuses**; sqrt may defer; clamp/div eager | **+1 kernel saved** on square+sum |

### 1.2 Not every iteration — strategy / densify

| Site | When | Lazy behavior |
|------|------|---------------|
| `mrnf.cpp:631` | prune / refine | `(means-center).abs().max(1)` — abs may defer; max last-dim can fuse |
| `mrnf.cpp:1473` | edge guidance | `mul(0.25f).add(1.0f)` — **2-node fusable chain** on `[N]` |
| `mrnf.cpp:762,1131` | grow/prune | `masked_fill` — eager |
| `improved_gs_plus.cpp:372+` | densify | `masked_fill` — eager |
| `strategy_utils.cpp:163,202` | score masking | `masked_fill` — eager |
| `mcmc.cpp:312,488` | densify | `.log()` — single unary defer/materialize |
| `metrics.cpp:142–156` | eval only | `.square()` + mean — similar to sparsity |

---

## 2. Fusion win analysis (per expression)

### Host cost model (from code; not timed)

**Defer a unary (when ≥4 KB):**

1. `operator Tensor`: size check, `snapshot()` → COW registry lock on leaf storage (`tensor_expr_impl.hpp:39`, `tensor.cpp:341–344`, `357–378`)  
2. `make_deferred_expr_tensor`: `shared_ptr<LazyExprState>`, type-erased `std::function`, always `id_++` (`tensor.cpp:420–455`)  
3. `lazy_ir_record_deferred` under **global IR mutex** (`lazy_ir.cpp:410–421`)  
4. `lazy_executor_register_deferred_materializer` under **registry mutex** (`lazy_executor.cpp:547–570`)  
5. Optional fusion registry lock + `create_lazy_snapshot` (`lazy_executor.cpp:577–617`, `573–575`)

**Materialize (no fusion consume):**

1. Per-node mutex gate (`tensor.cpp:471`)  
2. `lazy_planner_build_plan_for_tensor` → topo under IR mutex (`lazy_executor.cpp:709–735`, `lazy_ir.cpp:259–285`)  
3. `execute_topological_nodes` + materializer / fusion launch (`lazy_executor.cpp:462–542`)  
4. Unregister materializer + fusion (`lazy_executor.cpp:620–650`)  
5. Move published storage into tensor; drop lazy state (`tensor.cpp:535–573`)

**Fused reduce consume (best path):** skips full planner for that chain; one registry lock + one fused kernel (`tensor_unified_ops.cpp:1174–1205`, `lazy_executor.cpp:937–963`).

**Speculation:** on CPU-bound steps, single-node defer+materialize is likely tens of microseconds of host work vs ~1–3 µs for an eager launch setup; fusion only wins when it **removes a full-tensor intermediate kernel** on large buffers.

### Per-expression net

| Expression | Eager kernels | Lazy kernels | Saved kernels | Host overhead | Net (training) |
|------------|---------------|--------------|---------------|---------------|----------------|
| **B6** `(α−m).abs().mean()` full | 2 (abs+mean) | 1 fused | **1** | medium (build+consume) | **Win** if AlphaConsistent + large HxW |
| **B7** `(α−m).sign()*s` | 2 | 2 (sign then mul materialize) | 0 | high | **Loss** |
| **B4** `bg.pow` then mul+mean | 3 | 3 | 0 | medium | **Loss** |
| **B1–B2** masked_fill chains | 2–3 + cmp kernels | same | 0 | none from lazy | **N/A** (not lazy; still multi-kernel) |
| **C** `sigmoid` alone | 1 | 1 | 0 | high | **Loss** |
| **C** `square().sum()` | 2 | 1 fused | **1** | medium | **Win** when sparsity on, large N |
| **D** abs+mean({0}) CHW | 2 | 2 | 0 | high (full image intermediate held) | **Loss** |
| **E** opacity sigmoid (gsplat) | 1 | 1 | 0 | high every forward | **Loss** |
| **E** rotation square+sum | 2 | 1 fused | **1** | medium | **Win** on gsplat path |
| **E** scaling exp | 1 | 1 | 0 | high | **Loss** |
| **MRNF** `mul().add()` edge | 2 | 1 fused (if both deferred before sink) | **1** | medium | **Win** only at densify edge sampling, not every step |
| Scalar loss `+`/`*` | 1 each | 1 each (eager) | 0 | IR record if id≠0 | **small always-on cost** |

### Always-on IR tax (even without defer)

Eager binaries still call `lazy_ir_record_binary` after eval (`tensor_impl.hpp:885–886`). That path:

- requires `debug_id() != 0` (true for `empty` results: `tensor_unified_ops.cpp:361`)  
- takes **global** `LazyIrRuntime::mutex` (`lazy_ir.cpp:347–351`)  
- inserts nodes + leafs; destructor unregisters (`tensor.cpp:932–934`)

**Every** temporary binary in training (loss `+`, mask sub, etc.) pays this.  
**Speculation:** this is a measurable host tax independent of fusion payoff.

---

## 3. 4 KB defer threshold: under vs over in the hot loop

| Hot-loop tensor | Size | Path |
|-----------------|------|------|
| Loss scalars `[1]` | 4 B | **eager** (`tensor_expr_impl.hpp:35–36`) |
| Opacity `[N,1]` or `[N]`, N≪1024 | &lt;4 KB | **eager** early training |
| Opacity N≥1024 | ≥4 KB | **defer** |
| Scaling `[N,3]` N≥342 | ≥4 KB | **defer** |
| Image / alpha / masks (any realistic res) | ≫4 KB | **defer** for unary/scalar |
| Tile error map mean result | often small | may go eager |

**Is 4 KB well-placed?**

- **Intent is sound:** skip IR/registry for tiny tensors (`lazy_executor.cpp:132–136` comment).  
- **Problem:** threshold only decides *whether to build a deferred node*, not *whether fusion will fire*.  
- Single unaries that materialize immediately (sigmoid → `ptr`, exp → pin) still defer at ≥4 KB → **worst of both worlds**.  
- **Speculation:** for “unary then immediate sink without multi-op chain”, a higher threshold (e.g. 64 KB–1 MB) or “defer only if fusable parent exists / consumer is reduce” would cut overhead without losing image-scale abs+mean wins.

---

## 4. Snapshot / COW machinery

### Mechanisms

| Mechanism | Role | Evidence |
|-----------|------|----------|
| `TensorLeaf::snapshot_impl` | registers leaf cell on own storage | `tensor.cpp:341–344` |
| `create_lazy_snapshot` | shared Tensor copy + register on source storage | `tensor.cpp:351–355` |
| Fusion `source` holds `shared_ptr<Tensor>` snapshot | keeps operand alive for fused launch | `lazy_executor.cpp:604–606` |
| `preserve_lazy_snapshots_before_write` | on write: **full `clone()`** of each live snapshot | `tensor.cpp:386–409` |
| Triggered from `ptr()` non-const, `data_ptr()`, `zero_`, `fill_`, etc. | `tensor_impl.hpp:1307–1309`, `1339–1341`; `tensor.cpp:1996`, `2026` |

### How often per training step (FastGS default)

| Situation | Snapshots created | COW clone risk |
|-----------|-------------------|----------------|
| Pure FastGS, no mask, no sparsity, no gsplat getters | ~0 deferred unaries on params | **low** |
| Sparsity on | sigmoid + square fusion snapshots of `diff` / opacities | if Adam/`zero_` mutates shared storage **before** materialize/unregister, **full param clone** |
| Mask AlphaConsistent | abs/sign snapshots of α−mask temps (temps own storage) | low for model params |
| Gsplat `get_opacity/scaling/rotation` | snapshots of `_opacity/_scaling/_rotation` every forward until pin materializes | **window of COW risk** if concurrent write; sequential train: materialize before Adam |

**Worst-case cost:** one full tensor clone per pending snapshot (`tensor.cpp:407`) — e.g. clone of `_opacity` (`N×4` B) or means (`N×12` B) under mutation while lazy lives.

**Registry locks per deferred op:** IR mutex + materializer mutex + fusion mutex (3); unregister on destroy does both materializer + fusion locks (`lazy_executor.cpp:620–650`).

---

## 5. Memory: peaks and lifetime

### With memory planner (default ON)

- On multi-node materialize: `compute_release_schedule` drops cache entries after last consumer (`lazy_executor.cpp:256–295`, `485–498`).  
- Peak tracked in diagnostics `peak_cache_bytes` (`lazy_executor.cpp:251–254`).  
- Context is **stack-scoped per root materialization** (`lazy_executor.cpp:794–798`) — no cross-step cache.

### Without planner

- Cached materializations held until context ends — higher peak for multi-node graphs.  
- Training rarely builds multi-node deferred DAGs (binaries are eager), so planner **rarely activates meaningfully** on the hot path.

### Cases lazy holds data longer than eager

1. **Fusion `source` snapshot** keeps a Tensor view of the operand until recipe consumed/unregistered — extends refcount on storage (`lazy_executor.cpp:42–47`, `604–606`).  
2. **Deferred abs** on full image before mean: deferred object has null `data_` but materializer/`snapshot` keep inputs alive; with fusion miss on CHW `mean({0})`, intermediate abs buffer is still allocated on materialize — same as eager, plus host graph.  
3. **Assigned deferred** (`opa_sigmoid_ = opacities.sigmoid()`) delays allocation until `.ptr()` — **can reduce peak** if later unused (not the case here; ptr is always taken).  
4. **Nested deferred views** (`squeeze`/`reshape` on lazy): new deferred wrapping source can re-materialize (`tensor_impl.hpp:971–978`, `tensor_movement_ops.cpp:224–227`) — peak amplifier if fusion misses. **Speculation:** rare on FastGS hot path.

### Peak intermediates: abs+mean fusion win

| Mode | Intermediates |
|------|----------------|
| Eager abs then mean | full abs buffer + reduce scratch |
| Fused transform-reduce | **no** full abs buffer; only reduce output (`launch_fused_transform_reduce`) |

This is the main **memory** win of lazy, and it only hits B6 / sparsity square+sum / rotation square+sum / last-dim patterns.

---

## 6. Recommendations (ranked, with evidence)

### Verdict

| Option | Fit |
|--------|-----|
| (a) Keep as-is | **No** — residual sites lose or break even; always-on IR tax |
| (b) Bypass eager at listed sites | **Yes, targeted** |
| (c) Improve subsystem | **Yes, high leverage** |

### Ranked concrete changes

| Rank | Change | Why | Touch points |
|------|--------|-----|--------------|
| **1** | **Stop recording IR for eager binary/unary results** (or gate `lazy_ir_active` behind “has live deferred consumers” / debug flag) | Always-on global mutex on every binary in training; no fusion benefit for binaries | `tensor_impl.hpp:885–886`, `910`; `tensor_expr_impl.hpp:170–172`, `1011–1013`; `lazy_ir.cpp:189–191` |
| **2** | **Defer only fusable multi-op chains or producer→reduce**, not every large unary | Kills sigmoid/exp/sign/pow single-op overhead | `tensor_expr_impl.hpp:34–46`; optionally only register fusion in `LFS_DEFINE_UNARY_OP_FUSABLE` and defer if parent already deferred or caller is reduce |
| **3** | **Bypass lazy for known training sinks** | Zero risk to correctness of custom kernels | `sparsity_optimizer.cpp:104` — compute sigmoid into preallocated buffer or fused kernel (backward already fused at 146–155); `splat_data.cpp:625–640` — dedicated activation kernels or non-deferred path for rasterizer; `trainer.cpp:1874–1882` keep fusion for abs+mean but make sign eager or fused with scale |
| **4** | **Fuse densify L1 error for CHW** | FastGS is CHW; current `mean({0})` misses segmented fusion | `trainer.cpp:4998–5002` → custom `absdiff_mean_c` kernel (mirror `launch_ssim_to_error_map` at `ssim.cuh:380–385`) **or** extend fusion to first-dim reduce |
| **5** | **Mask preprocess single kernel** | Not lazy, but multi-kernel hot when masks on | `trainer.cpp:1768–1769`, `1837–1839` — banded mask kernel (already noted in `lfs-hotpath.md:322`) |
| **6** | **Raise default threshold or add “min fuse length ≥2”** | 4 KB is low for modern GPU launch cost | `lazy_executor.cpp:111`; measure with existing diagnostics `fused_launches` / `root_fallbacks` (`lazy_executor.hpp:64–77`) |
| **7** | **Make `pow` fusable or provide `pow_scalar` in fused chain** | Mask opacity path | `tensor_impl.hpp:1885`; `LazyPointwiseOpKind` (`lazy_executor.hpp:22–43`) lacks Pow |
| **8** | **COW: skip snapshot when operand is a temporary exclusive owner** | Reduce `preserve_lazy_snapshots_before_write` blast radius | `tensor.cpp:351–378`, `386–409` |
| **9** | Keep memory planner ON | Low cost when graphs tiny; correct for multi-node tests | default `lazy_executor.cpp:161–168` |

### What not to do

- Do **not** disable lazy globally without replacing abs+mean / square+sum fusion — those are real kernel savings when their features are on.  
- Do **not** convert binary ops to deferred without a binary fusion backend — code already documents this as intentional (`tensor_impl.hpp:856–858`).

---

## 7. Bottom line: does lazy pay for itself in the training hot loop?

| Path | Pays for itself? |
|------|------------------|
| FastGS + no mask + no sparsity + SSIM densify error | **No** — lazy almost never runs; IR tax on incidental binaries only |
| + sparsity ADMM | **Partial** — square+sum wins 1 kernel; sigmoid loses |
| + AlphaConsistent mask | **Partial** — abs+mean wins; sign/pow/masked_fill do not |
| + λ_dssim=0 densify L1 error | **No** — fusion miss on CHW |
| Gsplat rasterizer path | **Mixed** — rotation fuse wins; opacity/scale single unary loses every forward |

**Overall:** the lazy executor is a **correctness/optimization scaffold** that pays off mainly for **unary→reduce** and **unary chains** on large tensors. The training hot loop was largely rewritten to **bypass** it (fused L1/SSIM, FastGS raw activations, fused regs/ADMM bwd). What remains is often **host tax without kernel savings**. Optimize by **narrowing defer to fusion-productive patterns** and **removing always-on IR registration for eager ops**, not by relying on the current default defer-everything-≥4 KB behavior.

---

### Speculative items (not proven by timers)

- Absolute host microseconds per defer/materialize.  
- Whether IR mutex shows up in Nsight Systems on multi-threaded GUI+train.  
- Whether raising threshold to 64 KB would regress any non-training consumer.

*Read-only audit; no files modified.*
