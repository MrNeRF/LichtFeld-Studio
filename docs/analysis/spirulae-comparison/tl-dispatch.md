I'll audit host-side per-op dispatch overhead in the tensor library by reading the listed core files and tracing how training hot paths call into them.Reading core dispatch headers and the simple-op path next.Checking one more detail on `track_allocation` and in-place ops used by training:# Host-Side Per-Op Dispatch Overhead Audit — LichtFeld Tensor Library

**Scope:** CPU work between `t.add(x)` / similar API entry and CUDA kernel launch.  
**Sources:** `src/core/tensor/` (esp. listed files) + training call sites in `src/training/`.  
**Method:** Static analysis only. Microsecond numbers are **estimates** (marked as such), not measured profiles.

---

## 0. Executive summary

| Path | When used | Host work before kernel | Order-of-magnitude vs ~5–10 µs launch |
|------|-----------|---------------------------|--------------------------------------|
| **Eager tensor–tensor** (`add`/`mul`/…) | Always eager; no defer | Validation, **2× TensorLeaf heap cells**, **≥3 Tensor shallow copies**, shape vectors, `empty`+pool, stream prep, **global lazy-IR mutex + string**, launch | **Often ≥ launch cost** on small/medium tensors (**est. ~5–40 µs**) |
| **Unary / scalar** via expr → `Tensor` | Defers if `bytes ≥ 4096` | Deferred: IR + materializer registry + `std::function` heap; pay full cost on materialize | Build: **est. ~2–15 µs**; materialize: **est. tens–hundreds µs** for multi-node graphs |
| **In-place** (`add_`/…) | Training densify/opt | Lighter (no BinaryExpr), still validation + possible materialize + launch | **est. ~1–5 µs** + launch |
| **Fused custom training kernels** | SSIM/L1 etc. | Bypass generic binary path | Outside this audit’s “simple op” cost |

**Structural hotspot:** Tensor is not a cheap handle. Copy always allocates a **new `TensorState`**. Expression leaves **heap-allocate full `Tensor` cells**. Lazy IR is **always on** and **locks a process-wide mutex** on record/lookup, including shape **string** construction.

---

## 1. Cost anatomy of one simple op: `t.add(x)` (tensor–tensor)

### 1.1 Call chain (eager)

```
Tensor::operator+ / add
  → binary_op_with_promotion          [tensor_impl.hpp:859-887]
    → validate_binary_op              [835-852]
    → promote_dtypes / optional .to()
    → broadcast_shape                 [tensor.cpp:2352-2355]
    → BinaryExpr(TensorLeaf(lhs), TensorLeaf(rhs), ...)
    → expr.eval()                     // NOT operator Tensor(); never defers
    → BinaryExprEvaluator::eval       [tensor_expr_impl.hpp:327+]
    → lazy_ir_record_binary           [tensor_impl.hpp:886; lazy_ir.cpp:337-351]
```

`operator+` is a thin template forwarding to `add` ([tensor_impl.hpp:2504-2505](src/core/tensor/internal/tensor_impl.hpp), [1724-1726](src/core/tensor/internal/tensor_impl.hpp)).

**Comment in source** states binary ops are intentionally always eager (no fusion benefit for binary) ([854-858](src/core/tensor/internal/tensor_impl.hpp)).

### 1.2 Shape vector copies / allocations

| Site | What | File:line |
|------|------|-----------|
| `TensorShape` storage | `std::vector<size_t> dims_` (heap for rank ≥ 1) | [tensor_impl.hpp:208-209](src/core/tensor/internal/tensor_impl.hpp) |
| `Tensor::strides_` | Separate `std::vector<size_t>` | [445](src/core/tensor/internal/tensor_impl.hpp) |
| `shape.strides()` | **Allocates a new vector** every call | [236-246](src/core/tensor/internal/tensor_impl.hpp) |
| `empty` / `LoadOp::Empty` | `result.strides_ = args.shape.strides()` → 1 vector alloc | [tensor_unified_ops.cpp:355-357](src/core/tensor/tensor_unified_ops.cpp) |
| `broadcast::shape` | Always builds `std::vector<size_t> result(max_rank)` | [tensor_broadcast.hpp:31-50](src/core/tensor/internal/tensor_broadcast.hpp) |
| `broadcast_shape` | Wraps that vector in a new `TensorShape` | [tensor.cpp:2352-2355](src/core/tensor/tensor.cpp) |
| BinaryExpr members | Stores **copy** of `TensorShape` | [tensor_expr.hpp:200-211](src/core/tensor/internal/tensor_expr.hpp) |
| Tensor shallow copy | Copies `shape_` and `strides_` vectors | [tensor.cpp:636-642](src/core/tensor/tensor.cpp) |

**Per successful same-shape float add (typical):** roughly **4–8 host heap buffers** for shapes/strides alone (broadcast result, expr shape, empty result strides, leaf/eval tensor copies). Not `int64_t` specifically—dims are `size_t`.

### 1.3 String construction (labels / profiling / IR)

| Path | Unconditional? | File:line |
|------|----------------|-----------|
| Lazy IR node stores `std::string op_name` and `std::string shape` from `tensor.shape().str()` | **Yes, every IR record** | [lazy_ir.cpp:142-152](src/core/tensor/lazy_ir.cpp), [str:2555-2568](src/core/tensor/tensor.cpp) |
| Binary record name `"binary"` | Yes | [tensor_impl.hpp:886](src/core/tensor/internal/tensor_impl.hpp) |
| Unary eval records `typeid(UnaryOp).name()` (RTTI, often mangled) | Yes if IR active | [tensor_expr_impl.hpp:170-172](src/core/tensor/internal/tensor_expr_impl.hpp) |
| `Tensor::profiling_enabled_` logging | Only if enabled (default **false**) | [463](src/core/tensor/internal/tensor_impl.hpp), [654-658](src/core/tensor/tensor.cpp) |
| `OpTraceGuard` / `TensorOpTracer` | Only if tracer enabled or tensor tracked (default off) | [tensor_trace.hpp:34-39,167-174](src/core/include/core/tensor_trace.hpp) |
| `LFS_ASSERT_MSG` message args | Only on failure | [assert.hpp:20-26](src/core/include/core/assert.hpp) |
| Allocation profiler labels | Compile-time gated `LFS_ALLOCATION_PROFILING_ENABLED` (default **0**) | [allocation_profiler.hpp:25-27](src/core/tensor/internal/allocation_profiler.hpp), [memory_pool.hpp:360-364](src/core/tensor/internal/memory_pool.hpp) |
| `dtype_name` in `record_tensor` | Only when profiling enabled | [tensor_unified_ops.cpp:414-418](src/core/tensor/tensor_unified_ops.cpp) |

**Release default:** profiling/trace strings are mostly off; **IR shape/`op_name` strings still allocate**.

### 1.4 Mutex acquisitions

| Lock | When | File:line |
|------|------|-----------|
| `LazyIrRuntime::mutex` | Every `lazy_ir_record_*`, `tensor_lazy_expr_id`, unregister, set inputs | [lazy_ir.cpp:32-33,330-333,347-351](src/core/tensor/lazy_ir.cpp) |
| `DeferredMaterializerRegistry::mutex` | Register/unregister deferred materializers | [lazy_executor.cpp:38,553-557,620-628](src/core/tensor/lazy_executor.cpp) |
| `PointwiseFusionRegistry::mutex` | Scalar/unary fusion recipe register/consume | [42-51,589-591,946-949](src/core/tensor/lazy_executor.cpp) |
| `LazyExprState::gate` | Materialize deferred | [tensor_impl.hpp:2664](src/core/tensor/internal/tensor_impl.hpp), [tensor.cpp:471](src/core/tensor/tensor.cpp) |
| `StorageMeta::lazy_snapshot_mutex` | Snapshot register / COW before write | [380-381](src/core/tensor/internal/tensor_impl.hpp), [tensor.cpp:367,395](src/core/tensor/tensor.cpp) |
| `CudaMemoryPool::map_mutex_` | **Every** CUDA alloc track | [memory_pool.hpp:565-567](src/core/tensor/internal/memory_pool.hpp) |
| `stream_routing_mutex_` (shared) | Pool allocate path | [133](src/core/tensor/internal/memory_pool.hpp) |
| `CudaEventPool::mutex_` | Cross-stream wait / bridge | [cuda_event_pool.hpp:41-42](src/core/tensor/internal/cuda_event_pool.hpp), [cuda_stream_context.cpp:29-52](src/core/tensor/cuda_stream_context.cpp) |

**Eager same-stream binary (common):** at least **1 global IR mutex** (record) + **1 pool map mutex** (result alloc). Cross-stream adds event-pool lock + CUDA event APIs.

Stream TLS itself is **not** locked: `thread_local cudaStream_t` ([cuda_stream_context.cpp:13-20](src/core/tensor/cuda_stream_context.cpp)).

### 1.5 `shared_ptr` / heap churn (critical)

| Event | Allocations | File:line |
|-------|-------------|-----------|
| Default `Tensor` / member init | `state_ = make_shared<TensorState>()` **always** | [443](src/core/tensor/internal/tensor_impl.hpp) |
| **Copy ctor / copy assign** | **New** `make_shared<TensorState>(*other.state_)` (not shared impl) | [636-639](src/core/tensor/tensor.cpp), [703](src/core/tensor/tensor.cpp) |
| `TensorLeaf(Tensor)` | By-value Tensor (copy) + `make_shared<Tensor>(move)` | [324-325](src/core/tensor/tensor.cpp) |
| `TensorLeaf::eval_impl` success path | `return *tensor_ptr_` → **another full Tensor copy** | [333-338](src/core/tensor/tensor.cpp) |
| Binary eval | 2 leaf evals → **2 more copies** | [tensor_expr_impl.hpp:332-333](src/core/tensor/internal/tensor_expr_impl.hpp) |
| `empty` → `adopt_storage` | `make_shared<StorageOwner>` + aliasing `shared_ptr`s | [503-521](src/core/tensor/internal/tensor_impl.hpp), [407-410](src/core/tensor/tensor_unified_ops.cpp) |
| Deferred path | `make_shared<LazyExprState>()` + `std::function` type erasure | [439-440](src/core/tensor/tensor.cpp) |

**Binary add leaf stack (same dtype, contiguous):**

1. `TensorLeaf(lhs)`: Tensor copy + heap Tensor  
2. `TensorLeaf(rhs)`: same  
3. `left.eval()` / `right.eval()`: 2× Tensor copy  
4. `Tensor::empty`: default Tensor (state) + storage owner  
5. Return result (NRVO/move hopefully)

**Conservative count:** **≥4–6 `make_shared`-class heap objects** + several vector buffers **before** the kernel runs.

### 1.6 Stream lookups / prep

```
BinaryExprEvaluator (CUDA):
  prepare_inputs_for_stream({&left, &right})   [tensor_expr_impl.hpp:335-337]
  CUDAStreamGuard(execution_stream)
  Tensor::empty → state_->stream = getCurrentCUDAStream()  [unified_ops:362]
  pin_operands → materialize_if_deferred only  [impl:2646-2651]
  launch_*(..., result.stream())
```

`prepare_inputs_for_stream` ([cuda_stream_context.cpp:67-91](src/core/tensor/cuda_stream_context.cpp)):

1. TLS current stream, else first CUDA input’s home stream  
2. For each CUDA input: `sync_to_stream` → if home ≠ exec: `bridgeStreams` + `record_stream` ([879-891](src/core/tensor/tensor.cpp))  
3. Same-stream: early return after compare — **cheap**

Home stream is an **atomic load** in `TensorState::StreamHandle` ([414-430](src/core/tensor/internal/tensor_impl.hpp)).

### 1.7 Dtype dispatch mechanism

| Mechanism | Where | Kind |
|-----------|--------|------|
| `detail::dispatch_dtype` | [tensor_dtype_dispatch.hpp:20-45](src/core/tensor/internal/tensor_dtype_dispatch.hpp) | **`switch` on `DataType`** → template lambda; **not** a vtable, **not** a function table |
| Binary add path | Compile-time functor `ops::add_op{}` + runtime if-ladder on dtype in `BinaryExprEvaluator` | Switch-like branches on `Float16`/`Int64`/… ([348+](src/core/tensor/internal/tensor_expr_impl.hpp)) |
| `binary_op_with_promotion` | `promote_dtypes` then optional `.to()` | Runtime dtype promote |
| In-place / older generic path | Template `binary_op_generic<SrcT,OutT,Op>` | Compile-time type, still runtime shape/device checks |

**No virtual dispatch** on the simple arithmetic path. Cost is mostly **branching + template instantiation size**, not vcalls.

### 1.8 Result allocation path (`empty`)

```
Tensor::empty → load(LoadOp::Empty)     [factory:94-101]
  id_ = next_id_++                      // atomic always [unified:361]
  telemetry_record_materialization      // atomic always [369]
  allocate_cuda_storage → pool          // shared_lock + map lock [memory_pool]
  adopt_storage
```

Pool hit still pays **map mutex** via `track_allocation` ([565-567](src/core/tensor/internal/memory_pool.hpp)).

---

## 2. Lazy path: IR node build, materialize walk, training payment

### 2.1 Lazy IR is always active

```189:191:src/core/tensor/lazy_ir.cpp
    bool lazy_ir_active() {
        return true;
    }
```

There is **no production kill-switch** in this function.

### 2.2 Cost of building one IR node

`register_node_locked` ([120-158](src/core/tensor/lazy_ir.cpp)) under `runtime.mutex`:

1. Unregister existing mapping for tensor id  
2. Node limit check (default **65 536**, [46](src/core/tensor/lazy_ir.cpp))  
3. `normalize_inputs` (vector + linear scan)  
4. Bump dependents on inputs  
5. `unordered_map::emplace` of `LazyExprNode`:  
   - `std::string(op_name)`  
   - `std::vector` inputs  
   - **`tensor.shape().str()`** string  
6. `tensor_to_node[tensor_id] = node_id`  
7. Telemetry atomics  

**Per op record (est.):** 1 mutex, 1–2 hash ops, **2+ string/vector heaps**, refcount bookkeeping. **No weak_ptr in IR nodes**; weak_ptr is on executor registries.

### 2.3 When IR is recorded

| Op class | Record site | Deferred? |
|----------|-------------|-----------|
| Tensor–tensor binary | After eager eval | No |
| Unary/scalar via `operator Tensor` | If size ≥ threshold → deferred node; on eval → unary record with RTTI name | Yes if `bytes ≥ 4096` ([expr_impl:34-45](src/core/tensor/internal/tensor_expr_impl.hpp), threshold [111](src/core/tensor/lazy_executor.cpp)) |
| Reduce | After kernel | No (unless fusion short-circuit) |
| Views from deferred parent | Deferred view materializer | Yes |

`make_deferred_expr_tensor` ([420-455](src/core/tensor/tensor.cpp)):

- New `LazyExprState` + `std::function` materializer (typically **2 heaps**)  
- Always assigns `id_ = next_id_++`  
- `lazy_ir_record_deferred` (mutex)  
- `lazy_executor_register_deferred_materializer` (**another mutex**, copies `std::function`)

Fusion ops for fusable scalar/unary add **another registry lock** + possible `create_lazy_snapshot` (Tensor copy + snapshot list lock) ([577-617](src/core/tensor/lazy_executor.cpp), [351-354](src/core/tensor/tensor.cpp)).

### 2.4 Materialization walk cost

`materialize_deferred_slow` ([458-581](src/core/tensor/tensor.cpp)):

1. Gate mutex + reentrancy check  
2. Optional cache lookup  
3. Else `lazy_planner_execute_plan_for_tensor` ([737-815](src/core/tensor/lazy_executor.cpp)):  
   - `lazy_planner_build_plan_for_tensor` → `lazy_ir_collect_topological_subgraph` (**IR mutex**, full DFS, debug info **string copies** per node [259-285](src/core/tensor/lazy_ir.cpp))  
   - Fusion recipe collection (**fusion mutex**)  
   - Topo execute: per node registry lookup (**mutex**), materializer call or fused launch  
   - Root fallback often runs materializer **again** (`record_root_fallback`)  
4. Steal storage into `this`, clear `state_->lazy`  
5. Unregister materializer (**two mutexes**: deferred + fusion, [620-650](src/core/tensor/lazy_executor.cpp))

**Speculation:** multi-node chains can spend **far more host time planning/locking than launching**, especially if fusion fails and each node allocates.

### 2.5 When hot training pays this even if “eventual” kernel is raw

Training **does** use tensor arithmetic (not only custom kernels):

| Pattern | Example | Path paid |
|---------|---------|-----------|
| Loss accumulate | `loss_tensor_gpu = loss_tensor_gpu + tile_loss` ([trainer.cpp ~4333, 5068](src/training/trainer.cpp)) | Eager binary: full §1 cost each add |
| Abs / mean / scalar | `(alpha_2d - mask_f).abs()`, `alpha_error.mean() * w` ([1874-1880](src/training/trainer.cpp)) | Binary + unary (defer if large) + reduce + scalar |
| Error maps | `(corrected - gt).abs().mean(...)` ([4998-5002](src/training/trainer.cpp)) | Same |
| Contiguous / cast / permute | many sites | Movement + often `empty`+copy kernel |
| Metrics | `mask.sum()`, MSE, SSIM mean ([metrics.cpp](src/training/metrics/metrics.cpp)) | Reduce / binary |
| Densify / MCMC | `.contiguous()`, masks, index ops ([mcmc.cpp](src/training/strategies/mcmc.cpp), [mrnf.cpp](src/training/strategies/mrnf.cpp)) | High volume of tensor API |

**Even after materialization**, tensors remain in the IR map until destroy → `has_lazy_expr()` can still take the **IR mutex** ([1382-1384](src/core/tensor/internal/tensor_impl.hpp), [219-237](src/core/tensor/lazy_ir.cpp)). Reduce **probes fusion** when `has_lazy_expr()` ([1153-1156](src/core/tensor/tensor_unified_ops.cpp)) — so prior IR registration adds **extra lock(s)** on later reduces.

**Custom fused SSIM/L1 kernels** avoid the generic binary path for the heavy image loss core; host overhead is still paid on **surrounding** glue ops.

---

## 3. Validation overhead in release builds

### 3.1 What always runs

| Check | Release behavior | File:line |
|-------|------------------|-----------|
| `LFS_ASSERT` / `LFS_ASSERT_MSG` | **Always on** (throws via `assertion_failed`); only message work is failure-gated | [assert.hpp:10-26](src/core/include/core/assert.hpp) |
| `LFS_DEBUG_ASSERT*` | **Off** under `NDEBUG` | [28-48](src/core/include/core/assert.hpp) |
| `validate_binary_op` / `validate_unary_op` | Always: validity, device, broadcastability | [835-852,914-917](src/core/tensor/internal/tensor_impl.hpp) |
| `is_contiguous()` | **O(1) flag**, not a memory scan | [447,1493](src/core/tensor/internal/tensor_impl.hpp) |
| Contiguity **recompute** | Only on strided view creation (stride walk) | [1028-1037](src/core/tensor/internal/tensor_impl.hpp) |
| `contiguous_read` | Branch on flag; materialize only if false | [567-574](src/core/tensor/internal/tensor_impl.hpp) |
| `ptr<T>()` | `materialize_if_deferred` (branch), `require_valid`, dtype match `LFS_ASSERT`, stale view check | [1312-1336](src/core/tensor/internal/tensor_impl.hpp) |
| `tensor_contract::require_*` | Runtime validation helpers (used by `ptr`, some ops) | [270-314](src/core/tensor/internal/tensor_impl.hpp) |
| `has_nan` / `has_inf` | **Not** on every op; only explicit / `assert_finite` | [2600-2601](src/core/tensor/internal/tensor_impl.hpp), [3177+](src/core/tensor/tensor.cpp) |
| Op tracer | Default **disabled** | [tensor_trace.hpp:159](src/core/include/core/tensor_trace.hpp) |
| Tensor debug logging | `profiling_enabled_` default **false** | [463](src/core/tensor/internal/tensor_impl.hpp) |
| Lazy IR + telemetry atomics | **Always** | [lazy_ir_active], [lazy_config.cpp](src/core/tensor/lazy_config.cpp) |
| Empty-tensor overflow asserts | Always condition eval | [unified_ops:364-367](src/core/tensor/tensor_unified_ops.cpp) |

### 3.2 Bounds checks

- Accessor `operator()` uses `LFS_ASSERT_MSG` per index ([1093-1094](src/core/tensor/internal/tensor_impl.hpp)) — not on bulk kernels.  
- Kernel launches rely on host contracts; device-side debug asserts strip under `NDEBUG`.  
- **No** full-buffer NaN scan in the add/mul dispatch path.

### 3.3 Logging / profiler hooks (unconditional vs gated)

| Hook | Default release |
|------|-----------------|
| `telemetry_record_materialization` on every `empty` | **On** (atomics) [unified:369](src/core/tensor/tensor_unified_ops.cpp) |
| `next_id_++` on empty/copy/deferred | **On** (atomic) |
| VramProfiler in `track_allocation` | **On** (try/catch around diagnostics) [565-573](src/core/tensor/internal/memory_pool.hpp) |
| AllocationProfiler | Compile-time **off** by default |
| LOG_DEBUG on copy | Only if `profiling_enabled_` |

---

## 4. TensorState size / copy cost; training copy frequency

### 4.1 Layout (logical)

**`Tensor` (by value)** — [441-463](src/core/tensor/internal/tensor_impl.hpp):

| Field | Approx. (x86-64) |
|-------|------------------|
| `void* data_` | 8 |
| `shared_ptr<void> data_owner_` | 16 |
| `shared_ptr<TensorState> state_` | 16 |
| `TensorShape` (`vector` + `size_t`) | ~32 |
| `vector strides_` | 24 |
| `storage_offset_`, flags, device, dtype | ~16 |
| `shared_ptr<StorageMeta>` | 16 |
| `view_generation_snapshot_`, `id_`, `lazy_ir_registered_` | ~24 |
| **Total (est.)** | **~140–160 B** object + **heap** for vectors when rank ≥ 1 |

**`TensorState`** — [401-438](src/core/tensor/internal/tensor_impl.hpp):

| Field | Notes |
|-------|--------|
| `capacity`, `logical_size` | 16 B |
| alignment bools | packed |
| `StreamHandle` (`atomic<cudaStream_t>`) | 8 B + atomic ops on copy |
| `tracked` + `std::string name` | string may heap |
| `shared_ptr<LazyExprState> lazy` | 16 B |
| **Heap size (est.)** | **~80–120 B** + name |

**`StorageMeta`** (on real storage): atomics, **mutex**, weak_ptr vector, strings ([377-384](src/core/tensor/internal/tensor_impl.hpp)).

### 4.2 Copy cost (not LibTorch-cheap)

Despite the comment “SHALLOW COPY (LibTorch behavior)” ([635](src/core/tensor/tensor.cpp)):

- Data pointer / `data_owner_` / `storage_meta_` are shared.  
- **`TensorState` is deep-copied into a new `shared_ptr` every copy/assign** ([639](src/core/tensor/tensor.cpp), [703](src/core/tensor/tensor.cpp)).  
- Shape/stride vectors copied.  
- New `id_` via `next_id_++`.  
- `lazy_ir_registered_ = false` on copy (IR mapping not transferred).

**Move** is cheap for state (`std::move`), but moved-from rebuilds empty `TensorState` if null ([734-738](src/core/tensor/tensor.cpp)).

### 4.3 How often training copies Tensors by value

**High frequency (hot-ish glue):**

- Every `TensorLeaf` construction from operands  
- Every leaf `eval()` return  
- Loss expressions returning temporaries (`loss + x`, `.abs()`, `.mean()`)  
- `contiguous()` / `to()` / `clone()` produce new tensors  
- Returning tensors from helpers  

**Lower frequency (still expensive when they fire):**

- Optimizer param clones ([adam_optimizer.cpp:970](src/training/optimizer/adam_optimizer.cpp))  
- Densification realloc / cat / index_select chains  

**Speculation:** per training iteration, **tens to hundreds** of Tensor value copies on the glue path are plausible; most densify steps add more. Custom fused kernels reduce **elementwise kernel count** but not **handle copy** cost around them.

In-place APIs exist (`add_`, `mul_`, … [2315+](src/core/tensor/internal/tensor_impl.hpp)) and densify code uses them in places; loss accumulation often uses **out-of-place `+`**.

---

## 5. Per-op host overhead estimate vs raw kernel launch (~5–10 µs)

Assumptions: release, single stream, contiguous Float32, same shape, pool cache hit, IR already warm, no NaN checks. **Not measured.**

### 5.1 Eager `c = a.add(b)` (happy path)

| Category | Count / nature | **Est. host time** |
|----------|----------------|--------------------|
| Validation + promote + broadcast shape | few branches + 1 small vector | 0.1–0.5 µs |
| 2× TensorLeaf (copy + `make_shared<Tensor>`) | 2 state heaps + 2 Tensor heaps + vectors | 1–5 µs |
| 2× leaf eval (Tensor copy) | 2 state heaps | 0.5–2 µs |
| `empty` + pool + map lock + adopt_storage | 1–2 mutexes + refcount | 1–5 µs |
| Stream prep + guard (same stream) | TLS + 2 atomic stream loads | 0.05–0.3 µs |
| `lazy_ir_record_binary` | 1 mutex + strings + hash | 0.5–3 µs |
| Dtype branch + launch call | switch + `cudaLaunchKernel` | **5–10 µs** (driver) |
| **Host-only subtotal** | | **~5–20 µs** |
| **Total wall before async kernel runs** | | **~10–30 µs** (can match or exceed pure launch) |

If operands need **dtype cast**, **non-contiguous materialize**, or **cross-stream bridge**, add **another kernel + more host** (can jump to **50–200+ µs** class).

### 5.2 Deferred unary on large tensor (build only)

| Step | Est. |
|------|------|
| validate + UnaryExpr + size heuristic | &lt;1 µs |
| `make_deferred` + function wrap + 2 registry locks + IR string | **2–15 µs** |
| Later materialize (plan + locks + empty + launch) | **10–100+ µs** depending on chain |

Tiny tensors (`&lt; 4096` B) skip defer and still pay eager eval + IR ([expr_impl:34-37](src/core/tensor/internal/tensor_expr_impl.hpp)).

### 5.3 Comparison summary

| Op class | Host overhead vs 5–10 µs launch |
|----------|----------------------------------|
| Eager binary (best case) | **~1–2× launch** host-side alone |
| Eager binary (casts / noncontig / IR cold) | **Several× launch** |
| Deferred materialize of N-node chain | Can dominate iteration glue time |
| True in-place + direct launch helper | Closer to **&lt;1×** if no alloc |

**Implication for optimization plan:** cutting **host dispatch** can yield as much as faster small kernels; especially for **loss glue** and **1-element / small reductions**.

---

## 6. Top 10 dispatch-overhead reductions (ranked)

Each item: **why**, **file:line**, **fix sketch**, **regression risk**.

### 1. Stop allocating a new `TensorState` on every Tensor copy  
**Why:** Dominates leaf/eval/temp cost; copies should share metadata like storage.  
**Where:** [tensor.cpp:636-659](src/core/tensor/tensor.cpp), [662-712](src/core/tensor/tensor.cpp); default [impl:443](src/core/tensor/internal/tensor_impl.hpp).  
**Fix:** Make `TensorState` (or a `TensorImpl`) **shared** on shallow copy; keep stream/lazy/name on shared impl. Optionally null-state empty tensors without `make_shared`.  
**Risk:** Any code assuming copy isolates stream/name/lazy; mutation of shared metadata; debug id semantics.

### 2. Kill or gate always-on lazy IR in production training  
**Why:** Global mutex + string shape on **every** binary/unary/reduce record; `lazy_ir_active()==true` always.  
**Where:** [lazy_ir.cpp:189-191](src/core/tensor/lazy_ir.cpp), [120-158](src/core/tensor/lazy_ir.cpp), record call sites.  
**Fix:** Compile-time or runtime flag default **off** for training; enable for tests/debug. Keep fusion recipes on a side channel only when deferring.  
**Risk:** Tests/tools that introspect IR; fusion that relies on node ids (need alternate wiring).

### 3. Avoid `TensorLeaf` heap cells; hold `const Tensor*` / refcounted impl  
**Why:** `TensorLeaf(Tensor)` copies + `make_shared<Tensor>` per operand ([324-325](src/core/tensor/tensor.cpp)).  
**Where:** [tensor_expr.hpp:72-79](src/core/tensor/internal/tensor_expr.hpp), binary construction [881-884](src/core/tensor/internal/tensor_impl.hpp).  
**Fix:** `TensorLeaf` stores `shared_ptr<TensorState>` + data view fields, or non-owning `const Tensor*` with lifetime rules; snapshot path keeps owning cell.  
**Risk:** Lifetime bugs if expressions outlive operands; deferred snapshot COW ([351-409](src/core/tensor/tensor.cpp)).

### 4. Small-vector / inline shapes (rank ≤ 8)  
**Why:** Every shape/stride is `std::vector`; broadcast and empty reallocate.  
**Where:** [TensorShape:208-209](src/core/tensor/internal/tensor_impl.hpp), [strides_:445](src/core/tensor/internal/tensor_impl.hpp), [broadcast.hpp:31-33](src/core/tensor/internal/tensor_broadcast.hpp), [strides():236-246](src/core/tensor/internal/tensor_impl.hpp).  
**Fix:** `std::array<size_t, MAX_RANK>` + rank; cache strides on shape; avoid `shape.strides()` alloc in `empty`.  
**Risk:** ABI of Tensor; serialization; code assuming `dims()` returns stable `vector` reference.

### 5. Fast path for contiguous same-shape same-dtype binary (skip BinaryExpr)  
**Why:** Expr machinery + double eval copies add pure host work with no fusion benefit (already documented [854-858](src/core/tensor/internal/tensor_impl.hpp)).  
**Where:** `binary_op_with_promotion` [859-887](src/core/tensor/internal/tensor_impl.hpp); evaluator [327-365](src/core/tensor/internal/tensor_expr_impl.hpp).  
**Fix:**  
```text
validate → if same shape/device/dtype/contiguous:
  empty → pin → launch_binary → (optional IR) return
else: existing path
```  
**Risk:** Subtle broadcast/promotion edge cases; must keep dtype promotion identical.

### 6. IR / debug labels: `string_view` + no `shape().str()` in hot record  
**Why:** Unconditional `std::string` shape and op_name on every node ([142-149](src/core/tensor/lazy_ir.cpp)).  
**Fix:** Store `DataType`/`Device`/inline dims; format strings only in debug dump APIs.  
**Risk:** Debug tooling that expects `LazyExprDebugInfo::shape` string.

### 7. Flatten / specialize dtype path for Float32 training  
**Why:** Runtime ladders in evaluator + promote + optional `.to()`; training is almost all F32.  
**Where:** [dtype_dispatch.hpp:20-45](src/core/tensor/internal/tensor_dtype_dispatch.hpp), Binary evaluator dtype branches, macros [1724+](src/core/tensor/internal/tensor_impl.hpp).  
**Fix:** `add_f32` / `[[likely]] Float32` path without promote; keep generic as fallback.  
**Risk:** Mixed-dtype ops must not silently change promotion rules.

### 8. Lock-free or thread-local stream / IR / fusion registries  
**Why:** Contended process-wide mutexes on IR and fusion ([lazy_ir:33](src/core/tensor/lazy_ir.cpp), [lazy_executor:38-51](src/core/tensor/lazy_executor.cpp)). Stream TLS is already good; IR is not.  
**Fix:** Thread-local IR arenas for training single-thread; or sharded maps; or disable IR (§2).  
**Risk:** Cross-thread tensor sharing correctness; test isolation.

### 9. Slim `empty` / allocation tracking on hot path  
**Why:** Every result: atomics id/telemetry, pool shared_lock, **map_mutex**, VramProfiler try ([unified:361-369](src/core/tensor/tensor_unified_ops.cpp), [565-573](src/core/tensor/internal/memory_pool.hpp)).  
**Fix:** Optional “kernel scratch” allocator without global map for ephemeral outputs; batch telemetry; defer VramProfiler to debug.  
**Risk:** Leak tracking, OOM diagnostics, stream rehome correctness if map skipped.

### 10. Training glue: prefer in-place / fused loss accumulation APIs  
**Why:** `loss = loss + x` pays full binary host tax each time ([trainer.cpp](src/training/trainer.cpp) multiple sites).  
**Where:** Call sites + `add_` [2315+](src/core/tensor/internal/tensor_impl.hpp).  
**Fix:** `loss.add_(tile)`; fused `loss_add_scaled_(tensor, w)`; keep scalars on GPU without intermediate expr trees.  
**Risk:** Alias bugs if `tile` shares storage with `loss`; autograd-like assumptions (none here if manual).

**Honorable mentions (just outside top 10):**  
- Raise/eliminate lazy size heuristic thrash ([111,926-935](src/core/tensor/lazy_executor.cpp)) — deferral can cost more than it saves for short chains.  
- `has_lazy_expr()` should trust `state_->lazy` / a local flag, not mutex map ([1382-1384](src/core/tensor/internal/tensor_impl.hpp)).  
- `TensorLeaf::eval` should return `const Tensor&` or move shared cell without copy ([333-338](src/core/tensor/tensor.cpp)).  
- Event pool still mutex-bound on rare cross-stream paths ([cuda_event_pool](src/core/tensor/internal/cuda_event_pool.hpp)).

---

## 7. Walkthrough reference: cost stack for `a + b` (same F32 CUDA, contiguous)

```
CPU:
  operator+ → add
  validate_binary_op
  promote_dtypes (no-op)
  broadcast_shape → vector alloc
  TensorLeaf(a): Tensor copy (new TensorState + shape/stride vectors)
                 + make_shared<Tensor>
  TensorLeaf(b): same
  BinaryExpr holds shape copy
  eval():
    left.eval()  → Tensor copy
    right.eval() → Tensor copy
    prepare_inputs_for_stream (TLS, maybe sync)
    CUDAStreamGuard
    empty → next_id atomic, telemetry atomic, pool alloc+map lock, adopt_storage
    pin_operands (deferred check only)
    dtype branch
    launch_binary_op_generic  ← first real GPU enqueue
  lazy_ir_record_binary → global mutex, shape string, node insert
  return result (move)

vs ideal raw:
  ptrs + n + stream → launch
```

---

## 8. Confidence / speculation tags

| Claim | Confidence |
|-------|------------|
| Code paths, locks, always-on IR, copy allocating TensorState, vector shapes | **High** (direct source) |
| µs ranges and training op counts per iteration | **Speculation** — needs `perf`/Tracy/nvtx host timers |
| Fusion planner often “helps” vs pure eager for 1–2 ops | **Speculation** — heuristic may hurt short chains |
| VramProfiler cost relative to map lock | **Medium** — always called but try/catch wrapped |

---

## 9. Recommended measurement plan (for the optimization workstream)

1. Host timers around `binary_op_with_promotion` vs `launch_*` only.  
2. Counters: Tensor copy ctor hits, IR mutex hold time, `empty` rate, deferred materialize rate.  
3. A/B: IR off, shared TensorState, binary fast path — each isolation, with training golden outputs.  
4. Focus first on **loss glue** and **reduce/mean/item** paths that run every iteration even when raster/SSIM are fused.

---

**Bottom line:** Host dispatch for a “simple” tensor op is heavyweight: **multiple heap allocations, shape vector traffic, a global IR lock with string formatting, and pool tracking**, often in the same **5–30 µs** band as a CUDA launch. The largest structural wins are **shared TensorState**, **optional IR**, **no-expr binary fast path**, and **inline shapes**—with training call sites switching hot accumulators to **in-place** ops so fewer full dispatches run at all.
