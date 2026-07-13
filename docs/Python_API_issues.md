# Python API Issues

Issues discovered during documentation and binding review, comparing public docs against current Python bindings under `src/python`.

---

## Current Issues

### 1. `linspace()` accepts `dtype` but does not apply it

**Location:** `src/python/lfs/py_tensor.cpp:1401-1405`

`PyTensor::linspace()` takes `dtype`, but currently only forwards `device` to `Tensor::linspace(...)`.

```cpp
PyTensor PyTensor::linspace(float start, float end, int64_t steps,
                            const std::string& device,
                            const std::string& dtype) {
    auto t = Tensor::linspace(start, end, static_cast<size_t>(steps), parse_device(device));
    return PyTensor(t);
}
```

**Impact:** `lf.Tensor.linspace(..., dtype="...")` behaves as if dtype was ignored.

### 2. API reference previously documented `lf.ui.get_gpu_memory()`, but no binding exists

**Location:** `src/python/stubs/lichtfeld/ui/__init__.pyi`

The current UI stub exposes `get_fps()` and `get_git_commit()`, but not
`get_gpu_memory()`. The helper package does expose
`lfs_plugins.get_gpu_memory()`, which returns a coarse CUDA memory count via
PyTorch when available.

**Impact:** Plugins following older docs and calling `lf.ui.get_gpu_memory()`
will fail with `AttributeError`.

### 3. `selection.pick_at_screen()` mixes queried hit data with cursor hover index

**Location:** `src/python/stubs/lichtfeld/selection.pyi`

The stub documents that `pick_at_screen(screen_x, screen_y)` returns
`depth` and `world_position` for the queried coordinates, but `PickResult.index`
reflects the gaussian under the current cursor rather than the queried
coordinates.

**Impact:** Plugins using off-cursor point picking should not treat
`PickResult.index` as the hit index for the requested coordinates.

---

## Implementation Gaps

### Package Management

| Function | Status |
|----------|--------|
| `packages.uninstall_async()` | Not implemented (only sync `uninstall()`) |

### UI Styling API

| Function | Status |
|----------|--------|
| `get_style_color()` / `set_style_color()` | Not exposed |
| `get_style_var()` / `set_style_var()` | Not exposed |

The push/pop style stack APIs are available (`push_style_var`, `push_style_var_vec2`, `push_style_color`, `pop_*`).

### GPU Memory UI API

| Function | Status |
|----------|--------|
| `lf.ui.get_gpu_memory()` | Not exposed; use `lfs_plugins.get_gpu_memory()` if a plugin only needs a coarse helper value |

---

## Documentation Corrections Made

### API docs updated in this pass

| File | Update |
|------|--------|
| `docs/plugins/api-reference.md` | Added Python API surface map; filled in scene selection, `lf.selection`, camera/dataset, rendering, IO, pipeline, native ops, MCP, packages, scripts, keymap, animation, mesh; corrected export formats, transform wording, dialog coverage, missing GPU-memory binding |
| `docs/Python_UI.md` | Kept as compatibility redirect to canonical plugin UI docs |

---

## Recommendations

### High Priority

1. Apply `dtype` in `PyTensor::linspace()` or remove the argument from the public signature.
2. Decide whether `selection.pick_at_screen()` should return an index for the queried coordinate or keep the current cursor-index semantics and rename/document the field more explicitly.
### Medium Priority
3. Add async uninstall API for package parity with `install_async()`.
4. Add `lf.ui.get_gpu_memory()` or keep docs exclusively on `lfs_plugins.get_gpu_memory()`.

### Low Priority

5. Add direct style getters/setters if runtime theme customization from Python is needed beyond push/pop stacks.

---

## Files Reviewed

| File | Purpose |
|------|---------|
| `src/python/stubs/lichtfeld/__init__.pyi` | Top-level Python API surface |
| `src/python/stubs/lichtfeld/ui/__init__.pyi` | UI API surface |
| `src/python/stubs/lichtfeld/plugins.pyi` | Plugin API surface |
| `src/python/stubs/lichtfeld/scene.pyi` | Scene graph, cameras, point clouds, selection groups |
| `src/python/stubs/lichtfeld/selection.pyi` | Gaussian selection primitives |
| `src/python/stubs/lichtfeld/io.pyi` | File load/save API |
| `src/python/stubs/lichtfeld/ops.pyi` | Native operator registry |
| `src/python/stubs/lichtfeld/packages.pyi` | Package manager API |
| `src/python/stubs/lichtfeld/mcp.pyi` | Python MCP tool registration |
| `src/python/stubs/lichtfeld/scripts.pyi` | Scripts panel state/execution |
| `src/python/stubs/lichtfeld/keymap.pyi` | Input binding API |
| `src/python/stubs/lichtfeld/animation.pyi` | Animation clip/track API |
| `src/python/stubs/lichtfeld/mesh.pyi` | OpenMesh/tensor mesh API |
| `src/python/lfs/py_tensor.cpp` | Tensor bindings |
| `src/python/lfs/py_ui.cpp` | UI bindings |
| `src/python/lfs_plugins/*.py` | Plugin framework runtime API |
