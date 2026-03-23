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

---

### 2. Theme-switch API is limited to dark/light only

**Location:** `src/python/lfs/py_ui.cpp:4348-4363`

`lf.ui.set_theme(name)` currently only handles `"dark"` and `"light"`.

**Impact:** additional theme assets are not selectable through the Python API.

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

---

## Documentation Corrections Made

### API docs updated in this pass

| File | Update |
|------|--------|
| `docs/plugins/api-reference.md` | Corrected UI/file-dialog names, operator return type, plugin manager returns, tensor coverage, pyproject requirements |
| `docs/Python_UI.md` | Rewritten to current panel metadata, dialog APIs, hooks, and widget usage |

---

## Recommendations

### High Priority

1. Apply `dtype` in `PyTensor::linspace()` or remove the argument from the public signature.
### Medium Priority
2. Extend `lf.ui.set_theme()` to support all shipped themes (or document that only dark/light are runtime-switchable).
3. Add async uninstall API for package parity with `install_async()`.

### Low Priority

4. Add direct style getters/setters if runtime theme customization from Python is needed beyond push/pop stacks.

---

## Files Reviewed

| File | Purpose |
|------|---------|
| `src/python/stubs/lichtfeld/__init__.pyi` | Top-level Python API surface |
| `src/python/stubs/lichtfeld/ui/__init__.pyi` | UI API surface |
| `src/python/stubs/lichtfeld/plugins.pyi` | Plugin API surface |
| `src/python/lfs/py_tensor.cpp` | Tensor bindings |
| `src/python/lfs/py_ui.cpp` | UI bindings |
| `src/python/lfs_plugins/*.py` | Plugin framework runtime API |
