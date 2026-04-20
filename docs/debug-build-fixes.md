# Debug Build Fixes for LichtFeld Studio

This document describes the fixes implemented to enable Debug builds on Windows with vcpkg dependencies.

## Issue 1: Python ABI Mismatch (Linker Error LNK2019)

### Root Cause
When building in Debug mode on Windows, the MSVC compiler defines `_DEBUG`. Python's headers check for `_DEBUG` and automatically enable `Py_DEBUG`, which activates reference counting debug macros (`Py_NegativeRefcount`, `Py_DECREF_DecRefTotal`). These symbols only exist in debug Python (`python312_d.lib`), but vcpkg only provides release Python.

### Symptoms
```
error LNK2019: unresolved external symbol __imp__Py_NegativeRefcount
error LNK2019: unresolved external symbol __imp__Py_DECREF_DecRefTotal
```

### Solution
Created `src/python/python_compat.hpp` that temporarily undefs `_DEBUG` before including `Python.h`:

```cpp
#if defined(_WIN32) && defined(_DEBUG)
#  pragma push_macro("_DEBUG")
#  undef _DEBUG
#  include <Python.h>
#  pragma pop_macro("_DEBUG")
#else
#  include <Python.h>
#endif
```

Updated all files that include `Python.h` directly to use this wrapper instead.

---

## Issue 2: vcpkg Copies Release DLLs in Debug Builds

### Root Cause
vcpkg's `VCPKG_APPLOCAL_DEPS` feature copies dependencies to the build output, but it was copying Release DLLs (from `vcpkg_installed/x64-windows/bin/`) instead of Debug DLLs (from `vcpkg_installed/x64-windows/debug/bin/`) for Debug builds.

### Symptoms
Runtime crashes with assertions like "cannot construct from null pointer and non-zero size" due to ABI mismatch between Debug-compiled code and Release DLLs.

### Solution
Added a post-build command in `CMakeLists.txt` that copies DLLs from both vcpkg release and debug directories. This ensures all needed DLLs are available:
- Release DLLs (like `zlib1.dll`) are always copied
- Debug DLLs (like `zlibd1.dll`, `rmlui.dll` debug version) are also copied

```cmake
if(WIN32 AND DEFINED VCPKG_TARGET_TRIPLET)
    set(_vcpkg_debug_bin_dir "${CMAKE_BINARY_DIR}/vcpkg_installed/${VCPKG_TARGET_TRIPLET}/debug/bin")
    set(_vcpkg_release_bin_dir "${CMAKE_BINARY_DIR}/vcpkg_installed/${VCPKG_TARGET_TRIPLET}/bin")
    
    # Copy release DLLs first
    add_custom_command(TARGET ${PROJECT_NAME} POST_BUILD
        COMMAND ${CMAKE_COMMAND} -E copy_directory
            "${_vcpkg_release_bin_dir}"
            "$<TARGET_FILE_DIR:${PROJECT_NAME}>"
    )
    
    # Copy debug DLLs second (they have different names, so this adds to rather than replaces)
    add_custom_command(TARGET ${PROJECT_NAME} POST_BUILD
        COMMAND ${CMAKE_COMMAND} -E copy_directory
            "${_vcpkg_debug_bin_dir}"
            "$<TARGET_FILE_DIR:${PROJECT_NAME}>"
    )
endif()
```

---

## Issue 3: CMake Python Library Override Not Working

### Root Cause
The `lfs_force_python_release_abi()` function in `src/python/CMakeLists.txt` checked `IMPORTED_LOCATION_DEBUG` on Windows, but the import library (`.lib`) is stored in `IMPORTED_IMPLIB_DEBUG`, not `IMPORTED_LOCATION_DEBUG` (which holds the DLL path).

### Solution
Modified the function to check `IMPORTED_IMPLIB_DEBUG` on Windows and update both the import library and DLL location to use release versions.

---

## Issue 4: Null Optional Access Crash in Modal Overlay

### Root Cause
In `src/visualizer/gui/rml_modal_overlay.cpp`, RmlUI event processing (mouse clicks, key presses) could trigger button handlers that call `dismiss()`, resetting the `active_` optional. Later code then accessed `active_->has_input` without re-checking validity.

### Solution
Added a guard before accessing `active_` after RmlUI event processing:

```cpp
// Re-check active_ since RmlUI event processing above may have triggered
// dismiss() or cancel() callbacks that reset it
if (!active_)
    return;
```

---

## Files Modified

| File | Change |
|------|--------|
| `src/python/python_compat.hpp` | **New** - Python.h wrapper that undefs `_DEBUG` |
| `src/python/python_runtime.cpp` | Use `python_compat.hpp` |
| `src/python/gil.hpp` | Use `python_compat.hpp` |
| `src/python/plugin_runner.cpp` | Use `python_compat.hpp` |
| `src/python/runner.cpp` | Use `python_compat.hpp` |
| `src/python/lfs/py_cameras.cpp` | Use `python_compat.hpp` |
| `src/visualizer/gui/panels/python_console_panel.cpp` | Use `python_compat.hpp` |
| `tests/test_python_integration.cpp` | Use `python_compat.hpp` |
| `src/python/CMakeLists.txt` | Fix `lfs_force_python_release_abi()` for Windows |
| `CMakeLists.txt` | Add vcpkg DLL copy post-build command |
| `src/visualizer/gui/rml_modal_overlay.cpp` | Add null check for `active_` |
| `src/visualizer/gui/gpu_memory_query.cpp` | Remove COM Release in static destructor |
| `src/visualizer/gui/rmlui/rml_input_utils.hpp` | Remove RmlUI calls from destructor |
| `src/visualizer/gui/rmlui/rml_panel_host.cpp` | Remove destroyContext call from destructor |
| `src/python/runner.cpp` | Wrap preload thread in RAII struct with safe destructor |

---

## Issue 5: DXGI COM Release Crash on Shutdown

### Root Cause
The `DxgiMemoryState` static object in `gpu_memory_query.cpp` holds a COM pointer (`IDXGIAdapter3*`) and releases it in the destructor. At static destruction time during application shutdown, the DXGI/DirectX runtime may already be unloaded, causing a crash when `Release()` is called.

### Solution
Removed the `Release()` call from the destructor. The OS will clean up the COM reference when the process exits. This is the standard pattern for static COM objects.

### File Modified
| File | Change |
|------|--------|
| `src/visualizer/gui/gpu_memory_query.cpp` | Remove COM Release in static destructor |

---

## Issue 6: RmlUI Event Listener Removal Crash on Shutdown

### Root Cause
The `TextInputEscapeRevertController` class calls `RemoveEventListener` in its destructor. When panels containing this controller are destroyed during application shutdown, RmlUI may already be torn down, causing a crash when accessing RmlUI's internal event specification data.

### Solution
Changed the destructor to only clear the internal bindings map without calling `RemoveEventListener`. RmlUI will clean up the event listeners automatically when elements are destroyed.

### File Modified
| File | Change |
|------|--------|
| `src/visualizer/gui/rmlui/rml_input_utils.hpp` | Remove RmlUI calls from destructor |

---

## Issue 7: RmlPanelHost destroyContext Crash on Shutdown

### Root Cause
The `RmlPanelHost` destructor called `manager_->destroyContext()` to unregister its RmlUI context. During application shutdown, if `RmlUIManager` was destroyed before the panels, this call would access an invalid manager or try to operate on RmlUI that was already shut down.

### Solution
Removed the `destroyContext` call from the destructor. The `RmlUIManager::shutdown()` method already handles cleanup of all contexts.

### File Modified
| File | Change |
|------|--------|
| `src/visualizer/gui/rmlui/rml_panel_host.cpp` | Remove destroyContext call from destructor |

---

## Issue 8: Plugin Preload Thread Crash on Shutdown

### Root Cause
The `g_plugin_preload_thread` static variable is a `std::thread` that may still be joinable when the static destructor runs during DLL unload. In C++, destroying a joinable `std::thread` calls `std::terminate()`, causing a crash.

### Solution
Wrapped the thread in a RAII struct `PluginPreloadThread` that joins the thread in its destructor, ensuring proper cleanup at static destruction time.

### File Modified
| File | Change |
|------|--------|
| `src/python/runner.cpp` | Wrap preload thread in RAII struct with destructor that joins |
