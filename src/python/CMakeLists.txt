#SPDX - FileCopyrightText : 2025 LichtFeld Studio Authors
#
#SPDX - License - Identifier : GPL - 3.0 - or -later

message(STATUS "Configuring Python components (runner + nanobind module)")

#Python for interpreter + embedding + nanobind
find_package(Python REQUIRED COMPONENTS Interpreter Development Development.Module Development.SABIModule)
find_package(nanobind CONFIG REQUIRED)

option(LFS_DEV_IMPORT_SOURCE_PYTHON
    "Import built-in Python packages from src/python in non-portable builds"
    ON)

# vcpkg extension modules use release ABI; debug libpython has Py_DEBUG which
# changes _PyRuntimeState layout, causing ABI mismatch crashes.
# Force release Python library for debug builds on both Python::Python and Python::Module
# (nanobind uses Python::Module for extension modules)
function(lfs_force_python_release_abi target_name)
    # On Windows, check the import library (.lib); on Linux, check the library location
    if(WIN32)
        get_target_property(_py_debug_lib ${target_name} IMPORTED_IMPLIB_DEBUG)
    else()
        get_target_property(_py_debug_lib ${target_name} IMPORTED_LOCATION_DEBUG)
    endif()
    if(NOT _py_debug_lib)
        return()
    endif()

    if(_py_debug_lib MATCHES "python3\\.([0-9]+)d\\.a$")
        # Linux: libpython3.XXd.a -> libpython3.XX.a
        string(REGEX REPLACE "debug/lib/libpython3\\.([0-9]+)d\\.a$"
                             "lib/libpython3.\\1.a" _py_release_lib "${_py_debug_lib}")
        if(EXISTS "${_py_release_lib}")
            set_target_properties(${target_name} PROPERTIES
                IMPORTED_LOCATION_DEBUG "${_py_release_lib}")
            message(STATUS "Python: forced release ABI for ${target_name} (Linux)")
        endif()
    elseif(_py_debug_lib MATCHES "python3([0-9]+)_?d\\.lib$")
        # Windows: python312_d.lib -> python312.lib (import library)
        # We already have _py_debug_lib as the import library path
        string(REGEX REPLACE "python3([0-9]+)_?d\\.lib$"
                             "python3\\1.lib" _py_release_implib "${_py_debug_lib}")
        string(REPLACE "/debug/lib/" "/lib/" _py_release_implib "${_py_release_implib}")
        
        if(EXISTS "${_py_release_implib}")
            # Set the release import library for Debug config
            set_target_properties(${target_name} PROPERTIES
                IMPORTED_IMPLIB_DEBUG "${_py_release_implib}")
            
            # Also update the DLL location to use release python312.dll
            get_target_property(_py_debug_dll ${target_name} IMPORTED_LOCATION_DEBUG)
            if(_py_debug_dll)
                string(REGEX REPLACE "python3([0-9]+)_?d\\.dll$"
                                     "python3\\1.dll" _py_release_dll "${_py_debug_dll}")
                string(REPLACE "/debug/bin/" "/bin/" _py_release_dll "${_py_release_dll}")
                if(EXISTS "${_py_release_dll}")
                    set_target_properties(${target_name} PROPERTIES
                        IMPORTED_LOCATION_DEBUG "${_py_release_dll}")
                endif()
            endif()
            
            message(STATUS "Python: forced release ABI for ${target_name} (Windows) - using ${_py_release_implib}")
        endif()
    endif()
endfunction()

# Apply to both Python::Python (for embedded Python) and Python::Module (for nanobind)
lfs_force_python_release_abi(Python::Python)
if(TARGET Python::Module)
    lfs_force_python_release_abi(Python::Module)
endif()
if(TARGET Python::SABIModule)
    lfs_force_python_release_abi(Python::SABIModule)
endif()

# Shared library for Python runtime state - must be shared to avoid duplicate statics
# when linked into both the main executable and the Python module
add_library(lfs_python_runtime SHARED
    python_runtime.cpp
)

target_include_directories(lfs_python_runtime
    PUBLIC
        ${CMAKE_CURRENT_SOURCE_DIR}
        ${CMAKE_SOURCE_DIR}/src
        ${CMAKE_SOURCE_DIR}/src/core/include
)

target_link_libraries(lfs_python_runtime
    PUBLIC
        Python::Python
        glm::glm
    PRIVATE
        lfs_logger
)

target_compile_definitions(lfs_python_runtime
    PRIVATE
        LFS_PYTHON_RUNTIME_EXPORTS
)

set_target_properties(lfs_python_runtime PROPERTIES
    POSITION_INDEPENDENT_CODE ON
    CXX_STANDARD 23
    CXX_STANDARD_REQUIRED ON
    LIBRARY_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}
    RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}
)
get_property(_IS_MULTI_CONFIG GLOBAL PROPERTY GENERATOR_IS_MULTI_CONFIG)
if(WIN32 AND _IS_MULTI_CONFIG)
    set_target_properties(lfs_python_runtime PROPERTIES
        LIBRARY_OUTPUT_DIRECTORY              ${CMAKE_BINARY_DIR}/$<CONFIG>
        RUNTIME_OUTPUT_DIRECTORY              ${CMAKE_BINARY_DIR}/$<CONFIG>
        RUNTIME_OUTPUT_DIRECTORY_DEBUG        ${CMAKE_BINARY_DIR}/Debug
        RUNTIME_OUTPUT_DIRECTORY_RELEASE      ${CMAKE_BINARY_DIR}/Release
        RUNTIME_OUTPUT_DIRECTORY_RELWITHDEBINFO ${CMAKE_BINARY_DIR}/RelWithDebInfo
        RUNTIME_OUTPUT_DIRECTORY_MINSIZEREL   ${CMAKE_BINARY_DIR}/MinSizeRel
    )
endif()

# Static library for other Python utilities
add_library(lfs_python_utils STATIC
    runner.cpp
    python_buffer_analysis.cpp
    package_manager.cpp
    plugin_runner.cpp
    subprocess.cpp
    uv_runner.cpp
)

target_include_directories(lfs_python_utils
    PUBLIC
        ${CMAKE_CURRENT_SOURCE_DIR}
        ${CMAKE_SOURCE_DIR}/src
)

target_link_libraries(lfs_python_utils
    PUBLIC
        Python::Python
        lfs_core
        lfs_python_runtime
        lfs::tree_sitter
        lfs::tree_sitter_python
)

set(LFS_PYTHON_UTILS_DEFINITIONS
    LFS_PYTHON_EXECUTABLE="${Python_EXECUTABLE}"
)
if(LFS_DEV_IMPORT_SOURCE_PYTHON AND NOT BUILD_PORTABLE)
    list(APPEND LFS_PYTHON_UTILS_DEFINITIONS
        LFS_DEV_PYTHON_SOURCE_DIR="${CMAKE_CURRENT_SOURCE_DIR}"
    )
    message(STATUS "Python dev mode: built-in Python imports from ${CMAKE_CURRENT_SOURCE_DIR}")
endif()

target_compile_definitions(lfs_python_utils
    PUBLIC
        ${LFS_PYTHON_UTILS_DEFINITIONS}
)

set_target_properties(lfs_python_utils PROPERTIES
    POSITION_INDEPENDENT_CODE ON
    CXX_STANDARD 23
    CXX_STANDARD_REQUIRED ON
)

# nanobind module (Python import name: lichtfeld)
# NOTE: We intentionally do NOT use STABLE_ABI here because:
# 1. The main app embeds Python and links to python312.dll
# 2. STABLE_ABI would make the .pyd link to python3.dll instead
# 3. On Windows, having both DLLs loaded causes double Python initialization
# 4. This makes the .pyd version-specific but avoids the DLL conflict
nanobind_add_module(
    lfs_py
    NB_STATIC
    lfs/module.cpp
    lfs/py_animation.cpp
    lfs/py_command.cpp
    lfs/py_gizmo.cpp
    lfs/py_keymap.cpp
    lfs/py_tensor.cpp
    lfs/py_splat_data.cpp
    lfs/py_scene.cpp
    lfs/py_scripts.cpp
    lfs/py_cameras.cpp
    lfs/py_io.cpp
    lfs/py_packages.cpp
    lfs/py_params.cpp
    lfs/py_plugins.cpp
    lfs/py_ui.cpp
    lfs/py_ui_context.cpp
    lfs/py_ui_theme.cpp
    lfs/py_ui_operators.cpp
    lfs/py_modal_registry.cpp
    lfs/py_ui_modals.cpp
    lfs/notification_bridge.cpp
    lfs/py_ui_hooks.cpp
    lfs/py_ui_menus.cpp
    lfs/py_ui_panels.cpp
    lfs/py_uilist.cpp
    lfs/py_viewport.cpp
    viewport_overlay.cpp
    lfs/py_mcp.cpp
    lfs/py_mesh.cpp
    lfs/py_mesh2splat.cpp
    lfs/py_splat_simplify.cpp
    lfs/openmesh/handles.cpp
    lfs/openmesh/io.cpp
    lfs/py_rendering.cpp
    lfs/py_selection.cpp
    lfs/py_operator.cpp
    lfs/py_pipeline.cpp
    lfs/py_prop_registry.cpp
    lfs/py_signals.cpp
    lfs/py_rml.cpp
    lfs/rml_python_panel_adapter.cpp
    lfs/py_rml_im_mode_layout.cpp
    lfs/rml_im_mode_layout.cpp
    lfs/rml_im_mode_panel_adapter.cpp
)

set_target_properties(lfs_py PROPERTIES
    CUDA_ARCHITECTURES ""
    CUDA_SEPARABLE_COMPILATION OFF
    LINKER_LANGUAGE CXX
    POSITION_INDEPENDENT_CODE ON
    CXX_VISIBILITY_PRESET hidden
    VISIBILITY_INLINES_HIDDEN ON
    OUTPUT_NAME lichtfeld
)

# Do not add --exclude-libs,ALL here while lfs_visualizer is static on Linux.
# The embedded app and the lichtfeld extension currently share GUI registries
# and hook bridges through ELF symbol interposition.
target_compile_definitions(lfs_py PRIVATE
    LFS_PYTHON_EXECUTABLE="${Python_EXECUTABLE}"
    $<$<CONFIG:Debug>:DEBUG_BUILD>
    $<$<CONFIG:Release>:RELEASE_BUILD>
)


if(CMAKE_CXX_LINK_GROUP_USING_RESCAN_SUPPORTED)
    set(LFS_PY_INTERNAL_LINK_LIBS
        "$<LINK_GROUP:RESCAN,lfs_core,lfs_event_bridge,lfs_geometry,lfs_visualizer,lfs_mcp,lfs_python_runtime,lfs_training,lfs_io,lfs_rendering_tensor,lfs_sequencer>"
    )
else()
    set(LFS_PY_INTERNAL_LINK_LIBS
        lfs_core
        lfs_event_bridge
        lfs_geometry
        lfs_visualizer
        lfs_mcp
        lfs_python_runtime
        lfs_training
        lfs_io
        lfs_rendering_tensor
        lfs_sequencer
    )
endif()

target_link_libraries(lfs_py PRIVATE
    ${LFS_PY_INTERNAL_LINK_LIBS}
    $<IF:$<PLATFORM_ID:Windows>,OpenMeshCore,OpenMeshCoreStatic>
    $<IF:$<PLATFORM_ID:Windows>,OpenMeshTools,OpenMeshToolsStatic>
    OpenImageIO::OpenImageIO
    CUDA::cudart
    CUDA::curand
    ${CMAKE_DL_LIBS}
    imgui::imgui
    implot::implot
    $<IF:$<PLATFORM_ID:Windows>,RmlUi::RmlUi,lfs_rmlui>
)

target_include_directories(lfs_py PRIVATE
    ${CMAKE_SOURCE_DIR}/include
    ${CMAKE_BINARY_DIR}/include
    ${CMAKE_SOURCE_DIR}/src
    ${CMAKE_SOURCE_DIR}/src/visualizer
)

# Multi-config generators (MSVC) output to Release/Debug subdirs - copy to parent for Python to find
if(CMAKE_CONFIGURATION_TYPES)
    add_custom_command(TARGET lfs_py POST_BUILD
        COMMAND ${CMAKE_COMMAND} -E copy_if_different
            "$<TARGET_FILE:lfs_py>"
            "${CMAKE_CURRENT_BINARY_DIR}/$<TARGET_FILE_NAME:lfs_py>"
        COMMENT "Copying $<TARGET_FILE_NAME:lfs_py> to ${CMAKE_CURRENT_BINARY_DIR}"
    )
endif()

# Windows: Copy required DLLs next to the .pyd for Python module loading
# Python's DLL search doesn't use the app's directory, only the module's directory
if(WIN32)
    # Copy project DLLs
    add_custom_command(TARGET lfs_py POST_BUILD
        COMMAND ${CMAKE_COMMAND} -E copy_if_different
            "$<TARGET_FILE:lfs_python_runtime>"
            "$<TARGET_FILE:lfs_event_bridge>"
            "$<TARGET_FILE:lfs_core>"
            "$<TARGET_FILE:lfs_visualizer>"
            "$<TARGET_FILE:lfs_mcp>"
            "${CMAKE_CURRENT_BINARY_DIR}"
        COMMENT "Copying project DLLs to Python module directory"
    )

    # Copy CUDA DLLs
    file(GLOB CUDA_DLLS_1 "${CUDAToolkit_BIN_DIR}/cudart64*.dll" "${CUDAToolkit_BIN_DIR}/curand64*.dll")
    file(GLOB CUDA_DLLS_2 "${CUDAToolkit_BIN_DIR}/x64/cudart64*.dll" "${CUDAToolkit_BIN_DIR}/x64/curand64*.dll")
    set(CUDA_DLLS_FOR_PYD ${CUDA_DLLS_1} ${CUDA_DLLS_2})
    foreach(DLL ${CUDA_DLLS_FOR_PYD})
        if(EXISTS "${DLL}")
            add_custom_command(TARGET lfs_py POST_BUILD
                COMMAND ${CMAKE_COMMAND} -E copy_if_different "${DLL}" "${CMAKE_CURRENT_BINARY_DIR}"
                COMMENT "Copying ${DLL} to Python module directory"
            )
        endif()
    endforeach()

    # Copy vcpkg runtime DLLs
    add_custom_command(TARGET lfs_py POST_BUILD
        COMMAND ${CMAKE_COMMAND} -E copy_directory_if_different
            "${CMAKE_BINARY_DIR}/vcpkg_installed/${VCPKG_TARGET_TRIPLET}/bin"
            "${CMAKE_CURRENT_BINARY_DIR}"
        COMMENT "Copying vcpkg DLLs to Python module directory"
    )

    # Copy nvimgcodec
    if(TARGET nvimgcodec)
        add_custom_command(TARGET lfs_py POST_BUILD
            COMMAND ${CMAKE_COMMAND} -E copy_if_different
                "$<TARGET_FILE:nvimgcodec>"
                "${CMAKE_CURRENT_BINARY_DIR}"
            COMMENT "Copying nvimgcodec to Python module directory"
        )
    endif()
endif()

# Type stubs for IDE support (output to typings/ to avoid runtime conflicts)
option(BUILD_PYTHON_STUBS "Generate Python type stubs (requires loading the module)" ON)

# Skip stubgen for Debug builds on Windows: vcpkg provides a release Python
# interpreter, while Debug extension loading still has ABI-sensitive edges.
# Stubs are primarily useful for IDE support and can be generated from Release builds.
get_property(LFS_GENERATOR_IS_MULTI_CONFIG GLOBAL PROPERTY GENERATOR_IS_MULTI_CONFIG)
if(WIN32 AND NOT LFS_GENERATOR_IS_MULTI_CONFIG AND CMAKE_BUILD_TYPE STREQUAL "Debug")
    message(STATUS "Skipping Python stubs for Debug build (vcpkg Python is release-only)")
    set(BUILD_PYTHON_STUBS OFF)
endif()

if(BUILD_PYTHON_STUBS)
    set(GENERATED_STUBS_DIR "${CMAKE_CURRENT_BINARY_DIR}/typings")
    set(COMMITTED_STUBS_DIR "${CMAKE_CURRENT_SOURCE_DIR}/stubs")
    set(PYTHON_STUB_WORKFLOW_SCRIPT "${CMAKE_CURRENT_SOURCE_DIR}/scripts/manage_stubs.py")
    set(GENERATED_STUBS_MARKER "${GENERATED_STUBS_DIR}/lichtfeld/py.typed")
    if(WIN32 AND LFS_GENERATOR_IS_MULTI_CONFIG)
        set(GENERATED_STUBS_MARKER "${GENERATED_STUBS_DIR}/$<CONFIG>/lichtfeld/py.typed")
    endif()

    if(WIN32)
        # On Windows, we need to set PATH for DLL loading before running stubgen
        # Find the stubgen script (NB_DIR is set by find_package(nanobind))
        if(EXISTS "${NB_DIR}/src/stubgen.py")
            set(NB_STUBGEN "${NB_DIR}/src/stubgen.py")
        elseif(EXISTS "${NB_DIR}/stubgen.py")
            set(NB_STUBGEN "${NB_DIR}/stubgen.py")
        else()
            message(WARNING "nanobind stubgen.py not found at ${NB_DIR}, disabling stub generation")
            set(BUILD_PYTHON_STUBS OFF)
        endif()

        if(BUILD_PYTHON_STUBS)
            # Python 3.8+ on Windows requires os.add_dll_directory() for DLL loading
            # Create a wrapper script that sets up DLL paths before running stubgen
            # Use file(GENERATE) to evaluate generator expressions at generation time
            set(STUBGEN_WRAPPER "${CMAKE_CURRENT_BINARY_DIR}/run_stubgen_$<CONFIG>.py")
            file(GENERATE OUTPUT "${STUBGEN_WRAPPER}" CONTENT "
import os, pathlib, sys

stub_marker = pathlib.Path(r'${GENERATED_STUBS_MARKER}')
if '$<CONFIG>' == 'Debug':
    stub_marker.parent.mkdir(parents=True, exist_ok=True)
    stub_marker.touch()
    sys.exit(0)

# Add DLL directories (Python 3.8+ requirement on Windows)
dll_dirs = [
    r'$<TARGET_FILE_DIR:lfs_python_runtime>',
    r'${CMAKE_BINARY_DIR}/vcpkg_installed/${VCPKG_TARGET_TRIPLET}/bin',
    r'${CMAKE_BINARY_DIR}/external/nvImageCodec/src',
    r'${CUDAToolkit_BIN_DIR}',
    r'${CUDAToolkit_BIN_DIR}/x64',
]

for d in dll_dirs:
    if os.path.isdir(d):
        os.add_dll_directory(d)

# Add module path
sys.path.insert(0, r'$<TARGET_FILE_DIR:lfs_py>')

# Run stubgen
import runpy
sys.argv = ['stubgen', '-m', 'lichtfeld',
            '-M', r'${GENERATED_STUBS_DIR}/lichtfeld/py.typed',
            '-O', r'${GENERATED_STUBS_DIR}',
            '-r']
runpy.run_path(r'${NB_STUBGEN}', run_name='__main__')

# Normalize CRLF -> LF and add space after comma in templates for consistency with linux
import glob, pathlib, re
for pyi in glob.glob(r'${GENERATED_STUBS_DIR}/**/*.pyi', recursive=True):
    p = pathlib.Path(pyi)
    content = p.read_bytes().decode('utf-8').replace('\\r\\n', '\\n')
    # Add space after comma in C++ templates (e.g., VectorT<float,3> -> VectorT<float, 3>)
    while True:
        content, count = re.subn(r'(<[^>]+),([a-zA-Z0-9_])', r'\\1, \\2', content)
        if count == 0:
            break
    p.write_bytes(content.encode('utf-8'))

stub_marker.parent.mkdir(parents=True, exist_ok=True)
stub_marker.touch()
")

            add_custom_command(
                OUTPUT "${GENERATED_STUBS_MARKER}"
                COMMAND "${Python_EXECUTABLE}" "${STUBGEN_WRAPPER}"
                WORKING_DIRECTORY "${CMAKE_CURRENT_BINARY_DIR}"
                DEPENDS lfs_py lfs_python_runtime "${NB_STUBGEN}"
                COMMENT "Generating Python stubs for lichtfeld (Windows)"
            )
            add_custom_target(lichtfeld_stub ALL DEPENDS "${GENERATED_STUBS_MARKER}")
        endif()
    else()
        nanobind_add_stub(lichtfeld_stub
            MODULE lichtfeld
            OUTPUT_PATH ${GENERATED_STUBS_DIR}
            PYTHON_PATH ${CMAKE_CURRENT_BINARY_DIR}
            DEPENDS lfs_py
            RECURSIVE
            MARKER_FILE ${GENERATED_STUBS_DIR}/lichtfeld/py.typed
        )
    endif()

    file(GLOB_RECURSE COMMITTED_STUB_FILES CONFIGURE_DEPENDS "${COMMITTED_STUBS_DIR}/*")

    add_custom_target(refresh_python_stubs
        COMMAND "${Python_EXECUTABLE}" "${PYTHON_STUB_WORKFLOW_SCRIPT}"
            sync
            --generated "${GENERATED_STUBS_DIR}"
            --committed "${COMMITTED_STUBS_DIR}"
        DEPENDS lichtfeld_stub "${PYTHON_STUB_WORKFLOW_SCRIPT}"
        COMMENT "Syncing generated Python stubs into the source tree"
        VERBATIM
    )

    set(PYTHON_STUB_CHECK_STAMP "${CMAKE_CURRENT_BINARY_DIR}/python_stub_check.stamp")
    set(PYTHON_STUB_CHECK_COMMAND
        "${Python_EXECUTABLE}" "${PYTHON_STUB_WORKFLOW_SCRIPT}"
        check
        --generated "${GENERATED_STUBS_DIR}"
        --committed "${COMMITTED_STUBS_DIR}"
    )
    if(WIN32 AND LFS_GENERATOR_IS_MULTI_CONFIG)
        set(PYTHON_STUB_CHECK_STAMP "${CMAKE_CURRENT_BINARY_DIR}/python_stub_check_$<CONFIG>.stamp")
        set(PYTHON_STUB_CHECK_WRAPPER "${CMAKE_CURRENT_BINARY_DIR}/check_python_stubs_$<CONFIG>.py")
        file(GENERATE OUTPUT "${PYTHON_STUB_CHECK_WRAPPER}" CONTENT "
import pathlib, subprocess, sys

stamp = pathlib.Path(r'${PYTHON_STUB_CHECK_STAMP}')
if '$<CONFIG>' == 'Debug':
    stamp.parent.mkdir(parents=True, exist_ok=True)
    stamp.touch()
    sys.exit(0)

result = subprocess.run([
    r'${Python_EXECUTABLE}',
    r'${PYTHON_STUB_WORKFLOW_SCRIPT}',
    'check',
    '--generated', r'${GENERATED_STUBS_DIR}',
    '--committed', r'${COMMITTED_STUBS_DIR}',
])
if result.returncode != 0:
    sys.exit(result.returncode)
stamp.parent.mkdir(parents=True, exist_ok=True)
stamp.touch()
")
        set(PYTHON_STUB_CHECK_COMMAND "${Python_EXECUTABLE}" "${PYTHON_STUB_CHECK_WRAPPER}")
    endif()

    add_custom_command(
        OUTPUT "${PYTHON_STUB_CHECK_STAMP}"
        COMMAND ${PYTHON_STUB_CHECK_COMMAND}
        COMMAND ${CMAKE_COMMAND} -E touch "${PYTHON_STUB_CHECK_STAMP}"
        DEPENDS lichtfeld_stub "${PYTHON_STUB_WORKFLOW_SCRIPT}" ${COMMITTED_STUB_FILES}
        COMMENT "Checking committed Python stubs against generated stubs"
        VERBATIM
    )
    add_custom_target(check_python_stubs ALL
        DEPENDS "${PYTHON_STUB_CHECK_STAMP}"
    )
endif()

# Copy lfs_plugins Python module to build directory (recursively to include subdirectories)
add_custom_target(lfs_plugins_copy ALL
    COMMAND ${CMAKE_COMMAND} -E copy_directory_if_different ${CMAKE_CURRENT_SOURCE_DIR}/lfs_plugins $<TARGET_FILE_DIR:lfs_py>/lfs_plugins
    COMMENT "Copying lfs_plugins Python module"
)

add_custom_target(lfs_python_support_copy ALL
    COMMAND ${CMAKE_COMMAND} -E make_directory $<TARGET_FILE_DIR:lfs_py>
    COMMAND ${CMAKE_COMMAND} -E copy_if_different
        ${CMAKE_CURRENT_SOURCE_DIR}/_lfs_panel_contract.py
        ${CMAKE_CURRENT_SOURCE_DIR}/lfs_splat_lod_hierarchy.py
        $<TARGET_FILE_DIR:lfs_py>
    COMMENT "Copying internal Python support modules"
)

add_dependencies(lfs_py lfs_plugins_copy lfs_python_support_copy)
if(TARGET lichtfeld_stub)
    add_dependencies(lichtfeld_stub lfs_python_support_copy)
endif()

# UV package manager for Python package installation support
include(${CMAKE_SOURCE_DIR}/cmake/FetchUV.cmake)
fetch_uv()
if(BUILD_PORTABLE AND NOT EXISTS "${UV_BINARY_PATH}")
    message(FATAL_ERROR "Portable build requires bundled uv, but FetchUV failed (expected ${UV_BINARY_PATH})")
endif()
copy_uv_to_build()
install_uv()

# =============================================================================
# INSTALLATION - Python SDK components for plugin development
# =============================================================================
set(PYTHON_INSTALL_DIR "${CMAKE_INSTALL_LIBDIR}/python")

# Install lichtfeld Python module (.so/.pyd) to bin for portable builds
if(BUILD_PORTABLE)
    install(TARGETS lfs_py
        LIBRARY DESTINATION "${CMAKE_INSTALL_BINDIR}"
        RUNTIME DESTINATION "${CMAKE_INSTALL_BINDIR}"  # Windows DLLs
    )
else()
    install(TARGETS lfs_py
        LIBRARY DESTINATION "${PYTHON_INSTALL_DIR}"
        RUNTIME DESTINATION "${PYTHON_INSTALL_DIR}"  # Windows DLLs
    )
endif()

# Install lfs_python_runtime shared library (required by lichtfeld module and exe)
install(TARGETS lfs_python_runtime
    LIBRARY DESTINATION "${CMAKE_INSTALL_LIBDIR}"
    RUNTIME DESTINATION "${CMAKE_INSTALL_BINDIR}"  # Windows DLLs
)

# Install lfs_plugins Python package to bin for portable builds
if(BUILD_PORTABLE)
    install(DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}/lfs_plugins"
        DESTINATION "${CMAKE_INSTALL_BINDIR}"
        FILES_MATCHING PATTERN "*.py"
    )
    install(FILES "${CMAKE_CURRENT_SOURCE_DIR}/_lfs_panel_contract.py"
        DESTINATION "${CMAKE_INSTALL_BINDIR}"
    )
else()
    install(DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}/lfs_plugins"
        DESTINATION "${PYTHON_INSTALL_DIR}"
        FILES_MATCHING PATTERN "*.py"
    )
    install(FILES "${CMAKE_CURRENT_SOURCE_DIR}/_lfs_panel_contract.py"
        DESTINATION "${PYTHON_INSTALL_DIR}"
    )
endif()

# Stubs: use generated when available, otherwise committed from source tree
if(BUILD_PYTHON_STUBS)
    set(STUBS_DIR "${GENERATED_STUBS_DIR}")
else()
    set(STUBS_DIR "${CMAKE_CURRENT_SOURCE_DIR}/stubs")
endif()

if(BUILD_PORTABLE)
    install(DIRECTORY "${STUBS_DIR}/" DESTINATION "${CMAKE_INSTALL_BINDIR}" OPTIONAL)
else()
    install(DIRECTORY "${STUBS_DIR}/" DESTINATION "${PYTHON_INSTALL_DIR}" OPTIONAL)
endif()

# =============================================================================
# PORTABLE BUILD - Bundle vcpkg Python interpreter and standard library
# =============================================================================
if(BUILD_PORTABLE)
    message(STATUS "Portable build: bundling Python ${Python_VERSION}")

    # Get vcpkg Python directories
    get_filename_component(VCPKG_PYTHON_TOOLS_DIR "${Python_EXECUTABLE}" DIRECTORY)

    if(WIN32)
        # Windows: Python layout is tools/python3/{python.exe, Lib/, DLLs/, python312.dll}
        # Install to root Lib/ and DLLs/ to match getPythonHome() expectations
        set(VCPKG_PYTHON_STDLIB "${VCPKG_PYTHON_TOOLS_DIR}/Lib")
        set(VCPKG_PYTHON_DLLS "${VCPKG_PYTHON_TOOLS_DIR}/DLLs")
        set(PYTHON_INSTALL_STDLIB "Lib")

        # Install Python executable to bin/
        install(PROGRAMS "${Python_EXECUTABLE}"
            DESTINATION "${CMAKE_INSTALL_BINDIR}"
            COMPONENT runtime
        )

        # Install helper executables needed by Windows venv tooling.
        # Without these, uv/python may reject the interpreter for venv creation.
        set(PYTHON_HELPER_EXES
            "${VCPKG_PYTHON_TOOLS_DIR}/pythonw.exe"
            "${VCPKG_PYTHON_TOOLS_DIR}/venvlauncher.exe"
            "${VCPKG_PYTHON_TOOLS_DIR}/venvwlauncher.exe"
            "${VCPKG_PYTHON_TOOLS_DIR}/../bin/pythonw.exe"
            "${VCPKG_PYTHON_TOOLS_DIR}/../bin/venvlauncher.exe"
            "${VCPKG_PYTHON_TOOLS_DIR}/../bin/venvwlauncher.exe"
        )
        set(PYTHON_HELPER_INSTALL_LIST "")
        foreach(_helper IN LISTS PYTHON_HELPER_EXES)
            if(EXISTS "${_helper}")
                list(APPEND PYTHON_HELPER_INSTALL_LIST "${_helper}")
            endif()
        endforeach()
        if(PYTHON_HELPER_INSTALL_LIST)
            list(REMOVE_DUPLICATES PYTHON_HELPER_INSTALL_LIST)
            install(PROGRAMS ${PYTHON_HELPER_INSTALL_LIST}
                DESTINATION "${CMAKE_INSTALL_BINDIR}"
                COMPONENT runtime
            )
            message(STATUS "  Will bundle Python helper exes: ${PYTHON_HELPER_INSTALL_LIST}")
        else()
            message(WARNING "Python helper exes not found (pythonw/venvlauncher). "
                            "Windows portable plugin venv creation may fail.")
        endif()

        # Install python3.dll and python312.dll to bin/ (where exe is)
        file(GLOB PYTHON_DLLS "${VCPKG_PYTHON_TOOLS_DIR}/python*.dll")
        if(PYTHON_DLLS)
            install(FILES ${PYTHON_DLLS}
                DESTINATION "${CMAKE_INSTALL_BINDIR}"
                COMPONENT runtime
            )
        endif()

        # Also check parent dir for python DLLs (vcpkg sometimes puts them there)
        file(GLOB PYTHON_DLLS_PARENT "${VCPKG_PYTHON_TOOLS_DIR}/../bin/python*.dll")
        if(PYTHON_DLLS_PARENT)
            install(FILES ${PYTHON_DLLS_PARENT}
                DESTINATION "${CMAKE_INSTALL_BINDIR}"
                COMPONENT runtime
            )
        endif()

        # Install Python standard library
        if(EXISTS "${VCPKG_PYTHON_STDLIB}")
            install(DIRECTORY "${VCPKG_PYTHON_STDLIB}/"
                DESTINATION "${PYTHON_INSTALL_STDLIB}"
                COMPONENT runtime
                PATTERN "__pycache__" EXCLUDE
                PATTERN "*.pyc" EXCLUDE
                PATTERN "test" EXCLUDE
                PATTERN "tests" EXCLUDE
                PATTERN "idle_test" EXCLUDE
                PATTERN "tkinter" EXCLUDE
                PATTERN "turtledemo" EXCLUDE
                PATTERN "idlelib" EXCLUDE
                PATTERN "lib2to3" EXCLUDE
                PATTERN "ensurepip" EXCLUDE
            )
            message(STATUS "  Will bundle Python stdlib from: ${VCPKG_PYTHON_STDLIB}")
        else()
            message(WARNING "Python stdlib not found at ${VCPKG_PYTHON_STDLIB}")
        endif()

        # Install DLLs directory (compiled extensions like _socket.pyd, etc.)
        if(EXISTS "${VCPKG_PYTHON_DLLS}")
            install(DIRECTORY "${VCPKG_PYTHON_DLLS}/"
                DESTINATION "DLLs"
                COMPONENT runtime
                PATTERN "__pycache__" EXCLUDE
            )
        endif()

        # Install python312._pth next to python312.dll in bin/ to isolate
        # the embedded interpreter from system Python packages (e.g. Anaconda).
        # This file overrides CPython's default path computation on Windows.
        install(CODE "
            file(WRITE \"\${CMAKE_INSTALL_PREFIX}/${CMAKE_INSTALL_BINDIR}/python312._pth\"
                \".\\n../Lib\\n../DLLs\\nimport site\\n\")
            message(STATUS \"Installed python312._pth for path isolation\")
        " COMPONENT runtime)

    else()
        # Linux: Python layout is tools/python3/python3.12, lib/python3.12/
        set(VCPKG_PYTHON_STDLIB "${VCPKG_PYTHON_TOOLS_DIR}/../../lib/python3.12")
        get_filename_component(VCPKG_PYTHON_STDLIB "${VCPKG_PYTHON_STDLIB}" ABSOLUTE)
        set(PYTHON_INSTALL_STDLIB "${CMAKE_INSTALL_LIBDIR}/python3.12")

        # Install Python executable
        install(PROGRAMS "${Python_EXECUTABLE}"
            DESTINATION "${CMAKE_INSTALL_BINDIR}"
            RENAME "python3"
            COMPONENT runtime
        )

        # Create python3.12 symlink
        install(CODE "
            execute_process(COMMAND \${CMAKE_COMMAND} -E create_symlink
                python3 \"\${CMAKE_INSTALL_PREFIX}/${CMAKE_INSTALL_BINDIR}/python3.12\")
        " COMPONENT runtime)

        # Install libpython if it exists (shared library builds)
        file(GLOB PYTHON_SHARED_LIBS "${VCPKG_PYTHON_TOOLS_DIR}/../../lib/libpython3.12.so*")
        if(PYTHON_SHARED_LIBS)
            install(FILES ${PYTHON_SHARED_LIBS}
                DESTINATION "${CMAKE_INSTALL_LIBDIR}"
                COMPONENT runtime
            )
        endif()

        # Install Python standard library
        if(EXISTS "${VCPKG_PYTHON_STDLIB}")
            install(DIRECTORY "${VCPKG_PYTHON_STDLIB}/"
                DESTINATION "${PYTHON_INSTALL_STDLIB}"
                COMPONENT runtime
                PATTERN "__pycache__" EXCLUDE
                PATTERN "*.pyc" EXCLUDE
                PATTERN "test" EXCLUDE
                PATTERN "tests" EXCLUDE
                PATTERN "idle_test" EXCLUDE
                PATTERN "tkinter" EXCLUDE
                PATTERN "turtledemo" EXCLUDE
                PATTERN "idlelib" EXCLUDE
                PATTERN "lib2to3" EXCLUDE
                PATTERN "ensurepip" EXCLUDE
                PATTERN "unittest" EXCLUDE
                PATTERN "lib-dynload" EXCLUDE
            )

            # Install lib-dynload separately (compiled extensions)
            if(EXISTS "${VCPKG_PYTHON_STDLIB}/lib-dynload")
                install(DIRECTORY "${VCPKG_PYTHON_STDLIB}/lib-dynload"
                    DESTINATION "${PYTHON_INSTALL_STDLIB}"
                    COMPONENT runtime
                )
            endif()
        endif()
    endif()

    message(STATUS "  Python executable: ${Python_EXECUTABLE}")
    message(STATUS "  Python stdlib: ${VCPKG_PYTHON_STDLIB}")
endif()
