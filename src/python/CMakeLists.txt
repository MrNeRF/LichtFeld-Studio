#SPDX - FileCopyrightText : 2025 LichtFeld Studio Authors
#
#SPDX - License - Identifier : GPL - 3.0 - or -later

message(STATUS "Configuring Python components (runner + nanobind module)")

#Python for interpreter + embedding + nanobind
find_package(Python REQUIRED COMPONENTS Interpreter Development Development.Module Development.SABIModule)
find_package(nanobind CONFIG REQUIRED)

# vcpkg extension modules use release ABI; debug libpython has Py_DEBUG which
# changes _PyRuntimeState layout, causing ABI mismatch crashes.
get_target_property(_py_debug_lib Python::Python IMPORTED_LOCATION_DEBUG)
if(_py_debug_lib AND _py_debug_lib MATCHES "python3\\.([0-9]+)d\\.a$")
    # Linux: libpython3.XXd.a -> libpython3.XX.a
    string(REGEX REPLACE "debug/lib/libpython3\\.([0-9]+)d\\.a$"
                         "lib/libpython3.\\1.a" _py_release_lib "${_py_debug_lib}")
    if(EXISTS "${_py_release_lib}")
        set_target_properties(Python::Python PROPERTIES
            IMPORTED_LOCATION_DEBUG "${_py_release_lib}")
        message(STATUS "Python: forced release ABI for debug build (Linux)")
    endif()
elseif(_py_debug_lib AND _py_debug_lib MATCHES "python3([0-9]+)_?d\\.lib$")
    # Windows: python312_d.lib or python312d.lib -> python312.lib
    string(REGEX REPLACE "python3([0-9]+)_?d\\.lib$"
                         "python3\\1.lib" _py_release_lib "${_py_debug_lib}")
    # Also check debug/ -> lib/ path variant
    string(REPLACE "/debug/lib/" "/lib/" _py_release_lib_alt "${_py_release_lib}")
    if(EXISTS "${_py_release_lib}")
        set(_py_win_release "${_py_release_lib}")
    elseif(EXISTS "${_py_release_lib_alt}")
        set(_py_win_release "${_py_release_lib_alt}")
    endif()
    if(_py_win_release)
        set_target_properties(Python::Python PROPERTIES
            IMPORTED_LOCATION_DEBUG "${_py_win_release}")
        # Override import library for Windows linker
        get_target_property(_py_debug_implib Python::Python IMPORTED_IMPLIB_DEBUG)
        if(_py_debug_implib)
            string(REGEX REPLACE "python3([0-9]+)_?d\\.lib$"
                                 "python3\\1.lib" _py_release_implib "${_py_debug_implib}")
            string(REPLACE "/debug/lib/" "/lib/" _py_release_implib_alt "${_py_release_implib}")
            if(EXISTS "${_py_release_implib}")
                set_target_properties(Python::Python PROPERTIES
                    IMPORTED_IMPLIB_DEBUG "${_py_release_implib}")
            elseif(EXISTS "${_py_release_implib_alt}")
                set_target_properties(Python::Python PROPERTIES
                    IMPORTED_IMPLIB_DEBUG "${_py_release_implib_alt}")
            endif()
        endif()
        message(STATUS "Python: forced release ABI for debug build (Windows)")
    endif()
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
    RUNTIME_OUTPUT_DIRECTORY_DEBUG ${CMAKE_BINARY_DIR}
    RUNTIME_OUTPUT_DIRECTORY_RELEASE ${CMAKE_BINARY_DIR}
    RUNTIME_OUTPUT_DIRECTORY_RELWITHDEBINFO ${CMAKE_BINARY_DIR}
    RUNTIME_OUTPUT_DIRECTORY_MINSIZEREL ${CMAKE_BINARY_DIR}
)

# Static library for other Python utilities
add_library(lfs_python_utils STATIC
    runner.cpp
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
)

target_compile_definitions(lfs_python_utils
    PUBLIC
        LFS_PYTHON_EXECUTABLE="${Python_EXECUTABLE}"
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
    lfs/openmesh/handles.cpp
    lfs/openmesh/io.cpp
    lfs/py_rendering.cpp
    lfs/py_selection.cpp
    lfs/py_operator.cpp
    lfs/py_pipeline.cpp
    lfs/py_prop_registry.cpp
    lfs/py_signals.cpp
)

set_target_properties(lfs_py PROPERTIES
    CUDA_ARCHITECTURES ""
    CUDA_SEPARABLE_COMPILATION OFF
    LINKER_LANGUAGE CXX
    POSITION_INDEPENDENT_CODE ON
    OUTPUT_NAME lichtfeld
)

target_compile_definitions(lfs_py PRIVATE
    LFS_PYTHON_EXECUTABLE="${Python_EXECUTABLE}"
)

target_link_libraries(lfs_py PRIVATE
    lfs_core
    lfs_event_bridge
    lfs_geometry
    lfs_visualizer
    lfs_mcp
    lfs_python_runtime
    lfs_training
    lfs_io
    lfs_rendering
    $<IF:$<PLATFORM_ID:Windows>,OpenMeshCore,OpenMeshCoreStatic>
    $<IF:$<PLATFORM_ID:Windows>,OpenMeshTools,OpenMeshToolsStatic>
    lfs_sequencer
    OpenImageIO::OpenImageIO
    CUDA::cudart
    CUDA::curand
    imgui::imgui
    implot::implot
    glfw
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
        COMMAND ${CMAKE_COMMAND} -E copy_directory
            "${CMAKE_BINARY_DIR}/vcpkg_installed/x64-windows/bin"
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
if(BUILD_PYTHON_STUBS)
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
import os, sys

# Add DLL directories (Python 3.8+ requirement on Windows)
dll_dirs = [
    r'$<TARGET_FILE_DIR:lfs_python_runtime>',
    r'${CMAKE_BINARY_DIR}/vcpkg_installed/x64-windows/bin',
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
            '-M', r'${CMAKE_CURRENT_BINARY_DIR}/typings/lichtfeld/py.typed',
            '-O', r'${CMAKE_CURRENT_BINARY_DIR}/typings',
            '-r']
runpy.run_path(r'${NB_STUBGEN}', run_name='__main__')

# Normalize CRLF -> LF so git doesn't warn about line endings
import glob, pathlib
for pyi in glob.glob(r'${CMAKE_CURRENT_BINARY_DIR}/typings/**/*.pyi', recursive=True):
    p = pathlib.Path(pyi)
    content = p.read_bytes()
    if b'\\r\\n' in content:
        p.write_bytes(content.replace(b'\\r\\n', b'\\n'))
")

            add_custom_command(
                OUTPUT ${CMAKE_CURRENT_BINARY_DIR}/typings/lichtfeld/py.typed
                COMMAND "${Python_EXECUTABLE}" "${STUBGEN_WRAPPER}"
                WORKING_DIRECTORY "${CMAKE_CURRENT_BINARY_DIR}"
                DEPENDS lfs_py lfs_python_runtime "${NB_STUBGEN}"
                COMMENT "Generating Python stubs for lichtfeld (Windows)"
            )
            add_custom_target(lichtfeld_stub ALL DEPENDS ${CMAKE_CURRENT_BINARY_DIR}/typings/lichtfeld/py.typed)
        endif()
    else()
        nanobind_add_stub(lichtfeld_stub
            MODULE lichtfeld
            OUTPUT_PATH ${CMAKE_CURRENT_BINARY_DIR}/typings
            PYTHON_PATH ${CMAKE_CURRENT_BINARY_DIR}
            DEPENDS lfs_py
            RECURSIVE
            MARKER_FILE ${CMAKE_CURRENT_BINARY_DIR}/typings/lichtfeld/py.typed
        )
    endif()

    add_custom_command(TARGET lichtfeld_stub POST_BUILD
        COMMAND ${CMAKE_COMMAND} -E copy_directory
            "${CMAKE_CURRENT_BINARY_DIR}/typings/"
            "${CMAKE_CURRENT_SOURCE_DIR}/stubs/"
        COMMENT "Syncing stubs to source tree"
    )
endif()

# Copy lfs_plugins Python module to build directory (recursively to include subdirectories)
add_custom_target(lfs_plugins_copy ALL
    COMMAND ${CMAKE_COMMAND} -E copy_directory ${CMAKE_CURRENT_SOURCE_DIR}/lfs_plugins ${CMAKE_CURRENT_BINARY_DIR}/lfs_plugins
    COMMENT "Copying lfs_plugins Python module"
)
add_dependencies(lfs_py lfs_plugins_copy)

# UV package manager for Python package installation support
include(${CMAKE_SOURCE_DIR}/cmake/FetchUV.cmake)
fetch_uv()
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
else()
    install(DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}/lfs_plugins"
        DESTINATION "${PYTHON_INSTALL_DIR}"
        FILES_MATCHING PATTERN "*.py"
    )
endif()

# Stubs: use generated when available, otherwise committed from source tree
if(BUILD_PYTHON_STUBS)
    set(STUBS_DIR "${CMAKE_CURRENT_BINARY_DIR}/typings")
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
                PATTERN "unittest" EXCLUDE
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
