# SPDX-FileCopyrightText: 2025 LichtFeld Studio Authors
#
# SPDX-License-Identifier: GPL-3.0-or-later

include_guard(GLOBAL)

function(_lfs_pkg_config_module_available out_var module)
    execute_process(
        COMMAND "${LFS_PKG_CONFIG_EXECUTABLE}" --exists "${module}"
        RESULT_VARIABLE _pkg_result
        OUTPUT_QUIET
        ERROR_QUIET
    )

    set(${out_var} FALSE PARENT_SCOPE)
    if(_pkg_result EQUAL 0)
        set(${out_var} TRUE PARENT_SCOPE)
    endif()
endfunction()

function(_lfs_check_pkg_modules out_ready_var out_missing_var)
    set(_missing_modules)
    foreach(_module IN LISTS ARGN)
        _lfs_pkg_config_module_available(_module_available "${_module}")
        if(NOT _module_available)
            list(APPEND _missing_modules "${_module}")
        endif()
    endforeach()

    set(${out_ready_var} TRUE PARENT_SCOPE)
    if(_missing_modules)
        set(${out_ready_var} FALSE PARENT_SCOPE)
    endif()
    set(${out_missing_var} "${_missing_modules}" PARENT_SCOPE)
endfunction()

function(_lfs_format_list out_var)
    if(ARGN)
        list(JOIN ARGN ", " _joined)
        set(${out_var} "${_joined}" PARENT_SCOPE)
    else()
        set(${out_var} "none" PARENT_SCOPE)
    endif()
endfunction()

function(_lfs_append_target_runtime_dirs out_var target_name)
    if(NOT TARGET "${target_name}")
        set(${out_var} "${${out_var}}" PARENT_SCOPE)
        return()
    endif()

    set(_resolved_target "${target_name}")
    get_target_property(_aliased_target "${target_name}" ALIASED_TARGET)
    if(_aliased_target)
        set(_resolved_target "${_aliased_target}")
    endif()

    set(_runtime_dirs "${${out_var}}")
    foreach(_location_property
            IMPORTED_LOCATION
            IMPORTED_LOCATION_RELEASE
            IMPORTED_LOCATION_RELWITHDEBINFO
            IMPORTED_LOCATION_MINSIZEREL
            IMPORTED_LOCATION_DEBUG)
        get_target_property(_location "${_resolved_target}" "${_location_property}")
        if(_location AND EXISTS "${_location}")
            get_filename_component(_runtime_dir "${_location}" DIRECTORY)
            list(APPEND _runtime_dirs "${_runtime_dir}")
        endif()
    endforeach()

    list(REMOVE_DUPLICATES _runtime_dirs)
    set(${out_var} "${_runtime_dirs}" PARENT_SCOPE)
endfunction()

function(lfs_linux_preflight_gui_backends)
    if(NOT LFS_ENFORCE_LINUX_GUI_BACKENDS)
        return()
    endif()

    if(DEFINED CMAKE_SYSTEM_NAME)
        if(NOT CMAKE_SYSTEM_NAME STREQUAL "Linux")
            return()
        endif()
    elseif(NOT CMAKE_HOST_SYSTEM_NAME STREQUAL "Linux")
        return()
    endif()

    find_program(LFS_PKG_CONFIG_EXECUTABLE NAMES pkg-config)
    if(NOT LFS_PKG_CONFIG_EXECUTABLE)
        message(FATAL_ERROR
            "Linux GUI preflight failed: pkg-config is required before vcpkg resolves SDL3.\n"
            "Install pkg-config, then re-run CMake.\n"
            "To bypass this intentionally, configure with -DLFS_ENFORCE_LINUX_GUI_BACKENDS=OFF."
        )
    endif()

    find_program(LFS_WAYLAND_SCANNER_EXECUTABLE NAMES wayland-scanner)

    set(_x11_modules
        x11
        xext
    )
    set(_wayland_modules
        "wayland-client >= 1.18"
        wayland-egl
        wayland-cursor
        egl
        "xkbcommon >= 0.5.0"
    )

    _lfs_check_pkg_modules(_have_x11 _missing_x11 ${_x11_modules})
    _lfs_check_pkg_modules(_have_wayland _missing_wayland ${_wayland_modules})

    if(_have_wayland AND NOT LFS_WAYLAND_SCANNER_EXECUTABLE)
        set(_have_wayland FALSE)
        list(APPEND _missing_wayland wayland-scanner)
    endif()

    if(_have_x11 OR _have_wayland)
        set(_x11_status "missing")
        if(_have_x11)
            set(_x11_status "available")
        endif()

        set(_wayland_status "missing")
        if(_have_wayland)
            set(_wayland_status "available")
        endif()

        message(STATUS "Linux GUI preflight: X11=${_x11_status}, Wayland=${_wayland_status}")
        return()
    endif()

    _lfs_format_list(_missing_x11_text ${_missing_x11})
    _lfs_format_list(_missing_wayland_text ${_missing_wayland})

    message(FATAL_ERROR
        "Linux GUI preflight failed.\n"
        "\n"
        "SDL3's vcpkg port can build successfully without X11 or Wayland when the required system "
        "development files are missing, which produces a LichtFeld Studio binary that cannot open a GUI.\n"
        "\n"
        "Detected backend prerequisites:\n"
        "  X11: missing ${_missing_x11_text}\n"
        "  Wayland: missing ${_missing_wayland_text}\n"
        "\n"
        "Install at least one usable backend before configuring.\n"
        "Debian/Ubuntu examples:\n"
        "  X11 core: sudo apt install libx11-dev libxext-dev\n"
        "  Wayland core: sudo apt install libwayland-dev libxkbcommon-dev libegl-dev libegl1-mesa-dev\n"
        "\n"
        "To bypass this intentionally, configure with -DLFS_ENFORCE_LINUX_GUI_BACKENDS=OFF."
    )
endfunction()

function(lfs_validate_linux_sdl_video_backends)
    if(NOT LFS_ENFORCE_LINUX_GUI_BACKENDS OR NOT CMAKE_SYSTEM_NAME STREQUAL "Linux")
        return()
    endif()

    if(CMAKE_CROSSCOMPILING)
        message(STATUS "SDL3 video backend validation: skipped while cross-compiling")
        return()
    endif()

    include(CMakePushCheckState)
    include(CheckCXXSourceRuns)

    cmake_push_check_state(RESET)
    list(APPEND CMAKE_REQUIRED_LIBRARIES SDL3::SDL3)

    set(_sdl_runtime_dirs)
    _lfs_append_target_runtime_dirs(_sdl_runtime_dirs SDL3::SDL3)
    foreach(_runtime_dir IN LISTS _sdl_runtime_dirs)
        list(APPEND CMAKE_REQUIRED_LINK_OPTIONS "-Wl,-rpath,${_runtime_dir}")
    endforeach()

    unset(LFS_SDL3_HAS_LINUX_GUI_BACKEND CACHE)
    unset(LFS_SDL3_HAS_LINUX_GUI_BACKEND)

    check_cxx_source_runs(
        "
        #include <SDL3/SDL.h>
        #include <cstring>

        int main() {
            if (!SDL_Init(0)) {
                return 2;
            }

            bool has_gui_backend = false;
            const int num_drivers = SDL_GetNumVideoDrivers();
            for (int i = 0; i < num_drivers; ++i) {
                const char* driver = SDL_GetVideoDriver(i);
                if (!driver) {
                    continue;
                }

                if (std::strcmp(driver, \"x11\") == 0 || std::strcmp(driver, \"wayland\") == 0) {
                    has_gui_backend = true;
                    break;
                }
            }

            SDL_Quit();
            return has_gui_backend ? 0 : 1;
        }
        "
        LFS_SDL3_HAS_LINUX_GUI_BACKEND
    )

    cmake_pop_check_state()

    if(LFS_SDL3_HAS_LINUX_GUI_BACKEND)
        message(STATUS "SDL3 video backend validation: x11/wayland backend detected")
        return()
    endif()

    set(_triplet "${VCPKG_TARGET_TRIPLET}")
    if(NOT _triplet)
        set(_triplet "x64-linux")
    endif()

    message(FATAL_ERROR
        "SDL3 was found, but the resolved SDL build does not expose an X11 or Wayland video backend.\n"
        "\n"
        "This usually means vcpkg reused a stale cached SDL3 package built before the Linux GUI "
        "development files were installed.\n"
        "\n"
        "Recovery:\n"
        "  1. Remove the cached SDL3 package for this build:\n"
        "     cd \"$ENV{VCPKG_ROOT}\"\n"
        "     ./vcpkg remove sdl3:${_triplet} --recurse "
        "--x-install-root=\"${CMAKE_BINARY_DIR}/vcpkg_installed\" "
        "--x-packages-root=\"$ENV{VCPKG_ROOT}/packages\" "
        "--x-buildtrees-root=\"$ENV{VCPKG_ROOT}/buildtrees\"\n"
        "  2. Reconfigure with binary-cache bypass enabled:\n"
        "     VCPKG_BINARY_SOURCES='clear;default,write' cmake -B build -G Ninja --fresh\n"
        "\n"
        "To bypass this intentionally, configure with -DLFS_ENFORCE_LINUX_GUI_BACKENDS=OFF."
    )
endfunction()
