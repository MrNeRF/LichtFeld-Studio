set(_lfs_sdl3_x11_apt_packages
    "libx11-dev libxext-dev libxcursor-dev libxi-dev libxfixes-dev libxrandr-dev libxrender-dev libxss-dev libxtst-dev")
set(_lfs_sdl3_wayland_apt_packages
    "libwayland-dev libwayland-bin wayland-protocols libxkbcommon-dev libegl-dev")

function(_lfs_find_library out_var)
    find_library(_lfs_found_library NAMES ${ARGN})
    set(${out_var} "${_lfs_found_library}" PARENT_SCOPE)
endfunction()

function(_lfs_find_header out_var)
    find_path(_lfs_found_header_dir NAMES ${ARGN})
    set(${out_var} "${_lfs_found_header_dir}" PARENT_SCOPE)
endfunction()

function(_lfs_append_missing_library list_var display_name)
    _lfs_find_library(_lfs_library ${ARGN})
    if(NOT _lfs_library)
        set(_lfs_missing_items "${${list_var}}")
        list(APPEND _lfs_missing_items "${display_name}")
        set(${list_var} "${_lfs_missing_items}" PARENT_SCOPE)
    endif()
endfunction()

function(_lfs_require_sdl3_linux_desktop_deps)
    if(NOT VCPKG_TARGET_IS_LINUX)
        return()
    endif()

    if("x11" IN_LIST FEATURES)
        set(_x11_missing_items)

        foreach(_header IN ITEMS
                X11/Xlib.h
                X11/extensions/Xext.h
                X11/Xcursor/Xcursor.h
                X11/extensions/XInput2.h
                X11/extensions/Xfixes.h
                X11/extensions/Xrandr.h
                X11/extensions/Xrender.h
                X11/extensions/scrnsaver.h
                X11/extensions/XTest.h
                X11/extensions/Xdbe.h
                X11/extensions/shape.h
                X11/extensions/sync.h)
            _lfs_find_header(_lfs_header_dir "${_header}")
            if(NOT _lfs_header_dir)
                list(APPEND _x11_missing_items "${_header}")
            endif()
        endforeach()

        _lfs_append_missing_library(_x11_missing_items "libX11" X11)
        _lfs_append_missing_library(_x11_missing_items "libXext" Xext)
        _lfs_append_missing_library(_x11_missing_items "libXcursor" Xcursor)
        _lfs_append_missing_library(_x11_missing_items "libXi" Xi)
        _lfs_append_missing_library(_x11_missing_items "libXfixes" Xfixes)
        _lfs_append_missing_library(_x11_missing_items "libXrandr" Xrandr)
        _lfs_append_missing_library(_x11_missing_items "libXrender" Xrender)
        _lfs_append_missing_library(_x11_missing_items "libXss" Xss)
        _lfs_append_missing_library(_x11_missing_items "libXtst" Xtst)

        if(_x11_missing_items)
            list(REMOVE_DUPLICATES _x11_missing_items)
            string(JOIN ", " _x11_missing_text ${_x11_missing_items})
            message(FATAL_ERROR
                "SDL3 was requested with the 'x11' feature, but the required X11 development packages were not found before SDL3 configuration.\n"
                "\n"
                "Missing headers/libraries: ${_x11_missing_text}\n"
                "\n"
                "For Debian/Ubuntu/Pop!_OS install:\n"
                "  sudo apt install ${_lfs_sdl3_x11_apt_packages}\n"
                "\n"
                "Then re-run configuration so vcpkg rebuilds SDL3 with X11 support."
            )
        endif()
    endif()

    if("wayland" IN_LIST FEATURES)
        set(_wayland_missing_items)

        foreach(_header IN ITEMS
                wayland-client.h
                wayland-egl.h
                wayland-cursor.h
                xkbcommon/xkbcommon.h
                EGL/egl.h)
            _lfs_find_header(_lfs_header_dir "${_header}")
            if(NOT _lfs_header_dir)
                list(APPEND _wayland_missing_items "${_header}")
            endif()
        endforeach()

        _lfs_append_missing_library(_wayland_missing_items "libwayland-client" wayland-client)
        _lfs_append_missing_library(_wayland_missing_items "libwayland-egl" wayland-egl)
        _lfs_append_missing_library(_wayland_missing_items "libwayland-cursor" wayland-cursor)
        _lfs_append_missing_library(_wayland_missing_items "libxkbcommon" xkbcommon)
        _lfs_append_missing_library(_wayland_missing_items "libEGL" EGL)

        find_program(_lfs_wayland_scanner NAMES wayland-scanner)
        if(NOT _lfs_wayland_scanner)
            list(APPEND _wayland_missing_items wayland-scanner)
        endif()

        if(_wayland_missing_items)
            list(REMOVE_DUPLICATES _wayland_missing_items)
            string(JOIN ", " _wayland_missing_text ${_wayland_missing_items})
            message(FATAL_ERROR
                "SDL3 was requested with the 'wayland' feature, but the required Wayland development packages were not found before SDL3 configuration.\n"
                "\n"
                "Missing headers/libraries/tools: ${_wayland_missing_text}\n"
                "\n"
                "For Debian/Ubuntu/Pop!_OS install:\n"
                "  sudo apt install ${_lfs_sdl3_wayland_apt_packages}\n"
                "\n"
                "Optional extras:\n"
                "  sudo apt install libdecor-0-dev\n"
                "  sudo apt install libdrm-dev libgbm-dev\n"
                "\n"
                "If both the X11 and Wayland package sets are installed before SDL3 is configured, SDL3 should be built with support for both backends. "
                "Then re-run configuration so vcpkg rebuilds SDL3 with Wayland support."
            )
        endif()
    endif()
endfunction()

_lfs_require_sdl3_linux_desktop_deps()

include("${VCPKG_ROOT_DIR}/ports/sdl3/portfile.cmake")
