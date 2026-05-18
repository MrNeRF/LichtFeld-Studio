# SPDX-FileCopyrightText: 2025 LichtFeld Studio Authors
#
# SPDX-License-Identifier: GPL-3.0-or-later
#
# SetupNativeFileDialog.cmake
#
# Resolves the `nfd::nfd` link target with a platform-appropriate backend.
#
#   Windows / macOS
#     The vcpkg-built `nativefiledialog-extended` package is used as-is.
#     Backend is Win32 IFileDialog / Cocoa NSOpenPanel, which honor the
#     caller-supplied default folder reliably.
#
#   Linux
#     NFD upstream source is fetched (hermetic SHA512-pinned tarball) and
#     built as a subproject with `NFD_PORTAL=OFF`, so the library links
#     directly to GTK3 instead of going through xdg-desktop-portal. The
#     portal backend frequently has its `current_folder` hint silently
#     dropped by xdg-desktop-portal-gnome, which makes pickers open at
#     "Recent" regardless of the path the application passes. GTK's
#     GtkFileChooser honors `gtk_file_chooser_set_current_folder()`
#     deterministically, which is exactly the contract every caller
#     downstream of `nfd::nfd` already assumes.
#
# Post-condition: `nfd::nfd` is available as a link target.

include_guard(GLOBAL)

if(WIN32 OR APPLE)
    find_package(nfd CONFIG REQUIRED)
    return()
endif()

# ---------------------------------------------------------------------------
# Linux preflight: confirm GTK3 development files are available.
#
# NFD's GTK backend calls `pkg_check_modules(GTK3 REQUIRED gtk+-3.0)` during
# its own configure step. We probe first so the diagnostic points at the
# real fix (install libgtk-3-dev) rather than at a CMake error inside a
# FetchContent subproject.
# ---------------------------------------------------------------------------

find_package(PkgConfig REQUIRED)
# IMPORTED_TARGET so callers can link `PkgConfig::LFS_GTK3` directly for the
# folder picker, which sidesteps NFD for the SELECT_FOLDER path (NFD returns
# the highlighted item; we want the currently-open folder).
pkg_check_modules(LFS_GTK3 QUIET IMPORTED_TARGET gtk+-3.0)
if(NOT LFS_GTK3_FOUND)
    message(FATAL_ERROR
        "nativefiledialog-extended (Linux) requires GTK3 development files.\n"
        "\n"
        "pkg-config could not locate `gtk+-3.0`. Install the dev headers and re-run CMake.\n"
        "\n"
        "  Debian/Ubuntu:  sudo apt install libgtk-3-dev\n"
        "  Fedora/RHEL:    sudo dnf install gtk3-devel\n"
        "  Arch:           sudo pacman -S gtk3\n"
        "  openSUSE:       sudo zypper install gtk3-devel\n"
        "\n"
        "Why this is required:\n"
        "  We build NFD's GTK backend instead of its xdg-desktop-portal backend so file\n"
        "  and folder pickers reliably open at the directory the application requests.\n"
        "  The portal backend (vcpkg's default for `nativefiledialog-extended`) often\n"
        "  ignores `current_folder` on GNOME and lands the user on Recent."
    )
endif()
message(STATUS "nativefiledialog-extended: using GTK ${LFS_GTK3_VERSION} backend (portal disabled)")

# ---------------------------------------------------------------------------
# Fetch NFD upstream and build it with the GTK backend.
#
# Pin matches the vcpkg port's pin for `nativefiledialog-extended` 1.3.0;
# the URL formula and tarball are identical, so the SHA512 carries over.
# ---------------------------------------------------------------------------

include(FetchContent)

set(NFD_BUILD_TESTS OFF CACHE BOOL "" FORCE)
set(NFD_PORTAL OFF CACHE BOOL "" FORCE)
set(NFD_INSTALL OFF CACHE BOOL "" FORCE)

FetchContent_Declare(
    nativefiledialog_extended
    URL https://github.com/btzy/nativefiledialog-extended/archive/v1.3.0.tar.gz
    URL_HASH SHA512=1f2e17dd9ee5b416dfe1362b6eac6499c83c527a83478361769420f1d29bf21e0a81e4b6d45255703aba9be61c8379f7745fe182d74687a9c4f3309bd4fdf09e
    EXCLUDE_FROM_ALL
)
FetchContent_MakeAvailable(nativefiledialog_extended)

# Insulate NFD from project-wide `-Werror` policies; this is third-party code.
if(TARGET nfd)
    set_target_properties(nfd PROPERTIES COMPILE_WARNING_AS_ERROR OFF)
endif()
