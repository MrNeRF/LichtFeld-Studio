vcpkg_from_github(
    OUT_SOURCE_PATH SOURCE_PATH
    REPO haasn/libplacebo
    REF v7.360.1
    SHA512 209B1713CFF34F06149AF16FB3EA52E3662A566EF5DF6B29811AD295AA8CB6388F827A93FC8E0EED1A72F35B3B3AAE835520C933079E706A51D11136A8128799
    HEAD_REF master
)

# The upstream source archive intentionally excludes its Git submodules.  Two
# of them are required at build time to generate libplacebo's shader sources.
# Fetch precisely the revisions pinned by libplacebo v7.360.1, rather than
# relying on a host Python installation or on network access during Meson.
vcpkg_from_github(
    OUT_SOURCE_PATH JINJA_SOURCE
    REPO pallets/jinja
    REF 15206881c006c79667fe5154fe80c01c65410679
    SHA512 E1082222A4660E60F05E970E7C5B6F2FAB377BA01C273BCB6FE0EAD457EA5D4764C1D95FB3264B6BC371E122D574517AC35B6AE3858B50BC4918ACD08A3F75DE
    HEAD_REF main
)
vcpkg_from_github(
    OUT_SOURCE_PATH MARKUPSAFE_SOURCE
    REPO pallets/markupsafe
    REF 297fc8e356e6836a62087949245d09a28e9f1b13
    SHA512 8E16146B42DE9F0939B706C1652D4C5FE8E67E1F7E0C5A0E37D698D9AB10DCADF3E26B12E4BE2B37209C33703996351B02C54AF7CEB2D9EAF24AEDE7CECDF648
    HEAD_REF main
)
file(COPY "${JINJA_SOURCE}/src/"
     DESTINATION "${SOURCE_PATH}/3rdparty/jinja/src")
file(COPY "${MARKUPSAFE_SOURCE}/src/"
     DESTINATION "${SOURCE_PATH}/3rdparty/markupsafe/src")

# libplacebo v7 is written in GNU C (notably, its GPU callback tables use
# __typeof__). Native MSVC parses those declarations incorrectly, so do not
# attempt to emulate the extensions with source patches. Use Visual Studio's
# clang-cl component only for this port: it targets the same MSVC ABI and CRT,
# while LichtFeld Studio and every other dependency retain their normal MSVC
# compiler configuration. Linux and MinGW use their native compiler unchanged.
if(VCPKG_TARGET_IS_WINDOWS AND NOT VCPKG_TARGET_IS_MINGW)
    # Builds are launched from the x64 Native Tools Command Prompt, which
    # supplies VSINSTALLDIR for the active Visual Studio installation.
    file(TO_CMAKE_PATH "$ENV{VSINSTALLDIR}" _lfs_vs_installation)
    set(_lfs_clang_cl "${_lfs_vs_installation}/VC/Tools/Llvm/x64/bin/clang-cl.exe")
    if(NOT EXISTS "${_lfs_clang_cl}")
        message(FATAL_ERROR
            "libplacebo on Windows requires clang-cl from Visual Studio. Run CMake "
            "from the x64 Native Tools Command Prompt and install the 'C++ Clang "
            "Compiler for Windows' and 'MSBuild support for LLVM (clang-cl) toolset' "
            "individual components.")
    endif()

    # vcpkg supplies the normal MSVC native file first. Meson merges this file
    # afterwards, replacing just C/C++ compiler selection while preserving the
    # target triplet's CRT, include paths and linker settings.
    set(_lfs_clang_native_file "${CURRENT_BUILDTREES_DIR}/clang-cl-${TARGET_TRIPLET}.ini")
    file(WRITE "${_lfs_clang_native_file}"
        "[binaries]\n"
        "c = ['${_lfs_clang_cl}']\n"
        "cpp = ['${_lfs_clang_cl}']\n")
    set(VCPKG_MESON_NATIVE_FILE_DEBUG "${_lfs_clang_native_file}")
    set(VCPKG_MESON_NATIVE_FILE_RELEASE "${_lfs_clang_native_file}")
endif()

# The project's pinned vcpkg baseline ships Meson 0.58, while current
# libplacebo requires >= 0.63. This helper replaces the Meson CMake helpers
# only while this overlay port is configured; it does not alter vcpkg or any
# other dependency's Meson version.
include("${CURRENT_HOST_INSTALLED_DIR}/share/vcpkg-tool-meson/vcpkg-port-config.cmake")

# Keep the package focused on the Vulkan renderer used by LichtFeld.  The
# internal Dolby Vision implementation is enabled; libdovi is optional and
# not required for the supported FFmpeg side-data mapping path.
set(LIBPLACEBO_MESON_OPTIONS
    -Dvulkan=enabled
    # The application creates pl_vulkan with default parameters. Keep the
    # Vulkan loader enabled so libplacebo can resolve vkGetInstanceProcAddr.
    -Dvk-proc-addr=enabled
    -Dshaderc=enabled
    -Dglslang=disabled
    -Ddovi=enabled
    -Dlibdovi=disabled
    -Dopengl=disabled
    -Dd3d11=disabled
    -Dlcms=disabled
    -Dxxhash=disabled
    -Ddemos=false
    -Dtests=false
    -Dbench=false
    -Dfuzz=false
)

if(VCPKG_TARGET_IS_WINDOWS AND NOT VCPKG_TARGET_IS_MINGW)
    list(APPEND LIBPLACEBO_MESON_OPTIONS
        "-Dvulkan-registry=${CURRENT_INSTALLED_DIR}/share/vulkan/registry/vk.xml")
endif()

vcpkg_configure_meson(
    SOURCE_PATH "${SOURCE_PATH}"
    OPTIONS
        ${LIBPLACEBO_MESON_OPTIONS}
)

vcpkg_install_meson()
vcpkg_copy_pdbs()
vcpkg_fixup_pkgconfig()

file(REMOVE_RECURSE "${CURRENT_PACKAGES_DIR}/debug/include"
                    "${CURRENT_PACKAGES_DIR}/debug/share")

file(INSTALL "${CMAKE_CURRENT_LIST_DIR}/usage"
     DESTINATION "${CURRENT_PACKAGES_DIR}/share/${PORT}")
vcpkg_install_copyright(FILE_LIST "${SOURCE_PATH}/LICENSE")
