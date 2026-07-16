vcpkg_from_github(
    OUT_SOURCE_PATH SOURCE_PATH
    REPO haasn/libplacebo
    REF v7.360.1
    SHA512 209B1713CFF34F06149AF16FB3EA52E3662A566EF5DF6B29811AD295AA8CB6388F827A93FC8E0EED1A72F35B3B3AAE835520C933079E706A51D11136A8128799
    HEAD_REF master
)

# Vendor the pinned shader-generation sources omitted from libplacebo's archive.
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

# libplacebo uses GNU C extensions; use clang-cl only for this Windows port.
if(VCPKG_TARGET_IS_WINDOWS AND NOT VCPKG_TARGET_IS_MINGW)
    # VSINSTALLDIR is supplied by the x64 Native Tools Command Prompt.
    file(TO_CMAKE_PATH "$ENV{VSINSTALLDIR}" _lfs_vs_installation)
    set(_lfs_clang_cl "${_lfs_vs_installation}/VC/Tools/Llvm/x64/bin/clang-cl.exe")
    if(NOT EXISTS "${_lfs_clang_cl}")
        message(FATAL_ERROR
            "libplacebo on Windows requires clang-cl from Visual Studio. Run CMake "
            "from the x64 Native Tools Command Prompt and install the 'C++ Clang "
            "Compiler for Windows' and 'MSBuild support for LLVM (clang-cl) toolset' "
            "individual components.")
    endif()

    # Override only Meson's C/C++ compiler; keep vcpkg's MSVC settings.
    set(_lfs_clang_native_file "${CURRENT_BUILDTREES_DIR}/clang-cl-${TARGET_TRIPLET}.ini")
    file(WRITE "${_lfs_clang_native_file}"
        "[binaries]\n"
        "c = ['${_lfs_clang_cl}']\n"
        "cpp = ['${_lfs_clang_cl}']\n")
    set(VCPKG_MESON_NATIVE_FILE_DEBUG "${_lfs_clang_native_file}")
    set(VCPKG_MESON_NATIVE_FILE_RELEASE "${_lfs_clang_native_file}")
endif()

# Use the overlay's newer Meson helper without changing other ports.
include("${CURRENT_HOST_INSTALLED_DIR}/share/vcpkg-tool-meson/vcpkg-port-config.cmake")

# Build only the Vulkan renderer and built-in Dolby Vision support used here.
set(LIBPLACEBO_MESON_OPTIONS
    -Dvulkan=enabled
    # Required when pl_vulkan resolves Vulkan entry points internally.
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

set(_lfs_vulkan_registry "${CURRENT_INSTALLED_DIR}/share/vulkan/registry/vk.xml")
if(EXISTS "${_lfs_vulkan_registry}")
    list(APPEND LIBPLACEBO_MESON_OPTIONS
        "-Dvulkan-registry=${_lfs_vulkan_registry}")
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
