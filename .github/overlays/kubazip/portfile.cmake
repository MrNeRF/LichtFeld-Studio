# Same source as the registry port (release v0.3.15), fetched by commit id.
# The tag's generated archive no longer matches the checksum recorded in the
# registry, so a download by tag fails wherever the archive is not cached.
vcpkg_from_git(
    OUT_SOURCE_PATH SOURCE_PATH
    URL https://github.com/kuba--/zip.git
    REF 5f68005531b1505129baa27de2dfbf26c0d4172a
    PATCHES
        fix-name-conflict.diff
        disable-werror.patch
        fix-project-version.patch # https://github.com/kuba--/zip/commit/dd80aeab4293d1e11e31af20fabfd538d88625ef
)

vcpkg_cmake_configure(
    SOURCE_PATH "${SOURCE_PATH}"
    OPTIONS
        -DCMAKE_DISABLE_TESTING=ON
)

vcpkg_cmake_install()
vcpkg_cmake_config_fixup(CONFIG_PATH "lib/cmake/zip" PACKAGE_NAME "zip-kuba--")

if(VCPKG_LIBRARY_LINKAGE STREQUAL "dynamic")
    vcpkg_replace_string("${CURRENT_PACKAGES_DIR}/include/kubazip/zip/zip.h" "#ifndef ZIP_SHARED" "#if 0")
endif()

file(REMOVE_RECURSE "${CURRENT_PACKAGES_DIR}/debug/include")

# legacy polyfill
file(INSTALL "${CURRENT_PORT_DIR}/kubazipConfig.cmake" "${CURRENT_PORT_DIR}/usage" DESTINATION "${CURRENT_PACKAGES_DIR}/share/${PORT}")

vcpkg_install_copyright(
    FILE_LIST
        "${SOURCE_PATH}/LICENSE.txt"
        "${SOURCE_PATH}/src/miniz.h"
)
