# Replace the transitive zlib provider too, keeping a single ABI throughout vcpkg.
vcpkg_from_github(
    OUT_SOURCE_PATH SOURCE_PATH
    REPO zlib-ng/zlib-ng
    REF "2.3.3"
    SHA512 e2057c764f1d5aaee738edee7e977182c5b097e3c95489dcd8de813f237d92a05daaa86d68d44b331d9fec5d1802586a8f6cfb658ba849874aaa14e72a8107f5
    HEAD_REF develop
)
string(COMPARE EQUAL "${VCPKG_LIBRARY_LINKAGE}" "dynamic" BUILD_SHARED)
vcpkg_cmake_configure(
    SOURCE_PATH "${SOURCE_PATH}"
    OPTIONS
        -DZLIB_COMPAT=ON
        -DBUILD_SHARED_LIBS=${BUILD_SHARED}
        -DZLIB_ENABLE_TESTS=OFF
        -DWITH_GTEST=OFF
        -DWITH_NATIVE_INSTRUCTIONS=OFF
        -DINSTALL_UTILS=OFF
)
vcpkg_cmake_install()
vcpkg_copy_pdbs()
vcpkg_cmake_config_fixup(PACKAGE_NAME ZLIB CONFIG_PATH lib/cmake/ZLIB)
vcpkg_fixup_pkgconfig()
if(VCPKG_TARGET_IS_WINDOWS)
    if(BUILD_SHARED)
        set(PC_LIBRARY zlib)
    else()
        set(PC_LIBRARY zlibstatic)
    endif()
    vcpkg_replace_string("${CURRENT_PACKAGES_DIR}/lib/pkgconfig/zlib.pc" " -lz" " -l${PC_LIBRARY}")
    if(NOT VCPKG_BUILD_TYPE)
        vcpkg_replace_string("${CURRENT_PACKAGES_DIR}/debug/lib/pkgconfig/zlib.pc" " -lz" " -l${PC_LIBRARY}d")
    endif()
endif()
file(REMOVE_RECURSE "${CURRENT_PACKAGES_DIR}/debug/include" "${CURRENT_PACKAGES_DIR}/debug/share")
vcpkg_install_copyright(FILE_LIST "${SOURCE_PATH}/LICENSE.md")
