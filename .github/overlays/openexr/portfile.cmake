vcpkg_from_github(
    OUT_SOURCE_PATH SOURCE_PATH
    REPO AcademySoftwareFoundation/openexr
    REF "v3.4.15"
    SHA512 bdb45dc5af9669f845ec2c5c7cdda45ba46d8e178eb1a2754b8737944c9baef13a48e52e524c6bddefbf3aac770f899fb38147e26059a08dd70fdab879e14ed0
    HEAD_REF main
    PATCHES core-only.patch
)

vcpkg_check_features(OUT_FEATURE_OPTIONS OPTIONS
    FEATURES cpp OPENEXR_BUILD_CPP_LIBS
)
vcpkg_cmake_configure(
    SOURCE_PATH "${SOURCE_PATH}"
    OPTIONS
        ${OPTIONS}
        -DBUILD_TESTING=OFF
        -DBUILD_WEBSITE=OFF
        -DOPENEXR_BUILD_EXAMPLES=OFF
        -DOPENEXR_BUILD_TOOLS=OFF
        -DOPENEXR_INSTALL_TOOLS=OFF
        -DOPENEXR_INSTALL_PKG_CONFIG=ON
        -DCMAKE_REQUIRE_FIND_PACKAGE_libdeflate=ON
        -DCMAKE_REQUIRE_FIND_PACKAGE_openjph=ON
)
vcpkg_cmake_install()
vcpkg_copy_pdbs()
vcpkg_cmake_config_fixup(CONFIG_PATH lib/cmake/OpenEXR)
vcpkg_fixup_pkgconfig()
file(REMOVE_RECURSE "${CURRENT_PACKAGES_DIR}/debug/include" "${CURRENT_PACKAGES_DIR}/debug/share")
vcpkg_install_copyright(FILE_LIST "${SOURCE_PATH}/LICENSE.md" "${SOURCE_PATH}/PATENTS")
