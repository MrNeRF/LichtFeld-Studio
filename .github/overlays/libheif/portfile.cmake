vcpkg_from_github(
    OUT_SOURCE_PATH SOURCE_PATH
    REPO  strukturag/libheif
    REF "v${VERSION}"
    SHA512 c953cb9c3a4c0f1052f4d1e0475143cfea6c0a1ab787acb33a2fd173e0460b665717a56a99b864226dfe62bd139c92a66fe5dec75b91e2a333b77a0571e3c5a8
    HEAD_REF master
    PATCHES
        cxx-linkage-pkgconfig.diff
        find-modules.diff
        gdk-pixbuf.patch
        symbol-exports.diff
        aom-reinitialization.patch
        context-allocation.patch
        metadata-inflate.patch
)

vcpkg_find_acquire_program(PKGCONFIG)
set(ENV{PKG_CONFIG} "${PKGCONFIG}")
vcpkg_check_features(OUT_FEATURE_OPTIONS FEATURE_OPTIONS
    FEATURES avif-encode WITH_AOM_ENCODER)

vcpkg_cmake_configure(
    SOURCE_PATH "${SOURCE_PATH}"
    OPTIONS
        ${FEATURE_OPTIONS}
        -DBUILD_TESTING=OFF
        -DWITH_HEADER_COMPRESSION=ON
        -DCMAKE_REQUIRE_FIND_PACKAGE_ZLIB=ON
        -DCMAKE_DISABLE_FIND_PACKAGE_Brotli=ON
        -DCMAKE_COMPILE_WARNING_AS_ERROR=OFF
        -DPLUGIN_DIRECTORY=  # empty
        -DWITH_DAV1D=ON
        -DWITH_DAV1D_PLUGIN=OFF
        -DENABLE_PLUGIN_LOADING=OFF
        -DBUILD_DOCUMENTATION=OFF
        -DWITH_LIBDE265=OFF
        -DWITH_X265=OFF
        -DWITH_KVAZAAR=OFF
        -DWITH_FFMPEG_DECODER=OFF
        -DWITH_AOM_DECODER=OFF
        -DWITH_AOM_ENCODER_PLUGIN=OFF
        -DWITH_SvtEnc=OFF
        -DWITH_RAV1E=OFF
        -DWITH_UVG266=OFF
        -DWITH_VVDEC=OFF
        -DWITH_VVENC=OFF
        -DWITH_X264=OFF
        -DWITH_OpenH264_ENCODER=OFF
        -DWITH_JPEG_DECODER=OFF
        -DWITH_JPEG_ENCODER=OFF
        -DWITH_OpenJPEG_DECODER=OFF
        -DWITH_OpenJPEG_ENCODER=OFF
        -DWITH_OPENJPH_DECODER=OFF
        -DWITH_OPENJPH_ENCODER=OFF
        -DWITH_UNCOMPRESSED_CODEC=OFF
        -DWITH_WEBCODECS=OFF
        -DWITH_EXAMPLES=OFF
        -DWITH_EXAMPLE_HEIF_THUMB=OFF
        -DWITH_EXAMPLE_HEIF_VIEW=OFF
        -DWITH_GDK_PIXBUF=OFF
        -DWITH_LIBSHARPYUV=OFF
        -DWITH_OpenH264_DECODER=OFF
        -DVCPKG_LOCK_FIND_PACKAGE_Brotli=OFF
        -DVCPKG_LOCK_FIND_PACKAGE_Doxygen=OFF
        -DVCPKG_LOCK_FIND_PACKAGE_LIBDE265=OFF
        -DVCPKG_LOCK_FIND_PACKAGE_PNG=OFF
        -DVCPKG_LOCK_FIND_PACKAGE_TIFF=OFF
    OPTIONS_RELEASE
        "-DPLUGIN_INSTALL_DIRECTORY=${CURRENT_PACKAGES_DIR}/plugins/libheif"
    OPTIONS_DEBUG
        "-DPLUGIN_INSTALL_DIRECTORY=${CURRENT_PACKAGES_DIR}/debug/plugins/libheif"
    MAYBE_UNUSED_VARIABLES
        VCPKG_LOCK_FIND_PACKAGE_Brotli
        WITH_OpenH264_ENCODER
        WITH_WEBCODECS
)
vcpkg_cmake_install()
vcpkg_copy_pdbs()
vcpkg_cmake_config_fixup(CONFIG_PATH "lib/cmake/libheif")
set(heif_config "${CURRENT_PACKAGES_DIR}/share/libheif/libheif-config.cmake")
file(READ "${heif_config}" heif_config_text)
file(WRITE "${heif_config}"
    "include(CMakeFindDependencyMacro)\nfind_dependency(ZLIB)\n${heif_config_text}")
if("avif-encode" IN_LIST FEATURES)
    # Static exports refer to AOM::aom. Keep dependency discovery with the
    # provider package so every consumer gets a complete imported target.
    set(heif_config "${CURRENT_PACKAGES_DIR}/share/libheif/libheif-config.cmake")
    file(READ "${heif_config}" heif_config_text)
    file(WRITE "${heif_config}"
        "include(CMakeFindDependencyMacro)\nfind_dependency(AOM CONFIG HINTS \"\${CMAKE_CURRENT_LIST_DIR}/../aom\")\n${heif_config_text}")
endif()
vcpkg_fixup_pkgconfig()

if (VCPKG_LIBRARY_LINKAGE STREQUAL "dynamic")
    vcpkg_replace_string("${CURRENT_PACKAGES_DIR}/include/libheif/heif_library.h" "!defined(LIBHEIF_STATIC_BUILD)" "1")
else()
    vcpkg_replace_string("${CURRENT_PACKAGES_DIR}/include/libheif/heif_library.h" "!defined(LIBHEIF_STATIC_BUILD)" "0")
endif()

file(REMOVE_RECURSE "${CURRENT_PACKAGES_DIR}/debug/include")
file(REMOVE_RECURSE "${CURRENT_PACKAGES_DIR}/debug/share")
file(REMOVE_RECURSE "${CURRENT_PACKAGES_DIR}/lib/libheif" "${CURRENT_PACKAGES_DIR}/debug/lib/libheif")

file(GLOB maybe_plugins "${CURRENT_PACKAGES_DIR}/plugins/libheif/*")
if(maybe_plugins STREQUAL "")
    file(REMOVE_RECURSE "${CURRENT_PACKAGES_DIR}/plugins" "${CURRENT_PACKAGES_DIR}/debug/plugins")
endif()

vcpkg_install_copyright(FILE_LIST "${SOURCE_PATH}/COPYING")
