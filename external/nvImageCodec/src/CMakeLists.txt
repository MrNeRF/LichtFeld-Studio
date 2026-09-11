# SPDX-FileCopyrightText: Copyright (c) 2023 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

set(NVIMGCODEC_LIBRARY_NAME nvimgcodec)

add_subdirectory(imgproc)

set(NVIMGCODEC_SRCS
    ${CMAKE_CURRENT_SOURCE_DIR}/../include/nvimgcodec.h
    nvimgcodec_capi.cpp
    plugin_framework.cpp
    codec_registry.cpp
    codec.cpp
    image_decoder.cpp
    image_encoder.cpp
    image_parser.cpp
    code_stream.cpp
    file_io_stream.cpp
    std_file_io_stream.cpp
    image.cpp
    image_parser_factory.cpp
    image_encoder_factory.cpp
    image_decoder_factory.cpp
    image_generic_decoder.cpp
    image_generic_encoder.cpp
    nvimgcodec_director.cpp
    processing_results.cpp
    decode_state_batch.cpp
    encode_state_batch.cpp
    thread_pool.cpp
    default_executor.cpp
    builtin_modules.cpp
    parsers/exif.cpp
    parsers/jpeg.cpp
    parsers/jpeg2k.cpp
    parsers/parsers_ext_module.cpp
)

set(NVIMGCODEC_HEADERS
    builtin_modules.h
    code_stream.h
    codec.h
    codec_registry.h
    debug_messenger.h
    decode_state_batch.h
    default_debug_messenger.h
    default_executor.h
    directory_scaner.h
    encode_state_batch.h
    environment.h
    file_ext_codec.h
    file_io_stream.h
    host_dev.h
    icode_stream.h
    icodec.h
    icodec_registry.h
    idebug_messenger.h
    idecode_state.h
    idirectory_scaner.h
    iencode_state.h
    ienvironment.h
    iexecutor.h
    iimage.h
    iimage_decoder.h
    iimage_decoder_factory.h
    iimage_encoder.h
    iimage_encoder_factory.h
    iimage_parser.h
    iimage_parser_factory.h
    iiostream_factory.h
    ilibrary_loader.h
    ilogger.h
    image.h
    image_decoder.h
    image_decoder_factory.h
    image_encoder.h
    image_encoder_factory.h
    image_generic_codec.h
    image_generic_decoder.h
    image_generic_encoder.h
    image_parser.h
    image_parser_factory.h
    io_stream.h
    iostream_factory.h
    library_loader.h
    log.h
    log_ext.h
    logger.h
    mem_io_stream.h
    mmaped_file_io_stream.h
    nvimgcodec_director.h
    nvimgcodec_type_utils.h
    plugin_framework.h
    processing_results.h
    std_file_io_stream.h
    thread_pool.h
    user_executor.h
)

if(UNIX)
  list(APPEND NVIMGCODEC_SRCS mmaped_file_io_stream.cpp)
endif()

# Build the library
message(STATUS "props ProductVersionFiles ${ProductVersionFiles}")

add_library(${NVIMGCODEC_LIBRARY_NAME}_obj OBJECT ${NVIMGCODEC_SRCS} ${NVIMGCODEC_HEADERS} ${ProductVersionFiles})
add_library(${NVIMGCODEC_LIBRARY_NAME} SHARED $<TARGET_OBJECTS:${NVIMGCODEC_LIBRARY_NAME}_obj> $<TARGET_OBJECTS:nvimgcodec_imgproc_static>)
add_library(${NVIMGCODEC_LIBRARY_NAME}_static STATIC $<TARGET_OBJECTS:${NVIMGCODEC_LIBRARY_NAME}_obj> $<TARGET_OBJECTS:nvimgcodec_imgproc_static>)
target_compile_definitions(${NVIMGCODEC_LIBRARY_NAME}_obj PRIVATE
    "NVIMGCODEC_DEFAULT_EXTENSIONS_PATH=\"${NVIMGCODEC_DEFAULT_EXTENSIONS_PATH}\"")

set(LIBTOPACK ${NVIMGCODEC_LIBRARY_NAME} ${NVIMGCODEC_LIBRARY_NAME}_static)

find_package(CUDAToolkit REQUIRED)
add_subdirectory(dynlink)


if (WITH_DYNAMIC_CUDA)
    set(CUDA_DRIVER_LIB dynlink_cuda)
else()
    set(CUDA_DRIVER_LIB CUDA::cuda_driver)
endif()

target_link_libraries(${NVIMGCODEC_LIBRARY_NAME} PRIVATE ${CUDA_CUDART_LIB} ${CUDA_DRIVER_LIB})

if(UNIX)
    # CXX flags
    target_link_libraries(${NVIMGCODEC_LIBRARY_NAME} PRIVATE "${NVIMGCODEC_COMMON_DEPENDENCIES}")
    target_link_libraries(${NVIMGCODEC_LIBRARY_NAME}_static PRIVATE "${NVIMGCODEC_COMMON_DEPENDENCIES}")
endif()


# Configure library version
include(ConfigureVersion)
configure_library_version(NVIMGCODEC_LIBRARY_NAME
    "${CMAKE_CURRENT_SOURCE_DIR}/../include/nvimgcodec_version.h.in"
    "${CMAKE_CURRENT_SOURCE_DIR}/../include/nvimgcodec_version.h"
)

set(NVIMGCODEC_PUBLIC_HEADERS "${CMAKE_CURRENT_SOURCE_DIR}/../include/nvimgcodec.h")
set(NVIMGCODEC_PUBLIC_HEADERS "${NVIMGCODEC_PUBLIC_HEADERS}\;${CMAKE_CURRENT_SOURCE_DIR}/../include/nvimgcodec_version.h")

if(UNIX)
    set_target_properties(${NVIMGCODEC_LIBRARY_NAME} PROPERTIES
        POSITION_INDEPENDENT_CODE ON
        PUBLIC_HEADER ${NVIMGCODEC_PUBLIC_HEADERS}
        VERSION ${PROJECT_VERSION}
        SOVERSION ${PROJECT_VERSION_MAJOR})
else()
    # Windows: set explicit output directories for multi-config generators
    # Must set per-config properties to prevent config subdirectory being added
    set_target_properties(${NVIMGCODEC_LIBRARY_NAME} PROPERTIES
        POSITION_INDEPENDENT_CODE ON
        PUBLIC_HEADER ${NVIMGCODEC_PUBLIC_HEADERS}
        VERSION ${PROJECT_VERSION}
        OUTPUT_NAME ${NVIMGCODEC_LIBRARY_NAME}_${PROJECT_VERSION_MAJOR}
        ARCHIVE_OUTPUT_NAME ${NVIMGCODEC_LIBRARY_NAME})
    # Set per-config output directories to flatten structure for multi-config generators
    foreach(_config DEBUG RELEASE RELWITHDEBINFO MINSIZEREL)
        set_target_properties(${NVIMGCODEC_LIBRARY_NAME} PROPERTIES
            "RUNTIME_OUTPUT_DIRECTORY_${_config}" "${CMAKE_CURRENT_BINARY_DIR}"
            "LIBRARY_OUTPUT_DIRECTORY_${_config}" "${CMAKE_CURRENT_BINARY_DIR}"
            "ARCHIVE_OUTPUT_DIRECTORY_${_config}" "${CMAKE_CURRENT_BINARY_DIR}")
    endforeach()
endif()

# Installation (https://github.com/forexample/package-example) {

# Layout. This works for all platforms:
#   * <prefix>/${NVIMGCODEC_INSTALL_CMAKEDIR}/
#   * <prefix>/${NVIMGCODEC_INSTALL_LIBDIR}/
#   * <prefix>/${NVIMGCODEC_INSTALL_INCLUDEDIR}/
set(config_install_dir "${NVIMGCODEC_INSTALL_CMAKEDIR}")

set(generated_dir "${CMAKE_CURRENT_BINARY_DIR}/generated")

# Configuration
set(version_config "${generated_dir}/${PROJECT_NAME}ConfigVersion.cmake")
set(project_config "${generated_dir}/${PROJECT_NAME}Config.cmake")
set(TARGETS_EXPORT_NAME "${PROJECT_NAME}Targets")
set(namespace "${PROJECT_NAME}::")

# Include module with fuction 'write_basic_package_version_file'
include(CMakePackageConfigHelpers)

# Configure '<PROJECT-NAME>ConfigVersion.cmake'
# Use:
#   * PROJECT_VERSION
write_basic_package_version_file(
    "${version_config}" COMPATIBILITY SameMajorVersion
)

# Configure '<PROJECT-NAME>Config.cmake'
# Use variables:
#   * TARGETS_EXPORT_NAME
#   * PROJECT_NAME
# PATH_VARS below generate the PACKAGE_<var> placeholders used by Config.cmake.in.
configure_package_config_file(
    "${CMAKE_CURRENT_SOURCE_DIR}/../cmake/template/Config.cmake.in"
    "${project_config}"
    INSTALL_DESTINATION "${config_install_dir}"
    PATH_VARS
        NVIMGCODEC_INSTALL_INCLUDEDIR
        NVIMGCODEC_INSTALL_LIBDIR
        NVIMGCODEC_INSTALL_BINDIR
        NVIMGCODEC_INSTALL_EXTENSIONS_DIR
        NVIMGCODEC_INSTALL_EXTENSION_LIBDIR
)

# Targets:
install(
    TARGETS ${LIBTOPACK}
    EXPORT "${TARGETS_EXPORT_NAME}"
    LIBRARY DESTINATION "${NVIMGCODEC_INSTALL_LIBDIR}" COMPONENT lib
    ARCHIVE DESTINATION "${NVIMGCODEC_INSTALL_LIBDIR}" COMPONENT lib
    RUNTIME DESTINATION "${NVIMGCODEC_INSTALL_BINDIR}" COMPONENT lib
    PUBLIC_HEADER DESTINATION "${NVIMGCODEC_INSTALL_INCLUDEDIR}" COMPONENT lib
)

# License file - install to nvImageCodec-specific directory (not root)
set(NVIMGCODEC_ROOT "${CMAKE_CURRENT_SOURCE_DIR}/..")
if(EXISTS "${NVIMGCODEC_ROOT}/LICENSE.txt")
    install(
        FILES "${NVIMGCODEC_ROOT}/LICENSE.txt" "${NVIMGCODEC_ROOT}/Acknowledgements.txt"
        DESTINATION "share/nvImageCodec"
        COMPONENT lib
        OPTIONAL
    )
endif()

# Config
#   * <prefix>/${NVIMGCODEC_INSTALL_CMAKEDIR}/nvimgcodecConfig.cmake
#   * <prefix>/${NVIMGCODEC_INSTALL_CMAKEDIR}/nvimgcodecConfigVersion.cmake
install(
    FILES "${project_config}" "${version_config}"
    DESTINATION "${config_install_dir}"
    COMPONENT lib
)

# Config
# * <prefix>/${NVIMGCODEC_INSTALL_CMAKEDIR}/nvimgcodecTargets.cmake
install(
    EXPORT "${TARGETS_EXPORT_NAME}"
    NAMESPACE "${namespace}"
    DESTINATION "${config_install_dir}"
    COMPONENT lib
)

# Rewrite the just-installed *Targets*.cmake files so imported-target locations
# are resolved relative to the find_package-discovered directories
# (nvimgcodec_LIB_DIR / nvimgcodec_INCLUDE_DIR / nvimgcodec_BIN_DIR, defined by
# nvimgcodecConfig.cmake) instead of the build-time
# "${_IMPORT_PREFIX}/<fixed-subdir>".
#
# install(EXPORT ...) bakes each imported location as
# "${_IMPORT_PREFIX}/<subdir>", where _IMPORT_PREFIX is derived by climbing a
# fixed number of directories up from the targets file. Distro (DEB/RPM) and the
# Windows installer repackage the tree into a different layout (e.g.
# .../v0.9/lib/13/... on Windows), invalidating both the climbed prefix and the
# baked subdir, so find_package() fails with "references the file ... but this
# file does not exist". Expressing the locations relative to the discovered
# directories decouples the imported targets from the on-disk layout.
#
# This runs *after* the install(EXPORT ...) above so the files already exist.
# In the CODE string, @CONFIGURE-TIME@ CMake variables are expanded now, while
# \${...} are kept literal for install time (or written verbatim into the
# targets files as the replacement text). The more specific extension libdir is
# replaced before the plain libdir so the longer prefix wins.
install(CODE "
    # Honour DESTDIR: staged packaging installs (e.g. DESTDIR=<stage> cmake
    # --install) land the files at \$ENV{DESTDIR}\${CMAKE_INSTALL_PREFIX}, while
    # a plain --prefix install leaves DESTDIR empty. Concatenating both (no
    # separator, as the prefix is absolute) works for either case.
    set(_nvimgcodec_install_root \"\$ENV{DESTDIR}\${CMAKE_INSTALL_PREFIX}\")
    file(GLOB _nvimgcodec_tgt_files
        \"\${_nvimgcodec_install_root}/${config_install_dir}/${TARGETS_EXPORT_NAME}*.cmake\")
    if(NOT _nvimgcodec_tgt_files)
        message(WARNING \"nvimgcodec: no exported targets files found to patch under \${_nvimgcodec_install_root}/${config_install_dir}\")
    endif()
    foreach(_f IN LISTS _nvimgcodec_tgt_files)
        file(READ \"\${_f}\" _c)
        string(REPLACE \"\\\${_IMPORT_PREFIX}/${NVIMGCODEC_INSTALL_EXTENSION_LIBDIR}\" \"\\\${nvimgcodec_EXTENSION_LIB_DIR}\" _c \"\${_c}\")
        string(REPLACE \"\\\${_IMPORT_PREFIX}/${NVIMGCODEC_INSTALL_LIBDIR}\" \"\\\${nvimgcodec_LIB_DIR}\" _c \"\${_c}\")
        string(REPLACE \"\\\${_IMPORT_PREFIX}/${NVIMGCODEC_INSTALL_INCLUDEDIR}\" \"\\\${nvimgcodec_INCLUDE_DIR}\" _c \"\${_c}\")
        string(REPLACE \"\\\${_IMPORT_PREFIX}/${NVIMGCODEC_INSTALL_BINDIR}\" \"\\\${nvimgcodec_BIN_DIR}\" _c \"\${_c}\")
        string(REGEX REPLACE \"libnvimgcodec\\\\.so\\\\.[0-9]+\\\\.[0-9]+\\\\.[0-9]+\" \"libnvimgcodec.so\" _c \"\${_c}\")
        # The static archive is an optional artifact: some redistributable
        # packages (e.g. the runtime/dev split) ship the shared library plus the
        # CMake package but omit libnvimgcodec_static.a. Drop the static target
        # from the generated existence-check so find_package() does not hard-fail
        # when the archive is absent. The nvimgcodec::nvimgcodec_static target is
        # still declared, so linking against it (when it is shipped) keeps working.
        string(REGEX REPLACE \"list\\\\(APPEND _cmake_import_check_files_for_nvimgcodec::nvimgcodec_static [^)]*\\\\)\" \"\" _c \"\${_c}\")
        file(WRITE \"\${_f}\" \"\${_c}\")
        message(STATUS \"nvimgcodec: patched imported-target paths in \${_f}\")
    endforeach()
" COMPONENT lib)

# }
