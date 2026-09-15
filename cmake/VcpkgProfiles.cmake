# SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
#
# SPDX-License-Identifier: GPL-3.0-or-later

include_guard(GLOBAL)

# Validate the opt-in overlay before project() runs vcpkg. Standard presets,
# custom triplets and user-managed overlays keep their existing behavior.
function(lfs_validate_vcpkg_profile)
    file(REAL_PATH "${CMAKE_CURRENT_FUNCTION_LIST_DIR}/triplets/release-only" _release_overlay)
    set(_uses_release_overlay FALSE)
    foreach(_overlay IN LISTS VCPKG_OVERLAY_TRIPLETS)
        file(REAL_PATH "${_overlay}" _overlay BASE_DIRECTORY "${CMAKE_SOURCE_DIR}")
        if(_overlay STREQUAL _release_overlay)
            set(_uses_release_overlay TRUE)
        endif()
    endforeach()
    if(NOT _uses_release_overlay)
        return()
    endif()

    # OpenMesh caches CMAKE_CONFIGURATION_TYPES even for single-config Ninja.
    # Only the generator property describes the active build system; the cache
    # entry must not reject an existing tree or its next regeneration.
    get_property(_multi_config GLOBAL PROPERTY GENERATOR_IS_MULTI_CONFIG)
    if(_multi_config OR NOT CMAKE_BUILD_TYPE MATCHES "^(Release|RelWithDebInfo|MinSizeRel)$")
        message(FATAL_ERROR
            "The LichtFeld Release-only vcpkg profile requires a single-config "
            "Release, RelWithDebInfo or MinSizeRel build. Use a separate build "
            "directory with the standard triplet for Debug or multi-config builds "
            "(for example, cmake --preset debug). "
            "Active generator: '${CMAKE_GENERATOR}', build type: '${CMAKE_BUILD_TYPE}'.")
    endif()

    if(NOT VCPKG_TARGET_TRIPLET MATCHES "^x64-(windows|linux)$" OR
       NOT VCPKG_HOST_TRIPLET STREQUAL VCPKG_TARGET_TRIPLET)
        message(FATAL_ERROR
            "The LichtFeld Release-only vcpkg profile requires matching native "
            "x64-windows or x64-linux target and host triplets. Use the standard "
            "presets for other architectures or custom triplets.")
    endif()

    # Changes to an active recipe must rerun vcpkg during normal regeneration.
    set_property(DIRECTORY APPEND PROPERTY CMAKE_CONFIGURE_DEPENDS
        "${_release_overlay}/${VCPKG_TARGET_TRIPLET}.cmake")
    message(STATUS "vcpkg profile: Release-only (${VCPKG_TARGET_TRIPLET}, target and host)")
endfunction()
