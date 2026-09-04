# SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
# SPDX-License-Identifier: GPL-3.0-or-later

foreach(_lfs_required_variable IN ITEMS
        LFS_VISUALIZER_SOURCE_DIR
        LFS_RENDERING_SOURCE_DIR
        LFS_DESTINATION_DIR
        LFS_ALLOWED_DESTINATION_ROOT)
    if(NOT DEFINED ${_lfs_required_variable} OR "${${_lfs_required_variable}}" STREQUAL "")
        message(FATAL_ERROR "${_lfs_required_variable} is required")
    endif()
endforeach()

cmake_path(ABSOLUTE_PATH LFS_ALLOWED_DESTINATION_ROOT NORMALIZE
    OUTPUT_VARIABLE _lfs_allowed_destination_root)
cmake_path(ABSOLUTE_PATH LFS_DESTINATION_DIR NORMALIZE
    OUTPUT_VARIABLE _lfs_destination_dir)
cmake_path(IS_PREFIX _lfs_allowed_destination_root "${_lfs_destination_dir}"
    NORMALIZE _lfs_destination_is_allowed)

if(NOT _lfs_destination_is_allowed OR
   _lfs_destination_dir STREQUAL _lfs_allowed_destination_root)
    message(FATAL_ERROR
        "Refusing to synchronize outside a child of the visualizer build directory: "
        "${_lfs_destination_dir}")
endif()

function(_lfs_copy_resource_file _lfs_source_file _lfs_relative_destination _lfs_optional)
    set(_lfs_destination_file "${_lfs_destination_dir}/${_lfs_relative_destination}")
    if(EXISTS "${_lfs_source_file}" AND NOT IS_DIRECTORY "${_lfs_source_file}")
        if(IS_DIRECTORY "${_lfs_destination_file}")
            message(FATAL_ERROR
                "Visualizer resource destination is unexpectedly a directory: "
                "${_lfs_destination_file}")
        endif()
        get_filename_component(_lfs_destination_parent "${_lfs_destination_file}" DIRECTORY)
        file(MAKE_DIRECTORY "${_lfs_destination_parent}")
        file(COPY_FILE
            "${_lfs_source_file}"
            "${_lfs_destination_file}"
            ONLY_IF_DIFFERENT
            INPUT_MAY_BE_RECENT
        )
    elseif(NOT _lfs_optional)
        message(FATAL_ERROR "Required visualizer resource does not exist: ${_lfs_source_file}")
    elseif(IS_DIRECTORY "${_lfs_destination_file}")
        message(FATAL_ERROR
            "Optional visualizer resource destination is unexpectedly a directory: "
            "${_lfs_destination_file}")
    elseif(EXISTS "${_lfs_destination_file}")
        file(REMOVE "${_lfs_destination_file}")
    endif()
endfunction()

function(_lfs_sync_resource_directory
        _lfs_source_dir
        _lfs_relative_destination
        _lfs_glob_pattern
        _lfs_recurse
        _lfs_remove_stale
        _lfs_source_optional)
    set(_lfs_preserve_relative_files ${ARGN})
    set(_lfs_destination_subdir "${_lfs_destination_dir}/${_lfs_relative_destination}")

    if(NOT IS_DIRECTORY "${_lfs_source_dir}")
        if(_lfs_source_optional)
            return()
        endif()
        message(FATAL_ERROR "Visualizer resource directory does not exist: ${_lfs_source_dir}")
    endif()

    file(MAKE_DIRECTORY "${_lfs_destination_subdir}")
    if(_lfs_recurse)
        file(GLOB_RECURSE _lfs_source_files LIST_DIRECTORIES FALSE
            RELATIVE "${_lfs_source_dir}"
            "${_lfs_source_dir}/${_lfs_glob_pattern}"
        )
    else()
        file(GLOB _lfs_source_files LIST_DIRECTORIES FALSE
            RELATIVE "${_lfs_source_dir}"
            "${_lfs_source_dir}/${_lfs_glob_pattern}"
        )
    endif()
    list(SORT _lfs_source_files)

    foreach(_lfs_relative_file IN LISTS _lfs_source_files)
        set(_lfs_source_file "${_lfs_source_dir}/${_lfs_relative_file}")
        set(_lfs_destination_file "${_lfs_destination_subdir}/${_lfs_relative_file}")
        get_filename_component(_lfs_destination_parent "${_lfs_destination_file}" DIRECTORY)
        file(MAKE_DIRECTORY "${_lfs_destination_parent}")
        file(COPY_FILE
            "${_lfs_source_file}"
            "${_lfs_destination_file}"
            ONLY_IF_DIFFERENT
            INPUT_MAY_BE_RECENT
        )
    endforeach()

    if(_lfs_remove_stale)
        if(_lfs_recurse)
            file(GLOB_RECURSE _lfs_destination_files LIST_DIRECTORIES FALSE
                RELATIVE "${_lfs_destination_subdir}"
                "${_lfs_destination_subdir}/${_lfs_glob_pattern}"
            )
        else()
            file(GLOB _lfs_destination_files LIST_DIRECTORIES FALSE
                RELATIVE "${_lfs_destination_subdir}"
                "${_lfs_destination_subdir}/${_lfs_glob_pattern}"
            )
        endif()
        foreach(_lfs_relative_file IN LISTS _lfs_destination_files)
            list(FIND _lfs_source_files "${_lfs_relative_file}" _lfs_source_index)
            list(FIND _lfs_preserve_relative_files "${_lfs_relative_file}" _lfs_preserve_index)
            if(_lfs_source_index EQUAL -1 AND _lfs_preserve_index EQUAL -1)
                file(REMOVE "${_lfs_destination_subdir}/${_lfs_relative_file}")
            endif()
        endforeach()
    endif()
endfunction()

file(MAKE_DIRECTORY "${_lfs_destination_dir}/assets/fonts")

# Preserve the old optional overlay behavior. This directory is absent in the
# current tree, but files added to it later are picked up without reconfiguring.
_lfs_sync_resource_directory(
    "${LFS_VISUALIZER_SOURCE_DIR}/resources/assets"
    "assets"
    "*"
    OFF
    OFF
    ON
)

_lfs_copy_resource_file(
    "${LFS_VISUALIZER_SOURCE_DIR}/gui/assets/JetBrainsMono-Regular.ttf"
    "assets/JetBrainsMono-Regular.ttf"
    ON
)
_lfs_copy_resource_file(
    "${LFS_VISUALIZER_SOURCE_DIR}/gui/assets/lichtfeld-icon.png"
    "assets/lichtfeld-icon.png"
    ON
)
_lfs_copy_resource_file(
    "${LFS_RENDERING_SOURCE_DIR}/resources/assets/JetBrainsMono-Regular.ttf"
    "assets/fonts/JetBrainsMono-Regular.ttf"
    ON
)

_lfs_sync_resource_directory(
    "${LFS_VISUALIZER_SOURCE_DIR}/gui/assets/icon"
    "assets/icon"
    "*.png"
    OFF
    ON
    OFF
)
_lfs_sync_resource_directory(
    "${LFS_VISUALIZER_SOURCE_DIR}/gui/assets/icon/scene"
    "assets/icon/scene"
    "*.png"
    OFF
    ON
    OFF
)
_lfs_sync_resource_directory(
    "${LFS_VISUALIZER_SOURCE_DIR}/gui/assets/icon/sequencer"
    "assets/icon/sequencer"
    "*.png"
    OFF
    ON
    OFF
)
_lfs_sync_resource_directory(
    "${LFS_VISUALIZER_SOURCE_DIR}/gui/assets/environments"
    "assets/environments"
    "*"
    OFF
    ON
    OFF
)
_lfs_sync_resource_directory(
    "${LFS_VISUALIZER_SOURCE_DIR}/gui/rmlui/resources"
    "assets/rmlui"
    "*"
    OFF
    ON
    OFF
    "soft_shadow.png"
)
_lfs_copy_resource_file(
    "${LFS_VISUALIZER_SOURCE_DIR}/gui/assets/rmlui/soft_shadow.png"
    "assets/rmlui/soft_shadow.png"
    OFF
)
_lfs_sync_resource_directory(
    "${LFS_VISUALIZER_SOURCE_DIR}/gui/resources/locales"
    "locales"
    "*.json"
    OFF
    ON
    OFF
)
_lfs_copy_resource_file(
    "${LFS_VISUALIZER_SOURCE_DIR}/gui/resources/locale_index.json"
    "locale_index.json"
    OFF
)
