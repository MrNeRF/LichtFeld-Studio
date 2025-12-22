# SPDX-FileCopyrightText: 2025 LichtFeld Studio Authors
#
# SPDX-License-Identifier: GPL-3.0-or-later

# BundleDependencies.cmake
# Uses file(GET_RUNTIME_DEPENDENCIES) to bundle all runtime dependencies
# for a portable build. This is the CMake-recommended approach per miriameng's review.

# This script is called during install via:
#   cmake --install build --prefix ./dist

# Function to bundle runtime dependencies for a target
function(bundle_runtime_dependencies)
    set(options)
    set(oneValueArgs TARGET DESTINATION)
    set(multiValueArgs DIRECTORIES)
    cmake_parse_arguments(BUNDLE "${options}" "${oneValueArgs}" "${multiValueArgs}" ${ARGN})

    if(NOT BUNDLE_TARGET OR NOT BUNDLE_DESTINATION)
        message(FATAL_ERROR "bundle_runtime_dependencies requires TARGET and DESTINATION")
    endif()

    # Build the search directories list
    set(_search_dirs "")
    foreach(_dir IN LISTS BUNDLE_DIRECTORIES)
        if(EXISTS "${_dir}")
            list(APPEND _search_dirs "${_dir}")
        endif()
    endforeach()

    # Install the target's runtime dependencies
    install(CODE "
        message(STATUS \"Bundling runtime dependencies for ${BUNDLE_TARGET}...\")

        # Get the installed executable/library path
        set(_target_file \"\$<TARGET_FILE:${BUNDLE_TARGET}>\")

        file(GET_RUNTIME_DEPENDENCIES
            RESOLVED_DEPENDENCIES_VAR _resolved_deps
            UNRESOLVED_DEPENDENCIES_VAR _unresolved_deps
            CONFLICTING_DEPENDENCIES_PREFIX _conflicts
            EXECUTABLES \${_target_file}
            DIRECTORIES ${_search_dirs}
            PRE_EXCLUDE_REGEXES
                \"api-ms-.*\"
                \"ext-ms-.*\"
            POST_EXCLUDE_REGEXES
                \".*system32.*\"
                \".*System32.*\"
                \".*SYSTEM32.*\"
                \".*Windows.*\"
                \"/lib/x86_64-linux-gnu/.*\"
                \"/lib64/.*\"
                \"/usr/lib.*\"
        )

        # Log what we found
        foreach(_dep IN LISTS _resolved_deps)
            message(STATUS \"  Bundling: \${_dep}\")
        endforeach()

        if(_unresolved_deps)
            message(WARNING \"Unresolved dependencies:\")
            foreach(_dep IN LISTS _unresolved_deps)
                message(WARNING \"  \${_dep}\")
            endforeach()
        endif()

        # Copy resolved dependencies
        foreach(_dep IN LISTS _resolved_deps)
            file(INSTALL \${_dep}
                DESTINATION \"\${CMAKE_INSTALL_PREFIX}/${BUNDLE_DESTINATION}\"
                FOLLOW_SYMLINK_CHAIN
            )
        endforeach()
    " COMPONENT runtime)
endfunction()

# Simpler version that works at install time using generator expressions
function(install_runtime_dependencies_for)
    set(options)
    set(oneValueArgs TARGET LIB_DESTINATION)
    set(multiValueArgs SEARCH_PATHS)
    cmake_parse_arguments(ARG "${options}" "${oneValueArgs}" "${multiValueArgs}" ${ARGN})

    if(NOT ARG_TARGET)
        message(FATAL_ERROR "install_runtime_dependencies_for: TARGET is required")
    endif()

    if(NOT ARG_LIB_DESTINATION)
        set(ARG_LIB_DESTINATION "lib")
    endif()

    # Build directories string for the install script
    set(_dirs_str "")
    foreach(_path IN LISTS ARG_SEARCH_PATHS)
        string(APPEND _dirs_str "\"${_path}\" ")
    endforeach()

    # Generate install code that runs at install time
    install(CODE "
        set(_exe_file \"$<TARGET_FILE:${ARG_TARGET}>\")
        set(_dest_dir \"\${CMAKE_INSTALL_PREFIX}/${ARG_LIB_DESTINATION}\")
        set(_search_dirs ${_dirs_str})

        message(STATUS \"=== Analyzing runtime dependencies for ${ARG_TARGET} ===\")
        message(STATUS \"Executable: \${_exe_file}\")
        message(STATUS \"Search paths: \${_search_dirs}\")

        file(GET_RUNTIME_DEPENDENCIES
            RESOLVED_DEPENDENCIES_VAR _resolved
            UNRESOLVED_DEPENDENCIES_VAR _unresolved
            CONFLICTING_DEPENDENCIES_PREFIX _conflicts
            EXECUTABLES \"\${_exe_file}\"
            DIRECTORIES \${_search_dirs}
            PRE_EXCLUDE_REGEXES
                [[api-ms-.*]]
                [[ext-ms-.*]]
                [[API-MS-.*]]
                [[EXT-MS-.*]]
            POST_EXCLUDE_REGEXES
                [[.*[Ss]ystem32.*]]
                [[.*[Ww]indows.*]]
                [[/lib/x86_64-linux-gnu/.*]]
                [[/lib64/.*]]
                [[/usr/lib.*]]
                [[.*ld-linux.*]]
                [[.*libc\\.so.*]]
                [[.*libpthread.*]]
                [[.*libdl\\.so.*]]
                [[.*libm\\.so.*]]
                [[.*librt\\.so.*]]
                [[.*libstdc\\+\\+.*]]
                [[.*libgcc_s.*]]
        )

        message(STATUS \"Resolved dependencies: \${_resolved}\")

        if(_unresolved)
            message(STATUS \"Unresolved (system) dependencies: \${_unresolved}\")
        endif()

        # Create destination directory
        file(MAKE_DIRECTORY \"\${_dest_dir}\")

        # Copy each resolved dependency
        list(LENGTH _resolved _num_deps)
        message(STATUS \"Copying \${_num_deps} runtime dependencies to \${_dest_dir}\")

        foreach(_dep IN LISTS _resolved)
            get_filename_component(_dep_name \"\${_dep}\" NAME)
            message(STATUS \"  Installing: \${_dep_name}\")
            file(INSTALL \"\${_dep}\"
                DESTINATION \"\${_dest_dir}\"
                FOLLOW_SYMLINK_CHAIN
            )
        endforeach()

        message(STATUS \"=== Runtime dependencies installed ===\")
    " COMPONENT runtime)
endfunction()
