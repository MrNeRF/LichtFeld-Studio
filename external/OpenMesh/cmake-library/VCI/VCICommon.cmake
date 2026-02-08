include (MSVCMacros)
include (VCIPaths)

if (EXISTS ${CMAKE_SOURCE_DIR}/${CMAKE_PROJECT_NAME}.cmake)
  include (${CMAKE_SOURCE_DIR}/${CMAKE_PROJECT_NAME}.cmake)
endif ()

# prevent build in source directory

# add option to disable in-source build
option (
  BLOCK_IN_SOURCE_BUILD
  "Disable building inside the source tree"
  ON
)

if ( BLOCK_IN_SOURCE_BUILD )
  if ("${CMAKE_BINARY_DIR}" STREQUAL "${CMAKE_SOURCE_DIR}")
      message (SEND_ERROR "Building in the source directory is not supported.")
      message (FATAL_ERROR "Please remove the created \"CMakeCache.txt\" file, the \"CMakeFiles\" directory and create a build directory and call \"${CMAKE_COMMAND} <path to the sources>\".")
  endif ("${CMAKE_BINARY_DIR}" STREQUAL "${CMAKE_SOURCE_DIR}")
endif()

# allow only Debug, Release and RelWithDebInfo builds
set (CMAKE_CONFIGURATION_TYPES "Debug;Release;RelWithDebInfo" CACHE STRING "")
mark_as_advanced (CMAKE_CONFIGURATION_TYPES)

# set Debus as default build target
if (NOT CMAKE_BUILD_TYPE)
  set (CMAKE_BUILD_TYPE Debug CACHE STRING
      "Choose the type of build, options are: Debug, Release."
      FORCE)
endif ()

# create our output directroy
if (NOT EXISTS ${CMAKE_BINARY_DIR}/Build)
  file (MAKE_DIRECTORY ${CMAKE_BINARY_DIR}/Build)
endif ()

# set directory structures for the different platforms
if (CMAKE_HOST_SYSTEM_NAME MATCHES Windows)
  set (VCI_PROJECT_DATADIR ".")
  set (VCI_PROJECT_LIBDIR "lib")
  set (VCI_PROJECT_BINDIR ".")
  set (VCI_PROJECT_PLUGINDIR "Plugins")
  if (NOT EXISTS ${CMAKE_BINARY_DIR}/Build/${VCI_PROJECT_LIBDIR})
    file (MAKE_DIRECTORY ${CMAKE_BINARY_DIR}/Build/${VCI_PROJECT_LIBDIR})
  endif ()
elseif (APPLE)
  set (VCI_PROJECT_DATADIR "share/${CMAKE_PROJECT_NAME}")
  set (VCI_PROJECT_LIBDIR "lib${LIB_SUFFIX}")
  set (CMAKE_LIBRARY_OUTPUT_DIR "${CMAKE_BINARY_DIR}/Build/${VCI_PROJECT_LIBDIR}")
  set (VCI_PROJECT_PLUGINDIR "${VCI_PROJECT_LIBDIR}/plugins")
  set (VCI_PROJECT_BINDIR "bin")
else ()
  set (VCI_PROJECT_DATADIR "share/${CMAKE_PROJECT_NAME}")
  set (VCI_PROJECT_LIBDIR "lib${LIB_SUFFIX}")
  set (VCI_PROJECT_PLUGINDIR "${VCI_PROJECT_LIBDIR}/plugins")
  set (VCI_PROJECT_BINDIR "bin")
endif ()

# allow a project to modify the directories
if (COMMAND vci_modify_project_dirs)
  vci_modify_project_dirs ()
endif ()

if (NOT EXISTS ${CMAKE_BINARY_DIR}/Build/${VCI_PROJECT_DATADIR})
 file (MAKE_DIRECTORY ${CMAKE_BINARY_DIR}/Build/${VCI_PROJECT_DATADIR})
endif ()


# sets default build properties
macro (vci_set_target_props target)
  if (WIN32)
    set_target_properties (
      ${target} PROPERTIES
      BUILD_WITH_INSTALL_RPATH 1
      SKIP_BUILD_RPATH 0
    )
  elseif (APPLE AND NOT VCI_PROJECT_MACOS_BUNDLE)
    # save rpath
    set_target_properties (
      ${target} PROPERTIES
      INSTALL_RPATH "@executable_path/../${VCI_PROJECT_LIBDIR}"
      MACOSX_RPATH 1
      #BUILD_WITH_INSTALL_RPATH 1
      SKIP_BUILD_RPATH 0
    )  
  elseif (NOT APPLE)

    set_target_properties (
      ${target} PROPERTIES
      INSTALL_RPATH "$ORIGIN/../${VCI_PROJECT_LIBDIR}"
      BUILD_WITH_INSTALL_RPATH 1
      SKIP_BUILD_RPATH 0
      RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/Build/${VCI_PROJECT_BINDIR}"
      LIBRARY_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/Build/${VCI_PROJECT_LIBDIR}"
    )
  endif ()
endmacro ()

include (AddFileDependencies)
include (VCICompiler)

# define INCLUDE_TEMPLATES for everything we build
add_definitions (-DINCLUDE_TEMPLATES)

# unsets the given variable
macro (vci_unset var)
    set (${var} "" CACHE INTERNAL "")
endmacro ()

# sets the given variable
macro (vci_set var value)
    set (${var} ${value} CACHE INTERNAL "")
endmacro ()

# test for OpenMP
macro (vci_openmp)
  if (NOT OPENMP_FOUND)
     find_package(OpenMP)
     if (OPENMP_FOUND)
       set (CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}")
       set (CMAKE_C_FLAGS "${CMAKE_C_FLAGS} ${OpenMP_C_FLAGS}")
       add_definitions(-DUSE_OPENMP)

       # export includes and libraries for usage within openflipper plugins
       set(OPENMP_INCLUDE_DIRS ${OpenMP_CXX_INCLUDE_DIR})
       set(OPENMP_LIBRARIES ${OpenMP_libomp_LIBRARY})        
    endif()
  endif()
endmacro ()


# append all files with extension "ext" in the "dirs" directories to "ret"
# excludes all files starting with a '.' (dot)
macro (vci_append_files ret ext)
  foreach (_dir ${ARGN})
    file (GLOB _files "${_dir}/${ext}")
    foreach (_file ${_files})
      get_filename_component (_filename ${_file} NAME)
      if (_filename MATCHES "^[.]")
	list (REMOVE_ITEM _files ${_file})
      endif ()
    endforeach ()
    list (APPEND ${ret} ${_files})
  endforeach ()
endmacro ()

# append all files with extension "ext" in the "dirs" directories and its subdirectories to "ret"
# excludes all files starting with a '.' (dot)
macro (vci_append_files_recursive ret ext)
  foreach (_dir ${ARGN})
    file (GLOB_RECURSE _files "${_dir}/${ext}")
    foreach (_file ${_files})
      get_filename_component (_filename ${_file} NAME)
      if (_filename MATCHES "^[.]")
  list (REMOVE_ITEM _files ${_file})
      endif ()
    endforeach ()
    list (APPEND ${ret} ${_files})
  endforeach ()
endmacro ()

# get all files in directory, but ignore svn
macro (vci_get_files_in_dir ret dir)
  file (GLOB_RECURSE __files RELATIVE "${dir}" "${dir}/*")
  foreach (_file ${__files})
    if ( (NOT _file MATCHES ".*svn.*") AND (NOT _file MATCHES ".DS_Store") )
      list (APPEND ${ret} "${_file}")
    endif ()
  endforeach ()
endmacro ()

# drop all "*T.cc" files from list
macro (vci_drop_templates list)
  foreach (_file ${${list}})
    if (_file MATCHES "T.cc$")
      set_source_files_properties(${_file} PROPERTIES HEADER_FILE_ONLY TRUE)
      message("Deprecated naming scheme! The file ${_file} ends with T.cc indicating it is a template only implementation file. Please rename to T_impl.hh to avoid problems with several IDEs.")
    endif ()
  endforeach ()
endmacro ()

# copy the whole directory without svn files
function (vci_copy_after_build target src dst)
  vci_unset (_files)
  vci_get_files_in_dir (_files ${src})
  foreach (_file ${_files})
    add_custom_command(TARGET ${target} POST_BUILD
      COMMAND ${CMAKE_COMMAND} -E copy_if_different "${src}/${_file}" "${dst}/${_file}"
    )
  endforeach ()
endfunction ()

# install the whole directory without svn files
function (vci_install_dir src dst)
  vci_unset (_files)
  vci_get_files_in_dir (_files ${src})
  foreach (_file ${_files})
    get_filename_component (_file_PATH ${_file} PATH)
    install(FILES "${src}/${_file}"
      DESTINATION "${dst}/${_file_PATH}"
    )
  endforeach ()
endfunction ()

# extended version of add_executable that also copies output to out Build directory
function (vci_add_executable _target)
  add_executable (${_target} ${ARGN})

  # set common target properties defined in common.cmake
  vci_set_target_props (${_target})

  if (NOT VCI_COMMON_DO_NOT_COPY_POST_BUILD)
    if (WIN32 OR (APPLE AND NOT VCI_PROJECT_MACOS_BUNDLE))
      add_custom_command (TARGET ${_target} POST_BUILD
                          COMMAND ${CMAKE_COMMAND} -E
                          copy_if_different
                            $<TARGET_FILE:${_target}>
                            ${CMAKE_BINARY_DIR}/Build/${VCI_PROJECT_BINDIR}/$<TARGET_FILE_NAME:${_target}>)
    endif (WIN32 OR (APPLE AND NOT VCI_PROJECT_MACOS_BUNDLE))
  endif()
  
  if (NOT VCI_PROJECT_MACOS_BUNDLE OR NOT APPLE)
    install (TARGETS ${_target} DESTINATION ${VCI_PROJECT_BINDIR})
  endif ()
endfunction ()

# extended version of add_library that also copies output to out Build directory
function (vci_add_library _target _libtype)

  if (${_libtype} STREQUAL SHAREDANDSTATIC)
    set (_type SHARED)
    if (NOT WIN32 OR MINGW)
      set (_and_static 1)
    else ()
      set (_and_static 0)
    endif ()    
  else ()
    set (_type ${_libtype})
    set (_and_static 0)
  endif ()

  add_library (${_target} ${_type} ${ARGN} )

  # set common target properties defined in common.cmake
  vci_set_target_props (${_target})

  if (_and_static)
    add_library (${_target}Static STATIC ${ARGN})

    # set common target properties defined in common.cmake
    vci_set_target_props (${_target}Static)

    set_target_properties(${_target}Static PROPERTIES OUTPUT_NAME ${_target})
    
    if (NOT APPLE)
      set_target_properties (${_target}Static PROPERTIES 
                             LIBRARY_OUTPUT_DIRECTORY "${CMAKE_LIBRARY_OUTPUT_DIRECTORY}"
                            )
    endif ()
  endif ()

  if (NOT VCI_COMMON_DO_NOT_COPY_POST_BUILD)
    if ( (WIN32 AND MSVC) OR (APPLE AND NOT VCI_PROJECT_MACOS_BUNDLE))
      if (${_type} STREQUAL SHARED)
        add_custom_command (TARGET ${_target} POST_BUILD
                            COMMAND ${CMAKE_COMMAND} -E
                            copy_if_different
                              $<TARGET_FILE:${_target}>
                              ${CMAKE_BINARY_DIR}/Build/${VCI_PROJECT_LIBDIR}/$<TARGET_FILE_NAME:${_target}>)
      add_custom_command (TARGET ${_target} POST_BUILD
                            COMMAND ${CMAKE_COMMAND} -E
                            copy_if_different
                              $<TARGET_LINKER_FILE:${_target}>
                              ${CMAKE_BINARY_DIR}/Build/${VCI_PROJECT_LIBDIR}/$<TARGET_LINKER_FILE_NAME:${_target}>)
      elseif (${_type} STREQUAL MODULE)
        if (NOT EXISTS ${CMAKE_BINARY_DIR}/Build/${VCI_PROJECT_PLUGINDIR})
          file (MAKE_DIRECTORY ${CMAKE_BINARY_DIR}/Build/${VCI_PROJECT_PLUGINDIR})
        endif ()
        add_custom_command (TARGET ${_target} POST_BUILD
                            COMMAND ${CMAKE_COMMAND} -E
                            copy_if_different
                              $<TARGET_FILE:${_target}>
                              ${CMAKE_BINARY_DIR}/Build/${VCI_PROJECT_PLUGINDIR}/$<TARGET_FILE_NAME:${_target}>)
      elseif (${_type} STREQUAL STATIC)
      add_custom_command (TARGET ${_target} POST_BUILD
                            COMMAND ${CMAKE_COMMAND} -E
                            copy_if_different
                              $<TARGET_FILE:${_target}>
                              ${CMAKE_BINARY_DIR}/Build/${VCI_PROJECT_LIBDIR}/$<TARGET_FILE_NAME:${_target}>)
    endif()


    # make an extra copy for windows into the binary directory
      if (${_type} STREQUAL SHARED AND WIN32)
        add_custom_command (TARGET ${_target} POST_BUILD
                            COMMAND ${CMAKE_COMMAND} -E
                            copy_if_different
                              $<TARGET_FILE:${_target}>
                              ${CMAKE_BINARY_DIR}/Build/${VCI_PROJECT_BINDIR}/$<TARGET_FILE_NAME:${_target}>)
    endif ()

    endif( (WIN32 AND MSVC) OR (APPLE AND NOT VCI_PROJECT_MACOS_BUNDLE))

    if (_and_static)
      add_custom_command (TARGET ${_target}Static POST_BUILD
                          COMMAND ${CMAKE_COMMAND} -E
                          copy_if_different
                            $<TARGET_FILE:${_target}Static>
                            ${CMAKE_BINARY_DIR}/Build/${VCI_PROJECT_LIBDIR}/$<TARGET_FILE_NAME:${_target}Static>)

    endif ()
  endif ()
 

  # Block installation of libraries by setting VCI_NO_LIBRARY_INSTALL
  if ( NOT VCI_NO_LIBRARY_INSTALL )
    if (NOT VCI_PROJECT_MACOS_BUNDLE OR NOT APPLE)
      if (${_type} STREQUAL SHARED OR ${_type} STREQUAL STATIC )
        install (TARGETS ${_target}
                 RUNTIME DESTINATION ${VCI_PROJECT_BINDIR}
                 LIBRARY DESTINATION ${VCI_PROJECT_LIBDIR}
                 ARCHIVE DESTINATION ${VCI_PROJECT_LIBDIR})
        if (_and_static)
          install (TARGETS ${_target}Static
                   DESTINATION ${VCI_PROJECT_LIBDIR})
        endif ()
      elseif (${_type} STREQUAL MODULE)
        install (TARGETS ${_target} DESTINATION ${VCI_PROJECT_PLUGINDIR})
      endif ()
    endif ()
  endif()

endfunction ()

