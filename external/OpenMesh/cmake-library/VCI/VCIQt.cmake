#unset cached qt variables which are set by all qt versions. version is the major number of the qt version (e.g. 4 or 5, not 4.8)
macro (vci_unset_qt_shared_variables version)
  if (VCI_INTERNAL_QT_LAST_VERSION)
    if (NOT ${VCI_INTERNAL_QT_LAST_VERSION} EQUAL ${version})
      unset(QT_BINARY_DIR)
      unset(QT_PLUGINS_DIR)
      unset(VCI_INTERNAL_QT_LAST_VERSION)
    endif()
  endif()
  set (VCI_INTERNAL_QT_LAST_VERSION "${version}" CACHE INTERNAL "Qt Version, which was used on the last time")
endmacro()


# Macro to help find qt
#
# Input Variables used:
# QT_INSTALL_PATH : Path to a qt installation which contains lib and include folder (If not given, automatic search is active)
# QT_VERSION : (5/6) Default to qt 5 or 6
# QT_REQUIRED_PACKAGES : Required qt packages
# QT5_REQUIRED_PACKAGES : Required qt packages specific to qt5 only
# QT6_REQUIRED_PACKAGES : Required qt packages specific to qt6 only
#
# QT5_FOUND : Qt5 found
# QT6_FOUND : Qt6 found
# QT_FOUND : Any qt found?
#
macro (vci_qt)

  if(POLICY CMP0020)
    # Automatically link Qt executables to qtmain target on Windows
    cmake_policy(SET CMP0020 NEW)
  endif(POLICY CMP0020)

  set (QT_INSTALL_PATH "" CACHE PATH "Path to Qt directory which contains lib and include folder")

  if (EXISTS "${QT_INSTALL_PATH}")
    set (CMAKE_PREFIX_PATH "${CMAKE_PREFIX_PATH};${QT_INSTALL_PATH}")
    set (QT_INSTALL_PATH_EXISTS TRUE)
  else()
    set (QT_INSTALL_PATH_EXISTS FALSE)
  endif(EXISTS "${QT_INSTALL_PATH}")

  set(QT_FINDER_FLAGS "" CACHE STRING "Flags for the Qt finder e.g.
                                                       NO_DEFAULT_PATH if no system installed Qt shall be found")

  # compute default search paths
  set(SUPPORTED_QT_VERSIONS 6.2.2 6.2.1 6.1.2 6.0.4 6.0.3 5.15.2 5.12.2 5.11.3 5.11 5.10 5.9 5.8 5.7 5.6)
  if (NOT QT_INSTALL_PATH_EXISTS)
    message("-- No QT path specified (or does not exist) - searching in common locations...")
    foreach (suffix gcc_64 clang_64)
      foreach(version ${SUPPORTED_QT_VERSIONS})
        foreach(prefix ~/sw/Qt ~/Qt)
          # default institute qt path
          list(APPEND QT_DEFAULT_PATH "${prefix}/${version}/${suffix}")
          set (CMAKE_PREFIX_PATH "${CMAKE_PREFIX_PATH};${prefix}/${version}/${suffix}")
        endforeach()
      endforeach()
    endforeach()
  endif()

  # QtX11Extras only on Linux
  #if(NOT WIN32 AND NOT APPLE)
  #  list(APPEND QT_REQUIRED_PACKAGES X11Extras)
  #endif()

  # search for qt5 and qt6 installations offering requested components
  set(QT_VERSION "6" CACHE STRING "Override default qt version preference of qt6.")
  set(QT_DEFAULT_MAJOR_VERSION ${QT_VERSION})

  # find qt
  find_package (Qt${QT_DEFAULT_MAJOR_VERSION} COMPONENTS ${QT_REQUIRED_PACKAGES} PATHS ${QT_DEFAULT_PATH} ${QT_FINDER_FLAGS})

  # find version specific components
  if (Qt${QT_DEFAULT_MAJOR_VERSION}_FOUND AND QT${QT_DEFAULT_MAJOR_VERSION}_REQUIRED_PACKAGES)
    find_package (Qt${QT_DEFAULT_MAJOR_VERSION} COMPONENTS ${QT${QT_DEFAULT_MAJOR_VERSION}_REQUIRED_PACKAGES} PATHS ${QT_DEFAULT_PATH} ${QT_FINDER_FLAGS})
  endif()

  # set helpers
  set(QT6_FOUND ${Qt6_FOUND})
  set(QT5_FOUND ${Qt5_FOUND})
  if (QT6_FOUND OR QT5_FOUND)
    set(Qt_FOUND 1)
  else()
    set(Qt_FOUND 0)
  endif()
  set(QT_FOUND ${Qt_FOUND})

  # Use qt without version
  if (NOT QT_FOUND)
    set(QT_DEFAULT_MAJOR_VERSION 0)
  endif()

  # output some info
  if (QT5_FOUND)
    message("-- Found QT5")
  endif()
  if (QT6_FOUND)
    message("-- Found QT6")
  endif()

  if (NOT QT_FOUND)
    message("QT not found.")
  endif()

  if(QT_FOUND)
    if(Qt${QT_DEFAULT_MAJOR_VERSION}_VERSION) # use the new version variable if it is set
      set(Qt_VERSION_STRING ${Qt${QT_DEFAULT_MAJOR_VERSION}_VERSION})
    endif()

    message("-- Using QT${QT_DEFAULT_MAJOR_VERSION} (${Qt_VERSION_STRING})")

    set(QT_TARGET "Qt${QT_DEFAULT_MAJOR_VERSION}")

    string(REGEX REPLACE "^([0-9]+)\\.[0-9]+\\.[0-9]+.*" "\\1"
      QT_VERSION_MAJOR "${Qt_VERSION_STRING}")
    string(REGEX REPLACE "^[0-9]+\\.([0-9]+)\\.[0-9]+.*" "\\1"
      QT_VERSION_MINOR "${Qt_VERSION_STRING}")
    string(REGEX REPLACE "^[0-9]+\\.[0-9]+\\.([0-9]+).*" "\\1"
      QT_VERSION_PATCH "${Qt_VERSION_STRING}")

    vci_unset_qt_shared_variables(${QT_DEFAULT_MAJOR_VERSION})

    add_compile_definitions(QT_VERSION_MAJOR=${QT_VERSION_MAJOR})
    add_compile_definitions(QT_VERSION_MINOR=${QT_VERSION_MINOR})
    add_compile_definitions(QT_VERSION_PATCH=${QT_VERSION_PATCH})

    if (QT5_FOUND)
      #set plugin dir
      list(GET Qt5Gui_PLUGINS 0 _plugin)
      if (_plugin)
        get_target_property(_plugin_full ${_plugin} LOCATION)
        get_filename_component(_plugin_dir ${_plugin_full} PATH)
        set (QT_PLUGINS_DIR "${_plugin_dir}/../" CACHE PATH "Path to the qt plugin directory")
      elseif(QT_INSTALL_PATH_EXISTS)
        set (QT_PLUGINS_DIR "${QT_INSTALL_PATH}/plugins/" CACHE PATH "Path to the qt plugin directory")
      elseif()
        set (QT_PLUGINS_DIR "QT_PLUGIN_DIR_NOT_FOUND" CACHE PATH "Path to the qt plugin directory")
      endif(_plugin)
	endif()
	
	if (QT6_FOUND)
	    #TODO : FInd a cleaner solution!!!!
		
	    get_target_property(_QT_BINARY_DIR "Qt${QT_DEFAULT_MAJOR_VERSION}::uic" LOCATION)
		get_filename_component(_QT_BINARY_DIR "${_QT_BINARY_DIR}" DIRECTORY)
		set (QT_PLUGINS_DIR "${_QT_BINARY_DIR}/../plugins" CACHE PATH "Path to the qt plugin directory")      
	endif()
	
	
	
    # set binary dir for fixupbundle
    if(QT_INSTALL_PATH_EXISTS)
      set(_QT_BINARY_DIR "${QT_INSTALL_PATH}/bin")
    else()
      get_target_property(_QT_BINARY_DIR "Qt${QT_DEFAULT_MAJOR_VERSION}::uic" LOCATION)
      get_filename_component(_QT_BINARY_DIR ${_QT_BINARY_DIR} PATH)
    endif(QT_INSTALL_PATH_EXISTS)

    set (QT_BINARY_DIR "${_QT_BINARY_DIR}" CACHE PATH "Qt binary Directory")
    mark_as_advanced(QT_BINARY_DIR)

    set (CMAKE_INSTALL_RPATH_USE_LINK_PATH TRUE)

    foreach(QT_PACKAGE IN LISTS QT_REQUIRED_PACKAGES)
      include_directories(${Qt${QT_DEFAULT_MAJOR_VERSION}${QT_PACKAGE}_INCLUDE_DIRS})
      # add_definitions(${Qt${QT_DEFAULT_MAJOR_VERSION}${QT_PACKAGE}_DEFINITIONS})
      list(APPEND QT_LIBRARIES ${Qt${QT_DEFAULT_MAJOR_VERSION}${QT_PACKAGE}_LIBRARIES})
    endforeach()

    if (QT${QT_DEFAULT_MAJOR_VERSION}_REQUIRED_PACKAGES)
      foreach(QT_PACKAGE IN LISTS QT${QT_DEFAULT_MAJOR_VERSION}_REQUIRED_PACKAGES)
        include_directories(${Qt${QT_DEFAULT_MAJOR_VERSION}${QT_PACKAGE}_INCLUDE_DIRS})
        # add_definitions(${Qt${QT_DEFAULT_MAJOR_VERSION}${QT_PACKAGE}_DEFINITIONS})
        list(APPEND QT_LIBRARIES ${Qt${QT_DEFAULT_MAJOR_VERSION}${QT_PACKAGE}_LIBRARIES})
      endforeach()
    endif()

    if(NOT MSVC)
      set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fPIC")
    endif()

    if(MSVC)
      set(QT_LIBRARIES ${QT_LIBRARIES} ${Qt_QTMAIN_LIBRARIES})
    endif()

    #adding QT_NO_DEBUG to all release modes.
    #  Note: for multi generators like msvc you cannot set this definition depending of
    #  the current build type, because it may change in the future inside the ide and not via cmake
    if (MSVC_IDE)
      set(CMAKE_C_FLAGS_RELEASE "${CMAKE_C_FLAGS_RELEASE} /DQT_NO_DEBUG")
      set(CMAKE_CXX_FLAGS_RELEASE "${CMAKE_C_FLAGS_RELEASE} /DQT_NO_DEBUG")

      set(CMAKE_C_FLAGS_MINSIZEREL "${CMAKE_C_FLAGS_RELEASE} /DQT_NO_DEBUG")
      set(CMAKE_CXX_FLAGS_MINSITEREL "${CMAKE_C_FLAGS_RELEASE} /DQT_NO_DEBUG")

      set(CMAKE_C_FLAGS_RELWITHDEBINFO "${CMAKE_C_FLAGS_RELEASE} /DQT_NO_DEBUG")
      set(CMAKE_CXX_FLAGS_RELWITHDEBINFO "${CMAKE_C_FLAGS_RELEASE} /DQT_NO_DEBUG")
    else(MSVC_IDE)
      if(NOT CMAKE_BUILD_TYPE STREQUAL "Debug")
        add_definitions(-DQT_NO_DEBUG)
      endif()
    endif()

  endif ()
endmacro ()

#generates qt translations
function (vci_add_translations _target _languages _sources)

  string (TOUPPER ${_target} _TARGET)
  # generate/use translation files
  # run with UPDATE_TRANSLATIONS set to on to build qm files
  option (UPDATE_TRANSLATIONS_${_TARGET} "Update source translation *.ts files (WARNING: make clean will delete the source .ts files! Danger!)")

  set (_new_ts_files)
  set (_ts_files)

  foreach (lang ${_languages})
    if (NOT EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/translations/${_target}_${lang}.ts" OR UPDATE_TRANSLATIONS_${_TARGET})
      list (APPEND _new_ts_files "translations/${_target}_${lang}.ts")
    else ()
      list (APPEND _ts_files "translations/${_target}_${lang}.ts")
    endif ()
  endforeach ()

  if (QT5_FOUND) 
    FIND_PACKAGE( Qt5LinguistTools )
  endif()

  if (QT6_FOUND)
    FIND_PACKAGE( Qt6LinguistTools )
  endif()

  # This block contains the translation files that should be updated.
  # The UPDATE_TRANSLATIONS_${_TARGET} flag has to be set for the update
  # to be enabled.
  # Afterwards you can run the target ${_TARGET}_lupdate
  set (_qm_files)
  if ( _new_ts_files )

    if ( QT5_FOUND)
       qt_create_translation(_qm_files ${_sources} ${_new_ts_files})
    endif()

    if (QT6_FOUND)
      qt_add_lupdate(${_target}
                     TS_FILES ${_new_ts_files}
	             )
    endif ()

  endif ()

  # Here the list of translation files that will be compiled into
  # qm files is generated along with the targets.
  if ( _ts_files )
    if (QT5_FOUND)
       qt5_add_translation(_qm_files2 ${_ts_files})
    endif() 
    if (QT6_FOUND)
      qt_add_lrelease(${_target}
	              TS_FILES ${_ts_files} 
		      QM_FILES_OUTPUT_VARIABLE _qm_files2)
    endif()
    list (APPEND _qm_files ${_qm_files2})

  endif ()

  # create a target for the translation files ( and object files )
  # Use this target, to update only the translations
  add_custom_target (tr_${_target} DEPENDS ${_qm_files})
  GROUP_PROJECT( tr_${_target} "Translations")

  # Build translations with the application
  add_dependencies(${_target} tr_${_target} )

  if (NOT EXISTS ${CMAKE_BINARY_DIR}/Build/${VCI_PROJECT_DATADIR}/Translations)
    file(MAKE_DIRECTORY ${CMAKE_BINARY_DIR}/Build/${VCI_PROJECT_DATADIR}/Translations )
  endif ()

  foreach (_qm ${_qm_files})
    get_filename_component (_qm_name "${_qm}" NAME)
    add_custom_command (TARGET tr_${_target} POST_BUILD
      COMMAND ${CMAKE_COMMAND} -E
      copy_if_different
      ${_qm}
      ${CMAKE_BINARY_DIR}/Build/${VCI_PROJECT_DATADIR}/Translations/${_qm_name})
  endforeach ()

  if (NOT VCI_PROJECT_MACOS_BUNDLE OR NOT APPLE)
    install (FILES ${_qm_files} DESTINATION "${VCI_PROJECT_DATADIR}/Translations")
  endif ()
endfunction ()

# Function that writes all generated qch files into one Help.qhcp project file
function (generate_qhp_file files_loc plugin_name)

  set(qhp_file "${files_loc}/${plugin_name}.qhp")
  # Read in template file
  file(STRINGS "${CMAKE_SOURCE_DIR}/OpenFlipper/Documentation/QtHelpResources/QtHelpProject.qhp" qhp_template)

  # Initialize new project file
  file(WRITE ${qhp_file} "")
  foreach (_line ${qhp_template})
    string(STRIP ${_line} stripped)
    if("${stripped}" STREQUAL "files")
      vci_get_files_in_dir (_files ${files_loc})
      foreach (_file ${_files})
        string(REGEX MATCH ".+[.]+((html)|(htm)|(xml))$" fileresult ${_file})
        string(LENGTH "${fileresult}" len)
        if(${len} GREATER 0)
          file(APPEND ${qhp_file} "<file>${_file}</file>\n")
        endif()
      endforeach()
    else()
      string(REGEX REPLACE "plugin" ${plugin} newline ${_line})
      file(APPEND ${qhp_file} "${newline}\n")
    endif()
  endforeach()
endfunction()
