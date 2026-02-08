# -helper macro to add a "doc" target with CMake build system. 
# and configure doxy.config.in to doxy.config
#
# target "doc" allows building the documentation with doxygen/dot on WIN32 and Linux
# Creates .chm windows help file if MS HTML help workshop 
# (available from http://msdn.microsoft.com/workshop/author/htmlhelp)
# is installed with its DLLs in PATH.
#
#
# Please note, that the tools, e.g.:
# doxygen, dot, latex, dvips, makeindex, gswin32, etc.
# must be in path.
#
# Note about Visual Studio Projects: 
# MSVS hast its own path environment which may differ from the shell.
# See "Menu Tools/Options/Projects/VC++ Directories" in VS 7.1
#
# author Jan Woetzel 2004-2006
# www.mip.informatik.uni-kiel.de/~jw

if ( NOT DOXYGEN_FOUND)
  FIND_PACKAGE(Doxygen)
endif()

IF (DOXYGEN_FOUND)

  # click+jump in Emacs and Visual Studio (for doxy.config) (jw)
  IF    (CMAKE_BUILD_TOOL MATCHES "(msdev|devenv)")
    SET(DOXY_WARN_FORMAT "\"$file($line) : $text \"")
  ELSE  (CMAKE_BUILD_TOOL MATCHES "(msdev|devenv)")
    SET(DOXY_WARN_FORMAT "\"$file:$line: $text \"")
  ENDIF (CMAKE_BUILD_TOOL MATCHES "(msdev|devenv)")
  
  # we need latex for doxygen because of the formulas
  FIND_PACKAGE(LATEX)
  IF    (NOT LATEX_COMPILER)
    MESSAGE(STATUS "latex command LATEX_COMPILER not found but usually required. You will probably get warnings and user inetraction on doxy run.")
  ENDIF (NOT LATEX_COMPILER)
  IF    (NOT MAKEINDEX_COMPILER)
    MESSAGE(STATUS "makeindex command MAKEINDEX_COMPILER not found but usually required.")
  ENDIF (NOT MAKEINDEX_COMPILER)
  IF    (NOT DVIPS_CONVERTER)
    MESSAGE(STATUS "dvips command DVIPS_CONVERTER not found but usually required.")
  ENDIF (NOT DVIPS_CONVERTER)
  
  # This macro generates a new doc target. vci_create_doc_target( targetName [directory with the doxy.config.in] [dependency])
  # if no parameter is used except the target, a target of the given name will be created from the current source directories doxyfile and added as a dependency to the doc target
  # The first additional parameter is used to specify a directory where the doxyfile will be used
  # The second parameter defines a target, that will depend on the newly generated one. (Default is doc)
  macro (vci_create_doc_target target)

    set( DOC_DIRECTORY  "${CMAKE_CURRENT_SOURCE_DIR}" )
    set( DOC_DEPENDENCY "doc")

    # collect arguments
    if ( ${ARGC} EQUAL 2)
      set( DOC_DIRECTORY "${ARGV1}" )
    elseif ( ${ARGC} EQUAL 3)
      set( DOC_DIRECTORY  "${ARGV1}" )
      set( DOC_DEPENDENCY "${ARGV2}" )
    elseif( $ARGC GREATER 3 )
      set( DOC_DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}" )
      # failed as we do not know how to handle the parameters
      MESSAGE(SEND_ERROR "Unknown parameter for vci_create_doc_target!")
    endif()

    if("${DOC_DEPENDENCY}" STREQUAL "doc")
      #generate the global doc target (but only once!)
      if ( NOT TARGET doc )
        ADD_CUSTOM_TARGET( doc )
        SET_TARGET_PROPERTIES( doc PROPERTIES EchoString "Building Documentation"  )
        GROUP_PROJECT(doc Documentation)
      endif()
    endif()

    # If there exists an doxy.config.in, we configure it.
    IF   (EXISTS "${DOC_DIRECTORY}/doxy.config.in")

      #MESSAGE(STATUS "configured ${DOC_DIRECTORY}/doxy.config.in --> ${CMAKE_CURRENT_BINARY_DIR}/doxy.config")
      CONFIGURE_FILE(${DOC_DIRECTORY}/doxy.config.in
                     ${CMAKE_CURRENT_BINARY_DIR}/doxy.config
                     @ONLY )
      # use (configured) doxy.config from (out of place) BUILD tree:
      SET(DOXY_CONFIG "${CMAKE_CURRENT_BINARY_DIR}/doxy.config")

    ELSE (EXISTS "${DOC_DIRECTORY}/doxy.config.in")

      # use static hand-edited doxy.config from SOURCE tree:
      SET(DOXY_CONFIG "${DOC_DIRECTORY}/doxy.config")

      IF   (EXISTS "${DOC_DIRECTORY}/doxy.config")

        MESSAGE(STATUS "WARNING: using existing ${DOC_DIRECTORY}/doxy.config instead of configuring from doxy.config.in file.")

      ELSE (EXISTS "${DOC_DIRECTORY}/doxy.config")

        IF   (EXISTS "${CMAKE_MODULE_PATH}/doxy.config.in")
          # using template doxy.config.in
          #MESSAGE(STATUS "configured ${CMAKE_CMAKE_MODULE_PATH}/doxy.config.in --> ${CMAKE_CURRENT_BINARY_DIR}/doxy.config")
          CONFIGURE_FILE(${CMAKE_MODULE_PATH}/doxy.config.in
            ${CMAKE_CURRENT_BINARY_DIR}/doxy.config
            @ONLY )
          SET(DOXY_CONFIG "${CMAKE_CURRENT_BINARY_DIR}/doxy.config")

        ELSE (EXISTS "${CMAKE_MODULE_PATH}/doxy.config.in")

          # failed completely...
          MESSAGE(SEND_ERROR "Please create ${DOC_DIRECTORY}/doxy.config.in (or doxy.config as fallback)")

        ENDIF(EXISTS "${CMAKE_MODULE_PATH}/doxy.config.in")

      ENDIF(EXISTS "${DOC_DIRECTORY}/doxy.config")
    ENDIF(EXISTS "${DOC_DIRECTORY}/doxy.config.in")

    ADD_CUSTOM_TARGET(${target} ${DOXYGEN_EXECUTABLE} ${DOXY_CONFIG})
    # Group by folders on MSVC
    GROUP_PROJECT( ${target} "Documentation")

    add_dependencies( ${DOC_DEPENDENCY} ${target} )

#     # Add winhelp target only once!
#     GET_TARGET_PROPERTY(target_location winhelp EchoString)
#     if ( NOT target_location STREQUAL "Building Windows Documentation" )
#       ADD_CUSTOM_TARGET( winhelp )
#       SET_TARGET_PROPERTIES( winhelp PROPERTIES EchoString "Building Windows Documentation"  )
#     endif()

    # create a windows help .chm file using hhc.exe
    # HTMLHelp DLL must be in path!
    # fallback: use hhw.exe interactively
#     IF    (WIN32)
#       FIND_PACKAGE(HTMLHelp)
#       IF   (HTML_HELP_COMPILER)
#         SET (TMP "${CMAKE_CURRENT_BINARY_DIR}\\Doc\\html\\index.hhp")
#         STRING(REGEX REPLACE "[/]" "\\\\" HHP_FILE ${TMP} )
#         # MESSAGE(SEND_ERROR "DBG  HHP_FILE=${HHP_FILE}")
#         ADD_CUSTOM_TARGET(winhelp-${target} ${HTML_HELP_COMPILER} ${HHP_FILE})
#         ADD_DEPENDENCIES (winhelp-${target} ${target})
#
#         add_dependencies(winhelp winhelp-${target})
#
#
#         IF (NOT TARGET_DOC_SKIP_INSTALL)
#         # install windows help?
#         # determine useful name for output file
#         # should be project and version unique to allow installing
#         # multiple projects into one global directory
#         IF   (EXISTS "${PROJECT_BINARY_DIR}/Doc/html/index.chm")
#           IF   (PROJECT_NAME)
#             SET(OUT "${PROJECT_NAME}")
#           ELSE (PROJECT_NAME)
#             SET(OUT "Documentation") # default
#           ENDIF(PROJECT_NAME)
#           IF   (${PROJECT_NAME}_VERSION_MAJOR)
#             SET(OUT "${OUT}-${${PROJECT_NAME}_VERSION_MAJOR}")
#             IF   (${PROJECT_NAME}_VERSION_MINOR)
#               SET(OUT  "${OUT}.${${PROJECT_NAME}_VERSION_MINOR}")
#               IF   (${PROJECT_NAME}_VERSION_PATCH)
#                 SET(OUT "${OUT}.${${PROJECT_NAME}_VERSION_PATCH}")
#               ENDIF(${PROJECT_NAME}_VERSION_PATCH)
#             ENDIF(${PROJECT_NAME}_VERSION_MINOR)
#           ENDIF(${PROJECT_NAME}_VERSION_MAJOR)
#           # keep suffix
#           SET(OUT  "${OUT}.chm")
#
#           #MESSAGE("DBG ${PROJECT_BINARY_DIR}/Doc/html/index.chm \n${OUT}")
#           # create target used by install and package commands
#           INSTALL(FILES "${PROJECT_BINARY_DIR}/Doc/html/index.chm"
#             DESTINATION "doc"
#             RENAME "${OUT}"
#           )
#         ENDIF(EXISTS "${PROJECT_BINARY_DIR}/Doc/html/index.chm")
#         ENDIF(NOT TARGET_DOC_SKIP_INSTALL)
#
#       ENDIF(HTML_HELP_COMPILER)
#       # MESSAGE(SEND_ERROR "HTML_HELP_COMPILER=${HTML_HELP_COMPILER}")
#     ENDIF (WIN32)
  endmacro ()
ENDIF(DOXYGEN_FOUND)
