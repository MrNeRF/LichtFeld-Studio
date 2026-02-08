################################################################################
# Custom search paths for libraries
################################################################################

if ( WIN32 )

  find_path(VCI_WINDOWS_LIBS_DIR general/README.md
    DOC "Default library search dir for on windows."
    HINTS "C:/libs/"
          "D:/libs/"
          "E:/libs/"
          "R:/libs/")

  if (VCI_WINDOWS_LIBS_DIR)
    # add path to general libs
    list(APPEND CMAKE_PREFIX_PATH "${VCI_WINDOWS_LIBS_DIR}/general/")

    # add path for Visual Studio specific libraries
    # TODO: remove VS_SEARCH_PATH when the finders do not depend on it anymore
    if ( CMAKE_GENERATOR MATCHES "^Visual Studio 11.*Win64" )
      list(APPEND CMAKE_PREFIX_PATH "${VCI_WINDOWS_LIBS_DIR}/vs2012/x64/")
      SET(VS_SEARCH_PATH "${VCI_WINDOWS_LIBS_DIR}/vs2012/x64/")
    elseif ( CMAKE_GENERATOR MATCHES "^Visual Studio 11.*" )
      list(APPEND CMAKE_PREFIX_PATH "${VCI_WINDOWS_LIBS_DIR}/vs2012/x32/")
      SET(VS_SEARCH_PATH "${VCI_WINDOWS_LIBS_DIR}/vs2012/x32/")
    elseif ( CMAKE_GENERATOR MATCHES "^Visual Studio 12.*Win64" )
      list(APPEND CMAKE_PREFIX_PATH "${VCI_WINDOWS_LIBS_DIR}/vs2013/x64/")
      SET(VS_SEARCH_PATH "${VCI_WINDOWS_LIBS_DIR}/vs2013/x64/")
    elseif ( CMAKE_GENERATOR MATCHES "^Visual Studio 12.*" )
      list(APPEND CMAKE_PREFIX_PATH "${VCI_WINDOWS_LIBS_DIR}/vs2013/x32/")
      SET(VS_SEARCH_PATH "${VCI_WINDOWS_LIBS_DIR}/vs2013/x32/")
    elseif ( CMAKE_GENERATOR MATCHES "^Visual Studio 14.*Win64" )
      list(APPEND CMAKE_PREFIX_PATH "${VCI_WINDOWS_LIBS_DIR}/vs2015/x64/")
      SET(VS_SEARCH_PATH "${VCI_WINDOWS_LIBS_DIR}/vs2015/x64/")
    elseif ( CMAKE_GENERATOR MATCHES "^Visual Studio 14.*" )
      list(APPEND CMAKE_PREFIX_PATH "${VCI_WINDOWS_LIBS_DIR}/vs2015/x32/")
      SET(VS_SEARCH_PATH "${VCI_WINDOWS_LIBS_DIR}/vs2015/x32/")
    elseif ( CMAKE_GENERATOR MATCHES "^Visual Studio 15.*Win64" )
      list(APPEND CMAKE_PREFIX_PATH "${VCI_WINDOWS_LIBS_DIR}/vs2017/x64/")
      SET(VS_SEARCH_PATH "${VCI_WINDOWS_LIBS_DIR}/vs2017/x64/")
    elseif ( CMAKE_GENERATOR MATCHES "^Visual Studio 15.*" )
      list(APPEND CMAKE_PREFIX_PATH "${VCI_WINDOWS_LIBS_DIR}/vs2017/x32/")
      SET(VS_SEARCH_PATH "${VCI_WINDOWS_LIBS_DIR}/vs2017/x32/")
    endif()
  endif()
endif( WIN32 )
