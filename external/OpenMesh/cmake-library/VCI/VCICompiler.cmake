################################################################################
# Custom settings for compiler flags and similar
################################################################################

if ( WIN32 )
  ################################################################################
  # Windows large memory support
  ################################################################################
  if ( NOT DEFINED WINDOWS_LARGE_MEMORY_SUPPORT )
    set( WINDOWS_LARGE_MEMORY_SUPPORT true CACHE BOOL "Enable or disable binary support for large memory" )
  endif()
  
  set( ADDITIONAL_CMAKE_EXE_LINKER_FLAGS )
  set( ADDITIONAL_CMAKE_SHARED_LINKER_FLAGS )
  set( ADDITIONAL_CMAKE_MODULE_LINKER_FLAGS )
  
  if ( WINDOWS_LARGE_MEMORY_SUPPORT )
    if (MSVC)
	  list(APPEND ADDITIONAL_CMAKE_EXE_LINKER_FLAGS       "/LARGEADDRESSAWARE" )
	  list(APPEND ADDITIONAL_CMAKE_SHARED_LINKER_FLAGS    "/LARGEADDRESSAWARE" )
	  list(APPEND ADDITIONAL_CMAKE_MODULE_LINKER_FLAGS    "/LARGEADDRESSAWARE" )
    endif()  
  endif()
  
  
  foreach( flag ${ADDITIONAL_CMAKE_EXE_LINKER_FLAGS} )
    if( NOT CMAKE_EXE_LINKER_FLAGS MATCHES "${flag}" )
      set( CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} ${flag} ")
    endif()
  endforeach()
  
  foreach( flag ${ADDITIONAL_CMAKE_SHARED_LINKER_FLAGS} )
    if( NOT CMAKE_SHARED_LINKER_FLAGS MATCHES "${flag}" )
      set( CMAKE_SHARED_LINKER_FLAGS "${CMAKE_SHARED_LINKER_FLAGS} ${flag} ")
    endif()
  endforeach()
  
  foreach( flag ${ADDITIONAL_CMAKE_MODULE_LINKER_FLAGS} )
    if( NOT CMAKE_MODULE_LINKER_FLAGS MATCHES "${flag}" )
      set( CMAKE_MODULE_LINKER_FLAGS "${CMAKE_MODULE_LINKER_FLAGS} ${flag} ")
    endif()
  endforeach()

endif( WIN32 )

if (UNIX)

  set ( ADDITIONAL_CXX_FLAGS )
  set ( ADDITIONAL_CXX_DEBUG_FLAGS )
  set ( ADDITIONAL_CXX_RELEASE_FLAGS )
  set ( ADDITIONAL_CXX_RELWITHDEBINFO_FLAGS )
  
  set ( ADDITIONAL_C_FLAGS )
  set ( ADDITIONAL_C_DEBUG_FLAGS )
  set ( ADDITIONAL_C_RELEASE_FLAGS )
  set ( ADDITIONAL_C_RELWITHDEBINFO_FLAGS )

  ################################################################################
  # Defaults
  ################################################################################

  # add our standard flags for Template inclusion
  list(APPEND ADDITIONAL_CXX_FLAGS          "-DINCLUDE_TEMPLATES" )
  list(APPEND ADDITIONAL_C_FLAGS            "-DINCLUDE_TEMPLATES" )
  
  ################################################################################
  # OS Defines
  ################################################################################  

  if (APPLE)
    add_definitions( -DARCH_DARWIN )
  endif()
  
  ################################################################################
  # Build/Release Defines
  ################################################################################
  IF( NOT CMAKE_SYSTEM MATCHES "SunOS*")
    list(APPEND ADDITIONAL_CXX_DEBUG_FLAGS          "-DDEBUG" )
    list(APPEND ADDITIONAL_CXX_RELEASE_FLAGS        "-DNDEBUG" )
    list(APPEND ADDITIONAL_CXX_RELWITHDEBINFO_FLAGS "-DDEBUG" )    
    
    list(APPEND ADDITIONAL_C_DEBUG_FLAGS            "-DDEBUG" )
    list(APPEND ADDITIONAL_C_RELEASE_FLAGS          "-DNDEBUG" )
    list(APPEND ADDITIONAL_C_RELWITHDEBINFO_FLAGS   "-DDEBUG" )
  ENDIF()  
  
  ################################################################################
  # Warnings
  ################################################################################
  
  # Add the standard compiler warnings
  if ( NOT COMPILER_WARNINGS )
   
    IF ( APPLE  )
      # Skip unused parameters as it has to be used for the documentation via doxygen and the interfaces
      set ( COMPILER_WARNINGS "-W" "-Wall" "-Wno-unused" "-Wextra" "-Wno-non-virtual-dtor" "-Wno-unused-parameter" "-Wno-variadic-macros" CACHE STRINGLIST "This list contains the warning flags used during compilation " )
    elseif ("${CMAKE_CXX_COMPILER} ${CMAKE_CXX_COMPILER_ARG1}" MATCHES ".*clang")
      set ( COMPILER_WARNINGS "-W" "-Wall" "-Wextra" "-Wno-non-virtual-dtor" "-Wno-unused-parameter" "-Wno-variadic-macros" CACHE STRINGLIST "This list contains the warning flags used during compilation " )
    ELSEIF ( CMAKE_SYSTEM MATCHES "SunOS*" )
      set ( COMPILER_WARNINGS "" CACHE STRINGLIST "This list contains the warning flags used during compilation " )
    ELSE ()
      set ( COMPILER_WARNINGS "-W" "-Wall" "-Wno-unused" "-Wextra" "-Wno-variadic-macros" CACHE STRINGLIST "This list contains the warning flags used during compilation " )
  ENDIF()

  endif ( NOT COMPILER_WARNINGS )

  list(APPEND ADDITIONAL_CXX_FLAGS          ${COMPILER_WARNINGS} )
  list(APPEND ADDITIONAL_C_FLAGS            ${COMPILER_WARNINGS} )


  if ("${CMAKE_CXX_COMPILER}" MATCHES "Clang")
      list(APPEND ADDITIONAL_CXX_FLAGS "-Weverything")
      list(APPEND ADDITIONAL_CXX_FLAGS "-Wno-c++98-compat")
      list(APPEND ADDITIONAL_CXX_FLAGS "-Wno-c++98-compat-pedantic")
      list(APPEND ADDITIONAL_CXX_FLAGS "-Wno-padded")
      list(APPEND ADDITIONAL_CXX_FLAGS "-Wno-old-style-cast")
      list(APPEND ADDITIONAL_CXX_FLAGS "-Wno-documentation-unknown-command")
      list(APPEND ADDITIONAL_CXX_FLAGS "-Wno-unreachable-code-return")
      # enable later:
      list(APPEND ADDITIONAL_CXX_FLAGS "-Wno-sign-conversion")
      list(APPEND ADDITIONAL_CXX_FLAGS "-Wno-deprecated")
      list(APPEND ADDITIONAL_CXX_FLAGS "-Wno-weak-vtables")
  endif()
  
  ################################################################################
  # STL Vector checks
  ################################################################################
  
  # Pre initialize stl vector check variable
  if ( NOT STL_VECTOR_CHECKS )
    set ( STL_VECTOR_CHECKS false CACHE BOOL "Include full stl vector checks in debug mode (This option is only used in debug Mode!)" )
  endif ( NOT STL_VECTOR_CHECKS )
  
  # Add a flag to check stl vectors
  if ( STL_VECTOR_CHECKS AND NOT CMAKE_SYSTEM MATCHES "SunOS*"  )
    list(APPEND ADDITIONAL_CXX_DEBUG_FLAGS          "-D_GLIBCXX_DEBUG" )
    list(APPEND ADDITIONAL_CXX_DEBUG_FLAGS          "-D_GLIBCXX_DEBUG_PEDANTIC")
    list(APPEND ADDITIONAL_CXX_RELEASE_FLAGS        "-D_GLIBCXX_DEBUG" )
    list(APPEND ADDITIONAL_CXX_RELEASE_FLAGS        "-D_GLIBCXX_DEBUG_PEDANTIC")
    list(APPEND ADDITIONAL_CXX_RELWITHDEBINFO_FLAGS "-D_GLIBCXX_DEBUG" )
    list(APPEND ADDITIONAL_CXX_RELWITHDEBINFO_FLAGS "-D_GLIBCXX_DEBUG_PEDANTIC")
    
    list(APPEND ADDITIONAL_C_DEBUG_FLAGS            "-D_GLIBCXX_DEBUG" )
    list(APPEND ADDITIONAL_C_DEBUG_FLAGS            "-D_GLIBCXX_DEBUG_PEDANTIC")
    list(APPEND ADDITIONAL_C_RELEASE_FLAGS          "-D_GLIBCXX_DEBUG" )
    list(APPEND ADDITIONAL_C_RELEASE_FLAGS          "-D_GLIBCXX_DEBUG_PEDANTIC")
    list(APPEND ADDITIONAL_C_RELWITHDEBINFO_FLAGS   "-D_GLIBCXX_DEBUG" )
    list(APPEND ADDITIONAL_C_RELWITHDEBINFO_FLAGS   "-D_GLIBCXX_DEBUG_PEDANTIC")
  endif()

  ################################################################################
  # Process the additional flags:
  ################################################################################

  cmake_policy(SET CMP0057 NEW) 

  get_property(languages GLOBAL PROPERTY ENABLED_LANGUAGES)

  IF ( "CXX" IN_LIST languages  )
    # Add the debug flags
    foreach( flag ${ADDITIONAL_CXX_FLAGS} ${ADDITIONAL_CXX_DEBUG_FLAGS} )
      list (FIND ${CMAKE_CXX_FLAGS_DEBUG} ${flag} _index)
      if (${_index} EQUAL -1)
        set( CMAKE_CXX_FLAGS_DEBUG "${CMAKE_CXX_FLAGS_DEBUG} ${flag} ")
      endif()
    endforeach()

    # Add the release flags
    foreach( flag ${ADDITIONAL_CXX_FLAGS} ${ADDITIONAL_CXX_RELEASE_FLAGS} )
      list (FIND ${CMAKE_CXX_FLAGS_RELEASE} ${flag} _index)
      if (${_index} EQUAL -1)
        set( CMAKE_CXX_FLAGS_RELEASE "${CMAKE_CXX_FLAGS_RELEASE} ${flag} ")
      endif()
    endforeach()

    # Add the release with debug info flags
    foreach( flag ${ADDITIONAL_CXX_FLAGS} ${ADDITIONAL_CXX_RELWITHDEBINFO_FLAGS} )
      list (FIND ${CMAKE_CXX_FLAGS_RELWITHDEBINFO} ${flag} _index)
      if (${_index} EQUAL -1)
        set( CMAKE_CXX_FLAGS_RELWITHDEBINFO "${CMAKE_CXX_FLAGS_RELWITHDEBINFO} ${flag} ")
      endif()
    endforeach()
  ENDIF()


  IF ( "C" IN_LIST languages  )
    # Add the debug flags
    foreach( flag ${ADDITIONAL_C_FLAGS} ${ADDITIONAL_C_DEBUG_FLAGS} )
      list (FIND ${CMAKE_C_FLAGS_DEBUG} ${flag} _index)
      if (${_index} EQUAL -1)
        set( CMAKE_C_FLAGS_DEBUG "${CMAKE_C_FLAGS_DEBUG} ${flag} ")
      endif()
    endforeach()

    # Add the release flags
    foreach( flag ${ADDITIONAL_C_FLAGS} ${ADDITIONAL_C_RELEASE_FLAGS} )
        list (FIND ${CMAKE_C_FLAGS_RELEASE} ${flag} _index)
      if (${_index} EQUAL -1)
        set( CMAKE_C_FLAGS_RELEASE "${CMAKE_C_FLAGS_RELEASE} ${flag} ")
      endif()
    endforeach()

    # Add the release with debug info flags
    foreach( flag ${ADDITIONAL_C_FLAGS} ${ADDITIONAL_C_RELWITHDEBINFO_FLAGS} )
      list (FIND ${CMAKE_C_FLAGS_RELWITHDEBINFO} ${flag} _index)
      if (${_index} EQUAL -1)
        set( CMAKE_C_FLAGS_RELWITHDEBINFO "${CMAKE_C_FLAGS_RELWITHDEBINFO} ${flag} ")
      endif()
    endforeach()

  endif()
endif ()
