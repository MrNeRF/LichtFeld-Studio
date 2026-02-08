# fill string with spaces
macro (vci_format_string str length return)
    string (LENGTH "${str}" _str_len)
    math (EXPR _add_chr "${length} - ${_str_len}")
    set (${return} "${str}")
    while (_add_chr GREATER 0)
        set (${return} "${${return}} ")
        math (EXPR _add_chr "${_add_chr} - 1")
    endwhile ()
endmacro ()

# print message with color escape sequences if CMAKE_COLOR_MAKEFILE is set
string (ASCII 27 _escape)
function (vci_color_message _str)
    if (CMAKE_COLOR_MAKEFILE AND NOT WIN32)
        message (${_str})
    else ()
        string (REGEX REPLACE "${_escape}.[0123456789;]*m" "" __str ${_str})
        message (${__str})
    endif ()
endfunction ()

# info header
function (vci_print_configure_header _id _name)
    vci_format_string ("${_name}" 40 _project)
    if ( ${_id}_VERSION )
        vci_format_string ("${${_id}_VERSION}" 40 _version)
    else()
        vci_format_string ("${PROJECT_VERSION}" 40 _version)
    endif()
    vci_color_message ("\n${_escape}[40;37m************************************************************${_escape}[0m")
    vci_color_message ("${_escape}[40;37m* ${_escape}[1;31mVCI ${_escape}[0;40;34mBuildsystem${_escape}[0m${_escape}[40;37m                                          *${_escape}[0m")
    vci_color_message ("${_escape}[40;37m*                                                          *${_escape}[0m")
    vci_color_message ("${_escape}[40;37m* Package : ${_escape}[32m${_project} ${_escape}[37m      *${_escape}[0m")
    vci_color_message ("${_escape}[40;37m* Version : ${_escape}[32m${_version} ${_escape}[37m      *${_escape}[0m")

    if ( NOT WIN32 )
      # Just artistic. remove 2 spaces for release to make it look nicer ;-)
      if (${CMAKE_BUILD_TYPE} MATCHES "Debug")    
         vci_color_message ("${_escape}[40;37m* Type    : ${_escape}[32m${CMAKE_BUILD_TYPE} ${_escape}[37m                                         *${_escape}[0m")
      else()
         vci_color_message ("${_escape}[40;37m* Type    : ${_escape}[32m${CMAKE_BUILD_TYPE} ${_escape}[37m                                       *${_escape}[0m")
      endif()
    endif()

    vci_color_message ("${_escape}[40;37m************************************************************${_escape}[0m")
endfunction ()

# info line
function (vci_print_configure_footer)
    vci_color_message ("${_escape}[40;37m************************************************************${_escape}[0m\n")
endfunction ()

