
if(MSVC_MACROS_INCLUDED)
  return()
endif(MSVC_MACROS_INCLUDED)
set(MSVC_MACROS_INCLUDED TRUE)

set (MSVC_GROUPING ON CACHE BOOL "Group Files by folder structure on MSVC.")

# Enable project folders to group targets in solution folders on MSVC.
if(${MSVC_GROUPING})
set_property(GLOBAL PROPERTY USE_FOLDERS ON) 
endif(${MSVC_GROUPING})

MACRO (GROUP_PROJECT targetname groupname)
if(${MSVC_GROUPING})
set_target_properties(${targetname} 
   PROPERTIES 
   FOLDER "${groupname}")
endif(${MSVC_GROUPING})
ENDMACRO (GROUP_PROJECT)