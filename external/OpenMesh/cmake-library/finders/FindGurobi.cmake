# Once done this will define
#  Gurobi_FOUND - System has Gurobi
#  Targets:
#       Gurobi::GurobiC   - only the C interface
#       Gurobi::GurobiCXX - C and C++ interface

if (NOT TARGET Gurobi::GurobiCXX)
find_path(GUROBI_HOME
          NAMES include/gurobi_c++.h
          PATHS
          $ENV{GUROBI_HOME}
          "/opt/gurobi/linux64/"
          NO_DEFAULT_PATH # avoid finding /usr
          )

find_path(GUROBI_INCLUDE_DIR
    NAMES gurobi_c++.h
    HINTS
    "${GUROBI_HOME}/include"
    )
mark_as_advanced(GUROBI_INCLUDE_DIR)

set(GUROBI_BIN_DIR "${GUROBI_HOME}/bin")
set(GUROBI_LIB_DIR "${GUROBI_HOME}/lib")

if (WIN32)
    file(GLOB GUROBI_LIBRARY_LIST
        RELATIVE ${GUROBI_BIN_DIR}
        ${GUROBI_BIN_DIR}/gurobi*.dll)
elseif(APPLE)
      file(GLOB GUROBI_LIBRARY_LIST
              RELATIVE ${GUROBI_LIB_DIR}
              ${GUROBI_LIB_DIR}/libgurobi*.dylib)
else()
    file(GLOB GUROBI_LIBRARY_LIST
        RELATIVE ${GUROBI_LIB_DIR}
        ${GUROBI_LIB_DIR}/libgurobi*.so)
endif()

# Ignore libgurobiXY_light.so, libgurobi.so (without version):
string(REGEX MATCHALL
    "gurobi([0-9]+)\\..*"
    GUROBI_LIBRARY_LIST
    "${GUROBI_LIBRARY_LIST}"
    )

string(REGEX REPLACE
    ".*gurobi([0-9]+)\\..*"
    "\\1"
    GUROBI_LIBRARY_VERSIONS
    "${GUROBI_LIBRARY_LIST}")
list(LENGTH GUROBI_LIBRARY_VERSIONS GUROBI_NUMVER)

#message("GUROBI LIB VERSIONS: ${GUROBI_LIBRARY_VERSIONS}")

if (GUROBI_NUMVER EQUAL 0)
    message(STATUS "Found no Gurobi library version, GUROBI_HOME = ${GUROBI_HOME}.")
elseif (GUROBI_NUMVER EQUAL 1)
    list(GET GUROBI_LIBRARY_VERSIONS 0 GUROBI_LIBRARY_VERSION)
else()
    # none or more than one versioned library -let's try without suffix,
    # maybe the user added a symlink to the desired library
    message(STATUS "Found more than one Gurobi library version (${GUROBI_LIBRARY_VERSIONS}), trying without suffix. Set GUROBI_LIBRARY if you want to pick a certain one.")
    set(GUROBI_LIBRARY_VERSION "")
endif()

if (WIN32)
    find_library(GUROBI_LIBRARY
        NAMES "gurobi${GUROBI_LIBRARY_VERSION}"
        PATHS
        ${GUROBI_BIN_DIR}
    )
    find_library(GUROBI_IMPLIB
        NAMES "gurobi${GUROBI_LIBRARY_VERSION}"
        PATHS
        ${GUROBI_LIB_DIR}
    )
    mark_as_advanced(GUROBI_IMPLIB)
else ()
    find_library(GUROBI_LIBRARY
        NAMES "gurobi${GUROBI_LIBRARY_VERSION}"
        PATHS
        ${GUROBI_LIB_DIR}
    )
endif()
mark_as_advanced(GUROBI_LIBRARY)

if(GUROBI_LIBRARY AND NOT TARGET Gurobi::GurobiC)
    add_library(Gurobi::GurobiC SHARED IMPORTED)
    target_include_directories(Gurobi::GurobiC INTERFACE ${GUROBI_INCLUDE_DIR})
    set_target_properties(Gurobi::GurobiC PROPERTIES IMPORTED_LOCATION ${GUROBI_LIBRARY})
    if (GUROBI_IMPLIB)
        set_target_properties(Gurobi::GurobiC PROPERTIES IMPORTED_IMPLIB ${GUROBI_IMPLIB})
    endif()
endif()

# Gurobi ships with some compiled versions of its C++ library for specific
# compilers, however it also comes with the source code. We will compile
# the source code outselves -- this is much safer, as it guarantees the same
# compiler version and flags.
# (Note: doing this is motivated by actual sometimes-subtle ABI compatibility bugs)

find_path(GUROBI_SRC_DIR NAMES "Model.h" PATHS "${GUROBI_HOME}/src/cpp/")
mark_as_advanced(GUROBI_SRC_DIR)

file(GLOB GUROBI_CXX_SRC CONFIGURE_DEPENDS ${GUROBI_SRC_DIR}/*.cpp)
if(TARGET Gurobi::GurobiC AND GUROBI_CXX_SRC AND NOT TARGET Gurobi::GurobiCXX)
    add_library(GurobiCXX STATIC EXCLUDE_FROM_ALL ${GUROBI_CXX_SRC})
    add_library(Gurobi::GurobiCXX ALIAS GurobiCXX)

    target_compile_options(GurobiCXX
        PRIVATE
        $<$<CXX_COMPILER_ID:GNU>:-Wno-deprecated-copy>
        $<$<CXX_COMPILER_ID:Clang>:-Wno-deprecated-copy>
        )

    if(MSVC)
        target_compile_definitions(GurobiCXX PRIVATE "WIN64")
    endif()

    target_include_directories(GurobiCXX PUBLIC ${GUROBI_INCLUDE_DIR})
    target_link_libraries(GurobiCXX PUBLIC Gurobi::GurobiC)
    # We need to be able to link this into a shared library:
    set_target_properties(GurobiCXX PROPERTIES POSITION_INDEPENDENT_CODE ON)

endif()

# legacy support:
set(GUROBI_INCLUDE_DIRS "${GUROBI_INCLUDE_DIR}")
set(GUROBI_LIBRARIES Gurobi::GurobiC Gurobi::GurobiCXX)

include(FindPackageHandleStandardArgs)
find_package_handle_standard_args(Gurobi  DEFAULT_MSG GUROBI_LIBRARY GUROBI_INCLUDE_DIR GUROBI_SRC_DIR)

endif (NOT TARGET Gurobi::GurobiCXX)
