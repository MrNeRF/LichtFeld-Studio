# Once done this will (on success) define the following targets:
#       Mosek::MosekC    - only the C interface
#       Mosek::FusionCXX - C and C++ interface

# The enviroment variable MOSEK_DIR can be used to initialize MOSEK_BASE

# Tested with Mosek 9.2 on Linux

if (NOT TARGET Mosek::MosekC)

find_path (MOSEK_BASE
    NAMES tools/platform
    HINTS $ENV{MOSEK_DIR}
    PATHS
        /opt/mosek/9.2
        /opt/mosek/9.3
        $ENV{MOSEK_DIR}
    DOC "Base path of your Mosek installation, e.g. /opt/mosek/9.2")

if (${CMAKE_HOST_SYSTEM_NAME} MATCHES "Windows")
    set(DEFAULT_MOSEK_PLATFORM "win64x86")
elseif (${CMAKE_HOST_SYSTEM_NAME} MATCHES "Darwin")
    set(DEFAULT_MOSEK_PLATFORM "osx64x86")
    # TODO: arm64 (also on other platforms). I currently don't have Mosek installed,
    # so I cant't test this.
elseif (${CMAKE_HOST_SYSTEM_NAME} MATCHES "Linux")
    set(DEFAULT_MOSEK_PLATFORM "linux64x86")
else()
    set(DEFAULT_MOSEK_PLATFORM "unknown")
endif()

set (MOSEK_PLATFORM "${DEFAULT_MOSEK_PLATFORM}" CACHE STRING "mosek platform, e.g. linux64x86 or osx64x86")
mark_as_advanced(MOSEK_PLATFORM)

set (MOSEK_PLATFORM_PATH "${MOSEK_BASE}/tools/platform/${MOSEK_PLATFORM}")

find_path (MOSEK_INCLUDE_DIR
           NAMES mosek.h
           PATHS "${MOSEK_PLATFORM_PATH}/h"
          )

find_path (MOSEK_LIBRARY_DIR
           NAMES libmosek64.dylib
                 libmosek64.so
                 mosek64_9_2.dll # TODO: multi-version detection?
           PATHS "${MOSEK_PLATFORM_PATH}/bin")

find_library (MOSEK_LIBRARY
              NAMES mosek64
              PATHS "${MOSEK_LIBRARY_DIR}")

if(MOSEK_LIBRARY AND NOT TARGET Mosek::MosekC)
    add_library(Mosek::MosekC STATIC IMPORTED)
    target_include_directories(Mosek::MosekC INTERFACE ${MOSEK_INCLUDE_DIR})
    set_target_properties(Mosek::MosekC PROPERTIES IMPORTED_LOCATION ${MOSEK_LIBRARY})
endif()

find_path(MOSEK_SRC_DIR NAMES "SolverInfo.cc" PATHS "${MOSEK_PLATFORM_PATH}/src/fusion_cxx/")

if(MOSEK_LIBRARY AND MOSEK_SRC_DIR AND NOT TARGET Mosek::FusionCXX)
    add_library(FusionCXX EXCLUDE_FROM_ALL
        "${MOSEK_SRC_DIR}/BaseModel.cc"
        "${MOSEK_SRC_DIR}/Debug.cc"
        "${MOSEK_SRC_DIR}/IntMap.cc"
        "${MOSEK_SRC_DIR}/SolverInfo.cc"
        "${MOSEK_SRC_DIR}/StringBuffer.cc"
        "${MOSEK_SRC_DIR}/mosektask.cc"
        "${MOSEK_SRC_DIR}/fusion.cc")
    target_link_libraries(FusionCXX PUBLIC Mosek::MosekC)
    set_target_properties(FusionCXX PROPERTIES POSITION_INDEPENDENT_CODE ON)
    target_compile_features(FusionCXX INTERFACE cxx_std_11)
    target_include_directories(FusionCXX PUBLIC ${MOSEK_SRC_DIR})

    add_library(Mosek::FusionCXX ALIAS FusionCXX)
endif()

# legacy support:
set(MOSEK_INCLUDE_DIRS ${MOSEK_INCLUDE_DIR})
set(MOSEK_LIBRARIES Mosek::MosekC Mosek::FusionCXX)

include(FindPackageHandleStandardArgs)
find_package_handle_standard_args(Mosek DEFAULT_MSG MOSEK_LIBRARY MOSEK_INCLUDE_DIR)

endif (NOT TARGET Mosek::MosekC)
