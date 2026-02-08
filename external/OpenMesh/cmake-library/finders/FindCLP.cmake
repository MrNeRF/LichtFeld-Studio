# Try to find CLP: https://www.coin-or.org/Clp/
# On success, this will define the target Coin::CLP

if(NOT TARGET Coin::CLP)

if(NOT TARGET Coin::CoinUtils)
    find_package(CoinUtils QUIET)
    if(NOT TARGET Coin::CoinUtils)
        message(STATUS "FindCLP: Could not find CoinUtils as prerequisite of CLP.")
        return()
    endif()
endif()

find_path(CLP_INCLUDE_DIR
          NAMES ClpConfig.h
          PATHS "$ENV{CLP_DIR}/include/coin"
                "/opt/homebrew/include/clp/coin"  #homebrew default path
                "/usr/local/include/clp/coin"     #homebrew default path
                "/usr/local/include/coin-or"
                "/usr/include/coin"
                "/usr/include/coin-or"
                 "C:\\libs\\clp\\include"
                 "C:\\libs\\cbc\\include"
                 "${VS_SEARCH_PATH}CBC-2.9.7/Clp/include"
                 "${VS_SEARCH_PATH}CBC-2.9.4/Clp/include"
              )

find_library( CLP_LIBRARY
              NAMES Clp libClp
              PATHS "$ENV{CLP_DIR}/lib"
                    "/opt/homebrew/lib"  # homebrew default path
                    "/usr/local/lib"     # homebrew default path
                    "/usr/local/lib/coin-or"
                    "/usr/local/lib/coin"
                    "/usr/lib/coin-or"
                    "/usr/lib/coin"
                    "/usr/lib"
                    "C:\\libs\\clp\\lib"
                    "C:\\libs\\cbc\\lib"
              )

set(CLP_INCLUDE_DIRS "${CLP_INCLUDE_DIR}" )
set(CLP_LIBRARIES "${CLP_LIBRARY}" )

include(FindPackageHandleStandardArgs)
find_package_handle_standard_args(CLP  DEFAULT_MSG
                                  CLP_LIBRARY CLP_INCLUDE_DIR)

if(CLP_FOUND)
    add_library(Coin::CLP SHARED IMPORTED)
    set_property(TARGET Coin::CLP PROPERTY IMPORTED_LOCATION ${CLP_LIBRARY})
    target_include_directories(Coin::CLP INTERFACE ${CLP_INCLUDE_DIR})
    target_link_libraries(Coin::CLP INTERFACE Coin::CoinUtils)
endif()


mark_as_advanced(CLP_INCLUDE_DIR CLP_LIBRARY)
endif(NOT TARGET Coin::CLP)
