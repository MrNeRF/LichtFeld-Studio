# Try to find Coin-ORs CoinUtils: https://www.coin-or.org/
# On success, this will define the target Coin::CoinUtils
if (NOT TARGET Coin::CoinUtils)

find_path(CoinUtils_INCLUDE_DIR
    NAMES CoinPragma.hpp
    PATHS "$ENV{CoinUtils_DIR}/include/coin"
          "/opt/homebrew/include/coinutils/coin"  #homebrew default path
          "/usr/local/include/coinutils/coin"     #homebrew default path
          "/usr/local/include/coin-or"
          "/usr/include/coinutils"
          "/usr/include/coin-or"
          "/usr/include/coin"
          "/usr/lib/coin-or"
          "C:\\libs\\coinutils\\include"
    )

find_library( CoinUtils_LIBRARY
    NAMES CoinUtils
    PATHS "$ENV{CoinUtils_DIR}/lib"
        "/opt/homebrew/lib"  # homebrew default path
        "/usr/local/lib"     # homebrew default path
        "/usr/local/lib/coin-or"
        "/usr/local/lib/coin"
        "/usr/lib/coin-or"
        "/usr/lib/coin"
        "/usr/lib"
    )

mark_as_advanced(CoinUtils_INCLUDE_DIR CoinUtils_LIBRARY)

include(FindPackageHandleStandardArgs)
find_package_handle_standard_args(CoinUtils DEFAULT_MSG CoinUtils_INCLUDE_DIR CoinUtils_LIBRARY)

if(CoinUtils_FOUND)
    add_library(Coin::CoinUtils SHARED IMPORTED)
    set_property(TARGET Coin::CoinUtils PROPERTY IMPORTED_LOCATION ${CoinUtils_LIBRARY})
    target_include_directories(Coin::CoinUtils INTERFACE ${CoinUtils_INCLUDE_DIR})
    set(CoinUtils_LIBRARIES "Coin::CoinUtils" )
endif()


endif (NOT TARGET Coin::CoinUtils)
