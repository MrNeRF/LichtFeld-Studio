# Native Apple Silicon linkage, with one dependency configuration.
set(VCPKG_TARGET_ARCHITECTURE arm64)
set(VCPKG_CRT_LINKAGE dynamic)
set(VCPKG_LIBRARY_LINKAGE static)
set(VCPKG_CMAKE_SYSTEM_NAME Darwin)
set(VCPKG_OSX_ARCHITECTURES arm64)
set(VCPKG_BUILD_TYPE release)

# Keep dependency binaries on the same macOS baseline as the native viewer.
set(VCPKG_OSX_DEPLOYMENT_TARGET 26.0)

# curl 8.19's function-only probe finds pipe2 in SDK 27 even when the
# deployment target cannot use it. Keep curl's existing pipe fallback.
if(PORT STREQUAL "curl" AND VCPKG_OSX_DEPLOYMENT_TARGET VERSION_LESS 27.0)
    list(APPEND VCPKG_CMAKE_CONFIGURE_OPTIONS "-DHAVE_PIPE2=OFF")
endif()
