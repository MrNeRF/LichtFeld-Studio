# Native x64-windows linkage, with one dependency configuration.
# Keep this recipe identical for Release and RelWithDebInfo consumers.
set(VCPKG_TARGET_ARCHITECTURE x64)
set(VCPKG_CRT_LINKAGE dynamic)
set(VCPKG_LIBRARY_LINKAGE dynamic)
set(VCPKG_BUILD_TYPE release)
