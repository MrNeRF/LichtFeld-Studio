# OpenEXR overlay

Pins OpenEXR 3.4.15 independently of the builtin registry version, so the
repository's existing vcpkg baseline can build the required codecs.

The default port builds OpenEXRCore with libdeflate and OpenJPH. It does not
build or depend on Imath, Iex, IlmThread, OpenEXR, or OpenEXRUtil libraries.
OpenEXRCore uses its upstream built-in half conversion implementation.
The generated IlmThread configuration header remains available for Core's
internal threading configuration; it does not introduce an IlmThread library
dependency.

The optional `cpp` feature builds the upstream C++ libraries and adds Imath.
The project's `tests` feature enables it for the independent EXR reader/writer
used by codec tests. The application continues to link only OpenEXRCore.

`core-only.patch` separates the upstream C++ build, removes Core's Imath
dependency, and adjusts installed CMake and pkg-config metadata accordingly.
