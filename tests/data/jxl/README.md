# JPEG XL fixture

`rgb8.jxl` is an original 2x2, lossless, straight RGB8 sRGB image. Every pixel
is `(255, 0, 0)`. It is generated separately from the application's decoder
by `generate.cpp` using the upstream libjxl 0.12.0 encoder API.

`rgb16.jxl` has the same dimensions and sRGB encoding, with every RGB UInt16
pixel equal to `(65535, 32768, 0)`. Pass `16` after the destination filename
to generate it.

`rgb32.jxl` is linear RGB Float32 with every pixel `(2, 0.5, -0.25)`.
Pass `32` to generate it. This verifies preservation of values outside the
integer range through the existing Float32 prepared-cache policy. All three
fixtures use the bare codestream representation.

The source and fixture are GPL-3.0-or-later, like these tests. No external
photograph or third-party image is redistributed. The fixed bytes allow import
and discovery tests to run without generating their own input at runtime.

Compile `generate.cpp` with libjxl and run it with the destination filename as
its sole argument. The application does not gain an export API from this tool.
