# Official JPEG XL static-image regressions

These bitstreams are copied unchanged from `libjxl/conformance`, revision
`b1d0f990b03e57bf6d137c365cd5dc8b470b9191`:
https://github.com/libjxl/conformance/tree/b1d0f990b03e57bf6d137c365cd5dc8b470b9191/testcases

- `grayscale.jxl`: `grayscale/input.jxl`, VarDCT grayscale with an original ICC
  different from the decoded-data ICC. Reference dimensions 200x200x1. Reference
  Float32 values at (y,x): (50,50)=0.08087914437055588,
  (100,100)=0.3699168562889099, (199,199)=1.001685380935669.
- `patches-lossless.jxl`: `patches_lossless/input.jxl`, lossless modular patches.
- `sunset-logo.jxl`: `sunset_logo/input.jxl`, 10-bit, orientation and two layers
  coalesced to one static image, with reconstructed color values outside [0,1].
  Author Jon Sneyers; marked CC0/public domain in the upstream `source` file.

The corpus BSD-3-Clause notice is included verbatim in `LICENSE`. Original
source declarations, where supplied upstream, remain applicable. No animation
or auxiliary-channel acceptance is implied by these static regressions.

The separate local audit checked all 40 upstream cases at this revision, using
SHA256-verified reference arrays/profiles and official per-case tolerances.
These small cases keep regressions permanent without adding the entire corpus
or requiring network access during tests.
