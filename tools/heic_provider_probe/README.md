# HEIC system-provider qualification

Standalone diagnostic tools, not an application feature. They are deliberately
absent from root build/install/package targets and link only OS frameworks.
No HEVC decoder, codec installer, downloader or runtime fallback is added.

Build on Windows or macOS:

```sh
cmake -S tools/heic_provider_probe -B /private/path/probe-build -DCMAKE_BUILD_TYPE=Release
cmake --build /private/path/probe-build
/private/path/probe-build/heic_provider_probe input.heic /private/path/results
```

Use trusted qualification images and separate output directories. Outputs may
contain image pixels, ICC profiles and metadata; keep private photographs and
their outputs outside the repository. The byte-buffer limit is a probe guard,
not a bound on OS decoder internal allocations. These tools do not qualify a
production security boundary or implement the application's image contracts.

Windows reports the selected decoder/vendor, frame zero, native pixel format,
dimensions, bits/channel layout, ICC color contexts and closest formats offered
by `IWICBitmapSourceTransform`. `native.bin` is the native packed buffer, without
an explicit format/color conversion. A format-converter's ability to synthesize
16-bit output would not demonstrate source precision. Frame zero is intentionally
reported as such, not assumed universally equivalent to HEIF primary selection.

macOS explicitly selects `CGImageSourceGetPrimaryImageIndex`, permits float
output, reports actual CGImage storage/layout and copies the data provider.
`native.bin` may use packed 10-bit storage; bytes-per-component must not be
inferred from the number of channels. `properties.plist` retains reported
geometry/orientation metadata. No drawing into an sRGB context, automatic
orientation, tone mapping or manual color conversion is performed by the probe.
OS decoder behavior still requires qualification independently.

An image handle or dimensions alone do not prove decode success: pixel copying
must succeed. These tools return nonzero when opening/copying fails, and they
do not install a missing system codec.

Linux is intentionally not given a production adapter here. Evaluate a selected
external executable/library by absolute path and record its version before every
qualification run. Do not use the runner's old system FFmpeg 6.1. The qualified
reference in this investigation is FFmpeg 8.0.1, already built for project tests.
HEVC decoder presence alone does not qualify HEIF parsing, primary selection,
tile-grid assembly, auxiliary alpha, ICC or geometric transforms.

See `docs/development/media-heic-system-providers.md` for results and remaining
integration gates. Historical original fixtures recovered for private tests are
not copied back into the source tree or shipped by these tools.
