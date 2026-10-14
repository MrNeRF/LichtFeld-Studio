# Original AV1/HEIF decoder fixtures

Procedural RGB/RGBA/monochrome images, licensed GPL-3.0-or-later with this repo.
No photographs or third-party image samples are included. `rgb8.heic` is an
original earlier HEVC grid retained for OS-provider and unsupported-codec tests;
no HEVC decoder or encoder is needed to execute the tests.

Regenerate the positive fixtures with:

```
uv run --no-project --with pillow-heif==0.18.0 python tests/data/heif/generate.py
```

That isolated wheel supplies AOM AV1 encoding and independent decoding. It is
not a build, runtime or CI dependency. The application uses dav1d and enables
no encoder. Tests consume committed `.avif` files and reference pixels.
`manifest.json` records actual decoded dimensions, precision, tags and hashes.

High-precision inputs originate from adjacent native 10/12-bit codes within
one RGB8 bucket, including RGBA12. Independent reference samples are left-aligned
little-endian; the application's UINT16 contract expands them to full range.
SDR P3, BT.709 and linear transfers also have scalar color references.

The generator explicitly appends a second NCLX property beside ICC, updating
box sizes, associations and absolute item offsets. Tests change only its
transfer field to sRGB/PQ/HLG and read back both declarations before asserting
precedence. Invalid ICC still fails. ICC dates are fixed for repeatability.

Container `irot`/`imir` properties are authored explicitly; EXIF alone does not
establish transform coverage. The RGB8 fixture is a 2x2 grid of original 64x64
AV1 tiles cropped to 128x96, exercising partial bottom tiles. Fixture-specific
container construction runs only during generation; no image encoder is used
by the test executable. The other fixtures are ordinary single images except
the primary-selection case. PQ/HLG without ICC is raw-readable but rejected by
implicit SDR display loading.

`system-*.heic` and their independent `system-*.reference` buffers are original
procedural HEVC fixtures recovered from the earlier fixture generator at
`e506c66c4549b8512d0627f73a0cccf30d295883`. They qualify only OS-provided decoding;
they do not add an HEVC codec dependency. Native 10/12-bit references are
left-aligned, as for the AV1 fixtures. `system-grid.reference` belongs to the
retained `rgb8.heic`, not to the newer AV1 grid.

The two system ICC fixtures also use the existing `add_fixture_property` helper
to append a real second NCLX declaration without re-encoding. Tests read it back
and exercise ICC precedence over sRGB/PQ/HLG. The AV1 regeneration command above
does not regenerate these historical HEVC files. No encoder is needed to run
either set of tests. Native availability, precision, alpha and RGB-grid limits
are explicit; see `docs/development/media-heif.md`.

`system-yuv420.heic` is an original 32x24 limited-range YCbCr 4:2:0 fixture,
authored with pillow-heif 0.18.0 (HEVC quality 90, chroma 420, NCLX primaries 1,
transfer 13, matrix 1, full-range flag 0). Source RGB is
`(48+3*x, 64+4*y, 80+x+y)`. Its reference is the independent wheel decoder's
RGB8 output. Native comparisons allow at most six RGB8 levels and a mean at
most two for YUV reconstruction differences (Apple measured maximum five, mean
1.39149); bitstream range/matrix/transfer were independently checked with
FFmpeg 8. File/memory/direct-target equivalence remains
exact. Tests also verify the declared YCbCr/420 format, and mutate only the NCLX
transfer for HDR error regressions.

`system-combined.heic` appends essential `imir=1` after the original rotated
fixture's `irot`, using the existing property helper without re-encoding.
Its reference horizontally mirrors the 24x32 rotated reference. Apple must
match these independent pixels; Windows explicitly rejects this unqualified
combination instead of relying on undocumented WIC transform composition.
