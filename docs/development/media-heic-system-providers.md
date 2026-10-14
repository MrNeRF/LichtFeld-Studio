# HEIC system-provider investigation

Status: local qualification only, 14 October 2026. No HEIC support is enabled in
the application by this branch. The integrated AVIF/libheif/dav1d provider and
its HEVC rejection remain unchanged. No bundled HEVC fallback is introduced.

The maintainer's distribution decision concerns HEVC decoders, not simply the
choice of libde265. An external/system provider proposal needs agreement before
release integration; changing libraries or calling a process is not a general
patent/licensing guarantee. Preserve the existing video stack independently.

## Reuse and scope

`lfs_image_codecs`, its file/memory/allocator entry points and existing raw/display
separation remain the integration points. The repository has no existing WIC or
ImageIO image reader to reuse. Do not introduce a second image service, atomic
implementation or unconditional new dataset extension before qualification.

`tools/heic_provider_probe` contains standalone Windows/macOS probes linked only
to OS APIs. They are not linked/installed with the app. Trusted test inputs and
their output pixels/profiles stay in the private qualification workspace.

## Measured provider matrix

These are observations on the tested systems, not promises for all OS versions,
extensions, hardware or camera files.

| Contract | Windows WIC | macOS ImageIO | External FFmpeg 8.0.1 on Linux |
| --- | --- | --- | --- |
| Provider in environment | Microsoft HEIF 1.2.57.0, HEVC 2.5.33.0 | macOS 27.2 frameworks | Existing private project reference build; no system libheif installed |
| iPhone 13 P3 photo | Native pixel copy succeeds at 4032x3024 | Native pixel copy succeeds at 4032x3024 | Not tested in this increment |
| Original native 10-bit RGB | Exposes 32bpp BGR/8-bit components | Packed RGB101010; unpacked codes exactly match independent reference | RGB48 normalization differs from independently normalized reference by at most one UINT16 level |
| Original native 12-bit RGB/RGBA | Actual pixel copying fails, HRESULT 0xc00d3e8c | CGImage exposes dimensions/layout, but copying pixels fails | RGB12 normalized samples exact; auxiliary alpha decoded separately and exact after normalization |
| Rotation/mirroring | Tested output matches independent transformed pixels exactly | Exposes untransformed pixels; adapter must apply metadata exactly once | Full geometry qualification still needed |
| Non-first primary fixture | Frame zero yields the expected 16x16 primary in this sample | Explicit primary index 1 yields correct 16x16 pixels | Default sample yields expected output size; general selection still to qualify |
| Display P3 ICC | Original profile bytes and raw RGB8 match independent reference exactly | Original profile bytes and raw RGB8 match independent reference exactly | Full ICC preservation/application not yet qualified |
| Partial grid | Original 128x96 grid copies successfully | Real iPhone grid copies successfully | Default CLI outputs only a tile. Explicit four-tile assembly plus crop yields the original 128x96 reference exactly |
| PQ/HLG HEVC12 samples | Pixel copy fails | Pixel copy fails | Samples decode; implicit SDR conversion remains prohibited by app policy |

AVIF12 is also a control: WIC exposes 8-bit output for the tested source;
ImageIO successfully copies its 16-bit buffer. Thus the ImageIO HEVC12 failure
is not evidence that the probe cannot copy any 16-bit ImageIO image. It still
does not establish the root cause or universal lack of HEVC12 support.

The Linux FFmpeg reference uses the explicit path
`/task/ffmpeg8-reference-build/ffmpeg`, version 8.0.1, with its matching ffprobe.
An initial accidental test of system FFmpeg 6.1 is excluded from conclusions.
Version checking and an explicit executable path are mandatory; no PATH fallback.

FFmpeg's default CLI conversion is not a complete HEIC adapter. Grid tile
coordinates, primary selection, clean aperture and auxiliary associations must
be handled. The RGBA12 test's default RGBA conversion produced opaque alpha;
selecting its separate gray12 auxiliary stream recovered the original alpha.
This demonstrates building blocks, not a completed general reader.

The fixture-specific explicit four-tile test uses coordinates verified against
ffprobe stream-group data. Do not copy its hardcoded layout into production.
FFmpeg describes the general canvas/offset contract in
[AVStreamGroupTileGrid](https://www.ffmpeg.org/doxygen/8.1/structAVStreamGroupTileGrid.html).

## Required decisions before integration

1. Keep AV1 on the current deterministic provider. Select HEVC providers from
   actual codec identity, not filename; never route AVIF through the less precise
   WIC path merely because it is installed.
2. Preserve numerical precision. WIC availability does not authorize silently
   reducing 10-bit training data to 8-bit or expanding 8-bit output and labeling
   it native 16-bit. Decide whether a limited display-only provider is useful;
   expose explicit unsupported behavior for unqualified raw contracts.
3. Normalize ImageIO packed 10-bit storage using the verified layout. Apply
   orientation/crop once. Preserve ICC precedence over NCLX and reject implicit
   HDR-to-SDR paths rather than drawing everything through an sRGB context.
4. Qualify file/memory/allocator reads, alpha/premultiplication, malformed and
   truncated inputs, primary/derived grids, Unicode, resource limits and worker
   concurrency. Production COM handling must coexist with initialized caller
   apartments; the probe is a standalone MTA process, not that implementation.
5. For Linux, agree whether an optional externally installed provider is within
   policy. Avoid loading a second libheif ABI into the process or enabling global
   plugin discovery. Establish executable/provider identity and compatibility,
   precise errors, cancellation and lifetime before any subprocess proposal.
6. Validate unavailable-provider behavior without installing/uninstalling user
   extensions; deterministic injected absence tests belong with integration.
7. Only then advertise HEIC dataset discovery. Reuse existing raw/display,
   quantization, resize and cache contracts; test cold/warm GPU consumers and
   confirm packages/notices include no new HEVC decoder.

Sources for API behavior:
[Microsoft HEIF extension codec](https://learn.microsoft.com/en-us/windows/win32/wic/heif-codec),
[WIC closest pixel format](https://learn.microsoft.com/en-us/windows/win32/api/wincodec/nf-wincodec-iwicbitmapsourcetransform-getclosestpixelformat),
[ImageIO primary selection](https://developer.apple.com/documentation/imageio/cgimagesourcegetprimaryimageindex(_:)),
[ImageIO float option](https://developer.apple.com/documentation/imageio/kcgimagesourceshouldallowfloat).

Export, metadata/DJI preservation, auxiliary depth integration and GUI changes
remain separate subsequent work. Do not expand this investigation into those
features or describe a diagnostic probe as finished application support.
