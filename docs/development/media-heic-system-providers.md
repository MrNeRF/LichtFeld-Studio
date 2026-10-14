# HEIC system-provider investigation

Status: local integration in progress, 14 October 2026. Existing explicit codec
entry points now select an OS adapter for a qualified HEVC subset. Dataset
discovery is not expanded. The libheif decoder remains AV1/dav1d only, with no
bundled HEVC fallback. Mac production integration still awaits its own tests;
the ImageIO probe results below do not substitute for them.

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

## Local integration progress

`image_heif.cpp` retains primary/derived-item validation, source precision,
ICC/NCLX handling and the bounded per-thread LCMS cache. AV1 uses dav1d as before.
HEVC selects `image_heic_system.cpp`; no alternate image service is introduced.
Both file and memory entry points share the adapter and existing target copying.
Metadata probing checks capabilities without copying all pixels. Encoded HEVC
leaf precision must agree across tiles and with the primary declaration.

Windows currently accepts RGB8 only, without silently reducing higher precision
or dropping auxiliary alpha. It selects the Microsoft WIC vendor. COM is local
to the call and respects existing STA/MTA initialization. The Linux adapter is
explicitly unavailable; the FFmpeg investigation does not enable a new fallback.

A stronger pixel comparison exposed a Windows limitation beyond successful
copying: the original planar-RGB tile grid returns incorrect RGB pixels despite
successful WIC calls. The adapter therefore rejects RGB grids on Windows. The
earlier matrix's successful grid copy describes only API success, not pixel
correctness. A real YUV iPhone grid decodes through the public display entry
point; compared with the previous independent-provider display reference, mean
absolute RGB8 difference is 1.00684, maximum 66, and 99.6948% of samples differ by
at most eight. This is one sample, not a relaxed acceptance threshold or an
ImageIO/independent-decoder equivalence claim.

Apple's local implementation accepts RGB8 and the qualified packed RGB101010
layout, normalizes native codes to full-range UINT16 and applies libheif's ordered
container rotations/mirrors once. A separate EXIF orientation is not applied a
second time. A single clean aperture before rotations/mirrors is handled by
ImageIO; repeated apertures or apertures after those transforms are rejected.
It rejects unqualified precision/alpha/layout and inconsistent
geometry instead of interpreting unknown storage. Production codec tests pass
on the Mac for this subset. A valid ICC combined with contradictory PQ/HLG NCLX
passes shared ICC validation, but ImageIO cannot provide the native pixels:
that combination returns an explicit unsupported error. Windows supports the
qualified combination; AVIF retains its existing ICC precedence. Native HDR
without a usable ICC remains unsupported rather than receiving implicit SDR
conversion.

Windows standalone codec/export groups pass, including original sample pixels,
file/memory equivalence, targets, ICC over PQ/HLG, malformed ICC, allocation
failure, Unicode aliases, concurrent calls and a caller already in STA. Linux
standalone groups pass with declared native-provider-unavailable skips;
existing AVIF tests execute. Formatting, I/O/backend checks and error baseline
have passed during development. Existing Windows CUDA/Vulkan and discovery
profiles also pass against the updated app library. Native RGB8 is exercised
through cold/warm cache and queue consumers on Windows CUDA/Vulkan and Mac
Metal/Vulkan; Mac also exercises packed RGB10. An isolated consumer with a system-only PATH
decodes the real photo without bundled libde265. Its observed working-set peaks
are roughly 500–580 MiB. Two versus twelve repeated decodes do not show growth
proportional to the decode count; this short diagnostic is not a general leak or
memory-budget guarantee. CMake stub regeneration fixes only a docstring wrapping
drift inherited from dev, and the subsequent stub check passes.

The approved integration transfer and Mac validation are complete. Standalone
codec/export groups pass on Windows, Mac and Linux. Mac has only the existing
unavailable `/dev/full` PNG/JPEG cases skipped; Linux explicitly lacks a native
HEVC provider. Dataset discovery recognizes `.heic` on Windows/macOS, while
preserving same-stem legacy companion counts and preview preference. Recognition
does not promise that every HEIC precision/layout or Windows extension is
available. Linux discovery remains disabled. Full installer qualification,
hosted CI and older OS versions remain outside these local results. No HEVC
decoder is added to the bundled libheif configuration.

## Additional local audit

The production Mac entry point decodes the real iPhone photo at 4032x3024.
Against the separately rendered ImageIO display reference, RGB8 maximum
difference is one and mean absolute difference is 0.04799. Against the previous
independent decoder reference the mean is 0.60190 and maximum is 62. These are
diagnostics for one photo, not acceptance thresholds or universal equivalence.
A second public sample, `examples/example.heic` from upstream libheif v1.21.2
(SHA256 `7f8b363e4936c0666a25f64f3a92fda10bd8e5453be4592530b65a55dd98f3f2`),
decodes at 1280x854 on Windows and Mac. Their mean absolute RGB8 difference is
0.83824 and maximum is 76. Public photos remain in the private validation dossier,
not in the committed fixtures. This is still a small real-photo corpus.

An isolated Windows regression compiles the actual private adapter with COM
factory creation returning `REGDB_E_CLASSNOTREG`. File, borrowed-memory and
metadata calls fail explicitly without modifying the output. The substitution
exists only in that test translation unit; production has no global test hook.
This tests the adapter boundary, not a full loader running on an OS with the
HEVC extension uninstalled. Extension availability is not changed on the host.

Recursive PE import inspection of the built Windows app resolves 90 dependencies
with none unresolved and no added libde265/x265/HEVC decoder. The isolated runtime
stage also passes the existing generated-license coverage check for 78 vcpkg
ports, including libheif, dav1d and lcms. The audit exposed an existing notice
generator crash on platform-conditional FFmpeg feature objects; its fix reports
conditions without reimplementing vcpkg expression evaluation and uses installed
status for the license. This does not qualify runtime-loaded plugins or a full
installer payload.

Mac uses SDK 27 for validation and the project deployment target is 26. The
adapter uses the historical CoreGraphics pixel-format mask name for SDK source
compatibility. An actual macOS 26 runtime and SDK 26 are not available in this
workspace, so neither is claimed as tested. Root-cache selection of Homebrew
OpenJPH and the isolated libheif dependency originally built for 27 were found
during the deployment audit. The private HEIF dependencies were rebuilt using
the project's macOS 26 triplet, and the root cache now selects the existing
vcpkg OpenJPH package. Standalone codec/export and app Metal/Vulkan/discovery
groups pass again; the codec has no Homebrew OpenJPH runtime dependency and the
affected builds no longer emit the newer-deployment warnings. This is build
and dependency validation, not execution on macOS 26.

Sources for API behavior:
[Microsoft HEIF extension codec](https://learn.microsoft.com/en-us/windows/win32/wic/heif-codec),
[WIC closest pixel format](https://learn.microsoft.com/en-us/windows/win32/api/wincodec/nf-wincodec-iwicbitmapsourcetransform-getclosestpixelformat),
[ImageIO primary selection](https://developer.apple.com/documentation/imageio/cgimagesourcegetprimaryimageindex(_:)),
[ImageIO float option](https://developer.apple.com/documentation/imageio/kcgimagesourceshouldallowfloat).

Export, metadata/DJI preservation, auxiliary depth integration and GUI changes
remain separate subsequent work. Do not expand this investigation into those
features or describe a diagnostic probe as finished application support.
