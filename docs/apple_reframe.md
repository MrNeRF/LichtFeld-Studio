# Apple Reframe photo reconstruction

This optional, experimental backend creates an editable `SplatData` from a
single still photo using Apple's installed Photos Reframe predictors. It does
not include ML-SHARP, PyTorch, model weights, or Apple's framework binaries.

Enable it explicitly when configuring a macOS build:

```sh
cmake -S . -B build <your usual macOS configure arguments> -DLFS_ENABLE_APPLE_REFRAME=ON
```

Append the option to your normal macOS configure command or preset.
A macOS 27 or newer SDK/toolchain is required for the helper.
The application deployment target remains macOS 26; only `bin/lfs-reframe` is
compiled for arm64/macOS 27. The application does not link AlchemistBase.
The helper is installed alongside the executable, including inside portable
app bundles. Portable installation discovers and ad-hoc signs all nested Mach-O
files, including `lfs-reframe`, before sealing and verifying the bundle. This
ad-hoc signature is for local distribution, not notarization. For notarized
releases, re-sign the helper and other nested code with the release Developer ID
and hardened-runtime options before signing and notarizing the outer bundle.

`LFS_ENABLE_APPLE_REFRAME` defaults to **OFF**. Keep it **OFF for App Store
builds**. The disabled build compiles the unavailable backend and neither
builds nor installs the helper, private Swift interface/linking adapters, or
model resolver. Use a fresh packaging destination when switching variants.

On a supported Mac, choose **File → Import → Create Splat from Photo - Apple
Reframe**. The item is absent unless all of these are true:

- This build includes the optional backend and its executable helper.
- The process runs natively on Apple Silicon with macOS 27 or newer.
- The helper successfully launches and loads both registered Photos models.

Model readiness is checked in a background worker at startup and refreshed on
menu access after 30 seconds. Menus read the cached result without waiting; the
item stays hidden until the first successful check. Operator items dispatch by
stable operator ID and do not consume positional callback indices. Inference
validates availability on the import worker rather than repeating the readiness
probe on the UI thread. Probe failures are logged at debug level. If assets are missing, use the
spatial-photo feature in Photos and let Apple's downloads finish. LichtFeld
does not download/repackage these assets or request Apple's reserved
entitlements. Apple's active UAF asset set is preferred; discovery falls back
only to a unique installed model pair. Ambiguous models hide the menu item.

Processing runs in an isolated process with a three-minute limit. Cancel stops
the helper and discards its temporary output. A failed or cancelled generation
does not add a scene node. Successful reconstruction adds a uniquely named
node, supports undo/redo and ordinary `.licht` persistence, and does not treat
the original photo as a reloadable splat file.

The helper respects photo orientation and color profiles, tone-maps HDR to SDR,
and caps its input long edge at 4096 pixels. Reframe returns half-float
positions, scalar-first quaternions, linear scales/opacities and degree-zero SH
color coefficients in linear light. Conversion changes SH0 from linear light
to the viewer's sRGB encoding, normalizes quaternions, takes the logarithm of
scales and converts alpha to logits with finite endpoint clamping. The resulting
tensors use the active GPU backend. A node transform restores the source image
aspect ratio from Apple's square canonical rays. The visible reconstruction's
conservative three-sigma bounds rest on the scene's Y=0 ground plane and are
centered horizontally. Oversized reconstructions are uniformly reduced to a
four-unit envelope suitable for the standard camera; smaller ones are not enlarged.
The standard dataset-to-world boundary keeps it upright in LichtFeld's scene axes.
A 180-degree rotation around world Y makes the source-facing side face the
standard scene camera, rotating about the grounded origin.
Import does not alter the camera, FOV, pivot, projection mode or Home position.
Background pixels are reconstructed too; this feature does not remove them.

Private interfaces and installed models can change across OS updates. A failed
capability probe hides the item; errors during inference leave the scene intact.
Single-photo reconstructions have incomplete geometry outside the source view.

Research reference: [SpatialSlideshow](https://github.com/elliotttate/SpatialSlideshow),
commit `aa82a5e1eb28d915f524769965f336e7c6dd9a18`. Its MIT notice is preserved in
`src/io/apple_reframe/NOTICE`. The private declarations are source-only linking
adapters, not Apple implementations.

Validation: run `AppleReframeConversion` through CTest and
`tests/python/test_apple_reframe_menu.py` through pytest. Model integration tests
are local opt-in tests with synthetic fixtures; they require an Apple Silicon
Mac, macOS 27+ and installed assets. Check Mach-O deployment versions with
`vtool -show-build` and inspect the OFF package to confirm it contains no helper.
