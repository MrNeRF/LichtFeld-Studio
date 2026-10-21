# Rendered image sequences

`lf.ui.export_image_sequence(width, height, framerate, directory, format="png")`
uses the existing timeline export job, camera scheduling, animated scene
evaluation, progress and cancellation. It writes one image per timeline sample
at `i / framerate`, using the same rounded clip frame count as video export.
The output directory is explicit; this API does not open a file dialog.

Formats are `png`, `jpeg`, `exr`, `tiff16`, `jxl` and `avif`. The existing image
writers and `FileFrameSink` own encoding and atomic publication. There is no
second codec implementation, and a directory of images is not an animated
JXL/AVIF file.

`filename_pattern="frame_%06d"` and `first_frame_number=0` control naming.
The shared extraction formatter is reused, including its append-number behavior
for plain prefixes. Directory components in the pattern are rejected.
`overwrite=False` protects each existing destination with the writer's no-replace
policy. Explicit overwrite applies per image, not as a transaction over the
entire directory. Other files and extraction metadata are not removed.

## Precision and rendering

PNG/JPEG use the existing SDR video render and RGB8 output. EXR/TIFF16/JXL/AVIF
use the existing Gaussian raw linear capture, preserving Float32 samples until
the selected writer's explicit conversion. This capture precedes PPISP and
environment export post-processing. It rejects mesh/environment compositing,
point-cloud mode and non-native reconstruction rather than silently quantizing
or ignoring those features. These are different rendering contracts, not six
interchangeable file extensions for the same processed image.
Equirectangular output follows the existing raster backend capabilities and
currently requires 3DGUT; an unsupported backend fails rather than exporting
perspective pixels under an equirectangular request.

The script defaults are EXR HALF/ZIP, TIFF16 sRGB, JXL Float32 lossless and AVIF
8-bit lossless after the existing SDR conversion. C++ callers can select the
existing writer options through `VideoExportOptions::image_sequence` or the
standalone `TensorImageSequence` adapter. Lossless refers to the writer's input
codes; it does not undo transfer conversion or integer quantization.

`transparent=True` selects straight RGBA for the float capture. The current
PNG/JPEG sink accepts RGB8 only, so transparent PNG/JPEG requests are rejected.
EXR retains its existing premultiplied storage convention. No HDR mastering
contract or auxiliary/depth output is inferred from float storage.

The standalone Tensor adapter requires explicit input color and HWC
Float16/Float32 RGB/RGBA. It handles CPU/GPU and noncontiguous inputs using the
existing synchronized Tensor readback. Linear Float32 output avoids an RGB8
intermediate; explicit sRGB/linear conversion uses the shared transfer curves.
Linear input supports BT709/BT2020 primaries; sRGB input requires BT709.
Declared linear premultiplied alpha and EXR nits/scene-relative units retain the
existing writer contracts. The adapter does not convert premultiplied sRGB input.
Nonfinite input is rejected. The default per-frame readback limit is 1 GiB;
this is not an aggregate encoder RAM or concurrent-job budget.
The shared sink tracks filenames, and the existing timeline job stores sampled
camera states, so bookkeeping grows with the number of frames.

## Lifetime and incomplete output

The sequence session accepts exactly the declared number of ordered frames.
Rendered frames have output timeline timestamps and no synthetic source video
timestamp. A failed write, cancellation or incomplete close cannot report a
complete sequence. Completed image files survive interruption; the current
image is published through the existing atomic writer.
Call session operations serially. Cancellation from another thread uses the
thread-safe job token or caller-supplied cancellation callback; concurrent calls
to `write`, `close` or `abort` on the same session are not supported.
Callback reentry into `open`, `write` or `close` is rejected before the Tensor
adapter can replace its options or readback storage. A callback may abort the
sequence; it cannot reopen it within the active call or publish another frame.

Automatic renderer resume is not provided. Existing image files are never
adopted solely because their names match. The video extraction checkpoint does
not identify scene content, animation, callbacks or render settings; a renderer
resume policy must establish those identities first. No completion manifest or
new GUI selector is introduced here. Common metadata/sidecar publication and the
GUI redesign remain separate work.
