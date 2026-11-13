---
title: Mask video frames
---

# Mask video frames

LichtFeld Studio uses native SAM2.1 image and video inference to mask subjects in extracted video frames. Choose a subject interactively or use automatic detection for a camera operator. It runs with the project's C++/CUDA tensor library and requires a CUDA GPU.

## Video Extractor

Open **Video Extractor** and enable **Mask operator / person (SAM2.1)** in the **Masking** section after **Resolution**. Masking is off by default. With masking off, extraction writes frames directly into the selected output folder.

With masking enabled, the selected folder becomes a dataset root:

```text
dataset/
  images/
    frame_1.jpg
  masks/
    frame_1.jpg.png
```

The grayscale PNG masks use **0 to exclude** a pixel and **255 to keep** it. Training's existing mask loader recognizes these sidecars. Extraction trim, resolution, rotation, naming, and sharpness settings still apply. Only successfully written frames from the current extraction are masked. Stopping extraction or masking leaves the files already written available for inspection.

Automatic detection compares a grid of image cells between frames one to three seconds apart and looks for textured cells near the image border that keep their content while the surrounding scene moves. It suits a visible camera operator in moving-camera footage. A stationary camera or a person moving independently usually needs **Use manual seed**: enter a point inside the person as normalized output-image coordinates, and the zero-based index of the extracted frame containing that point. The upper-left is `(0, 0)` and the lower-right is `(1, 1)`.

For circular fisheye frames, enable **Fisheye image circle** to exclude pixels outside the centered circle. **Recenter operator** rotates the operator into a virtual fisheye view before tracking and maps the result back to the source image. Set the source **Fisheye FOV** to the lens's full angular field of view and **View FOV** to the virtual view's angular field of view. This is an equidistant approximation for a single circular lens image; split multi-lens recordings into individual lens frames first. The CLI also accepts off-center circles and unequal horizontal/vertical radii.

**Dilation** expands the excluded region by the chosen radius in output-image pixels (default `12`, range `0-128`). **Model file** accepts a local SAM2.1 video `.lfw` path; leave it empty to use the verified download.

If automatic detection cannot localize an operator, the result shows a warning with the reason and writes only the image-circle mask (or keeps every pixel when no circle is enabled). Enable **Require an operator** to fail instead. This also applies when SAM2 cannot confirm the proposed seed. Batch results preserve each video's warning.

Seed boxes, background points, multiple foreground points, and off-center or elliptical circles are available through the CLI.

## Assisted frame preview

After extracting a single video, choose **Mask frames · SAM** to open the frames written by that extraction in Image Preview. The action also remains available for successfully written frames after stopping extraction. Image Preview offers the same tools through its **Mask · SAM2.1** button when opened on dataset images.

In an extracted sequence, the side panel has two sections: **Frame** for points, boxes and proposal generation, and **Sequence** for tracking and approval. Switching sections preserves your edits and the selected frame range. The save actions and status remain visible below the controls.

Tracking follows the extracted sequence and its sampling gaps. The frame status shows the approximate relative time reported by the extractor; it does not represent exact source PTS or analysis of every original video frame.

The **Mask · SAM2.1** button opens an editor in the side panel. Use **Foreground +** for positive points on the object and **Background −** for negative points outside it, or drag a **Box** (bounding box) around the subject. These are visual prompts: they guide segmentation rather than painting or erasing mask pixels. Click or drag an existing point to select or move it. The control list also lets you select points and the box with the keyboard. **Swap +/−** changes the selected point’s label, and **Delete** removes the selected control; the default shortcuts are Delete for the selected control and right-click for the control under the cursor. Removing a point removes a constraint; it does not turn that point into background. Drag a box edge or corner to resize it. In **Box**, drag inside the box to move it; draw outside the existing box to replace it. There is one box per proposal. **Undo** uses the shared history for point, box, and applied-mask changes, including edits made while browsing other images. **Pan** returns to normal pan gestures without adding points. Generate, removal, and mouse shortcuts can be rebound in the keymap preferences; Help and tooltips show the current bindings.

Choose **Generate** and inspect candidates **1**, **2**, and **3**. **Generate** always computes a new proposal from the current points and box. After adding controls, **Refine previous** optionally uses the previously selected proposal as a mask prior. Moving, deleting, or relabeling points, or changing an existing box, discards that prior. **Overlay** hides the proposal for comparison with the original. The displayed quality is SAM's estimate, not verified accuracy. Points and boxes belong to the original image, including when the preview is zoomed or rotated. For selection by description, switch to **Text and categories**, as described below.

Generating or selecting a proposal does not change the image's mask. For an image associated with a camera in the current dataset, choose **Apply: keep selection** or **Apply: exclude selection** to replace its mask at the original image resolution. Apply confirms the change, closes SAM, and updates the image’s mask overlay. Proposals and applied masks use the same theme color; the legend identifies whether the highlighted pixels are a proposal, kept mask pixels, or excluded mask pixels under the current inversion setting. The shared **Undo** and **Redo** actions restore mask changes. Applying requires an editable dataset before training initialization.

Save the project normally to store the PNG mask inside the `.licht` file. Reopening the project and resetting training preserve the image's mask. The source dataset files are preserved. Apply stores the chosen keep/exclude selection; **Invert** is a training setting and does not change the stored PNG. **Segment and Ignore** ignores Invert. Starting or resuming training warns when saved mask assets exist but the mask mode is still None.

Points and boxes are kept per image while browsing the current preview. **Clear points and box** clears these working controls through the shared history. **Release SAM** retains points and boxes and releases the model asynchronously. Close SAM before starting training. Escape first cancels a box gesture or exits SAM mode.

For frames opened directly from a single extraction, **Track subject** propagates the selected proposal over a chosen frame range. Set the first and last frames, keep the current seed inside the range, and choose forward, backward, or both directions. This uses SAM2.1 temporal memory in the finalized extracted-frame order, advancing between these samples rather than decoding the source video again. It does not require camera motion or restrict the subject to the small area expected by automatic operator detection. Video-capable weights are required; the extractor's model file or the verified SAM2.1 cache is reused.

Browse the proposals before saving. Thumbnail marks distinguish proposals (●), low confidence (!), and approved frames (✓); the frame status explains them. Correct any frame with points or a box, generate a new proposal, and track again from it. Manual keyframes and approved frames are protected when propagation crosses them. Editing a frame revokes its approval and preserves any previously saved mask until you explicitly save a replacement. Wait for tracking or saving to finish before editing controls. **Stop tracking** preserves proposals already produced; **Release SAM** also releases GPU resources while keeping the sequence's proposals available for inspection. Proposals have a bounded memory budget and are discarded when this sequence is closed or replaced; there is no persistent session recovery.

Approve frames individually with **Approve this frame**. Under **Save approved masks**, **Save · keep** writes the selected subject as white and **Save · exclude** writes it as black. A confirmation shows the number of approved frames and the extraction output folder. Only approved masks are saved, as original-resolution single-channel binary PNGs in `masks/<image filename>.png` under that folder, using the same naming convention as automatic extraction. Existing masks for approved frames are replaced atomically; other frames remain unchanged. Saving files is an export and cannot be undone. If saving stops or fails, already saved masks remain available and the completion count is shown. A frame without a saved mask remains unmasked.

The filmstrip overlays each visible frame's retained proposal using the same theme color as the main preview. **Overlay** shows or hides both. Approval badges remain visible. These small previews support a quick check; inspect uncertain boundaries in the main image before approving them.

To review many frames, set **First frame** and **Last frame**, then choose **Approve range…**. The confirmation counts masks awaiting approval, those with low model confidence, and frames without a proposal. Missing proposals remain unapproved so you can return to them later. Approval remains an explicit review decision, and one shared **Undo** step restores the entire group. Stale proposals reject the group without partially changing its approvals.

Frames extracted without camera association can therefore save masks directly. Dataset images opened outside an extraction sequence continue to use **Keep selection** or **Exclude selection** on the current camera, with project persistence and shared undo as described above. Automatic operator tracking remains available through the extraction options. Brush painting, independent object tracks and equirectangular projection tools are separate capabilities. Text/category search applies to the current frame; it does not redetect a category throughout the sequence.

## Select objects by text or category

Choose **Text and categories**, then add categories or enter English descriptions,
one per line. Prefer short object descriptions such as `round wooden table`;
spatial relations and complete sentences are not reliably understood. Quick categories include common vehicles, animals, plants and everyday
objects; descriptions let you request other objects. The official LLMDet Tiny
weights are prepared automatically when needed. **Find in this frame**
searches the current frame and asks the existing SAM2.1 model to segment the
detected objects. Each detected object receives one mask-prompt refinement with the
same box and cached image encoding; the initial mask is kept unless the predicted
mask quality increases. The result is one combined mask, which still needs review.
Manual point refinement is available for single-object proposals. Combined semantic
masks must be regenerated when the search changes; their point-refinement action is disabled. The category **People** uses
`person`, the detector's category name: it can match multiple people, not just
one. Descriptions such as `people` are also accepted; wording can change results.

**Advanced options > Detection threshold** applies to all categories and descriptions
in the next search. The default is **40%**; higher values reject more uncertain
detections. Double-click the label to restore the default. Editing search settings
keeps the current proposal until another search is submitted. Lowering the threshold can recover borderline detections, but also admits more false positives.
An absent category searched alone can match a prominent unrelated object; adding
other categories can change its score. Always review the mask, including apparently confident results.

Review the overlay before applying or approving it. **Points and box** returns to
the manual tools in the same editor, where the proposal can be refined. No match
is reported explicitly. A limited result requires further review; detection and
SAM quality estimates are not guarantees of accuracy.

With **No objects found**, **Apply: keep selection** is disabled. **Exclude selection**
remains available and keeps every pixel. An approved empty selection also blocks
**Save · keep** for the sequence; revoke its approval or use **Save · exclude**.
Tracking is also disabled for this empty result.
Use **Generate** with new points or a box to correct an empty result, without an empty mask prior.

On an extracted-frame sequence, **Track subject** propagates the selected mask
through the chosen range. It does not search subsequent frames for new members of
a category. For example, tracking a mask found with `person` does not find a new
person entering the video later. Searching every frame by category is a separate
capability.

Semantic inference uses native CUDA operations; this feature adds no Torch or
ONNX Runtime dependency. The model is loaded only when a semantic search is
requested and is released with SAM.
The fixed BERT uncased vocabulary is included in the installation. Enter up to
16 descriptions within the shared 256-token context; oversized input is rejected
without silently truncating it.


### Automatic semantic model preparation

Models are managed in **Preferences > Models**. The first semantic search downloads
[`llmdet-tiny-v1.lfw`](https://github.com/MrNeRF/LichtFeld-Studio/releases/tag/model-llmdet-tiny-v1)
from the LichtFeld Studio model release and verifies its pinned SHA-256. The
application uses the same managed download, cancellation and atomic publication
path as SAM, MoGe and LPIPS. It does not convert weights or require Python for
model preparation.

The existing model cache is used: `%USERPROFILE%/.lichtfeld/onnx` on Windows and
`~/.lichtfeld/onnx` on Linux. The native archive occupies about 692 MB; the
historical `onnx` directory name also covers native models. No source checkpoint
is downloaded or retained. Offline searches use the installed archive.

A failed or cancelled download does not install a partial model. Download,
integrity verification and removal are also available in Preferences, through
the same cache owner. Removal is blocked while the archive is in use.

### Prepare weights manually

To reproduce the native weights without Torch, obtain the pinned upstream
checkpoint printed by `python tools/nn_export/export_llmdet_weights.py --help`
and run:

```sh
python tools/nn_export/export_llmdet_weights.py model.safetensors --out llmdet-tiny-v1.lfw
```

The exporter uses the Python standard library, verifies the source SHA-256 before
reading checkpoint records, and uses the shared native weight writer. The upstream
[LLMDet project](https://github.com/iSEE-Laboratory/LLMDet) and weights use Apache-2.0.
For offline installation, put the resulting `llmdet-tiny-v1.lfw` in the model cache;
the application verifies its pinned digest before loading it. This developer
export reproduces the archive distributed by the model release; it is not part
of the application's automatic download path.

### Licenses and attribution

The LLMDet project and downloaded weights are declared Apache-2.0. The bundled
BERT vocabulary comes from the same pinned official LLMDet Tiny repository.
The LLMDet and Transformers license texts are included under
`lfs_plugins/assets/bert` and collected by the distribution's existing
third-party notice generator.

The separate `tests/data/nn/semantic_people.png` fixture is a resized frame from
*Tears of Steel*, licensed CC BY 3.0. Its companion `semantic_people.LICENSE.txt`
records the Blender Foundation attribution, source, license and transformation.
This test image retains its own license; it is not installed with the application.

## Command line

Extract and mask a video at one frame per second:

```sh
LichtFeld-Studio mask-video clip.mp4 --output dataset --fps 1
```

Mask an existing dataset's `images/` folder, using a point on a person in extracted frame 12:

```sh
LichtFeld-Studio mask-video dataset --fps 1 --seed-frame 12 --seed 0.5,0.8
```

Existing frames are processed in natural filename order. `--fps` supplies their temporal spacing; use the rate at which they were extracted. Each dataset represents one continuous video sequence. To process a new sequence, use a separate dataset folder.

```sh
LichtFeld-Studio mask-video clip.mp4 --output dataset --start 10 --end 30 \
  --fps 2 --seed 0.45,0.75 --seed-box 0.3,0.55,0.6,0.95
LichtFeld-Studio mask-video dataset --fisheye --recenter --fisheye-fov 200
LichtFeld-Studio mask-video dataset --circle 0.5,0.5,0.48,0.49 --recenter
```

| Option | Default | Meaning |
|---|---|---|
| `--seed x,y[,label]` | Automatic | Normalized point; repeat for multiple points. Label `1` selects the person, `0` excludes a background point. |
| `--seed-frame` | `0` | Zero-based index within extracted frames. |
| `--seed-box x0,y0,x1,y1` | None | Normalized bounds around the person. Without a point, the center becomes a foreground point. |
| `--fisheye` | Off | Centered circle with radii `0.5`. |
| `--circle cx,cy,rx,ry` | Whole image | Normalized elliptical image circle. |
| `--recenter` | Off | Recenter fisheye inference; requires a circle. |
| `--fisheye-fov`, `--view-fov` | `200`, `160` | Source and virtual-view fields of view in degrees. |
| `--dilation` | `12` | Expand excluded pixels by a circular radius in output-image pixels. |
| `--require-operator` | Off | Fail if no operator seed can be confirmed. Otherwise write circle-only masks for sequences without an operator. |
| `--overwrite` | Off | Permit replacement of generated frames and masks. |
| `--model` | Cached weights | Explicit local SAM2.1 video `.lfw` file. |
| `--no-download` | Off | Require weights already available locally. |
| `--download-only` | Off | Download and verify the default weights, then exit. |

## Tracking and memory

The default policy admits a prediction to spatial memory only when the object is present, its predicted IoU is at least `0.5`, and the selected mask occupies at most `40%` of the inference image. It keeps the seed and up to six recent reliable spatial memories, plus a bounded object-pointer history. Small foreground components and holes of at most eight connected pixels in the model’s `256 × 256` mask are removed or filled. Tracking runs forward and backward from the seed. Each run loads the model once in Float32, with a Float16 image encoder, and maintains one sequence of temporal state.

An empty prediction or three consecutive unreliable frames triggers recovery from the last reliable mask and its bounding box. A rejected recovery leaves the existing temporal track intact. Review the generated masks, especially around occlusions, fast motion, very small people, or inaccurate fisheye geometry. Predicted IoU is the model's confidence estimate, not a measured overlap with a ground-truth mask.

Decoded images, cell caches, and pending mask writes are bounded. Frame paths and a few kilobytes of cell statistics per frame remain proportional to the sequence length. Memory admission estimates the model and full-resolution buffers; it does not reserve GPU memory against other applications.

## Weights

The default asset is [`sam2.1-hiera-base-plus-video.lfw`](https://github.com/MrNeRF/LichtFeld-Studio/releases/download/model-sam2-video-v1/sam2.1-hiera-base-plus-video.lfw), converted from the official [SAM2.1 Hiera Base Plus checkpoint](https://github.com/facebookresearch/sam2). SAM2.1 weights are licensed under Apache-2.0. Downloads and cached files are checked against SHA-256:

```text
b4a981065f3d64fe98cd882f28e5bd3a076b39f5f0c696c7885dbe404bc18fed
```

Weights use the existing model cache at `~/.lichtfeld/onnx/`, shared with LPIPS and MoGe. The historical folder name does not indicate a runtime dependency: inference reads `.lfw` directly.
