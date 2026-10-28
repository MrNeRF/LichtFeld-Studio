---
title: Mask people in video frames
---

# Mask people in video frames

LichtFeld Studio uses native SAM2.1 image and video inference to exclude a camera operator or a prompted person from extracted video frames. It runs with the project's C++/CUDA tensor library and requires a CUDA GPU.

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
