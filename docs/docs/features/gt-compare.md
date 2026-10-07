# GT Compare: Fit and 1:1

GT Compare shows a dataset camera's ground-truth image alongside the rendered scene.
Use **Fit** to see the whole image, or **1:1** to inspect fine detail without shrinking
it to the viewport.

## Compare at native resolution

1. Select a dataset camera in the Scene panel and enable GT Compare. The default
   keyboard shortcut is **G**.
2. Choose **RGB** in the GT Compare toolbar.
3. Click **1:1**. Once ready, the image and the rendered scene show the same centered
   crop at the camera's native image scale.
4. Pan with the **Camera Pan** binding, which is **right-button drag** by default.
   Both sides follow the same crop. Drag the comparison divider to inspect either side.
5. Click **1:1** again to return to Fit.

In 1:1, one ground-truth image pixel occupies one **physical framebuffer pixel**.
This uses the display's pixel density, not the logical size of a UI control. Images
larger than the viewport are cropped; smaller images are centered with empty space
around them. Resizing the viewport keeps the crop's center where possible and clamps
it to the image edges. Returning to Fit and entering 1:1 again resets the crop.

The rendered scene uses the corresponding camera calibration and crop, so it zooms
and pans with the ground truth. Both sides use the crop's pixel dimensions, and the
ground-truth side samples exact image texels.

Appearance correction uses the full camera image coordinates for vignetting in 1:1.
Auto mode can still adapt exposure and color to the visible crop as you pan.

## Loading, cancellation, and failures

A spinner in the button and **Preparing 1:1…** below the toolbar appear while the
native image is being prepared. The Fit preview remains interactive. The button
becomes solid blue when the displayed frame is actually using 1:1.

Click **1:1** again to cancel the request. Loading feedback disappears immediately;
the view returns to Fit as the next frame is displayed. Cancellation does not wait
for an image decoder that is already running.

If loading or preparation fails, the toolbar shows **Couldn't prepare 1:1. Retrying…**
and a **Retry** action. Hover over the warning for the detailed reason; it is also
written to the log. Automatic retries have a two-second cooldown. **Retry** clears
the warning and requests another attempt immediately, reusing a decoded image or an
attempt already in progress when possible.

The warning remains visible during automatic attempts until 1:1 succeeds. Cancelling,
changing the camera or comparison context, and explicit Retry also clear it. If the
source file has moved or been deleted, restore access to it before retrying.

## Supported cameras and modes

1:1 is available in **RGB** comparison for pinhole, fisheye, and thin-prism fisheye
cameras with an image source. Distorted cameras also need usable undistortion
calibration. Depth and Normal comparison modes, and equirectangular cameras, do not
support 1:1. Switching away from RGB returns the setting to Fit.

For a distorted camera, the comparison uses the **undistorted image and its
calibration**. The output dimensions and focal lengths can differ from those of the
original distorted image. The 1:1 relationship applies to that native-resolution
undistorted output, not to the original distorted pixel grid.

A disabled 1:1 button explains why the current camera or mode is unavailable. An image
path can exist in the project even when the file cannot be read; that produces loading
feedback and a retryable warning instead.

## Repeated toggles and memory

After a successful load, LFS keeps the current camera's decoded RGB image in CPU
memory when returning to Fit. Repeated Fit/1:1 toggles reuse it and avoid another
file decode. GPU preparation may still be needed.

The retained image uses approximately **3 × width × height bytes**: about **146 MiB**
for an 8972 × 5670 image. Decoder working memory and GPU resources are additional.
Only the current image is retained. Changing the camera or source, leaving RGB GT
Compare, replacing the scene, invalidating the image cache, or closing LFS releases
it. A cancelled load that had not completed is not retained.

## Saved projects

Projects always open in **Fit**. Enable 1:1 when you want to inspect native detail.
The 1:1 toggle is temporary inspection state: it is not saved with the project.
The crop position, decoded image, loading status, and errors are also not saved.

Projects saved with usable undistortion calibration preserve it. For older projects,
LFS reconstructs undistortion from the saved source calibration when available.
If a distorted camera has no usable source calibration, 1:1 may be unavailable.
Reimport the original dataset with this version and save it again to record the
calibration.

Opening and resaving in a version without 1:1 support is not a guaranteed round trip:
it can discard the camera's undistortion metadata. Keep a copy made by a version
with 1:1 support if you need to return to native comparison later. Files also need
to use a container format supported by the version opening them.
