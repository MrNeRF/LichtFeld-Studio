#ifndef LFS_VK_CUSTOM_EFFECT_LAYOUT_INC
#define LFS_VK_CUSTOM_EFFECT_LAYOUT_INC

// Custom effect tail of the overlay parameter table, appended after the legacy
// entries for custom effect draws only. Slots are float4 and relative to the
// tail; integer lanes are bit casts and LFS_VK_CUSTOM_NONE marks an absent
// binding. Macro-only, so the Slang shaders and the C++ renderer and table
// builder (custom_effect_layout.h) share the constants.

#define LFS_VK_CUSTOM_NONE 0xffffffffu

// uint4(object count, binding count, global binding, reveal binding)
#define LFS_VK_CUSTOM_HEADER_SLOT 0u
// float4(time, progress, softness, radius)
#define LFS_VK_CUSTOM_TIMING_SLOT          1u
#define LFS_VK_CUSTOM_BOUNDS_MIN_SLOT      2u
#define LFS_VK_CUSTOM_BOUNDS_MAX_SLOT      3u
#define LFS_VK_CUSTOM_CENTER_SLOT          4u
#define LFS_VK_CUSTOM_CAMERA_POSITION_SLOT 5u
// float4(fx, fy, cx, cy) in full-image pixels
#define LFS_VK_CUSTOM_INTRINSICS_SLOT 6u
#define LFS_VK_CUSTOM_DISTORTION_SLOT 7u
// uint4(width, height, camera model, 0) of the full image
#define LFS_VK_CUSTOM_IMAGE_SLOT 8u
// Four world-to-view rows.
#define LFS_VK_CUSTOM_WORLD_TO_VIEW_SLOT 9u
#define LFS_VK_CUSTOM_HEADER_SLOTS       13u

// The header is followed by each binding's parameter slots, then by the
// effect of the published program set each binding runs, four to a uint4
// slot, then by one object record per scene node slot:
// uint4(node binding, object id, 0, 0).
#define LFS_VK_CUSTOM_PARAMETER_SLOTS 16u

// 3DGUT geometry written by custom projection for the custom raster, per
// render slot: float4(world center, raw opacity) and three float4 rows of the
// world-space inverse scale/rotation.
#define LFS_VK_CUSTOM_GEOMETRY_FLOAT4S 4u

#endif
