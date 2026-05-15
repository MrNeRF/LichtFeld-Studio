/* SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
 *
 * SPDX-License-Identifier: GPL-3.0-or-later */

#version 450

layout(location = 0) in vec3 in_position;
layout(location = 1) in vec3 in_color;

layout(set = 0, binding = 0) readonly buffer ModelTransforms {
    mat4 model_transforms[];
};

layout(set = 0, binding = 1) readonly buffer TransformIndices {
    int transform_indices[];
};

layout(set = 0, binding = 2) readonly buffer Visibility {
    uint visibility_mask[];
};

layout(push_constant) uniform PushConstants {
    mat4 view_proj;
    mat4 view;
    mat4 crop_to_local;
    vec4 crop_min;             // xyz = min, w unused
    vec4 crop_max;             // xyz = max, w unused
    vec4 voxel_focal_ortho;    // x = voxel_size * scaling_modifier, y = focal_y, z = ortho pixels_per_world, w unused
    ivec4 counts;              // x = n_transforms, y = n_visibility, z = flags, w = max_point_size
} pc;

const int FLAG_HAS_CROP        = 1 << 0;
const int FLAG_CROP_INVERSE    = 1 << 1;
const int FLAG_CROP_DESATURATE = 1 << 2;
const int FLAG_ORTHOGRAPHIC    = 1 << 3;
const int FLAG_HAS_INDICES     = 1 << 4;

#define PC_N_TRANSFORMS    pc.counts.x
#define PC_N_VISIBILITY    pc.counts.y
#define PC_FLAGS           pc.counts.z
#define PC_MAX_POINT_SIZE  pc.counts.w

layout(location = 0) out vec3 v_color;

void reject() {
    // Push the vertex outside the NDC cube; gl_PointSize = 0 also nukes any
    // residual fragment coverage from drivers that ignore offscreen positions.
    gl_Position = vec4(2.0, 2.0, 2.0, 1.0);
    gl_PointSize = 0.0;
    v_color = vec3(0.0);
}

void main() {
    int t_idx = 0;
    if (PC_N_TRANSFORMS > 0) {
        if ((PC_FLAGS & FLAG_HAS_INDICES) != 0) {
            t_idx = transform_indices[gl_VertexIndex];
            if (t_idx < 0) {
                t_idx = 0;
            } else if (t_idx >= PC_N_TRANSFORMS) {
                t_idx = PC_N_TRANSFORMS - 1;
            }
        }
        if (PC_N_VISIBILITY > 0 && t_idx < PC_N_VISIBILITY &&
            visibility_mask[t_idx] == 0u) {
            reject();
            return;
        }
    }

    vec3 ws = in_position;
    if (PC_N_TRANSFORMS > 0) {
        mat4 m = model_transforms[t_idx];
        vec4 transformed = m * vec4(in_position, 1.0);
        if (abs(transformed.w) > 1e-6) {
            ws = transformed.xyz / transformed.w;
        } else {
            ws = transformed.xyz;
        }
    }

    bool desaturate = false;
    if ((PC_FLAGS & FLAG_HAS_CROP) != 0) {
        vec3 local = (pc.crop_to_local * vec4(ws, 1.0)).xyz;
        bool inside = all(greaterThanEqual(local, pc.crop_min.xyz)) &&
                      all(lessThanEqual(local, pc.crop_max.xyz));
        bool visible = ((PC_FLAGS & FLAG_CROP_INVERSE) != 0) ? !inside : inside;
        if (!visible) {
            if ((PC_FLAGS & FLAG_CROP_DESATURATE) == 0) {
                reject();
                return;
            }
            desaturate = true;
        }
    }

    vec4 view_pos = pc.view * vec4(ws, 1.0);
    vec4 clip = pc.view_proj * vec4(ws, 1.0);

    if (abs(clip.w) <= 1e-6) {
        reject();
        return;
    }

    bool orthographic = (PC_FLAGS & FLAG_ORTHOGRAPHIC) != 0;
    float view_z_positive = -view_pos.z;
    if (!orthographic && view_z_positive < 1e-4) {
        reject();
        return;
    }

    float voxel = max(pc.voxel_focal_ortho.x, 1e-5);
    float radius_px;
    if (orthographic) {
        float pixels_per_world = max(pc.voxel_focal_ortho.z, 1e-5);
        radius_px = max(1.0, ceil(voxel * pixels_per_world * 0.5));
    } else {
        float focal_y = max(pc.voxel_focal_ortho.y, 1.0);
        radius_px = max(1.0, ceil(voxel * focal_y / max(view_z_positive, 1e-4)));
    }

    float max_size = float(PC_MAX_POINT_SIZE > 0 ? PC_MAX_POINT_SIZE : 64);
    float diameter_px = clamp(2.0 * radius_px, 1.0, max_size);

    vec3 color = clamp(in_color, vec3(0.0), vec3(1.0));
    if (desaturate) {
        float gray = dot(color, vec3(0.299, 0.587, 0.114));
        color = mix(color, vec3(gray), 0.75);
    }

    gl_Position = clip;
    gl_PointSize = diameter_px;
    v_color = color;
}
